#!/usr/bin/env python3
"""
Post the V2 runner's PP sampled-token recv when it is consumed, not right after
the forward, so the earlier PP stage's next forward is not stalled behind it.

THE BUG AS IT PRESENTS

On Tioga (MI250X), Inkling-Small-bf16 at TP8 x PP2 over 2 nodes (V2 model
runner, async scheduling), CUDA graphs win at concurrency 1 and then lose to
--enforce-eager from c=4 up. The gap widens with load. 8k in / 1k out, no MTP,
tok/s/user (and tok/s/GPU):

    c       1      4      16     64           256
    eager   16.8   8.3    7.7    7.1 / 199    5.5 / 453
    graphs  19.5   7.5    7.0    4.8 / 132    3.0 / 251

The kernels are not the cause. On 1 node with no PP, graphs are faster than
eager at every batch size (TP8 decode step at B = 1/16/128: 26/32/47 ms with
graphs vs 55/63/66 ms eager).

WHY

With PP and async scheduling, the V2 runner decodes each request every pp_size
steps. Running requests therefore split into pp_size microbatches, and stage 0
should run microbatch B's forward while stage 1 runs microbatch A's. After each
forward, a non-last rank calls PPHandler.receive(), which immediately enqueues
the recv side of the last rank's sampled-token broadcast on a side stream
(broadcast_stream, own "pp_broadcast" communicator). The data is only consumed
pp_size steps later, in get_prev_sampled_outputs().

That recv is an RCCL kernel. It starts as soon as stage 0's forward k is done
and then spins on the GPU until stage 1 finishes step k. On gfx90a the main
stream's next forward (k+1) slows to a fraction of its speed while it spins,
even though there is no data dependency between them. Most of stage 0's
forward k+1 therefore runs after stage 1 has finished step k. The stages
largely serialize, and a two-microbatch decode step costs close to S0 + S1
per microbatch instead of max(S0, S1).

Eager hides this because each forward is bound by CPU launch time (~30 ms
per stage in the bench below), which is longer than the stall. Graphs remove
the CPU bound and expose the serialization. In the bench below, eager alt is
61.6 ms at B=4 before this patch and 63.9 ms after.

Evidence: 1 node, TP4 x PP2, 21 layers, dummy weights, graphs. The figure is
the per-token interval. "same" puts all requests in one microbatch; "alt"
staggers them into two.

    B=4   same 36.5 ms   alt 60.4 ms
    B=16  same 35.6 ms   alt 80.9 ms

A pipelined alt should be no slower than same. A per-rank CUDA-event timeline
shows why it is:
  * Stage 1's forward runs 74.3 -> 89.4 ms, and its sampled-token broadcast
    completes at 90.1.
  * Stage 0's forward k+1 starts at 74.3 and needs ~19.5 ms on its own.
    It does not finish until 103.9, ~14 ms after the broadcast it was stuck
    behind.
  * While the recv spins, stage 0's forward runs at ~40% speed.
GPU_MAX_HW_QUEUES does not help: 8 makes it worse (alt 78.3 ms at B=4) and 2
does nothing (71.1 ms).

With this patch, same job shape (gbench jobs f61hzyNNL4WT before and
f61iFWwXYAmD after; graphs, alt):

    B=4   60.4 -> 34.6 ms
    B=16  80.9 -> 40.4 ms
    B=32          36.3 ms   (same: 39.3 ms)

On the 2-node setup above, now with MTP 3 and inkling_single_pass_kv_bounds.py,
graphs go from losing to eager at every
rung to beating it from c=4 to c=64. tok/s/user, graphs before -> after (eager):

    c=4   14.4 -> 25.7  (16.0)
    c=16   9.5 -> 21.7  (14.7)
    c=64   6.8 -> 15.6  (12.4)

Full table in docs/MI250X_PORTABILITY.md §9.1.

THE FIX

receive() now records only what the slot needs: num_reqs, idx_mapping,
need_sampled_mask and the generation snapshot. get_prev_sampled_outputs()
posts the same broadcast recvs when it pops the slot pp_size steps later. They
are posted on broadcast_stream after wait_stream(main_stream), so on the GPU
they start once forward k+1 is done. By then stage 1 has normally finished
step k and the data is already waiting.

Ordering is unchanged. Slots are posted in FIFO order, each with the same
broadcasts in the same order as before. A slot is posted even if every entry
is excluded (freed or not needing a sample), because the last rank's sends
must still be matched. The last rank's sends (a few KB) complete into RCCL's
connection buffers before the recv is posted. If a send did have to wait, it
would block only the sender's broadcast_stream, and nothing there waits on it
until this rank posts the recv. Steps that schedule nothing still pop a slot,
as before. Consumers see exactly what they saw before, a PendingRecv.

BLAST RADIUS

Only non-last PP ranks of the V2 model runner (PP > 1). At PP = 1 there is no
PPHandler. The sampled tokens, counts and drafts received are bit-identical;
only when the recv kernel is enqueued changes.

TURNING IT OFF: ZENITH_PP_DEFERRED_RECV=0

On Lux (MI355X, RCCL-ANP between nodes), MTP + PP2 hangs in the warmup decode
steps with this patch in place: stage 0's hidden-state send is never matched
and the RCCL watchdog fires after 600 s. GPU_MAX_HW_QUEUES=8 does not change
that. MTP-off PP2 on Lux and MTP + PP2 on Tioga run fine with it. The image
bakes this patch in, so with ZENITH_PP_DEFERRED_RECV=0 this script undoes it
instead (stock recv-after-forward). zenith_tuned_defaults.sh sets that for MTP
on 2 nodes. The price is the stage serialization described above: on Lux,
MTP-3 at TP8 x PP2 then scores 43.4 against 51.5 for MTP off, winning only
at c=1 (docs/LUX_HACKATHON.md section 6).

UPSTREAM

Not reported yet (vLLM bc0f47cd).
"""
import argparse
import importlib.util
import os
import sys

TARGET = "vllm/v1/worker/gpu/pp_utils.py"
MARKER = "zenith: deferred PP sampled-token recv"

OLD_DATACLASS = '''def compute_need_sampled_mask(input_batch: InputBatch) -> np.ndarray | None:
'''

NEW_DATACLASS = '''@dataclass
class _ZenithDeferredRecv:
    """{marker} (pp_broadcast_deferred_recv.py):
    what receive() knows at step T; the recvs are posted at step T+pp_size."""

    num_reqs: int
    idx_mapping: torch.Tensor  # [num_reqs]
    idx_mapping_np: np.ndarray  # [num_reqs]
    need_sampled_mask: np.ndarray  # [num_reqs]
    gen_at_receive_np: np.ndarray  # [num_reqs]


def compute_need_sampled_mask(input_batch: InputBatch) -> np.ndarray | None:
'''.format(marker=MARKER)

OLD_CONSUME = '''        if slot is None:
            return None

        # Skip requests which did not need sampled output and/or those already
'''

NEW_CONSUME = '''        if slot is None:
            return None
        # zenith: post the deferred recv now, even if every entry ends up
        # excluded below -- the last rank's broadcast must still be matched.
        slot = self._zenith_post_recv(slot)

        # Skip requests which did not need sampled output and/or those already
'''

OLD_RECEIVE = '''        num_reqs = input_batch.num_reqs
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            sampled_tokens = torch.empty(
'''

NEW_RECEIVE = '''        # zenith: do not post the recv here. Posted right after this rank's
        # forward, the RCCL recv kernel spins until the last rank finishes
        # this step and stalls this rank's next forward on the main stream
        # (gfx90a), serializing the PP stages. get_prev_sampled_outputs()
        # posts it pp_size steps later (pp_broadcast_deferred_recv.py).
        self.queue[-1] = _ZenithDeferredRecv(
            input_batch.num_reqs,
            input_batch.idx_mapping,
            input_batch.idx_mapping_np,
            need_sampled_mask,
            gen_at_receive_np,
        )
        return bool(need_sampled_mask.all())

    def _zenith_post_recv(self, d: _ZenithDeferredRecv) -> PendingRecv:
        num_reqs = d.num_reqs
        with torch.cuda.stream(self.broadcast_stream):
            # Ordered after the forward already queued on the main stream, so
            # the recv kernel never runs alongside it.
            self.broadcast_stream.wait_stream(self.main_stream)
            sampled_tokens = torch.empty(
'''

OLD_PENDING = '''        self.queue[-1] = PendingRecv(
            event,
            sampled_tokens,
            num_sampled,
            num_rejected,
            input_batch.idx_mapping,
            input_batch.idx_mapping_np,
            need_sampled_mask,
            gen_at_receive_np,
            draft_tokens,
        )
        return bool(need_sampled_mask.all())
'''

NEW_PENDING = '''        return PendingRecv(
            event,
            sampled_tokens,
            num_sampled,
            num_rejected,
            d.idx_mapping,
            d.idx_mapping_np,
            d.need_sampled_mask,
            d.gen_at_receive_np,
            draft_tokens,
        )
'''

SITES = [
    ("deferred-slot dataclass", OLD_DATACLASS, NEW_DATACLASS),
    ("post at consume (get_prev_sampled_outputs)", OLD_CONSUME, NEW_CONSUME),
    ("receive() records instead of posting", OLD_RECEIVE, NEW_RECEIVE),
    ("posted slot returned as PendingRecv", OLD_PENDING, NEW_PENDING),
]


def find_target():
    """Locate pp_utils.py inside whichever site-packages vLLM lives in.

    Deliberately does NOT `import vllm` (see rocm_aiter_ray_env_refresh.py):
    importing it touches the GPU at module scope and fails in a build container.
    """
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.origin:
        sys.exit("vllm is not installed; run this inside the container")
    root = os.path.dirname(os.path.dirname(os.path.abspath(spec.origin)))
    path = os.path.join(root, TARGET)
    if not os.path.exists(path):
        sys.exit(f"expected {path} to exist; vLLM layout changed")
    return path


def undo(path, src, check):
    """ZENITH_PP_DEFERRED_RECV=0: restore the stock recv-after-forward."""
    if MARKER not in src:
        print(f"deferred PP recv off (ZENITH_PP_DEFERRED_RECV=0): {path} is stock")
        return 0
    stock = src
    for name, old, new in reversed(SITES):
        n = stock.count(new)
        if n != 1:
            print(f"ERROR: cannot undo '{name}' in {path} (matched {n}x, "
                  "expected 1)", file=sys.stderr)
            return 1
        stock = stock.replace(new, old, 1)
    if check:
        print(f"would undo: {path}")
        return 0
    with open(path, "w") as fh:
        fh.write(stock)
    print(f"undone: {path}  (ZENITH_PP_DEFERRED_RECV=0; recv posted right "
          "after the forward again)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report status without writing")
    ap.add_argument("--path", help="patch this file instead of the installed "
                    "vLLM's (e.g. a copy bind-mounted over the image file)")
    args = ap.parse_args()

    path = args.path or find_target()
    with open(path) as fh:
        src = fh.read()

    if os.environ.get("ZENITH_PP_DEFERRED_RECV", "1") == "0":
        return undo(path, src, args.check)

    if MARKER in src:
        print(f"already patched: {path}")
        return 0

    patched = src
    problems = []
    for name, old, new in SITES:
        n = patched.count(old)
        if n != 1:
            problems.append(f"{name} (matched {n}x, expected 1)")
            continue
        patched = patched.replace(old, new, 1)

    # All four or none. Deferring the recv without posting it at consume time
    # leaves the last rank's broadcast unmatched and the PP group hangs.
    if problems:
        print(f"ERROR: patch target not found in {path}:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        print("vLLM changed PPHandler; check whether upstream already defers "
              "the recv and re-derive or retire this patch", file=sys.stderr)
        return 1

    if args.check:
        print(f"would patch: {path}")
        return 0

    with open(path, "w") as fh:
        fh.write(patched)
    print(f"patched: {path}  (PP sampled-token recv posted at consume time)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
