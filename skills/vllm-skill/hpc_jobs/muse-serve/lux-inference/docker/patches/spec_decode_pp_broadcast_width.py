#!/usr/bin/env python3
"""
Make the PP+async sampled-token broadcast carry the *right* token under MTP.

THE BUG AS IT PRESENTS

With async scheduling on (vLLM's default for MTP), the last PP stage
broadcasts its sampled tokens back to the earlier stages so they can build the
next step's input_ids. The sender asserts a width of 1:

    gpu_model_runner.py:4864  (_pp_broadcast_prev_sampled_token_ids)
        assert sampled_token_ids.dim() == 2 and sampled_token_ids.shape[-1] == 1

MTP makes the rejection sampler return [num_reqs, 1 + num_spec_tokens], so the
assert fires on the first decode step and the engine dies with EngineDeadError
(TP8 PP2 on two Lux nodes). Turning async scheduling *off* to dodge it lands
in a worse bug -- a closed-cycle deadlock in post_step -> take_draft_token_ids
-- so the broadcast is the path that has to work.

WHY THE OBVIOUS FIX (JUST WIDEN IT) IS WRONG

The tempting patch is to relax the assert and widen the receive buffer to
[num_reqs, 1+num_spec], since every consumer in _prepare_input_ids indexes
column 0 and would happily ignore the extra columns. That patch runs, serves
tokens, and produces subtly wrong output. Here is why.

Column 0 of those two tensors does not mean the same thing:

  * The broadcast at the original call site (line 4621) sends the raw
    `sampler_output.sampled_token_ids`, shape (num_reqs, num_spec+1), where
    rejected positions are -1. Column 0 is the FIRST accepted token of the
    step.

  * What every consumer of `prev_sampled_token_ids` actually expects is the
    LAST accepted token -- the one that becomes the next step's input. On the
    single-rank path that tensor is built at line 4984 as
    `next_token_ids.unsqueeze(1)`, and next_token_ids comes from
    `prepare_next_token_ids_padded` (llm_base_proposer.py:1052), which returns
    shape [batch_size] -- exactly one token per request, chosen by a Triton
    kernel that walks the row to the last non-rejected entry and substitutes a
    backup token for discarded requests.

So under MTP with an acceptance rate above zero, "widen it" feeds earlier PP
stages the first accepted token where the last accepted one belongs. Every
request that accepts more than one token per step gets its input_ids rewound.
That is silent corruption in the generated text, and it would have sailed
through a throughput benchmark looking like a clean 2-node MTP win. Numbers
would have been reported from it.

THE ACTUAL FIX

Broadcast the *reduced* tensor instead of the raw one, reducing it in place at
the existing call site.

The wire format therefore stays [num_reqs, 1] in every configuration, which is
why this patch does not touch the receiver, the recv buffer, or any of the
`[..., 0]` consumer indexings -- they were never the problem. Column 0 keeps
its one true meaning (last accepted token) on both the PP and non-PP paths.

The reduction is a gather, and it needs nothing but `sampler_output`:

    last  = (tok != -1).sum(dim=1, keepdim=True) - 1
    prev  = tok.gather(1, last.clamp_(min=0))

Rejection sampling accepts a prefix and marks every position after the first
rejection -1, so the count of non-(-1) entries minus one *is* the index of the
last accepted token. That reproduces what the `_eagle_prepare_next_token_padded`
Triton kernel computes, for the one column the broadcast carries.

WHY IT IS COMPUTED HERE AND NOT AFTER THE DRAFTER

An earlier version of this patch moved the broadcast to just before
`bookkeep`, so it could reuse the `prev_sampled_token_ids` that the drafter
block leaves behind. That ran -- it cleared the assert and the server came up
and answered /v1/models -- and then deadlocked on the first generate with
`TimeoutError: RPC call to sample_tokens timed out` (job 46977).

Moving a collective is not free. The drafter between the old and new positions
runs its own model forward carrying TP/EP collectives, so relocating the
broadcast past it reorders this rank's collective stream against the other PP
ranks', which issue their receive at the top of sample_tokens and never enter
the drafter at all (non-last ranks have no drafter -- bug #1). Reducing in
place keeps the broadcast exactly where upstream tested it: same position in
the step, same order relative to every other collective. Only the *contents*
of the message change, which is all that was ever wrong.

The drafter's own `prev_sampled_token_ids` is also cleared at line 4652 --
`self.input_batch.prev_sampled_token_ids = None` -- between the old call site
and the intended new one, so the relocated read was relying on the drafter
block to repopulate it, which is exactly the coupling that made the ordering
fragile. The gather has no such dependency.

Both sides still issue exactly one broadcast per step, on the same group, at
the same point. Collective pairing and ordering are untouched.

THE SECOND GAP THIS HAS TO CLOSE

Moving the broadcast is necessary but not sufficient. Non-last PP ranks have
no drafter (that is bug #1, handled by spec_decode_drafter_pp_guard.py), so
`self._draft_token_ids` is None there, and _prepare_input_ids bails out of the
draft-token scatter at line 1902:

    if self._draft_token_ids is None or not spec_flattened_indices:
        return

On an all-decode batch the preceding guard at 1866 computes
`num_common_tokens == total_without_spec`, so it does NOT copy input_ids_cpu to
the GPU -- the code assumes the GPU scatter will cover everything. With the
scatter skipped, the draft-token positions in input_ids.gpu keep whatever was
left there from the previous step. Wrong tokens, no error. Forcing the CPU copy
on exactly those ranks makes the contents deterministic instead of stale.

THE THIRD GAP -- WHY THE FIRST TWO FIXES STILL GAVE ZERO ACCEPTANCE

With sites 1 and 2 applied the engine came up, served /v1/models and actually
generated -- and then reported, every step (job 46979):

    Mean acceptance length: 1.00, Accepted: 0 tokens, Drafted: 11 tokens,
    Avg Draft acceptance rate: 0.0%

Nothing was ever accepted, and the step rate then fell to zero and
sample_tokens timed out. The reason is that forcing the CPU copy does not
actually supply the draft tokens, because under async scheduling *the scheduler
does not have them*. engine/core.py:640 gates the hand-off:

    if self.check_for_draft_tokens and not self.async_scheduling and model_executed:
        draft_token_ids = self.model_executor.take_draft_token_ids()

So with async on, `scheduled_spec_decode_tokens` carries a placeholder, exactly
as 46979's input dump shows:

    scheduled_spec_decode_tokens={cmpl-...: [-1]}, num_spec_tokens_to_schedule=1

The real draft token is meant to be patched into input_ids.gpu by the worker
itself, from worker-local GPU state, in the scatter at line 1917. That scatter
reads `self._draft_token_ids` -- which only the rank that ran the drafter has.
On PP>1 the drafting rank (last) and the rank that builds the next step's
input_ids (first) are different processes, and nothing carries the tensor
between them. Rank 0 therefore runs the target forward with -1 in every
speculative position, rank 1's rejection sampler compares its own draft against
that, and rejects unconditionally. Acceptance is pinned at zero by
construction.

This is invisible at PP=1 because there the drafting rank *is* the rank that
prepares inputs. PP splits the two roles apart and no message was ever defined
to reconnect them.

The fix is to broadcast the draft tokens too -- a second PP collective mirroring
the sampled-token one that already exists. Sites 4 and 5 add the send (on the
last rank, right after the spec block has settled `_draft_token_ids`) and the
receive (on the earlier ranks, immediately after the existing recv).

ORDERING, AGAIN

Site 4 *adds* a collective at the end of the sender's spec block rather than
moving one, so the PP-group issue order is B1,B2 on the sender and R1,R2 on the
receiver -- matched pairwise, in order. The drafter's own collectives ride the
TP/EP group, a different communicator, so they cannot interleave with these on
the PP communicator. That is the distinction from the failed relocation
attempt: reordering an existing collective against a foreign group is what
deadlocked 46977; appending a matched pair to one group is not.

Both broadcasts share the `_is_all_reqs_chunked_prefill()` gate, so the two
sides always agree on whether a message exists at all. The draft tensor is cast
to int32 and made contiguous before sending: the not-fitting branch builds
`_draft_token_ids` via `torch.zeros(1).expand(...)`, which is a zero-stride
view that NCCL cannot send.

Receiver shape is [num_reqs, num_spec_tokens], matching what the scatter at
line 1917 expects to flatten, and `prev_num_spec_tokens` is set alongside it so
the `prev_index * self.prev_num_spec_tokens` arithmetic at line 1848 indexes
the received tensor correctly -- on the sender that field is maintained by
_copy_draft_token_ids_to_cpu (line 4906), which drafter-less ranks never reach.

BLAST RADIUS

PP>1 with async scheduling and spec decode. Nothing else.

  * PP=1: the sender is unreachable (the call site is guarded by
    `pp.world_size > 1 and pp.is_last_rank`) and the receiver is guarded by
    `not pp.is_last_rank`. Every single-node number stands.
  * PP>1 without spec decode: `self.num_spec_tokens` is 0, the original call
    site keeps firing exactly where and as it did, and the 1866 guard's added
    clause is False because no draft tokens are scheduled. Bit-identical to
    upstream -- this matters, because the 2-node TP8/PP2 MTP-off curve is a
    published control.
  * PP>1 with MTP: previously EngineDeadError on step 1.

UPSTREAM

Build 0.2.1.dev32+gbc0f47cd0. Worth reporting with the drafter-guard bug; both
show PP+spec-decode was wired but never exercised end to end.
"""
import argparse
import os
import sys

TARGET = "vllm/v1/worker/gpu_model_runner.py"
MARKER = "zenith: MTP-aware PP broadcast"

# ---------------------------------------------------------------- site 1
# Reduce to the last accepted token in place, at the original call site.
# The broadcast does NOT move -- see "WHY IT IS COMPUTED HERE" in the module
# docstring; moving it past the drafter deadlocked sample_tokens (job 46977).
OLD_CALLSITE = '''        if self.use_async_scheduling:
            pp = get_pp_group()
            # For torchrun external_launcher PP mode with broadcast_pp_output=True,
            # PP outputs have been broadcasted to all ranks at logits computation.
            # Therefore, here is no need to send sampled token ids again in this case.
            if not self.broadcast_pp_output and pp.world_size > 1 and pp.is_last_rank:
                self._pp_broadcast_prev_sampled_token_ids(
                    sampler_output.sampled_token_ids
                )
'''

NEW_CALLSITE = '''        if self.use_async_scheduling:
            pp = get_pp_group()
            # For torchrun external_launcher PP mode with broadcast_pp_output=True,
            # PP outputs have been broadcasted to all ranks at logits computation.
            # Therefore, here is no need to send sampled token ids again in this case.
            if not self.broadcast_pp_output and pp.world_size > 1 and pp.is_last_rank:
                # {marker}: with spec decode on, sampled_token_ids is
                # [num_reqs, 1+num_spec] and its column 0 is the FIRST accepted
                # token -- but prev_sampled_token_ids must hold the LAST one,
                # which is what every [..., 0] consumer reads. Broadcasting the
                # raw tensor rewinds input_ids on any request that accepts more
                # than one token: silent corruption, not a crash. Reduce here,
                # in place, so the collective keeps its original position in the
                # step (relocating it past the drafter deadlocks sample_tokens).
                _send = sampler_output.sampled_token_ids
                if self.num_spec_tokens and _send.shape[-1] > 1:
                    # Rejection sampling accepts a prefix and marks the rest -1,
                    # so (count of non-(-1)) - 1 indexes the last accepted token.
                    _last = (_send != -1).sum(dim=1, keepdim=True) - 1
                    _send = _send.gather(1, _last.clamp_(min=0))
                self._pp_broadcast_prev_sampled_token_ids(_send)
'''.format(marker=MARKER)

# ---------------------------------------------------------------- site 3
# Non-last PP ranks have no drafter, so they cannot scatter draft tokens from
# the GPU tensor. Force the CPU copy that carries them instead.
OLD_COPY = '''        if num_common_tokens < total_without_spec:
            # If not all requests are decodes from the last iteration,
            # we need to copy the input_ids_cpu to the GPU first.
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
            if self.enable_prompt_embeds:
                self.inputs_embeds.copy_to_gpu(total_num_scheduled_tokens)
'''

NEW_COPY = '''        # {marker}: the second clause covers non-last PP ranks, which have no
        # drafter and so leave self._draft_token_ids None. Without it, the draft
        # scatter below returns early (line ~1902) while this guard has already
        # decided the CPU copy was unnecessary, leaving stale tokens at every
        # draft position. The scheduler already put the correct draft tokens in
        # input_ids_cpu (scheduled_spec_decode_tokens), so copying is sufficient;
        # the sampled-token scatter that follows fixes the one stale position
        # per request.
        if num_common_tokens < total_without_spec or (
            total_num_spec_tokens > 0 and self._draft_token_ids is None
        ):
            # If not all requests are decodes from the last iteration,
            # we need to copy the input_ids_cpu to the GPU first.
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
            if self.enable_prompt_embeds:
                self.inputs_embeds.copy_to_gpu(total_num_scheduled_tokens)
'''.format(marker=MARKER)

# ---------------------------------------------------------------- site 4
# The sender's assert now states the real invariant: width 1, always.
OLD_ASSERT = '''        # `prev_sampled_token_ids` is expected to have shape [num_reqs, 1].
        assert sampled_token_ids.dim() == 2 and sampled_token_ids.shape[-1] == 1, (
            "PP+async expects sampled_token_ids to have shape [num_reqs, 1]"
        )
'''

NEW_ASSERT = '''        # `prev_sampled_token_ids` is expected to have shape [num_reqs, 1].
        # {marker}: this assert is deliberately NOT relaxed. Callers now reduce
        # the spec-decode output to the last accepted token before calling, so
        # width 1 remains the true invariant on every path. Widening it here
        # would let the raw [num_reqs, 1+num_spec] tensor through, whose column
        # 0 is the first accepted token -- silently wrong, not loudly broken.
        assert sampled_token_ids.dim() == 2 and sampled_token_ids.shape[-1] == 1, (
            "PP+async expects sampled_token_ids to have shape [num_reqs, 1], got "
            f"{{tuple(sampled_token_ids.shape)}}"
        )
'''.format(marker=MARKER)

# ---------------------------------------------------------------- site 4
# Send the draft tokens from the last PP rank, which is the only rank that has
# a drafter. Placed at the end of the spec block, after every branch that can
# assign self._draft_token_ids has run, and before bookkeep.
OLD_DRAFT_SEND = '''                self._draft_probs = None
                self._draft_prob_req_ids = None
                self._copy_draft_token_ids_to_cpu(scheduler_output, zeros_only=True)

        with record_function_or_nullcontext("gpu_model_runner: bookkeep"):
'''

NEW_DRAFT_SEND = '''                self._draft_probs = None
                self._draft_prob_req_ids = None
                self._copy_draft_token_ids_to_cpu(scheduler_output, zeros_only=True)

            # {marker}: under async scheduling the scheduler never carries draft
            # tokens (engine/core.py gates take_draft_token_ids on
            # `not async_scheduling`), so they live only in this rank's
            # _draft_token_ids. At PP>1 the drafting rank and the rank that
            # builds the next step's input_ids are different processes, and
            # without this send the earlier ranks run the target forward with -1
            # in every speculative slot -- acceptance pinned at 0%.
            self._pp_broadcast_draft_token_ids()

        with record_function_or_nullcontext("gpu_model_runner: bookkeep"):
'''.format(marker=MARKER)

# ---------------------------------------------------------------- site 5
# Receive the draft tokens on the earlier PP ranks, and define both halves of
# the new collective next to the sampled-token pair they mirror.
OLD_DRAFT_RECV = '''            # receive sampled token ids from the last PP rank.
            if self.use_async_scheduling and not get_pp_group().is_last_rank:
                self._pp_receive_prev_sampled_token_ids_to_input_batch()
'''

NEW_DRAFT_RECV = '''            # receive sampled token ids from the last PP rank.
            if self.use_async_scheduling and not get_pp_group().is_last_rank:
                self._pp_receive_prev_sampled_token_ids_to_input_batch()
                # {marker}: and the draft tokens, which this rank cannot
                # produce (no drafter) and the scheduler does not carry.
                self._pp_receive_draft_token_ids()
'''.format(marker=MARKER)

# ---------------------------------------------------------------- site 6
# The two new methods, defined immediately after the sampled-token pair.
OLD_METHODS = '''    def take_draft_token_ids(self) -> DraftTokenIds | None:
'''

NEW_METHODS = '''    def _pp_broadcast_draft_token_ids(self) -> None:
        """{marker}: send this step's draft tokens to the earlier PP stages.

        Only the last PP rank has a drafter, and under async scheduling the
        scheduler does not relay draft tokens, so this is the only way the
        ranks that build input_ids can learn what was drafted.
        """
        pp = get_pp_group()
        if (
            not self.use_async_scheduling
            or self.broadcast_pp_output
            or pp.world_size <= 1
            or not pp.is_last_rank
            or not self.num_spec_tokens
        ):
            return
        # Gated identically to the sampled-token broadcast so both sides always
        # agree on whether a message exists this step.
        if self._is_all_reqs_chunked_prefill():
            return
        num_reqs = self.input_batch.num_reqs
        draft = self._draft_token_ids
        if not torch.is_tensor(draft):
            # The drafter was skipped entirely this step (or returned CPU
            # lists); send zeros so the receive still has a partner.
            send = torch.zeros(
                (num_reqs, self.num_spec_tokens),
                dtype=torch.int32,
                device=self.device,
            )
        else:
            # .contiguous() matters: the not-fitting branch builds this with
            # torch.zeros(1).expand(...), a zero-stride view NCCL cannot send.
            send = draft.to(dtype=torch.int32).contiguous()
        torch.distributed.broadcast(send, src=pp.rank, group=pp.device_group)

    def _pp_receive_draft_token_ids(self) -> None:
        """{marker}: receive the draft tokens broadcast by the last PP stage."""
        pp = get_pp_group()
        if not self.num_spec_tokens or self.broadcast_pp_output:
            return
        if self._is_all_reqs_chunked_prefill():
            return
        num_reqs = self.input_batch.num_reqs
        recv = torch.empty(
            (num_reqs, self.num_spec_tokens), dtype=torch.int32, device=self.device
        )
        torch.distributed.broadcast(recv, src=pp.last_rank, group=pp.device_group)
        self._draft_token_ids = recv
        # _prepare_input_ids indexes the flattened tensor as
        # `prev_index * self.prev_num_spec_tokens`. On the sender that field is
        # maintained by _copy_draft_token_ids_to_cpu, which this rank never
        # reaches, so set it here or the scatter reads the wrong columns.
        self.prev_num_spec_tokens = self.num_spec_tokens

    def take_draft_token_ids(self) -> DraftTokenIds | None:
'''.format(marker=MARKER)

SITES = [
    ("reduce to last accepted token at the call site", OLD_CALLSITE, NEW_CALLSITE),
    ("input_ids CPU copy for drafter-less PP ranks", OLD_COPY, NEW_COPY),
    ("sender assert message", OLD_ASSERT, NEW_ASSERT),
    ("draft-token broadcast send", OLD_DRAFT_SEND, NEW_DRAFT_SEND),
    ("draft-token broadcast receive", OLD_DRAFT_RECV, NEW_DRAFT_RECV),
    ("draft-token broadcast methods", OLD_METHODS, NEW_METHODS),
]


def find_target(explicit=None):
    if explicit:
        return explicit
    try:
        import vllm
        path = os.path.join(os.path.dirname(vllm.__file__),
                            TARGET.split("vllm/", 1)[1])
        if os.path.exists(path):
            return path
    except Exception:
        pass
    for root in sys.path:
        cand = os.path.join(root, TARGET)
        if os.path.exists(cand):
            return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None)
    # --check is what the Dockerfile's patch loop passes to every script.
    ap.add_argument("--dry-run", "--check", dest="dry_run", action="store_true")
    args = ap.parse_args()

    path = find_target(args.file)
    if not path:
        print(f"zenith: {TARGET} not found; nothing to patch")
        return 0

    with open(path) as fh:
        src = fh.read()

    if MARKER in src:
        print(f"zenith: {os.path.basename(path)} already has the MTP-aware PP "
              f"broadcast fix; no change")
        return 0

    patched = src
    problems = []
    for name, old, new in SITES:
        n = patched.count(old)
        if n != 1:
            problems.append(f"{name} (matched {n}x, expected 1)")
            continue
        patched = patched.replace(old, new, 1)

    # All six or none. Applying the assert hunk without the reduction hunk
    # would leave the engine asserting exactly as it does unpatched; applying
    # the reduction without the drafter-less CPU copy would serve stale draft
    # tokens on the earlier PP ranks; and applying either half of the
    # draft-token broadcast without the other hangs the PP group on an
    # unmatched collective. Every partial application is worse than none.
    if problems:
        print("zenith: WARNING -- anchor mismatch: " + "; ".join(problems))
        print("zenith: refusing to apply a partial patch (a half-applied "
              "broadcast fix corrupts decode instead of crashing)")
        return 1

    if args.dry_run:
        print(f"zenith: would patch {path} ({len(SITES)} sites)")
        return 0

    with open(path, "w") as fh:
        fh.write(patched)
    print(f"zenith: patched {path} -- PP+async broadcast is MTP-aware "
          f"({len(SITES)} sites)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
