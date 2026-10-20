#!/usr/bin/env python3
"""
Give Inkling `compute_logits_local` so vLLM's batch-sharded sampling can run.

Why this patch exists
---------------------
At c512 decode with MTP-1, the last PP stage's sample step costs ~5 ms of
its loop on MI250X: the lm_head GEMM (~1.1 ms), the TP all-gather of the
[T, 25128] logits shards into [T, 201024] on every rank (~1.5 ms), the
transpose copy that all-gather leaves behind (~0.55 ms), then the sampler
over all 2 x 256 rows, replicated on all 8 TP ranks (~2.3 ms).

The V2 model runner already has the fix: `--enable-batch-sharded-sampling`
(vllm/v1/worker/gpu/sample/batch_shard.py). Each TP rank computes its
vocab-shard logits for all rows, an all-to-all hands every rank the full-vocab
logits of 1/TP of the requests, each rank samples only those, and a small
all-gather of the sampled tokens restores the batch. The all-to-all moves
1/TP of the all-gather's bytes and the sampler sees 1/TP of the rows.

The runner only turns it on for models that implement
`compute_logits_local(hidden_states)` (the logits before the TP gather);
otherwise it logs "Disabling batch-sharded sampling: ... does not implement
compute_logits_local" and carries on as before. Inkling doesn't implement it.

The fix
-------
Add it to _TmlForCausalLMBase, the same one-liner as Llama's:

    def compute_logits_local(self, hidden_states):
        return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)

InklingLogitsProcessor already honours skip_gather on both of its paths (the
muP addmm returns before `_gather_logits`). The local logits are exactly the
shard the all-gather would have concatenated (the same addmm, alpha=1/mup,
same M, N and K, rows only permuted into owner order), the all-to-all
concatenates the shards in rank order as the all-gather does, and the runner
slices the vocab padding off with `[:, :vocab_size]` as compute_logits does;
Gumbel noise and rejection-sampling randomness are keyed per request (seed,
position, token id), not by batch slot. Sampled tokens should therefore be
bit-identical, but that could not be shown: in a 384-request temperature-0.7
test (21 layers, MTP-1, graphs) the flag changed tokens in every request
(f61rRSiuU3wd), and two runs with the flag off differ just as much
(f61rMLTWncKh). Two 2-node ladders passed the correctness gate.

Measured (TP8 x PP2, MTP-1, 16384 budget, dense capture sizes): the last
stage's sample step at 256 requests goes from 7.4 to 4.1 ms, and the envelope
score from 1467.3 (3 runs) to 1515.9 / 1508.5 (f61rnpCA6agF, f61s2AfCpM8K).

With the flag off nothing changes: the method is never called.

Both the amd/ and nvidia/ copies of model.py carry the same class; both are
patched so the two stay identical.

Upstream status
---------------
Present on the pinned build 0.2.1.dev32+gbc0f47cd0.
"""
import argparse
import importlib.util
import os
import sys

TARGETS = [
    "vllm/models/inkling/amd/model.py",
    "vllm/models/inkling/nvidia/model.py",
]

MARKER = "Zenith patch inkling_compute_logits_local.py"

OLD = '''    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)
'''
NEW = OLD + '''
    def compute_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # {marker}: the TP-local vocab shard
        # of the logits (no gather), for --enable-batch-sharded-sampling.
        return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)
'''.format(marker=MARKER)


def vllm_root():
    """Locate site-packages without `import vllm` (that touches the GPU)."""
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.origin:
        sys.exit("vllm is not installed; run this inside the container")
    return os.path.dirname(os.path.dirname(os.path.abspath(spec.origin)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report status without writing")
    ap.add_argument("--root", help="site-packages root (default: the installed vllm's)")
    args = ap.parse_args()

    root = args.root or vllm_root()
    rc = 0
    for rel in TARGETS:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            print(f"ERROR: expected {path} to exist; vLLM layout changed",
                  file=sys.stderr)
            rc = 1
            continue
        with open(path) as fh:
            src = fh.read()
        if MARKER in src:
            print(f"already patched: {path}")
            continue
        if "def compute_logits_local" in src:
            print(f"ERROR: {path} already defines compute_logits_local",
                  file=sys.stderr)
            print("upstream may have added it; retire the patch", file=sys.stderr)
            rc = 1
            continue
        got = src.count(OLD)
        if got != 1:
            print(f"ERROR: compute_logits anchor found {got} times in {path}",
                  file=sys.stderr)
            rc = 1
            continue
        if args.check:
            print(f"would patch: {path}")
            continue
        with open(path, "w") as fh:
            fh.write(src.replace(OLD, NEW, 1))
        print(f"patched: {path}  (_TmlForCausalLMBase.compute_logits_local)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
