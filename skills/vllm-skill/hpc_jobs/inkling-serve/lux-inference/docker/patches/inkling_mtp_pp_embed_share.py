#!/usr/bin/env python3
"""
Let Inkling's MTP drafter share the target's token embedding under PP>1.

Why this patch exists
---------------------
InklingMTP never builds its own token embedding. InklingMultiTokenPredictor
sets `self.embed_tokens = None` and relies on load_eagle_model to attach the
target's (the comment there: building a replicated copy "would transiently
double the 2.3 GiB table"). The V2 model runner does that attaching in
vllm/v1/worker/gpu/spec_decode/eagle/utils.py:

    def maybe_share_target_embed(draft_model, draft_inner, target_inner):
        ...
        if get_pp_group().world_size > 1 and not hasattr(
            draft_model, "has_own_embed_tokens"
        ):
            return

so with pipeline parallelism, a draft that doesn't declare
`has_own_embed_tokens` either way is left as it was built. For InklingMTP
that is None, and the last PP rank (the only one with a drafter) dies in
engine startup, during profile_run -> _dummy_run:

    multi_module_mtp/speculator.py:351 _prepare_inputs
      -> self.model.embed_input_ids -> self.embed_tokens.weight
    AttributeError: 'NoneType' object has no attribute 'weight'

Observed on tioga, Inkling-Small-bf16 TP8 PP3 with ZENITH_MTP=1
(job f61FRg2NdCvb, Worker_PP2). PP=1 never takes that early return, which is
why single-node MTP works.

The fix
-------
Declare `has_own_embed_tokens = False` on InklingMTP. That is the truth (it
has none), and it is exactly what the guard asks for: the early return is
skipped, `_should_share` returns True for a False flag, and the target's
embedding is attached as it is at PP=1.

That only works because the target has the table on the last PP rank. It
does: InklingModel builds embed_tokens (InklingReplicatedEmbedding)
unconditionally on every rank, not as a PPMissingLayer, and
_load_inkling_weights loads it through AutoWeightsLoader with no PP
filtering, so the last rank's copy holds the real weights. If upstream ever
makes it a PPMissingLayer, maybe_share_target_embed raises "needs the target
input embedding, but it is unavailable on this PP stage" instead of serving
with a garbage table -- a loud failure, which is what we want.

The LM head needs nothing: lm_head sharing has no PP guard, and the last PP
rank owns the target's lm_head.

Both the amd/ and nvidia/ copies of mtp.py carry the same class; both are
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
    "vllm/models/inkling/amd/mtp.py",
    "vllm/models/inkling/nvidia/mtp.py",
]

MARKER = "zenith: share the target embedding under PP"

OLD = '''class InklingMTP(nn.Module, SupportsMultiModalEmbeddings):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
'''
NEW = '''class InklingMTP(nn.Module, SupportsMultiModalEmbeddings):
    # {marker}
    # (see patches/inkling_mtp_pp_embed_share.py). maybe_share_target_embed
    # skips drafts that don't declare this when PP>1, which left
    # model.embed_tokens None on the last PP rank.
    has_own_embed_tokens = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
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
    args = ap.parse_args()

    root = vllm_root()
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
        if "has_own_embed_tokens" in src:
            print(f"ERROR: {path} already declares has_own_embed_tokens",
                  file=sys.stderr)
            print("upstream may have fixed this; retire the patch", file=sys.stderr)
            rc = 1
            continue
        got = src.count(OLD)
        if got != 1:
            print(f"ERROR: InklingMTP anchor found {got} times in {path}",
                  file=sys.stderr)
            rc = 1
            continue
        if args.check:
            print(f"would patch: {path}")
            continue
        with open(path, "w") as fh:
            fh.write(src.replace(OLD, NEW, 1))
        print(f"patched: {path}  (InklingMTP.has_own_embed_tokens = False)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
