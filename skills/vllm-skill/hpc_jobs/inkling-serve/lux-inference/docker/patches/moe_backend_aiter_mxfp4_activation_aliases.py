#!/usr/bin/env python3
"""
Expose the AITER MXFP4 activation-variant backends on the `--kernel-config`
`moe_backend` enum, so an MXFP4-expert MoE model can be served W4A4 instead
of W4A16.

Why this patch exists
---------------------
Two MXFP4 activation variants exist in vLLM for AITER:

    AITER_MXFP4_FP8    W4A8  -> AiterW4A8ExpertsMonolithic
    AITER_MXFP4_MXFP4  W4A4  -> AiterExperts

W4A8 wants `kFp8StaticTensorSym`, i.e. *static* per-tensor input scales, so
it is only reachable for checkpoints that ship `input_scale` tensors. A
checkpoint with `"activation_scheme": "dynamic"` has nothing to load. Do not
add a `aiter_mxfp4_fp8` alias expecting it to work for such a checkpoint.

W4A4 is supported by the installed kernel (`_supports_quant_scheme(
kMxfp4Static, mxfp4-dynamic)` and `_supports_current_device()` are True on
gfx950), and `map_mxfp4_backend()` in fused_moe/oracle/mxfp4.py already
carries the mapping:

    "aiter_mxfp4_mxfp4": [Mxfp4MoeBackend.AITER_MXFP4_MXFP4],

The only thing missing is the CLI surface. `MoEBackend` in config/kernel.py
is a `Literal[...]` that lists "aiter" and "aiter_triton_mxfp4_bf16" but not
the two `aiter_mxfp4_*` keys, so pydantic rejects them at argparse time:

    vllm serve: error: argument --kernel-config: 1 validation error for
    KernelConfig / moe_backend / Input should be 'auto', 'triton', ...
    [type=literal_error, input_value='aiter_mxfp4_mxfp4']

The gap is present on the pinned build bc0f47cd0 *and* on upstream main
1ea7c63f4, so rebasing does not fix it.

Why not `--quantization-config {"moe":{"activation":"mxfp4"}}`
--------------------------------------------------------------
That is the documented way to force an activation scheme, and it parses and
reaches the engine intact. But it is only consulted by
`select_mxfp4_moe_backend(moe, activation_key)` via
`_resolve_activation_key()` / `_user_moe_activation_override()`. A model
whose quant config calls a model-specific MXFP4 backend selector that takes
no activation_key never reads the override, and still logs
`Using 'AITER_MXFP4_BF16' Mxfp4 MoE backend.`

An explicit `moe_backend="aiter"` does not help either: it walks
`map_mxfp4_backend("aiter")` in order, and that list starts with
AITER_MXFP4_BF16, which succeeds -- so the W4A4 variant is never reached.
The alias must name the single backend.

What this changes
-----------------
One line: adds "aiter_mxfp4_mxfp4" to the MoEBackend Literal, plus a doc
bullet. No selection logic is touched -- the oracle already knows what to do
with the string, and every other model keeps the same default path it had
before, because nothing selects this backend unless the operator asks for it
by name.

Usage (inside the container):
    python3 moe_backend_aiter_mxfp4_activation_aliases.py [--check]

    --check   exit 0 if already patched or patch would apply cleanly,
              exit 1 if the target text is not found (upstream changed)

Then serve with:
    --kernel-config '{"moe_backend":"aiter_mxfp4_mxfp4"}'

and confirm in the log that the backend line reads AITER_MXFP4_MXFP4 rather
than AITER_MXFP4_BF16 -- both resolve to the AiterExperts class, so the
"Using AiterExperts" line is NOT diagnostic on its own.
"""

import argparse
import importlib.util
import os
import sys

TARGET = "vllm/config/kernel.py"

# The enum entry we key off. "aiter_triton_mxfp4_bf16" is the last AITER
# member of the Literal, so appending after it keeps the vendor family
# grouped the way upstream has it.
OLD_ENUM = '''    "aiter",
    "aiter_triton_mxfp4_bf16",
    "flydsl",
'''

NEW_ENUM = '''    "aiter",
    "aiter_triton_mxfp4_bf16",
    "aiter_mxfp4_mxfp4",
    "flydsl",
'''

OLD_DOC = '''    - "aiter_triton_mxfp4_bf16": Use the AITER Triton MXFP4 W4A16
      (moe_gemm_a16w4) MoE kernel (ROCm gfx942/gfx950/gfx1250)
'''

NEW_DOC = '''    - "aiter_triton_mxfp4_bf16": Use the AITER Triton MXFP4 W4A16
      (moe_gemm_a16w4) MoE kernel (ROCm gfx942/gfx950/gfx1250)
    - "aiter_mxfp4_mxfp4": Use the AITER MXFP4 MoE kernel with MXFP4
      (W4A4) activations rather than BF16. Names the single backend on
      purpose: "aiter" walks its variant list BF16-first and never reaches
      this one. Required for models whose MXFP4 MoE selector ignores
      quantization_config.moe.activation
'''

MARKER = '"aiter_mxfp4_mxfp4"'


def find_target():
    """Locate config/kernel.py inside whichever site-packages vLLM lives in.

    Deliberately does NOT `import vllm`: importing it executes
    vllm/platforms/rocm.py, which calls torch.cuda.get_device_properties() at
    module scope and dies in a build container with no GPU exposed.
    find_spec() resolves the package location from the filesystem without
    running __init__.py.
    """
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.origin:
        sys.exit("vllm is not installed; run this inside the container")
    root = os.path.dirname(os.path.dirname(os.path.abspath(spec.origin)))
    path = os.path.join(root, TARGET)
    if not os.path.exists(path):
        sys.exit(f"expected {path} to exist; vLLM layout changed")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report status without writing")
    args = ap.parse_args()

    path = find_target()
    with open(path) as fh:
        src = fh.read()

    if MARKER in src:
        print(f"already patched: {path}")
        return 0

    if OLD_ENUM not in src:
        print(f"ERROR: MoEBackend enum target not found in {path}",
              file=sys.stderr)
        print("vLLM reordered the Literal; re-derive the patch",
              file=sys.stderr)
        return 1

    # The alias is only useful if the oracle still maps it to a backend. If
    # upstream renamed the key, patching the enum would produce a config that
    # argparse accepts and then dies on at model load -- worse than failing
    # here. Read the oracle as text for the same reason find_target() avoids
    # importing vllm.
    oracle = os.path.join(os.path.dirname(os.path.dirname(path)),
                          "model_executor", "layers", "fused_moe", "oracle",
                          "mxfp4.py")
    if os.path.exists(oracle):
        with open(oracle) as fh:
            if '"aiter_mxfp4_mxfp4"' not in fh.read():
                print(f"ERROR: map_mxfp4_backend() in {oracle} no longer has "
                      "an 'aiter_mxfp4_mxfp4' key", file=sys.stderr)
                print("the enum alias would be accepted then fail at load; "
                      "re-derive the patch", file=sys.stderr)
                return 1
    else:
        print(f"WARNING: could not find {oracle} to cross-check the alias",
              file=sys.stderr)

    if args.check:
        print(f"would patch: {path}")
        return 0

    out = src.replace(OLD_ENUM, NEW_ENUM, 1)
    if OLD_DOC in out:
        out = out.replace(OLD_DOC, NEW_DOC, 1)
    else:
        # Docstring drift is cosmetic -- the enum entry is what makes the
        # flag work, so do not fail the patch over it.
        print("note: moe_backend docstring not in expected form; "
              "enum patched without the doc bullet", file=sys.stderr)

    with open(path, "w") as fh:
        fh.write(out)
    print(f"patched: {path}  (moe_backend += aiter_mxfp4_mxfp4)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
