#!/usr/bin/env python3
"""
Re-read the AITER env vars in a Ray worker after the driver's env is applied.

Why this patch exists
---------------------
`vllm/_aiter_ops.py` snapshots every AITER toggle into class attributes at
*module import* time:

    class rocm_aiter_ops:
        _AITER_ENABLED = envs.VLLM_ROCM_USE_AITER      # read once, at import
        _FMOE_ENABLED  = envs.VLLM_ROCM_USE_AITER_MOE
        ...

With the `mp` executor that is harmless: the driver exports
VLLM_ROCM_USE_AITER=1 before anything imports vLLM, so the snapshot is correct.

With the `ray` executor -- which is exactly what multinode uses -- it is not.
A RayWorkerProc actor is started by the raylet, whose environment does *not*
carry the driver's vars. The actor imports vLLM (and with it `_aiter_ops`,
snapshotting VLLM_ROCM_USE_AITER=False), and only afterwards does

    vllm/v1/executor/ray_executor_v2.py::initialize_worker()

copy the driver's env vars in with os.environ.setdefault(). By then the
snapshot is already stale and nothing refreshes it.

The two halves of vLLM then disagree, because they read the flag by different
routes:

  * the MoE oracle, fused_moe/oracle/unquantized.py, tests the *live*
    environment via `envs.is_set("VLLM_ROCM_USE_AITER")` and `envs.VLLM_*`
    (vllm.envs uses module __getattr__, so it always sees current os.environ)
    -> sees AITER enabled, selects UnquantizedMoeBackend.AITER

  * the kernel's is_supported_config() consults the *cached* class attributes
    via rocm_aiter_ops.is_fused_moe_enabled()
    -> sees AITER disabled, reports unsupported

so `_return_or_raise` raises, on every worker, and the engine dies:

    ValueError: Unquantized MoE backend ROCm AITER does not support the
    deployment configuration since kernel does not support current device
    rocm. AITER MoE is not enabled -- set VLLM_ROCM_USE_AITER=1 and
    VLLM_ROCM_USE_AITER_MOE=1 to enable it.

The error message is actively misleading: both variables *are* set to 1 in the
driver and in the actor's os.environ by the time it fires.

Upstream already provides the fix, it is just never called on this path:
`rocm_aiter_ops.refresh_env_variables()` exists precisely to re-read the env
after it changes ("after monkey patching the env variables in the unit test,
you can call this function to reload the env variables"). We call it at the end
of initialize_worker(), once the driver env has landed -- the same place vLLM
already accepts that os.environ has just changed underneath it.

Guarded by current_platform.is_rocm() so the patched file stays a no-op on
CUDA, and wrapped in a try/except so a future vLLM that drops
refresh_env_variables() degrades to today's behaviour instead of breaking
worker startup outright.

This is only reachable with --distributed-executor-backend ray, which is why
single-node (mp) runs of the same model on the same image work fine.

Usage (inside the container):
    python3 rocm_aiter_ray_env_refresh.py [--check]

    --check   exit 0 if already patched or patch would apply cleanly,
              exit 1 if the target text is not found (upstream changed)
"""

import argparse
import importlib.util
import os
import sys

TARGET = "vllm/v1/executor/ray_executor_v2.py"

# the exact body we expect to find (upstream, unpatched)
OLD = '''        if driver_env_vars:
            for key, value in driver_env_vars.items():
                os.environ.setdefault(key, value)
        for key, value in env_vars.items():
            os.environ[key] = value
'''

NEW = '''        if driver_env_vars:
            for key, value in driver_env_vars.items():
                os.environ.setdefault(key, value)
        for key, value in env_vars.items():
            os.environ[key] = value

        # vllm/_aiter_ops.py caches every AITER toggle into class attributes at
        # module import time. A Ray actor imports vLLM before the lines above
        # run, so those attributes were snapshotted from the raylet's env, not
        # the driver's, and VLLM_ROCM_USE_AITER=1 is missed. The MoE oracle
        # reads the live env and picks the AITER backend; the kernel's
        # is_supported_config() reads the stale cache and rejects it, and the
        # worker dies with "AITER MoE is not enabled" even though it is.
        # refresh_env_variables() is upstream's own re-read hook.
        if current_platform.is_rocm():
            try:
                from vllm._aiter_ops import rocm_aiter_ops

                rocm_aiter_ops.refresh_env_variables()
            except (ImportError, AttributeError):
                pass
'''

MARKER = "rocm_aiter_ops.refresh_env_variables()"


def find_target():
    """Locate the ray executor inside whichever site-packages vLLM lives in.

    Deliberately does NOT `import vllm`. Importing it executes
    vllm/platforms/rocm.py, which calls torch.cuda.get_device_properties() at
    module scope and dies with "RuntimeError: No CUDA GPUs are available" in a
    build container, where no GPU is exposed. find_spec() resolves the package
    location from the filesystem without running __init__.py, which is all we
    need to rewrite a file inside it.
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

    if OLD not in src:
        print(f"ERROR: patch target not found in {path}", file=sys.stderr)
        print("vLLM changed initialize_worker(); re-derive the patch",
              file=sys.stderr)
        return 1

    # the patched code calls current_platform; the module already imports it,
    # but check rather than assume -- a NameError here would break every worker
    if "from vllm.platforms import current_platform" not in src:
        print(f"ERROR: {path} does not import current_platform", file=sys.stderr)
        print("re-derive the patch to add the import", file=sys.stderr)
        return 1

    if args.check:
        print(f"would patch: {path}")
        return 0

    with open(path, "w") as fh:
        fh.write(src.replace(OLD, NEW, 1))
    print(f"patched: {path}  (AITER env re-read in Ray workers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
