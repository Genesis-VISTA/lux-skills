# vllm_src/

Git bundles for vLLM commits that are not on the vLLM remote. The Dockerfile copies this directory
to `/opt/vllm_src/`, and with `VLLM_BUNDLE=<file>` it fetches the bundle's branches into the clone
before checking out `VLLM_HASH`. Bundles are gitignored except the ones the repo's results depend on
(the table below); keep a copy of any other on shared storage next to the results it produced, and note
it here.

A bundle only needs to carry the local commits. Its prerequisites must be public commits that the
default clone already contains:

    git bundle create vllm_<name>.bundle refs/heads/<branch> ^<public parent> ...
    git bundle verify vllm_<name>.bundle     # run in a clone that has the parents

Build with it:

    VLLM_HASH=<local sha> VLLM_BUNDLE=vllm_<name>.bundle ZENITH_IMAGE_TAG=<tag> ./build_image.sh gfx950

| bundle | commit | what | where |
|---|---|---|---|
| `vllm_musemain_0e939af8.bundle` | `0e939af8ad477f52230534fca91053d96e7da59a` | vLLM main `642c18d64` (2026-10-15) merged into `muse-spark-support` `6cc692b5a` (vLLM PR #61879); the Muse Spark image | in this directory (22 KB, Apache-2.0 like the rest of vLLM) |
