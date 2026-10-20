# RCCL-ANP for Lux (Pollara 400G RoCE)

Cross-node RCCL on Lux needs AMD's RCCL net plugin, `librccl-anp.so`
("accelerated network path"), plus the vendor `libionic` verbs provider for
the Pollara (ionic) NICs. The Lux hosts ship neither as something a container
can use, and the inference image does not include them, so this package
carries a copy built on Lux. `run_container.sh` uses it by default for any
multi-node run: `docker/patches` is bind-mounted read-only at
`/opt/zenith_patches`, and `ZENITH_ANP_DIR` defaults to
`/opt/zenith_patches/anp`. A single-node run never touches it, because RCCL
uses XGMI inside a node.

| file | what |
|---|---|
| `librccl-anp.so` | amd-anp v1.4.0, built against RCCL `rocm-7.2.4` for gfx950 |
| `libionic.so.1.1.61` | the hosts' `libionic-amd` 61.0.26.08.28.001 verbs provider |
| `libionic.so.1`, `libionic-rdmav59.so` | symlinks to it: one file is both the ionic_dv library ANP links and the verbs provider, so both see the same QP structs. **Not shipped:** VISTA's skill import refuses symlinks, and copies would load the library twice (two inodes). `hpc_jobs/inkling-serve/job.lux.slurm` runs the engine from a per-job copy and creates both links there; run it, not this tree, for multi-node serving. |
| `libibverbs.d/ionic.driver` | `driver /opt/zenith_patches/anp/libionic`, so libibverbs loads this provider rather than the image's upstream one |
| `VERSION` | plugin tag, md5 prefix, and the RCCL/libionic it was built against |
| `build/` | `build_anp.sh` and `stage_anp.sh`, which produced these files |

The image's rdma-core 61 and the hosts' libionic-amd 61 both use
`IBVERBS_PRIVATE_59`, so only libionic needs to be staged, not libibverbs.

## What a healthy run looks like

* The launcher's RCCL health check prints the stage 0 -> 1 handoff at about
  2,800-2,900 Gbps aggregate over the 8 TP lanes (GPUDirect RDMA). On TCP
  sockets the same check reads ~237 Gbps.
* With `ZENITH_NCCL_DEBUG=INFO`, every cross-node channel reads
  `via NET/RCCL-ANP/<n>/GDRDMA`.

`run_container.sh` makes a fallback fail loudly instead of silently running
~12x slower. On multi-node runs it sets `NCCL_NET=RCCL-ANP`, so RCCL errors at
communicator creation if the plugin cannot load. It also sets
`ZENITH_HEALTH_MIN_PP_GBPS` (1500 with GDR, 500 without), so the health check
fails and nothing is served if the measured handoff falls below that.
`ZENITH_ALLOW_SOCKET_FALLBACK=1` turns both off, for debugging only.

## Rebuilding

You only need to rebuild if the hosts' libionic-amd or the image's RCCL
changes ABI. Building takes two hosts. A login node has the libdrm headers
and the internet access that RCCL's cmake needs. A compute node has
`libionic-amd-devel`; alternatively, copy `/usr/include/libionic` from one
into `$ANP_WORK/hostinc`.

1. Under `$ANP_WORK`:
   * clone `https://github.com/ROCm/rccl` at tag `rocm-7.2.4` into `rccl/`;
   * unpack the amd-anp v1.4.0 release (`https://github.com/rocm/amd-anp`) into `anp-v1.4.0/`;
   * unpack the boost 1.86 headers into `boost/`.
2. On a login node, run `ANP_WORK=... bash build/build_anp.sh rccl` (cmake + hipify only).
3. On a compute node (or a login node with `hostinc`), run
   `ANP_WORK=... ZENITH_IMAGE_TAR=... bash build/build_anp.sh anp`.
   This applies the two source fixes that libionic-amd 61 / RCCL 7.2.4 need:
   * the `IONIC_UDMA_MASK_*` defines;
   * the extra `ncclCudaCalloc` stream argument.
4. On a compute node, run `ANP_WORK=... bash build/stage_anp.sh v1.4.0`. It
   rewrites the binaries here and leaves the licences and this README alone.

## Licences

* **amd-anp** (`librccl-anp.so`): the BSD-style licence in
  `LICENSE-amd-anp.txt`, which is RCCL's (NVIDIA, with AMD and Microsoft
  modifications). Binary redistribution requires keeping that notice, which is
  why it is here.
* **libionic-amd**: `GPL-2.0 or Linux-OpenIB`, at the redistributor's option.
  This package redistributes it under the Linux-OpenIB (BSD) terms in
  `LICENSE-libionic-Linux-OpenIB.txt`. The GPL-2.0 text is included as well,
  since it is the other half of the dual licence.
