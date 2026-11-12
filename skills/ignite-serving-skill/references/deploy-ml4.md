# Host the pinned ML4 recipe on Ignite

This is a single-replica GPU Job workflow, using the PVC-based implementation
from [Zenith PR #2](https://github.com/jbelofamd/zenith/pull/2). Access to that
repository is required. Use the reviewed commit below even if `main` advances;
do not replace its unmerged vLLM with a generic image or a floating branch.

## Obtain and render the recipe

Use an authorized GitHub credential through normal Git authentication:

```bash
git clone https://github.com/jbelofamd/zenith.git /path/to/zenith
git -C /path/to/zenith fetch origin refs/pull/2/head
git -C /path/to/zenith checkout --detach ebf19b355b753d5a26c590463f782a0f1f43901f
```

An existing clean checkout at that commit also works. The public skill does
not redistribute Zenith's source bundle or patches. If access is unavailable,
report the dependency and request an authorized checkout; do not silently swap
the model stack. Model access and licensing are also the caller's responsibility.

From the skill directory, with Python 3.12+ and Git:

```bash
python3 scripts/prepare_ml4.py --zenith /path/to/zenith \
  --namespace YOUR_ASSIGNED_GPU_NAMESPACE --name ml4-trial \
  --out /path/to/new/ml4-plan --serve-hours 4 \
  --proxy http://proxy.ccs.ornl.gov:3128
```

`--out` must be new and should be outside version control: its ConfigMaps carry
the authorized Zenith sources. The helper renders five JSON files and a
`plan.json` with their hashes and deployment identity. It performs no cluster
writes. A distinct name yields separate Job/PVC/Service/ConfigMap names and pod
selectors, so a new replica cannot enter another replica's Service endpoints.

The profile is Mistral Large 4 Quark MXFP4 at model revision
`d084803a4a2bfac7274058937f6665750741d650`, TP4 on one MI355X node, FP8 KV cache,
DSpark K=3, 262,144 context, and the baseline Zenith tuning profile. It preserves
vLLM `bd4fa2633ea773f95128beb27ef7b4a5f5f3e8a3` and AITER
`84077b3c0461dac1f0f3f8b9590a9282e2035a2f`.

| Phase | Reservation / output |
|---|---|
| Storage | 256 GiB build PVC and 1 TiB model PVC, RWX `netapp-file` by default |
| Build | 1 GPU, 16 CPU, 128 GiB RAM; four-hour deadline |
| Download | 1 GPU for placement/egress, 4 CPU, 16 GiB RAM; six-hour deadline |
| Preflight | 4 GPUs, 8 CPU, 32 GiB RAM; 30-minute deadline |
| Serve | 4 GPUs, 16 CPU, 768 GiB RAM, 64 GiB shared memory; four hours by default |

Serving reserves at most 16 GPU-hours with the default deadline; build, download
and preflight can reserve up to 12 GPU-hours combined. These are upper bounds,
not measured utilization or official billing. Set `--serve-hours` from 1 to 48
within the allocation. There is no automatic renewal or autoscaler.

The [OLCF guide](https://docs.olcf.ornl.gov/systems/lux_user_guide.html#lux-kubernetes-user-guide)
documents GPU Jobs, one GPU container, walltime and proxy rules. CPU-only workers
currently lack internet egress, hence the staging Job's GPU placement. The tested
PVC class and resource sizes above are pilot configuration, not universal site
requirements. `--storage-class` can select another compatible RWX class; its
permissions/capacity must be checked. Build/serve run as UID/GID 1000 with device
group 998 in this profile. A different site device group needs an upstream recipe
adaptation and validation, not a privileged-container workaround.

The build uses an exact AMD base image and creates its Python/ROCm development
environment on the build PVC. No custom image registry, host Docker socket or
privileged pod is needed. Model staging is anonymous by default; optional
`--hf-secret NAME` refers to an existing Secret with key `token` and is passed
only to download. `--pull-secret NAME` supplies existing image-pull credentials.
Do not put tokens in the repository, generated files or proxy URLs.

## Inspect and apply the plan

Use kubectl with an explicit private kubeconfig and context on a host with the
site-supported route to Rancher. The client helper routes only its own
port-forward process; it does not change your operator kubectl configuration.
Use OLCF's gateway instructions for operator API access as well.

Check the assigned namespace, quota/limits, existing PVCs and create permissions.
The guide uses `<project>-gpu`; the original pilot used `ven136`. Do not infer
that another account has access to the pilot namespace.

Example Bash variables for the remaining commands:

```bash
K=(kubectl --kubeconfig /path/to/your/ignite.yaml --context YOUR_IGNITE_CONTEXT)
NS=YOUR_ASSIGNED_GPU_NAMESPACE
PLAN=/path/to/new/ml4-plan
NAME=ml4-trial
"${K[@]}" auth can-i create jobs.batch -n "$NS"
"${K[@]}" auth can-i create persistentvolumeclaims -n "$NS"
"${K[@]}" auth can-i create services -n "$NS"
"${K[@]}" auth can-i create configmaps -n "$NS"
"${K[@]}" get resourcequota,limitrange,pvc -n "$NS"
for phase in storage build download preflight serve; do
  "${K[@]}" apply --server-side --dry-run=server -f "$PLAN/$phase.json" || break
done
```

Resolve any dry-run error before applying. Use **server-side apply**: the source
ConfigMap is too large for client-side apply's last-applied annotation. Account
for quotas, PVC capacity and the GPU reservations before the authorized launch.
No production service or data should be replaced as part of an unrelated test.

```bash
"${K[@]}" apply --server-side -f "$PLAN/storage.json"
"${K[@]}" get pvc "$NAME-build" "$NAME-models" -n "$NS"
# Continue only when both claims are Bound.
"${K[@]}" apply --server-side -f "$PLAN/build.json"
"${K[@]}" apply --server-side -f "$PLAN/download.json"
"${K[@]}" get jobs "$NAME-build" "$NAME-download" -n "$NS"
"${K[@]}" logs -n "$NS" "job/$NAME-build"
"${K[@]}" logs -n "$NS" "job/$NAME-download"
```

Build/staging may run concurrently. Require both Jobs to be Complete. Checkpoint
staging verifies all files against reported sizes before writing its completion
marker; this is completeness checking, not a separate cryptographic weights audit.
Anonymous Hugging Face downloads may be rate-limited. Inspect the failure and
resume the same revision only after the previous writer is terminated. Do not
start concurrent builders or downloaders on the same PVC. A failed Job does not
retry automatically; retain its logs before deleting/recreating that named Job.

```bash
"${K[@]}" apply --server-side -f "$PLAN/preflight.json"
"${K[@]}" logs -n "$NS" "job/$NAME-preflight"
"${K[@]}" get job "$NAME-preflight" -n "$NS"
# Require Complete, including four-rank RCCL, before serving.
"${K[@]}" apply --server-side -f "$PLAN/serve.json"
"${K[@]}" get pods -n "$NS" -l "app.kubernetes.io/instance=$NAME" -o wide
"${K[@]}" logs -n "$NS" "job/$NAME" -c tp4-0
```

Serving mounts build/model PVCs read-only and writes runtime caches to emptyDir.
The first pilot startup took about 35.6 minutes including 9.8 minutes loading
weights; this is not a guarantee. Startup probes allow two hours. Require a Ready
pod and use [connect.md](connect.md) to verify health/models and a real answer.
The endpoint is internal ClusterIP and has no public ingress or separate API key.

## End or update a session

Job expiry stops the replica, even with active clients. Four GPUs remain reserved
between requests until the Job terminates. Shared weights do not provide shared
GPU scheduling or automatic scale-out. Additional replicas require separate
allocations and a routing design.

To end only this session while preserving PVC data:

```bash
"${K[@]}" delete job "$NAME" -n "$NS"
```

A Job's pod template is immutable. To restart/update an authorized session,
retain logs, terminate the old Job, wait for its pod to exit, then apply the
reviewed serving file. A different compiled recipe requires a separate build PVC;
do not bypass the pin/provenance checks or mutate a shared environment in use.
Avoid broad label-based deletion. PVCs have an independent lifecycle, and deleting
them can delete their data. The serving file itself contains no PVC definitions.
