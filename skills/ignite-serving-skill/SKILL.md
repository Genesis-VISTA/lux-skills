---
name: ignite-serving-skill
description: Connect to a shared vLLM endpoint on Lux Ignite Kubernetes, or prepare and operate a pinned Zenith Mistral Large 4 GPU Job. Use for Rancher/kubectl model hosting on Ignite; Slurm serving stays in vllm-skill.
compatibility: Needs a host shell with kubectl, OpenSSH and a route to the Ignite Rancher API. It does not run inside the VISTA sandbox and ships no VISTA HPC jobs.
---

# Model serving on Lux Ignite Kubernetes

Target the **Kubernetes side of OLCF Lux**, administered through Rancher.
Use this skill for an existing shared endpoint or a new single-node ML4 server.
The hosting recipe retains Zenith's pinned, unmerged vLLM/AITER optimizations.
It does not use `sbatch`, VISTA `submit_hpc_job`, Slurm/Flux, or a Docker daemon.
For a Slurm workload use `vllm-skill` instead.

## Running inside VISTA

If `/mnt/skills` exists, you are in the VISTA sandbox. It has no kubectl, OpenSSH
or git, reaches only public network destinations, and cannot take the user's
SecurID prompt, so do not try to connect, port-forward or apply manifests here,
and never ask for a kubeconfig to be copied into the sandbox. Explain the steps,
or render a plan with `prepare_ml4.py` if an authorized Zenith checkout is
already in the sandbox. Tell the user to run `connect.py` and any `kubectl apply`
in a terminal on their own machine. To use a running Ignite model from VISTA,
they can point a custom inference endpoint in VISTA's Settings at the local URL
`connect.py` prints; this route is untested.

## Choose the requested operation

- **Use an existing service:** read [references/connect.md](references/connect.md).
  Run [scripts/connect.py](scripts/connect.py) with the caller's kubeconfig,
  explicit context, verified API server URL, namespace and Service name. This
  forwards one local port and allocates no GPUs.
- **Host a new ML4 service:** read [references/deploy-ml4.md](references/deploy-ml4.md).
  [scripts/prepare_ml4.py](scripts/prepare_ml4.py) renders a plan from an authorized,
  clean checkout of Zenith commit `ebf19b355b753d5a26c590463f782a0f1f43901f`.
  It creates local files only. Inspect the plan and validate cluster admission
  before applying it within the user's authorized deployment scope.
- **Inspect or troubleshoot:** start with namespaced Job, pod, event and log
  reads for that workload. A forbidden cluster-wide node/namespace listing is
  not proof the user's namespaced service cannot run.

Run commands from this skill directory, or resolve `scripts/` and `references/`
relative to this file after installation. Python 3.12+ is needed for preparing
the Zenith plan; the client helper needs Python 3.10+, kubectl and OpenSSH.
The public skill includes no Zenith source bundle, model weights, kubeconfig or
credentials. Deployment requires access to the linked Zenith repository; using
an existing Service does not.

## Establish the actual target

Use the user's own Rancher kubeconfig and verify its context/server against the
Ignite cluster they selected. Never print the raw config or copy another user's
credentials. Ask only for missing target facts: allocated namespace, existing
Service/model name for a client, or a unique workload name for a deployment.
Existing session authorization and known values carry forward.

Follow the [OLCF Kubernetes guide](https://docs.olcf.ornl.gov/systems/lux_user_guide.html#lux-kubernetes-user-guide).
Its current GPU namespace convention is `<project>-gpu`; the November 2, 2026
pilot was admitted in `ven136`. **Use the namespace actually assigned to the
caller**, and validate it, rather than deriving one from a project code.
Namespace allocation and Kubernetes permissions are separate from OLCF login.

For deployment, inspect resource quotas, limits and PVCs within that namespace;
check Job/PVC/Service creation permissions and current site allocation policy.
Use GPU Jobs with one GPU container, matching GPU requests/limits and a walltime
of at most 48 hours. The wrapper defaults to a four-hour serving session.
Keep non-root execution, device groups and TLS verification intact.

## Success criteria and operation

For a new server, require completed build/staging Jobs, a passing four-GPU RCCL
preflight, a Ready serving pod, `/health`, the expected `/v1/models` entry and a
nonempty chat-completion answer. `--check --test` runs a short real request and
then closes the client's temporary connection. An allocated GPU or a running
pod alone does not establish inference readiness.

One ML4 replica reserves four MI355X GPUs while running, including idle time.
vLLM batches requests within that replica. There is no autoscaler or automatic
partition-wide load balancing. Job expiry ends serving; do not describe the
bounded recipe as an indefinite, highly available service.

When reporting readiness, give the namespace, Job/Service, local base URL, model
ID, resource reservation, deadline and exact checks performed. Report GPU
correctness, inference correctness and throughput measurements separately.
Inspect failures before retrying; do not replay failed inference automatically
or repeatedly replace a GPU Job. End a requested test by deleting only its named
Job, keeping the PVCs unless the user explicitly requests data removal.

See [references/validation.md](references/validation.md) for the pilot evidence,
the limits of the generalized workflow and reproducible local checks.

## License and acknowledgments

This skill's original wrappers and documentation are Apache-2.0 under
[LICENSE](LICENSE). Nicholas Malaya contributed the Ignite workflow with Codex
assistance. Jonathan Belof and the Zenith contributors provide the upstream
ML4 recipe. Zenith, vLLM, AITER, ROCm images and model weights retain their own
licenses. Attribution and source links are in `skills/ATTRIBUTION.md` in the
source repository; the operational skill does not depend on that file.
