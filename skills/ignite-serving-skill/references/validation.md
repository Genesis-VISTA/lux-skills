# Validation record

## Observed on Ignite, November 2, 2026

The precursor workflow ran in `ven136`, with a TP4 server on four MI355X GPUs:

- Non-root build of Zenith's unmerged vLLM/AITER stack on persistent storage.
- Anonymous staging of 66 checkpoint files, 577,934,666,965 bytes, size verified.
- Four-GPU RCCL preflight and vLLM ROCm platform checks passed.
- API health/model listing and real chat completions passed. Logs showed
  AITER/FlyDSL MXFP4 kernels; these were correctness checks, not a benchmark.
- The Linux/Bash and Windows client routes reached the same service from a
  Windows workstation. The source scripts used each caller's kubeconfig and
  retained TLS verification.

Zenith PR #2 adds generalized manifests, source provenance and deadlines. Those
resources passed server-side admission dry runs, but the final generalized build
has not undergone a second clean GPU build. This skill calls that pinned renderer
and adds separate names/selectors, explicit target selection and local tests.

## Contribution checks

On November 2, 2026:

- All six skills passed all five validator profiles with no errors. The four
  warnings concerned the existing Primus and vLLM document lengths.
- Repository attribution/license policy passed; 20 repository tests passed,
  including 10 focused Ignite tests.
- `unpack.sh` copied this skill into a separate installation directory.
- This wrapper rendered the pinned Zenith checkout. All five generated phases
  passed Ignite server-side admission dry runs, creating no GPU workloads.
- This client verified `/health`, the `le-chonk` model entry and a real chat
  completion through an existing SSH SOCKS tunnel, then closed its connection.
- The repository's pre-commit step could not run because the upstream checkout
  lacks `.pre-commit-config.yaml`. Python syntax and `git diff --check` passed.

From the repository root:

```bash
python3 tools/validate_skills.py skills/ignite-serving-skill --profile all --no-color
python3 tools/validate_repo_policy.py
python3 tools/run_tests.py
python3 -m unittest discover -s tests -p 'test_ignite_serving.py' -v
```

The focused tests check isolated Service selectors/ConfigMap references, pinned
checkout rejection, no-overwrite plan output, TLS/cluster mismatch rejection,
model selection, occupied ports and a real local HTTP fixture for the inference
probe. Preparing a plan and dry-running all five files on Ignite checks the
upstream integration without allocating new GPUs. A `connect.py --check --test`
against an already running authorized service verifies the actual client route.

Not established: fresh-user entitlement/authentication, native Lux-host client
execution, forced pod replacement, credential expiry, standalone Docker, complete
agent-client end-to-end discovery, quality evaluation, long-context limits,
multiuser throughput or latency under load. The 262,144-token configured context
and 1,024-sequence scheduler limit are configuration, not measured user capacity.
