# Getting Started

Lux Skills is a collection of reusable Agent Skills for running AI workloads on OLCF Lux.
There is nothing to compile at the repository root.

## Clone The Repo

```bash
git clone TODO:lux-skills-repository-url lux-skills
cd lux-skills
```

## Use The Skills Catalog

The repository organizes skills under `skills/<domain>/<skill>/`.
Most agent clients expect each skill to be a direct child of a configured skills directory, so flatten the catalog with `unpack.sh`:

```bash
export SKILLS_DIR="$HOME/.agents/skills"
./unpack.sh --target "$SKILLS_DIR" --mode copy
```

## Use The Skills In VISTA

The Lux skills ship VISTA HPC jobs under `hpc_jobs/`. Import the skill folder in VISTA (**Skills → Import**), load it into a project, and ask for a job such as `primus-skill/primus-smoke`, `vllm-skill/inkling-serve` (Inkling) `vllm-skill/muse-serve` (Muse Spark Open) `autogen-skill/icf-campaign` (the ICF agents, against an Inkling server) or `parallel-attention-skill/ringx-bench` (RingX kernel tests and benchmark) on `lux`.

## Next Steps

- Read the root `README.md` for the full project overview.
- Read each skill's `SKILL.md` for its Lux prerequisites, jobs and `script_args`.
- See `CONTRIBUTING.md` for contribution guidelines and issue reporting.
- See `Makefile` for the local checks (`make lint`, `make policy`, `make test`, and `make verify-new-skill`).
