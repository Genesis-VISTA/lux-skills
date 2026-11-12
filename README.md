# Lux Skills

**Portable AI-agent workflows for running AI workloads on OLCF Lux**

Lux Skills is a collection of reusable [Agent Skills](https://agentskills.io) for
training and serving large language models on the
[OLCF Lux](https://docs.olcf.ornl.gov/systems/lux_user_guide.html) AMD Instinct
MI355X cluster. HPC skills ship jobs in the
[VISTA](https://genesis-vista.github.io/docs/modsim-skill) `hpc_jobs/` format, so
VISTA can submit it with `submit_hpc_job(job="<skill>/<job>", cluster="lux")`,
and it also works by hand from a Lux login node with Claude Code, Codex, Cursor,
OpenCode or Gemini CLI. `ignite-serving-skill` targets Lux's Ignite Kubernetes
partition through kubectl, with a shared-service client and a pinned ML4 hosting
recipe. It does not submit VISTA HPC jobs.

---

## What Are Skills?

Skills are markdown files (`SKILL.md`) with YAML frontmatter that teach an AI agent how to handle domain-specific tasks. Each skill provides:

- **Domain expertise** — Lux-specific knowledge: Slurm partition, GPUs, the RoCE fabric, containers, Orion Lustre and node-local NVMe
- **Structured workflows** — Step-by-step instructions for setting up, smoke-testing, scaling and debugging jobs
- **Execution resources** — HPC job folders for VISTA or Kubernetes workflows for Ignite
- **Supporting resources** — Scripts, configs and reference results the agent can use

---

## Skills

| Skill | Target | Description |
|---|---|---|
| [`autogen-skill`](skills/autogen-skill/SKILL.md) | Lux HPC (Slurm + rootless podman) | Run AutoGen multi-agent workflows headless: the four-agent ICF capsule optimizer (a campaign or a laser-energy sweep, with an implosion movie) against a `vllm-skill` Inkling server or gpt-oss-20b served in the job. |
| [`ignite-serving-skill`](skills/ignite-serving-skill/SKILL.md) | Lux Ignite (Kubernetes + Rancher) | Connect to a shared model or host Zenith's pinned Mistral Large 4 stack on four MI355X GPUs using persistent storage, bounded GPU Jobs and kubectl. Hosting requires an authorized Zenith checkout. |
| [`parallel-attention-skill`](skills/parallel-attention-skill/SKILL.md) | GPU clusters (Lux, Frontier) | Choose, benchmark and integrate RingX context-parallel attention (ringX1 to ringX4) for long-sequence ViT and GPT training; a Lux job tests and benchmarks the kernels against ring attention. |
| [`primus-skill`](skills/primus-skill/SKILL.md) | Lux HPC (Slurm + apptainer) | Train LLMs with AMD Primus (Megatron backend): a Mistral-7B smoke test, or Mistral-Large-Instruct-2411 123B continued pretraining and its Hugging Face export. |
| [`verl-grpo-skill`](skills/verl-grpo-skill/SKILL.md) | Lux HPC (Slurm + apptainer) | Post-train Mistral-Large 123B with verl GRPO and verifiable science rewards: an 8-node smoke test, checkpointed control/public-only runs, and science evaluation. |
| [`vllm-skill`](skills/vllm-skill/SKILL.md) | Lux HPC (Slurm + rootless podman) | Serve LLMs with vLLM as an OpenAI-compatible endpoint (Inkling-MXFP4 by default, or Muse Spark Open on one node) and score them on a science benchmark suite. |

Lux has two sides: an HPC side (Slurm, ROCm, Lmod, Orion, NVMe) and a Kubernetes
side (Rancher, kubectl). Use `ignite-serving-skill` for Kubernetes hosting and
shared-service access; `vllm-skill` retains the Slurm serving workflow.

---

## Quick Start

There's nothing to compile: clone the repo, then expose the skills to your agent.
Skills live directly under `skills/<skill>/`. Agent clients generally expect
each skill to be a direct child of a configured skills directory; `unpack.sh`
copies or links the selected skills there.

```bash
git clone TODO:lux-skills-repository-url lux-skills
cd lux-skills
```

### Flatten skills into place (`unpack.sh`)

Choose the skills directory documented by your client, then let `unpack.sh`
flatten every skill into it:

```bash
export SKILLS_DIR="$HOME/.agents/skills"  # Codex, Cursor, OpenCode, Gemini CLI
# export SKILLS_DIR="$HOME/.claude/skills"  # Claude Code
./unpack.sh --target "$SKILLS_DIR" --mode copy
```

Use `--mode symlink` for live links back to the clone, `./unpack.sh vllm-skill`
for one skill, `./unpack.sh --list` to preview, or `./unpack.sh --help` for all
options. Existing same-named destination entries are replaced.

### Use an HPC skill in VISTA

Import a skill folder (for example `skills/vllm-skill/`) in VISTA
with **Skills → Import**, load it into a project, then ask for its job, for
example `vllm-skill/inkling-serve`, `primus-skill/primus-smoke`,
`autogen-skill/icf-campaign` or `parallel-attention-skill/ringx-bench` on `lux`. Each
`SKILL.md` lists its jobs, their `script_args` keys and the output files to read.
For GRPO, submit `verl-grpo-skill/grpo-train`; it defaults to the tutorial's
three-step smoke run.

### Use a Skill

Ask naturally, for example: “Smoke-test Primus on one Lux node” or “Start a
vLLM endpoint on Lux and point my agent at it.” Your client can select the
matching skill from its description.

For Ignite, load `ignite-serving-skill` into a terminal-capable agent and ask,
for example: "Connect to my existing Ignite model service using my kubeconfig."
For a new deployment, ask it to prepare and validate an ML4 Kubernetes plan.
The skill checks the caller's actual namespace and keeps the optimized Zenith
vLLM/AITER revisions pinned. See its validation record for tested scope.

---

## Repository Structure

```
lux-skills/
├── unpack.sh                  # flatten skills into an agent's skills dir
├── skills/
│   ├── autogen-skill/SKILL.md
│   ├── ignite-serving-skill/SKILL.md
│   ├── parallel-attention-skill/SKILL.md
│   ├── primus-skill/SKILL.md
│   ├── verl-grpo-skill/SKILL.md
│   └── vllm-skill/SKILL.md
├── CONTRIBUTING.md
├── LICENSE
├── NOTICE                     # third-party licensing and attribution inventory
├── README.md
└── skill_spec.md              # Agent Skills format specification
```

Each skill is a directory containing:
- `SKILL.md` — Required entry point with frontmatter and instructions
- `hpc_jobs/<job>/` — VISTA HPC jobs where applicable (`README.md`, `cluster_defaults.json`, `job.lux.slurm`, `setup_lux.sh`)
- Supporting files — Optional scripts, configs and reference results

---

## Contributing

See the [Contributing Guide](CONTRIBUTING.md) for the skill format, the checks
to run and the review process. Use the checklist in [CONTRIBUTING.md](CONTRIBUTING.md)
and the pull request template in [.github/pull_request_template.md](.github/pull_request_template.md)
before opening a PR.

## Contributors

Individual skills and their directories contain specific attribution details. The catalog-level and skill contributors include the following:

- Nicholas Malaya
- Junqi Yin

---

## License
This product is licensed under the root [LICENSE](LICENSE) found at the root of
this repository.
Some skills under `skills/` are sourced from third parties. Where a `LICENSE`
(or `LICENSE.txt`) file is present in a subdirectory, that license governs the
contents of that subdirectory and supersedes the root license for that subtree.
The repository-level summary of individual skill and third-party licensing
details is in [NOTICE](NOTICE); see the attribution file in each identified
subtree for the licensing information supplied with that content.

## Acknowledgment
This work was supported by the U.S. Department of Energy (DOE), Office of Science, Office of Advanced Scientific Computing Research in alignment with DOE's Genesis Mission.
