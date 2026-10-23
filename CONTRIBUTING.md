# Contributing to Lux Skills

Thank you for your interest in contributing to Lux Skills. This guide covers the [Claude Code skills format](https://code.claude.com/docs/en/skills) as a baseline, the [VISTA HPC skill format](https://genesis-vista.github.io/docs/modsim-skill) every skill here follows, plus our conventions for portable multi-client skills. The development repository is TODO: (not settled yet).

---

## Quick Start

1. **Fork** the repository and clone locally
2. **Create a branch** for your skill: `git checkout -b skill/your-skill-name`
3. **Develop** your skill following the specification below
4. **Validate** your skill in the client(s) you intend to support
5. **Submit** a pull request for review

### Where to Add Your Skill

Lux is the one target, so skills sit directly in `skills/` (`skills/<skill>/SKILL.md`),
with no domain folder:

- `primus-skill` (training), `vllm-skill` (inference) and `autogen-skill` (agents on a vLLM endpoint), all on the Lux HPC side
  (Slurm, ROCm, containers, Orion, NVMe)
- `parallel-attention-skill` (RingX context-parallel attention: choosing, benchmarking, integrating)

Add a new skill as `skills/<skill>/` and list its authors in `skills/ATTRIBUTION.md`.

State in each skill which side of Lux it targets (HPC or Kubernetes). A skill
that covers both gives each side its own clearly labeled section. Take every
Lux command, module, partition and path from the
[Lux User Guide](https://docs.olcf.ornl.gov/systems/lux_user_guide.html), or
mark it `TODO:`.

### VISTA HPC Jobs

Each job lives at `<skill>/hpc_jobs/<job>/` with a `README.md` whose first line
is `# <job>`, a `cluster_defaults.json` with a `lux` section (`iri.partition`,
`resources.gpus_per_node`, `duration` in seconds, at most 4 hours), and a
`job.lux.slurm` with no `#SBATCH` lines that writes under `$VISTA_OUT` and
accepts only allow-listed `KEY=VALUE` arguments. The `SKILL.md` `name` must
equal the skill folder name and must name each job as `<skill>/<job>`. Check a
skill with VISTA's `hpc-skill-author` validator before opening a PR.

### New Skill Pull Request Checklist

Use this checklist when your PR adds or changes a skill:

1. Create or update the skill subtree under `skills/<domain>/<skill>/`.
2. Keep the skill instructions in `SKILL.md` and any supporting files in the same subtree.
3. Add or update the relevant attribution file in that subtree when the skill has authors, upstream sources, or provenance notes.
4. Put new author names in the attribution source file first, not directly in the root README.
5. Add a subtree `LICENSE` or `LICENSE.txt` when imported content requires its own license; summarize third-party provenance in `NOTICE`.
6. Update the root `README.md` repository structure block if the visible tree changed.
7. If you introduce a new top-level skill family or a new attribution pattern, update `tools/repo_inventory.py` and its tests so the repository-policy check still understands the layout.
8. Run `make verify-new-skill` before opening the PR. This wraps skill validation, repo policy, tests, and lint in one command.

### Skill Portability

The compatibility workflow in [.github/workflows/validate-skills.yml](.github/workflows/validate-skills.yml) runs the validator across the supported client profiles. When writing skill docs, prefer skill-root-relative paths and plain-language references to bundled files. Avoid hard-coding `.claude/skills/` in portable instructions unless the step is truly Claude-only, and keep any client-specific syntax isolated and clearly labeled.
For guidance on AI/LLM-assisted contributions, refer to the section below: [Guidelines for AI/LLM-Assisted Contributions](#guidelines-for-ai-llm-assisted-contributions).
For a PR-oriented checklist, see [.github/pull_request_template.md](.github/pull_request_template.md).

### Upstream Provenance and Submodules

When a skill is derived from, mirrors, or is maintained alongside an upstream repository, please identify that repository in the relevant attribution or documentation file so the provenance is clear. Contributions delivered through `git submodule` are also acceptable when the relationship is documented clearly, including the upstream source, the reason for using a submodule, and any license or attribution obligations that apply.

### Local Hygiene

The repository includes [.pre-commit-config.yaml](.pre-commit-config.yaml) for workflow YAML, `unpack.sh`, and the main Markdown docs. If you have those tools installed locally, run `pre-commit run --all-files` before opening a pull request.

---

## Skill Structure

Each skill is a directory with `SKILL.md` as the entry point:

```
<skill-name>/
├── SKILL.md              # Main instructions (required)
├── reference.md          # Detailed documentation (optional)
├── examples/             # Example outputs (optional)
│   └── sample-output.md
└── scripts/              # Utility scripts (optional)
    └── validate.py
```

### SKILL.md Format

Every skill has YAML frontmatter followed by markdown instructions:

```yaml
---
name: climate-data-fetch
description: >
  Retrieve climate model outputs from ESGF federated archives.
  Use when fetching CMIP5, CMIP6, or E3SM data for analysis.
allowed-tools: Bash(python *), Bash(curl *), Read, Write
---

# Climate Data Retrieval

You are a climate data specialist with expertise in federated data archives
and CF conventions.

## Your Capabilities

- Search ESGF nodes for CMIP5, CMIP6, and E3SM datasets
- Validate data availability across DOE data nodes (LLNL, ANL, ORNL)
- Subset data by variable, time range, and spatial region
- Provide metadata summaries following CF conventions

## Workflow

1. **Clarify requirements**: Confirm variable, model, experiment, and time range
2. **Search availability**: Query ESGF to find matching datasets
3. **Select data node**: Prefer DOE nodes when multiple sources exist
4. **Retrieve data**: Download or generate access scripts
5. **Validate**: Check data integrity and report any quality flags

## Guidelines

- Always verify data availability before committing to a retrieval plan
- Report known data retractions or quality issues
- Include DOI and citation information with retrieved datasets
- Warn about large data volumes before initiating transfers

## Arguments

The user may provide: `$ARGUMENTS`

Expected format: `<variable> <model> [time_range]`

Examples:
- `tas CESM2` — surface temperature from CESM2, all available times
- `pr E3SM 2015-2100` — precipitation from E3SM, 2015-2100
```

---

## Frontmatter Reference

| Field | Required | Description |
|-------|----------|-------------|
| `name` | No | Skill name (defaults to directory name). Lowercase, hyphens, max 64 chars. |
| `description` | **Yes** | What the skill does and when to use it. Claude uses this to decide when to activate. Front-load key use cases; truncated at 250 chars in listings. |
| `allowed-tools` | No | Tools Claude can use without asking permission. See examples below. |
| `disable-model-invocation` | No | Set `true` to prevent automatic activation. User must invoke with `/name`. |
| `user-invocable` | No | Set `false` to hide from `/` menu. For background knowledge only. |
| `context` | No | Set `fork` to run in an isolated subagent. |
| `agent` | No | Subagent type when `context: fork`. Options: `Explore`, `Plan`, `general-purpose`. |
| `paths` | No | Glob patterns limiting when skill activates (e.g., `*.nc, *.zarr`). |

### Tool Permissions

The `allowed-tools` field grants Claude permission to use specific tools without asking:

```yaml
# Allow specific bash commands
allowed-tools: Bash(python *), Bash(curl *), Bash(sbatch *)

# Allow file operations
allowed-tools: Read, Write, Glob, Grep

# Allow web access
allowed-tools: WebFetch, WebSearch

# Combine multiple
allowed-tools: Read, Bash(python *), Bash(pytest *)
```

### Variable Substitutions

The placeholders below are Claude Code-specific. Use them only when authoring for that client.

| Variable | Description |
|----------|-------------|
| `$ARGUMENTS` | All arguments passed when invoking the skill |
| `$ARGUMENTS[N]` or `$N` | Specific argument by index (0-based) |
| `${CLAUDE_SKILL_DIR}` | Directory containing this skill's SKILL.md |
| `${CLAUDE_SESSION_ID}` | Current session ID |

### Dynamic Context Injection

Run shell commands before Claude sees the skill content:

```yaml
---
name: hpc-status
description: Check HPC job status and queue state
allowed-tools: Bash(squeue *), Bash(sacct *)
---

## Current Queue State

!`squeue -u $USER --format="%.10i %.20j %.8T %.10M %.6D %R" 2>/dev/null || echo "Queue unavailable"`

## Your Task

Based on the queue state above, help the user with: $ARGUMENTS
```

The `` !`command` `` syntax runs before Claude receives the prompt.

---

## Supporting Files

Keep `SKILL.md` focused (<500 lines). Move detailed content to supporting files:

```
my-skill/
├── SKILL.md           # Overview and main instructions
├── api-reference.md   # Detailed API documentation
├── examples.md        # Usage examples
└── scripts/
    └── helper.py      # Utility scripts Claude can execute
```

Reference supporting files from SKILL.md:

```markdown
## Additional Resources

- For ESGF API details, see [api-reference.md](api-reference.md)
- For usage examples, see [examples.md](examples.md)

To validate retrieved data, run:
```bash
python scripts/validate.py <dataset_path>
```
```

---

## Writing Effective Skills

### Prompt Engineering

**Be specific about scope**
```markdown
# Good
You specialize in CMIP6 climate data from ESGF. You can search, retrieve,
and validate datasets but do not perform climate analysis.

# Too vague
You help with climate data.
```

**Include guardrails**
```markdown
## Limitations

- Do not modify existing data files without explicit confirmation
- Warn before downloads exceeding 10 GB
- Do not store credentials in scripts or outputs
```

**Show expected formats**
```markdown
## Output Format

Provide results as:

| Variable | Model | Experiment | Time Range | Data Node | Size |
|----------|-------|------------|------------|-----------|------|
| tas | CESM2 | ssp585 | 2015-2100 | llnl | 2.3 GB |
```

### Scientific Rigor

- **Cite sources** for domain knowledge embedded in prompts
- **Document assumptions** and limitations explicitly
- **Include uncertainty** when relevant
- **Prefer established methods** over novel approaches
- **Reference standards** (CF conventions, ISO formats, community schemas)

### For HPC Skills

```yaml
---
name: hpc-job-submit
description: Submit and monitor jobs on Slurm/PBS HPC systems
allowed-tools: Bash(sbatch *), Bash(squeue *), Bash(scancel *), Read, Write
disable-model-invocation: true  # User must explicitly invoke
---

# HPC Job Submission

## Safety Guidelines

- Never submit jobs without user confirmation of the job script
- Always show estimated SU cost before submission
- Limit default node count to project allocation
- Include wall time estimates based on problem size

## Workflow

1. Review or create job script
2. Validate resource requests against allocation
3. Confirm with user before submission
4. Submit and report job ID
5. Provide monitoring commands
```

---

## Running Skills in Subagents

Use `context: fork` for tasks that benefit from isolation:

```yaml
---
name: deep-literature-search
description: Comprehensive literature search across multiple databases
context: fork
agent: Explore
allowed-tools: WebSearch, WebFetch, Read, Write
---

# Literature Search Agent

Conduct a thorough literature search for: $ARGUMENTS

## Search Strategy

1. Search Google Scholar, arXiv, and PubMed
2. Identify seminal papers and recent advances
3. Map citation relationships
4. Synthesize findings into a structured report

## Output

Create a markdown report with:
- Executive summary
- Key papers (with citations)
- Research gaps
- Suggested reading order
```

---

## Example Skills

### Minimal Skill

```yaml
---
name: fortran-help
description: Help with Fortran code, focusing on modern Fortran best practices
---

You are a Fortran expert specializing in modern Fortran (2003/2008/2018).

When helping with Fortran code:
- Prefer modern constructs over legacy patterns
- Use modules and explicit interfaces
- Recommend `implicit none` in all scopes
- Suggest array operations over explicit loops where appropriate
```

### Skill with Scripts

```yaml
---
name: netcdf-inspect
description: Inspect and summarize NetCDF file structure and contents
allowed-tools: Bash(python *), Read
---

# NetCDF Inspector

Analyze NetCDF files and provide structured summaries.

## Quick Inspection

Run the bundled inspection script:

```bash
python scripts/inspect_nc.py <dataset_path>
```

## Testing Your Skill

### Manual Testing

1. Install the skill in your Claude Code skills directory
2. Start a new Claude Code session
3. Check the skill is recognized: "What skills are available?"
4. Test invocation: `/your-skill-name test arguments`
5. Test automatic activation with matching prompts

### Test Cases to Cover

- **Happy path**: Skill works with valid inputs
- **Edge cases**: Missing arguments, unusual inputs
- **Error handling**: Invalid data, network failures
- **Guardrails**: Skill refuses inappropriate requests

---

## Pull Request Process

### Before Submitting

- [ ] `SKILL.md` has a clear, specific description
- [ ] Instructions are tested in Claude Code
- [ ] Supporting files are referenced from SKILL.md
- [ ] No sensitive data or credentials in code
- [ ] Scientific claims include citations or references

### Review Criteria

1. **Usefulness** — Does this skill address a real research need?
2. **Correctness** — Is the domain knowledge accurate?
3. **Safety** — Are appropriate guardrails in place?
4. **Clarity** — Are instructions unambiguous?
5. **Composability** — Can it work alongside other skills?

### Naming Conventions

- Use lowercase with hyphens: `climate-data-fetch`, not `ClimateDataFetch`
- Be specific: `cmip6-search` rather than `data-search`
- Include domain when helpful: `mpi-debug`, `slurm-submit`
- Choose the appropriate skills category (see "Where to Add Your Skill" above)
- If your skill spans multiple domains, discuss placement in your pull request

---

## Community Guidelines

This project and everyone participating in it is governed by our [Code of Conduct](./CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code.

### Communication

- Use GitHub issues for bugs and feature requests
- Tag issues appropriately: `new-skill`, `bug`, `enhancement`
- Be respectful and constructive

### Collaboration

- Credit contributors in skill descriptions
- Build on existing skills rather than duplicating
- Share learnings with the community

### Guidelines for AI/LLM-Assisted Contributions

- **Remain accountable for all your outputs and decisions.**
   Individuals remain fully responsible and accountable for the accuracy, quality, appropriateness, and consequences of their work. Use of AI does not transfer this responsibility to the AI model, agent, or other tool.
- **Understand your work.**
   Regardless of how code or PR was produced, this project requires that authors illustrate a thorough understanding of any proposed changes. You must review such code line-by-line; it is your responsibility to ensure that it is correct, and that it does not breach copyright. Always critically engage with AI outputs, do not trust them implicitly. AI-assisted code, analysis, and artifacts must be tested and validated at a level appropriate to their impact. Authors are responsible for ensuring that generated code is correct, secure, maintainable, non-obfuscated, appropriately scoped, documented, and reproducible where relevant.
- **Disclose AI-generated or AI-assisted work.**
   If AI/LLM tools were primarily used to generate code or artifacts, this should be clearly indicated in the PR.
- **Use of AI to review PRs.**
   All PRs must be reviewed by a human reviewer. An LLM review may be used in addition to a human reviewer since this can help spot issues that a human may have missed, but this should not be the sole reviewer. The human reviewer should be fully accountable and responsible for the review feedback or comments (see 1).
- **Proprietary or personal information.**
   For this project, proprietary or personal information should never be sent to code generators or AI tools.
- **Be transparent, assume goodwill, and share what you learn.**
   Contributors should be open about relevant AI use, disclose details of AI use as appropriate to the project, engage constructively with colleagues, and share experiences and lessons learned with the project.

---

## Recognition

Contributors are recognized through:
- Author credits in skill metadata
- Acknowledgment in the [Contributors](README.md#contributors) section of the README and in release notes
- Mention in release notes for the skills they touched

We value all contributions, including new skills, improvements, documentation, and feedback.
