# icf-campaign

Run the hackathon's four-agent ICF capsule optimizer headless on OLCF Lux. There are four agents: the PI, the
Design Explorer, the Defect Adversary and an execution proxy. They run AutoGen 0.2 over an MCP physics surrogate.
A job is either one design campaign at a given laser energy or a sweep over laser energies. It can also render
the best design's implosion movie and ask the PI for a written debrief. Part of the `autogen-skill` skill,
which explains how to read the results.

Supported clusters: Lux

The agents need an OpenAI-compatible LLM endpoint. There are two backends:

- `BACKEND=endpoint` (default): a running `vllm-skill/inkling-serve` job (Inkling-MXFP4). The job finds your newest
  one by itself, or takes `SERVE_JOB=<id>` or `ENDPOINT=<url>`. It runs on one whole node (the defaults).
  If the server is still starting, the job waits up to 40 min for it.
- `BACKEND=self`: the job serves gpt-oss-20b (`gptoss-20b-hedp`) itself with vLLM on GPU 0 of its own node. It
  needs the whole node, which the defaults (`exclusive_node_use=true, gpus_per_node=8`) give it. vLLM is up
  ~4 min after the job starts.

Default time: 1:00:00. The job ends by itself. Measured with Inkling-MXFP4: a 3 MJ campaign ~7 min, and a
3-point sweep (`2:4:1`) ~33 min. Each sweep point is a full 10-round campaign, so allow ~11 min per point and
raise `duration` for longer sweeps. With `BACKEND=self`, gpt-oss-20b replies faster: vLLM starts in ~4 min, and
a 3 MJ campaign takes ~3 min.

The job builds the app's Python venv itself, on node-local disk. From a compute node, Lustre is far too slow
for the venv's ~22,000 small files. The job uses the node's NVMe (`/mnt/bb/<jobid>`) when it has one, and `/tmp`
otherwise: jobs submitted without the nvme constraint, such as VISTA's S3M submissions, have no NVMe. The first job
installs the pinned packages from PyPI through the OLCF proxy (~20 s) and saves the venv as one tar (~630 MB) in
the job's shared directory. Later jobs unpack it in ~1 s. Either way the job then runs the app's offline tests.
The first line of the venv step in `job.log` says which disk the job used. `setup_lux.sh` only checks files and
builds nothing.

## Script args

`script_args` is a space-separated list of `KEY=VALUE` pairs. All are optional, and any other key is refused:

- `BACKEND=endpoint|self`: see above.
- `SERVE_JOB=<jobid>`: the inkling-serve job to use (default: your newest running one). `ENDPOINT=<url>`: any
  OpenAI base URL reachable from Lux compute nodes, e.g. `http://lux439:8000/v1`.
- `ENERGY_MJ=<MJ>`: one campaign at this laser energy (default 3).
- `SWEEP_MJ=<lo>:<hi>:<step>`: one campaign per laser energy instead, e.g. `2:4:1`. Each point costs about as
  much as one campaign.
- `MOVIE=0|1` (default 1): the implosion movie of the best design and its time histories.
- `DEBRIEF=0|1` (default 1): the PI's written analysis of the session.
- `REASONING_EFFORT=none|minimal|low|medium|high|max`, `MAX_TOKENS=<n>`, `TEMPERATURE=<t>`: the agents' LLM
  settings. The defaults are medium / 8192 / 1.0 for Inkling, and the server's / 2048 / 1.0 for other models.
- `MAX_DELTA=<f>` (default 0.3), `MAX_ROUGHNESS=<m>` (default 3e-6): the Defect Adversary's budget (drive asymmetry,
  surface roughness in m).

Examples: `script_args="ENERGY_MJ=2.5"`, `script_args="SWEEP_MJ=2:4:1 DEBRIEF=0"`,
`duration="2:00:00", script_args="BACKEND=self"`.

## Outputs

In the job's output directory:

- `summary.txt`: model and settings, the robust yield (the worst case over the adversary's defect probes) and the
  best single run, the best design's parameters, the sweep table, timings. It ends with `=== CAMPAIGN OK` or
  `=== CAMPAIGN FAILED: <reason>`, including for failures before the campaign starts.
- `transcript.md`: the agents' conversation, request by request. `debrief.md`: the PI's analysis.
- `campaign.json`: every simulation's design, defects and yield.
- `implosion.mp4` and the same movie as `implosion.gif` (480 px) for viewers without video; `time_histories.png`;
  `yield_vs_laser.png` (sweeps).
- `job.log`: the job's own log, including the venv step. `vllm.log`: vLLM's log (`BACKEND=self`).
  `venv_build.log` (first job only) and `offline_tests.log`. `app/`: the job's copy of the app.
