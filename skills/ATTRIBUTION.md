# Attribution

The skills in this directory run AI workloads on the HPC side of OLCF Lux
(Slurm, ROCm, containers, Orion Lustre and node-local NVMe). None of them uses
the Lux Kubernetes (Rancher/kubectl) side.

The skills' own code (job folders, wrappers, `env.lux.sh`, docs) is licensed
Apache-2.0; each skill carries its licence at `<skill>/LICENSE`. Third-party content the skills ship, run or read keeps its
own terms, listed in each skill's `SKILL.md` § "License and acknowledgments".

**Authors:**
- Junqi Yin, Oak Ridge National Laboratory (ORNL)

## autogen-skill

**Original repository:** https://github.com/jqyin/lux-autogen-skill (imported as
`lux-autogen-skill`, renamed `autogen-on-lux` → `autogen-skill`).

The app under `autogen-skill/hpc_jobs/icf-campaign/icf_4agent/` is AMD's
four-agent ICF tutorial app, MIT-licensed (Copyright (c) 2026 Advanced Micro
Devices, Inc.; `LICENSE` in that directory). AutoGen (pyautogen 0.2) and the
other Python packages are installed from PyPI by the job, not shipped.
`icf_4agent/tiktoken/o200k_base.tiktoken` is a copy of OpenAI's o200k_base
vocabulary, shipped for offline use (TODO: confirm its licence terms).

## parallel-attention-skill

Written for this repository from RingX
([jqyin/ringX-attention](https://github.com/jqyin/ringX-attention), MIT, Copyright
2025) and its SC '25 paper (Yin, Palash, Shankar, Wang; DOI
10.1145/3712285.3759859). It links to, and quotes short snippets from:

- RingX: the kernels, the benchmark and the tests;
- [ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention) (zhuzilin): the benchmark baselines;
- [climate-vit](https://github.com/jqyin/climate-vit): the ViT example;
- [forge](https://github.com/at-aaims/forge), `xforge` branch: the GPT example.

RingX and ring-flash-attention are not vendored: the `ringx-bench` job clones and
installs them on Lux. The job ships small adapted files: `bench_ringx.py` (RingX's
benchmark loop), `tests/test_ringX4_attn_func.py` (adapted from RingX's ringX3
test) and `shims/utils.py` (ring-flash-attention's test helpers), all from MIT code.

## primus-skill

**Original repository:** https://github.com/jqyin/lux-primus-skill (imported as
`lux-primus-skill`, renamed `primus-on-lux` → `primus-skill`).

Adapted from AMD's work: Primus (AMD-AGI/Primus, MIT) and the `rocm/primus`
image, and AMD's Mistral-Large MFU replication package. Megatron-LM (Primus
submodule) is BSD-3-Clause style, Copyright (c) NVIDIA Corporation.

## vllm-skill

**Original repository:** https://github.com/jqyin/lux-vllm-skill (imported as
`lux-vllm-skill`, renamed `vllm-on-lux` → `vllm-skill`).

The engine under `vllm-skill/hpc_jobs/inkling-serve/lux-inference/` is adapted
from AMD's Zenith inference package (`zenith_inference_v8_lux`); copyright stays
with AMD and its contributors. vLLM is Apache-2.0. See
`vllm-skill/hpc_jobs/inkling-serve/lux-inference/README.md` § Licensing.

The Muse Spark Open engine under `vllm-skill/hpc_jobs/muse-serve/lux-inference/`
is ported from AMD's Zenith v0.1.9 (https://github.com/jbelofamd/zenith, tag
`v0.1.9`, commit `70e5f0bf`), MIT-licensed (Copyright (c) 2026 The Zenith
authors; `LICENSE-zenith` in that directory), with Zenith -> Lux renames and
the Lux additions listed in its `README.md`, which are Apache-2.0.

## verl-grpo-skill

Written for this repository as a wrapper around tutorial 4 from the Lux
AI-for-Science package at `lux-primus/lux_hackathon_AI4Sci_tutorials`.
Its GRPO assets are also shared under `Genesis-Hackathon`, the skill's default
package root. The tutorial's verl GRPO driver, custom science reward, Ray/Slurm
scripts, prepared data, model weights and containers are read from that package;
none is vendored here. This subtree's original wrappers and documentation are
Apache-2.0. The Mistral model and its derivatives use the Mistral Research
Licence; the package's `README.hackathon` and tutorial 4 `README.md` give the
data and result attribution.
