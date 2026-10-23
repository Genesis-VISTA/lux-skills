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
