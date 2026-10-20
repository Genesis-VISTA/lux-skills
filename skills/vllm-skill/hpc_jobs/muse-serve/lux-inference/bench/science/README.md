# bench/science: physics, chemistry and materials-science suite

Three public benchmarks, run zero-shot with chain of thought against any OpenAI-compatible chat endpoint
(the servers in `docker/`, or anything else). The client uses only the Python standard library.

| benchmark | questions scored | what it tests | source, licence |
|---|---|---|---|
| GPQA Diamond, physics + chemistry | 179 (86 physics, 93 chemistry) | graduate-level questions written by domain experts to resist look-up; 4 options | [idavidrein/gpqa](https://github.com/idavidrein/gpqa), CC BY 4.0 |
| MaScQA | 646 | materials science from India's GATE graduate entrance exams, 14 topics: 284 multiple choice, 70 matching, 67 multiple choice with a numerical step, 225 numerical answers | [M3RG-IITD/MaScQA](https://github.com/M3RG-IITD/MaScQA), CC BY-NC-SA 4.0 (non-commercial) |
| MMLU-Pro physics + chemistry + engineering | 3,400 (1,299 / 1,132 / 969) | broad undergraduate coverage, 10 options; big enough for tight intervals | [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro), MIT |

GPQA is the hard test, MaScQA the materials test, and MMLU-Pro gives statistical power: at 179 questions GPQA's
95% interval is about +/-7 points, at 3,400 MMLU-Pro's is about +/-1.7.

## 1. Get the data (once, on a host with internet)

```bash
pip install pyarrow          # only MMLU-Pro needs it (its test split is a parquet file)
ZENITH_DATASETSPATH=/path/to/datasets python3 bench/science/fetch_science_data.py
# -> /path/to/datasets/science/{gpqa_diamond,mascqa,mmlu_pro_sci}.jsonl
```

The data is not shipped in this package; the script downloads it, pinned to fixed commits and checked against
sha256 hashes, so every site scores the same questions. On Lux, run it on a login node (they have direct internet;
compute nodes do not).

**Keep GPQA off the web.** Its authors distribute it in a password-protected zip so that the questions stay out of
web crawls (and so out of future training data), and ask that examples not be posted online. The same goes for the
suite's output JSON, which contains every question's reply. Do not commit either to a public repository.

Preparation details:

- **GPQA:** the four options are shuffled once per question with a fixed seed. Biology (19 questions) is fetched
  but skipped unless `--gpqa-domains all`.
- **MaScQA:**
  - Answers come from the repo's `scoresheets/all_questions.xlsx`.
  - One question is listed twice upstream and is kept once.
  - Three are dropped: the exam awarded marks to all for them.
  - Numerical answers are the exam's accepted range ("0.85 to 0.89"); two typos in that sheet are fixed.
  - Multiple-select keys ("A,C") must be matched exactly; "C OR D" accepts either.
- **MMLU-Pro:** the physics, chemistry and engineering rows of the test split.

## 2. Run it against a server

```bash
python3 bench/science/science_suite.py --host <head node> --port 8000 \
    --data-dir /path/to/datasets/science --concurrency 128 --out science.json | tee science.txt
```

- **Prompts:** temperature 0. Each prompt ends with an instruction to finish on "The answer is (X)" (or "The answer
  is N" for numerical questions). MMLU-Pro uses its own official zero-shot instruction.
- **Extraction:** the last "answer is ..."; failing that, "Answer: X"; for multiple choice, failing both, the last
  standalone option letter. Numbers are read in plain, comma-grouped, `e` or `x 10^n` form. A reply with no answer
  is scored wrong.
- **Reasoning models:** a reply cut off while thinking (no `content`) is scored from its `reasoning`, the same rule
  as `bench/inkling/mmlu_pro_quick.py`. `--effort` passes a chat-template `reasoning_effort`.
- **Smoke test:** `--n 20` runs the first 20 questions of each benchmark. `--benchmarks gpqa,mascqa` runs a subset.

The report gives:

- accuracy with a Wilson 95% interval for each benchmark;
- breakdowns by GPQA domain and subdomain, MaScQA question type and topic, and MMLU-Pro category;
- per-field accuracy: physics = GPQA physics + MMLU-Pro physics; chemistry likewise; materials = MaScQA;
  engineering = MMLU-Pro engineering;
- the **suite score**, the unweighted mean of the three benchmark accuracies, so the 3,400 MMLU-Pro questions do
  not swamp the other two.

Truncated, errored and unanswered replies are counted, so a broken deployment shows up as such rather than as a
low score.

## Comparing two models (e.g. a fine-tune against its base)

Serve both the same way, run the suite against each with the same flags, and compare per benchmark and per field
using the intervals. With temperature 0 a rerun of the same server reproduces almost exactly. Differences under
about 2 points on MMLU-Pro, 4 on MaScQA or 7 on GPQA are within noise.

Mistral-Large-Instruct-2411's baseline on Lux is in `docs/LUX_MISTRAL_LARGE.md`.
