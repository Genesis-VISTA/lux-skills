#!/usr/bin/env python3
"""Download and prepare the science suite's three datasets (run once, on a host with internet).

usage: fetch_science_data.py [--out-dir DIR]

DIR defaults to $ZENITH_DATASETSPATH/science (or ./science). Writes three JSONL files there, one question per line,
which science_suite.py reads:

  gpqa_diamond.jsonl   GPQA Diamond (198 questions: 86 physics, 93 chemistry, 19 biology; the suite scores physics
                       and chemistry). github.com/idavidrein/gpqa, CC BY 4.0. The upstream zip is password-protected
                       (the password is published in that repo's README) so that the questions stay out of web
                       crawls; the authors ask that examples not be posted online. Keep the extracted file off the
                       web, out of git, and out of any package: this script is how it gets distributed.
  mascqa.jsonl         MaScQA (649 unique questions; the upstream lists one twice): materials science, from India's
                       GATE graduate entrance exams, 14 topics, four types (MCQS, MATCH, MCQS-NUM multiple choice;
                       NUM numerical, scored against the exam's accepted range). github.com/M3RG-IITD/MaScQA,
                       CC BY-NC-SA 4.0 (non-commercial use). Answers come from the repo's scoresheets/all_questions.xlsx.
  mmlu_pro_sci.jsonl   MMLU-Pro test split, physics (1,299) + chemistry (1,132) + engineering (969) = 3,400 questions,
                       10 options each. huggingface.co/datasets/TIGER-Lab/MMLU-Pro, MIT. Needs pyarrow to read the
                       parquet file (pip install pyarrow); the other two need only the standard library.

Every download is pinned to a commit and checked against a sha256, so two sites that run this get the same
questions. GPQA's four options are shuffled here, once, with a fixed seed per question, so the answer letter does not
depend on when or where the suite runs.
"""
import argparse, csv, hashlib, io, json, os, random, re, sys, urllib.request, zipfile
import xml.etree.ElementTree as ET

GPQA_URL = "https://raw.githubusercontent.com/idavidrein/gpqa/56686c06f5e19865c153de0fdb11be3890014df7/dataset.zip"
GPQA_SHA = "461ae7329f15a3e35f8184d2dac24b990f34fdf12f366ca4062d8e6638cd08dc"
GPQA_PWD = b"deserted-untie-orchid"   # published in the upstream README
MASCQA_BASE = "https://raw.githubusercontent.com/M3RG-IITD/MaScQA/f524889b9cbcec15f4d6216a3c66f9e6c57f0712/"
MASCQA_FILES = {"mascqa-eval.json": "545669a0474df358e096aaac5e780602c06ddf132a7d84225c4b2c336b1e78a2",
                "scoresheets/all_questions.xlsx": "00c3e95d63c5765aef49f07e13bb00313c8a51856f81ff0a6986e542440003c3"}
MMLU_PRO_URL = ("https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/"
                "b189ec765aa7ed75c8acfea42df31fdae71f97be/data/test-00000-of-00001.parquet")
MMLU_PRO_SHA = "0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8"
MMLU_PRO_CATEGORIES = ("physics", "chemistry", "engineering")
SEED = 20261011


def fetch(url, sha):
    with urllib.request.urlopen(url, timeout=300) as r:
        data = r.read()
    got = hashlib.sha256(data).hexdigest()
    if sha and got != sha:
        sys.exit(f"sha256 mismatch for {url}: got {got}, expected {sha}")
    return data, got


def write(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def gpqa(out):
    data, _ = fetch(GPQA_URL, GPQA_SHA)
    z = zipfile.ZipFile(io.BytesIO(data))
    rows = list(csv.DictReader(io.StringIO(z.read("dataset/gpqa_diamond.csv", pwd=GPQA_PWD).decode("utf-8"))))
    res = []
    for i, r in enumerate(rows):
        right = r["Correct Answer"].strip()
        opts = [right] + [r[f"Incorrect Answer {k}"].strip() for k in (1, 2, 3)]
        random.Random(SEED + i).shuffle(opts)
        res.append({"id": r["Record ID"], "domain": r["High-level domain"].strip().lower(),
                    "subdomain": r["Subdomain"].strip(), "question": r["Question"].strip(),
                    "options": opts, "answer": "ABCD"[opts.index(right)]})
    write(os.path.join(out, "gpqa_diamond.jsonl"), res)
    return len(res)


def read_xlsx(data):
    """First sheet of an .xlsx as a list of {column letter: cell text} (stdlib only)."""
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    z = zipfile.ZipFile(io.BytesIO(data))
    shared = ["".join(t.text or "" for t in si.iter(ns + "t"))
              for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(ns + "si")]
    rows = []
    for row in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(ns + "row"):
        cells = {}
        for c in row.iter(ns + "c"):
            v = c.find(ns + "v")
            if v is not None:
                cells[re.match(r"[A-Z]+", c.get("r")).group()] = shared[int(v.text)] if c.get("t") == "s" else v.text
        rows.append(cells)
    return rows


def mascqa_answer(qtype, raw):
    """Normalize the scoresheet's answer cell. Returns None for questions the exam gave marks to all for."""
    s = raw.strip().upper()
    if s == "MARKS TO ALL":
        return None
    if qtype == "NUM":
        # ranges are written "lo to hi", "lo : hi" or one value; fix the sheet's T0 (zero) and 48O (letter O) typos
        s = re.sub(r"\bT0\b", "TO", s).replace("48O", "480")
        ranges = []
        for alt in s.split(" OR "):
            nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", alt)]
            if len(nums) not in (1, 2):
                raise ValueError(f"unparsed numeric answer {raw!r}")
            lo, hi = nums[0], nums[-1]
            ranges.append([min(lo, hi), max(lo, hi)])
        return {"ranges": ranges}
    # multiple choice: "A,C" is a multiple-select key (all of them, nothing else); "C OR D" / "A; B" accept either
    return {"choices": sorted({"".join(sorted(re.findall(r"[A-D]", alt)))
                               for alt in re.split(r"\s+OR\s+|;", s)})}


def mascqa(out):
    ev = json.loads(fetch(MASCQA_BASE + "mascqa-eval.json", MASCQA_FILES["mascqa-eval.json"])[0])
    sheet = read_xlsx(fetch(MASCQA_BASE + "scoresheets/all_questions.xlsx",
                            MASCQA_FILES["scoresheets/all_questions.xlsx"])[0])
    key = {r["A"]: r for r in sheet[1:] if r.get("A")}
    res, seen, dropped = [], set(), 0
    for topic, block in ev.items():
        for qid, qtype, q in zip(block["qids"], block["qstr"], block["questions"]):
            if qid in seen:   # G-XEC-2016-14 appears twice upstream, identical both times
                continue
            seen.add(qid)
            ans = mascqa_answer(qtype, key[qid]["C"])
            if ans is None:
                dropped += 1
                continue
            res.append({"id": qid, "type": qtype, "topic": topic, "question": q.strip(), **ans})
    write(os.path.join(out, "mascqa.jsonl"), res)
    return len(res), dropped


def mmlu_pro(out):
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("MMLU-Pro needs pyarrow to read its parquet file: pip install pyarrow")
    data, got = fetch(MMLU_PRO_URL, MMLU_PRO_SHA)
    t = pq.read_table(io.BytesIO(data)).to_pylist()
    res = [{"question_id": r["question_id"], "category": r["category"], "question": r["question"],
            "options": list(r["options"]), "answer": r["answer"]}
           for r in t if r["category"] in MMLU_PRO_CATEGORIES]
    write(os.path.join(out, "mmlu_pro_sci.jsonl"), res)
    return len(res), got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(os.environ.get("ZENITH_DATASETSPATH", "."), "science"))
    ap.add_argument("--only", default="gpqa,mascqa,mmlu_pro")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    only = a.only.split(",")
    if "gpqa" in only:
        print(f"gpqa_diamond.jsonl: {gpqa(a.out_dir)} questions")
    if "mascqa" in only:
        n, d = mascqa(a.out_dir)
        print(f"mascqa.jsonl: {n} questions ({d} dropped: the exam gave marks to all)")
    if "mmlu_pro" in only:
        n, h = mmlu_pro(a.out_dir)
        print(f"mmlu_pro_sci.jsonl: {n} questions (parquet sha256 {h})")
    print(f"wrote {a.out_dir}")


if __name__ == "__main__":
    main()
