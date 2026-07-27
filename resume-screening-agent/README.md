# Resume Screening Agent

An AI agent that ranks a folder of resumes against a Job Description (JD) using
NLP similarity (TF-IDF + cosine similarity) combined with rule-based skill,
education, and experience extraction. Outputs a ranked, scored shortlist with
human-readable reasoning for each candidate.

Built for the Rooman Technologies 24-Hour AI Agent Challenge (Junior AI
Research Associate — Selection Round).

## What it does

1. Reads a Job Description (`.txt`).
2. Reads a folder of resumes (`.txt`, `.pdf`, or `.docx`).
3. Extracts skills, education, and approximate years of experience from each
   resume using a keyword/regex-based parser.
4. Computes a relevance score for each resume against the JD using:
   - **TF-IDF + cosine similarity** on the full document text (captures
     overall semantic/textual overlap).
   - **Required-skill overlap** — the fraction of JD-listed required skills
     found in the resume (captures exact, high-signal matches TF-IDF alone
     can miss).
   - Final score = `0.6 * TF-IDF similarity + 0.4 * skill overlap`.
5. Ranks all candidates highest-to-lowest and writes the result to
   `output/ranked_output.json` and `output/ranked_output.csv`, each row
   including a plain-English reasoning string (matched skills, missing
   skills, experience, education).

## Installation

Requires Python 3.9+.

```bash
git clone <your-repo-url>
cd resume-screening-agent
pip install -r requirements.txt
```

No API key is required — this agent uses classical NLP (TF-IDF), not an LLM
API, so there is nothing to configure.

## Running it

```bash
python agent.py --jd sample_data/jd.txt --resumes sample_data/resumes --out output/ranked_output.json
```

- `--jd` — path to the job description text file
- `--resumes` — path to a folder containing resumes (`.txt`, `.pdf`, `.docx`)
- `--out` — where to write the ranked JSON (a CSV of the same name is also
  written automatically)

### Sample run included

This repo ships with a sample JD (`sample_data/jd.txt`, a Junior AI Research
Associate role) and 5 sample resumes (`sample_data/resumes/`) of varying
relevance, so reviewers can run the command above immediately and see a
real ranked shortlist without supplying their own data.

Sample console output:

```
Shortlist ranked against JD: jd.txt
------------------------------------------------------------
#1  resume_01_ananya.txt           score=0.338
#2  resume_03_priya.txt            score=0.3313
#3  resume_05_sneha.txt            score=0.2102
#4  resume_02_karthik.txt          score=0.082
#5  resume_04_ravi.txt             score=0.0225
```

Full per-candidate reasoning is in `output/ranked_output.json` /
`output/ranked_output.csv`, e.g.:

```json
{
  "rank": 1,
  "file": "resume_01_ananya.txt",
  "final_score": 0.338,
  "tfidf_similarity": 0.2467,
  "skill_overlap_score": 0.5,
  "matched_skills": ["git", "machine learning", "nlp", "pandas", "python", "rest api", "sql"],
  "missing_skills": ["aws", "docker", "postgresql", ...],
  "experience_years": 0.5,
  "education": "b.tech in computer science, xyz institu",
  "reasoning": "TF-IDF similarity to JD: 0.247. Matched 7/14 required skills (...). ..."
}
```

## Design choices & why

- **TF-IDF + cosine similarity over an LLM call**: given the 24-hour window,
  I chose a deterministic, dependency-light, zero-API-key method. It's fast,
  fully reproducible (no randomness or rate limits), and transparent —
  reviewers can see exactly why a score is what it is.
- **Skill overlap as a second signal**: TF-IDF alone rewards resumes that are
  verbose or reuse JD phrasing, and can under-rate a resume that lists exact
  required skills tersely. Blending in explicit skill overlap corrects for
  that.
- **Rule-based extraction over an ML/NER model**: a fixed skill vocabulary +
  regex is simple, fast to build correctly in the time available, and fully
  explainable — every extracted skill/education/experience value can be
  traced back to the exact rule that produced it.

## Tradeoffs & what I'd improve with more time

- **Skill vocabulary is a fixed list.** It won't catch skills/synonyms not in
  `SKILL_VOCAB` (e.g. "ML" vs "Machine Learning" are separate strings unless
  normalized). With more time I'd add synonym normalization or a small
  embedding-based skill matcher.
- **Experience extraction is a heuristic regex**, not a true date-range
  parser. It sums explicit "X years" / "X months" mentions, which can
  overcount if a resume repeats the same duration in multiple sections.
- **TF-IDF captures lexical overlap, not deep semantic meaning** — it won't
  know "PyTorch" and "deep learning framework" are related unless both
  literally appear. Swapping in sentence-embedding similarity (e.g.
  `sentence-transformers`) would improve semantic matching at the cost of a
  heavier dependency and slower startup — a reasonable next step outside the
  24-hour window.
- **No PDF/DOCX layout awareness** (tables, multi-column resumes can extract
  out of order). A production version would use a layout-aware parser.
- **Education extraction returns the first regex match only** — it doesn't
  disambiguate multiple degrees listed on one resume.

## Repo structure

```
resume-screening-agent/
├── agent.py                  # main script
├── requirements.txt
├── README.md
├── sample_data/
│   ├── jd.txt                 # sample Job Description
│   └── resumes/                # 5 sample resumes (varied relevance)
└── output/
    ├── ranked_output.json      # generated on run
    └── ranked_output.csv       # generated on run
```
