"""
Resume Screening Agent
-----------------------
Ranks a folder of resumes against a Job Description (JD) using
TF-IDF + cosine similarity (NLP similarity method), plus rule-based
skill/education/experience extraction for human-readable reasoning.

Usage:
    python agent.py --jd sample_data/jd.txt --resumes sample_data/resumes --out output/ranked_output.json

Author: Angadi Pallavi
"""

import argparse
import json
import csv
import os
import re
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# 1. FILE PARSING (PDF / DOCX / TXT)
# ---------------------------------------------------------------------------

def extract_text(file_path: Path) -> str:
    """Extract raw text from a resume file (.txt, .pdf, .docx)."""
    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        return file_path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            print("pypdf not installed. Run: pip install pypdf", file=sys.stderr)
            return ""
        reader = PdfReader(str(file_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if suffix == ".docx":
        try:
            import docx
        except ImportError:
            print("python-docx not installed. Run: pip install python-docx", file=sys.stderr)
            return ""
        doc = docx.Document(str(file_path))
        return "\n".join(p.text for p in doc.paragraphs)

    print(f"Unsupported file type skipped: {file_path.name}", file=sys.stderr)
    return ""


# ---------------------------------------------------------------------------
# 2. SKILL / EDUCATION / EXPERIENCE EXTRACTION (rule-based, transparent)
# ---------------------------------------------------------------------------

# A reasonably broad skill vocabulary. Extend this list as needed.
SKILL_VOCAB = [
    "python", "java", "javascript", "sql", "nosql", "mysql", "postgresql",
    "sqlite", "machine learning", "deep learning", "nlp", "pytorch",
    "tensorflow", "scikit-learn", "pandas", "numpy", "rest api", "restful",
    "flask", "fastapi", "django", "docker", "kubernetes", "aws", "gcp",
    "azure", "git", "github", "html", "css", "react", "node.js", "excel",
    "tableau", "power bi", "data analysis", "data science", "llm",
    "openai", "anthropic", "claude", "gpt", "communication", "accounting",
]

EDUCATION_PATTERNS = [
    r"b\.?\s?tech", r"m\.?\s?tech", r"b\.?\s?sc", r"m\.?\s?sc", r"b\.?\s?e\b",
    r"b\.?\s?com", r"mba", r"phd", r"bachelor", r"master",
]

EXPERIENCE_YEARS_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:\+)?\s*year", re.IGNORECASE
)
EXPERIENCE_MONTHS_PATTERN = re.compile(
    r"(\d+)\s*month", re.IGNORECASE
)


def extract_skills(text: str) -> list:
    text_lower = text.lower()
    found = [skill for skill in SKILL_VOCAB if skill in text_lower]
    return sorted(set(found))


def extract_education(text: str) -> str:
    text_lower = text.lower()
    for pattern in EDUCATION_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            # grab a short surrounding snippet for context
            start = max(0, match.start() - 5)
            end = min(len(text), match.end() + 40)
            return text[start:end].strip().replace("\n", " ")
    return "Not specified"


def extract_experience_years(text: str) -> float:
    """Rough heuristic: sum explicit year mentions, convert month mentions to years."""
    years = [float(y) for y in EXPERIENCE_YEARS_PATTERN.findall(text)]
    months = [int(m) for m in EXPERIENCE_MONTHS_PATTERN.findall(text)]
    total = sum(years) + sum(months) / 12
    return round(total, 2)


# ---------------------------------------------------------------------------
# 3. SCORING: TF-IDF + COSINE SIMILARITY
# ---------------------------------------------------------------------------

def compute_similarity_scores(jd_text: str, resume_texts: list) -> list:
    """
    Returns a list of similarity scores (0-1) between the JD and each resume,
    using TF-IDF vectorization and cosine similarity.
    """
    corpus = [jd_text] + resume_texts
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)

    jd_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]

    scores = cosine_similarity(jd_vector, resume_vectors)[0]
    return scores.tolist()


def skill_overlap_score(jd_skills: list, resume_skills: list) -> float:
    """Fraction of JD-required skills present in the resume (0-1)."""
    if not jd_skills:
        return 0.0
    overlap = set(jd_skills) & set(resume_skills)
    return round(len(overlap) / len(jd_skills), 3)


# ---------------------------------------------------------------------------
# 4. MAIN PIPELINE
# ---------------------------------------------------------------------------

def run(jd_path: str, resumes_dir: str, out_path: str):
    jd_file = Path(jd_path)
    resumes_folder = Path(resumes_dir)

    if not jd_file.exists():
        print(f"JD file not found: {jd_path}", file=sys.stderr)
        sys.exit(1)
    if not resumes_folder.exists():
        print(f"Resumes folder not found: {resumes_dir}", file=sys.stderr)
        sys.exit(1)

    jd_text = extract_text(jd_file)
    jd_skills = extract_skills(jd_text)

    resume_files = sorted([
        f for f in resumes_folder.iterdir()
        if f.suffix.lower() in (".txt", ".pdf", ".docx")
    ])

    if not resume_files:
        print("No resumes found in the given folder.", file=sys.stderr)
        sys.exit(1)

    resume_texts, records = [], []
    for f in resume_files:
        text = extract_text(f)
        resume_texts.append(text)
        records.append({
            "file": f.name,
            "text": text,
            "skills": extract_skills(text),
            "education": extract_education(text),
            "experience_years": extract_experience_years(text),
        })

    # NLP similarity (semantic-ish overlap of full document content)
    tfidf_scores = compute_similarity_scores(jd_text, resume_texts)

    results = []
    for rec, tfidf_score in zip(records, tfidf_scores):
        overlap_score = skill_overlap_score(jd_skills, rec["skills"])
        # Final score: weighted blend of TF-IDF similarity (60%) and
        # explicit required-skill overlap (40%). Blend keeps the ranking
        # robust to resumes that are well-written (high TF-IDF) but light
        # on the specific required skills, and vice versa.
        final_score = round(0.6 * tfidf_score + 0.4 * overlap_score, 4)

        matched = sorted(set(jd_skills) & set(rec["skills"]))
        missing = sorted(set(jd_skills) - set(rec["skills"]))

        reasoning = (
            f"TF-IDF similarity to JD: {round(tfidf_score, 3)}. "
            f"Matched {len(matched)}/{len(jd_skills)} required skills "
            f"({', '.join(matched) if matched else 'none'}). "
            f"Missing: {', '.join(missing) if missing else 'none'}. "
            f"Approx. experience: {rec['experience_years']} years. "
            f"Education: {rec['education']}."
        )

        results.append({
            "file": rec["file"],
            "final_score": final_score,
            "tfidf_similarity": round(tfidf_score, 4),
            "skill_overlap_score": overlap_score,
            "matched_skills": matched,
            "missing_skills": missing,
            "experience_years": rec["experience_years"],
            "education": rec["education"],
            "reasoning": reasoning,
        })

    # Rank: highest score first
    results.sort(key=lambda r: r["final_score"], reverse=True)
    for i, r in enumerate(results, start=1):
        r["rank"] = i

    # Write JSON
    out_json = Path(out_path)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Write CSV alongside
    out_csv = out_json.with_suffix(".csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "rank", "file", "final_score", "tfidf_similarity",
            "skill_overlap_score", "matched_skills", "missing_skills",
            "experience_years", "education", "reasoning",
        ])
        writer.writeheader()
        for r in results:
            row = r.copy()
            row["matched_skills"] = "; ".join(row["matched_skills"])
            row["missing_skills"] = "; ".join(row["missing_skills"])
            writer.writerow(row)

    # Print shortlist to console
    print(f"\nShortlist ranked against JD: {jd_file.name}\n" + "-" * 60)
    for r in results:
        print(f"#{r['rank']}  {r['file']:<30} score={r['final_score']}")
    print(f"\nFull results written to:\n  {out_json}\n  {out_csv}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume Screening Agent")
    parser.add_argument("--jd", required=True, help="Path to job description text file")
    parser.add_argument("--resumes", required=True, help="Path to folder of resumes")
    parser.add_argument("--out", default="output/ranked_output.json", help="Output JSON path")
    args = parser.parse_args()

    run(args.jd, args.resumes, args.out)
