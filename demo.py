# demo.py
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi
from pydantic import BaseModel, Field, ValidationError
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from app.llm_client import OllamaClient
from app.prompts import (
    build_assessment_messages,
    build_generic_qa_messages,
    build_intent_messages,
    build_json_repair_messages,
    build_triage_messages,
)

# ---------------------------
# Capstone requirements (only used for ASSESSMENT_GEN)
# ---------------------------

CAPSTONE_CONTEXT = """Add one research-grade requirement:
- Students must implement a prompt-injection test suite for their RAG pipeline
- They must report: attack success rate, false block rate, JSON validity rate, and citation coverage
- Include an ethics/compliance section (cyber law/reg angle)
"""


# ---------------------------
# Schemas (structured JSON)
# ---------------------------

class IntentOut(BaseModel):
    intent: str = Field(..., description="GENERIC_QA or ASSESSMENT_GEN")
    confidence: float = Field(..., ge=0.0, le=1.0)


class Threat(BaseModel):
    type: str
    severity: str
    evidence: str


class TriageOut(BaseModel):
    action: str = Field(..., description="ALLOW, ALLOW_WITH_GUARDRAILS, BLOCK")
    risk: int = Field(..., ge=0, le=100)
    threats: List[Threat] = Field(default_factory=list)
    rationale: str


# For outputs we keep it simple: always write answer.md + answer.pdf
# For assessment: also create assessment_brief.md/pdf, rubric.md/pdf, submission_checklist.md/pdf

# ---------------------------
# Simple PDF exporter (wrap lines)
# ---------------------------

def md_to_pdf(md_text: str, pdf_path: Path, title: str = "") -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    width, height = A4
    margin = 50
    y = height - margin

    def draw_line(line: str, y_pos: float) -> float:
        # basic wrapping by character count (good enough for demo)
        max_chars = 95
        parts = [line[i:i+max_chars] for i in range(0, len(line), max_chars)] or [""]
        for p in parts:
            nonlocal_y = y_pos
            c.drawString(margin, nonlocal_y, p)
            y_pos -= 14
            if y_pos < margin:
                c.showPage()
                y_pos = height - margin
        return y_pos

    if title.strip():
        c.setFont("Helvetica-Bold", 14)
        y = draw_line(title.strip(), y)
        y -= 8
        c.setFont("Helvetica", 11)

    for raw in md_text.splitlines():
        line = raw.replace("\t", "    ")
        y = draw_line(line, y)
    c.save()


# ---------------------------
# Local RAG (BM25 over .md)
# ---------------------------

@dataclass
class Doc:
    path: Path
    text: str
    tokens: List[str]


class LocalBM25RAG:
    def __init__(self, kb_dir: Path):
        self.kb_dir = kb_dir
        self.docs: List[Doc] = []
        self.bm25: Optional[BM25Okapi] = None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9_]+", text.lower())

    def build(self) -> None:
        md_files = sorted(self.kb_dir.glob("*.md"))
        docs: List[Doc] = []
        for p in md_files:
            try:
                t = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            toks = self._tokenize(t)
            if toks:
                docs.append(Doc(path=p, text=t, tokens=toks))
        self.docs = docs
        if not docs:
            self.bm25 = None
            return
        self.bm25 = BM25Okapi([d.tokens for d in docs])

    def retrieve(self, query: str, k: int = 3) -> str:
        if not self.bm25 or not self.docs:
            return ""
        q = self._tokenize(query)
        scores = self.bm25.get_scores(q)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        chunks = []
        for idx in ranked:
            d = self.docs[idx]
            # Take first ~1200 chars as snippet (demo-friendly)
            snippet = d.text.strip().replace("\r\n", "\n")[:1200]
            chunks.append(f"[{d.path.name}]\n{snippet}\n")
        return "\n".join(chunks).strip()


# ---------------------------
# JSON helpers
# ---------------------------

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> str:
    """
    Try to extract a JSON object from model output.
    """
    text = text.strip()
    # remove ```json fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    m = JSON_RE.search(text)
    return m.group(0) if m else text


def llm_json(
    llm: OllamaClient,
    model: str,
    messages: List[Dict[str, str]],
    schema_text: str,
    max_tokens: int,
    temperature: float = 0.2,
    repair_attempts: int = 2,
) -> Dict[str, Any]:
    """
    Call LLM expecting JSON. Repair if invalid.
    """
    last = ""
    for attempt in range(repair_attempts + 1):
        resp = llm.chat(messages=messages, model=model, temperature=temperature, max_tokens=max_tokens)
        last = resp.text
        candidate = extract_json(resp.text)
        try:
            return json.loads(candidate)
        except Exception:
            # repair
            messages = build_json_repair_messages(schema_text=schema_text, bad_output=resp.text)
            continue
    raise RuntimeError("Failed to produce valid JSON after repair attempts.\n\nLast output:\n" + last)


# ---------------------------
# Output folder naming
# ---------------------------

def slugify(s: str, max_len: int = 40) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s[:max_len] if len(s) > max_len else s


def make_case_dir(out_dir: Path, user_prompt: str, intent: str, action: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = slugify(user_prompt)
    name = f"{ts}_{slug}_{intent}_{action}"
    case_dir = out_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


# ---------------------------
# Generation logic
# ---------------------------

def answer_generic(user_prompt: str, rag_snippets: str) -> str:
    # Keep it neutral; optionally append sources line if snippets exist
    srcs = []
    for line in rag_snippets.splitlines():
        m = re.match(r"^\[(.+?)\]$", line.strip())
        if m:
            srcs.append(m.group(1))
    sources_line = ""
    if srcs:
        sources_line = "\n\nSources: " + ", ".join(dict.fromkeys(srcs).keys())
    # The model itself will produce the answer; this is only for postprocessing if needed
    return sources_line


def generate_assessment_artifacts(llm_answer: str) -> Dict[str, str]:
    """
    Minimal parser to split the model answer into 3 artifacts if possible.
    If not found, fall back to the whole text for brief and create simple placeholders.
    """
    # Heuristics: look for headings
    text = llm_answer.strip()
    def find_section(title: str) -> str:
        pat = re.compile(rf"(?im)^\s*#+\s*{re.escape(title)}\s*$")
        m = pat.search(text)
        if not m:
            return ""
        start = m.end()
        # next heading
        m2 = re.compile(r"(?im)^\s*#+\s*").search(text, start)
        end = m2.start() if m2 else len(text)
        return text[start:end].strip()

    brief = find_section("Assessment Brief") or ""
    rubric = find_section("Marking Rubric") or ""
    checklist = find_section("Submission Checklist") or ""

    if not brief:
        brief = text
    if not rubric:
        rubric = "## Marking Rubric\n\n- HD / D / C / P criteria (to be refined)\n- Alignment to learning outcomes\n- Security testing quality\n- Reporting and analysis quality\n- Ethics/compliance coverage\n"
    if not checklist:
        checklist = "## Submission Checklist\n\n- Report submitted (PDF)\n- Code repository link\n- Prompt-injection test suite included\n- Metrics reported (ASR, FBR, JSON validity, citation coverage)\n- Ethics/compliance section included\n"

    return {
        "assessment_brief.md": brief if brief.startswith("#") else "# Assessment Brief\n\n" + brief,
        "rubric.md": rubric if rubric.startswith("#") else "# Marking Rubric\n\n" + rubric,
        "submission_checklist.md": checklist if checklist.startswith("#") else "# Submission Checklist\n\n" + checklist,
    }


# ---------------------------
# Main loop
# ---------------------------

def run_one(
    llm: OllamaClient,
    rag: Optional[LocalBM25RAG],
    out_dir: Path,
    model: str,
    user_prompt: str,
    capstone: bool,
) -> Tuple[Path, IntentOut, TriageOut, str]:
    rag_snippets = rag.retrieve(user_prompt, k=3) if rag else ""

    # 1) Intent routing
    intent_schema = '{"intent":"GENERIC_QA|ASSESSMENT_GEN","confidence":0.0-1.0}'
    intent_json = llm_json(
        llm=llm,
        model=model,
        messages=build_intent_messages(user_prompt),
        schema_text=intent_schema,
        max_tokens=200,
        temperature=0.0,
    )
    intent_out = IntentOut(**intent_json)
    intent = intent_out.intent.strip().upper()
    if intent not in ("GENERIC_QA", "ASSESSMENT_GEN"):
        intent = "GENERIC_QA"

    # 2) Security triage
    triage_schema = (
        '{ "action":"ALLOW|ALLOW_WITH_GUARDRAILS|BLOCK", "risk":0-100, '
        '"threats":[{"type":"...","severity":"low|medium|high","evidence":"..."}], "rationale":"..." }'
    )
    triage_json = llm_json(
        llm=llm,
        model=model,
        messages=build_triage_messages(user_prompt, rag_snippets=rag_snippets),
        schema_text=triage_schema,
        max_tokens=350,
        temperature=0.0,
    )
    triage_out = TriageOut(**triage_json)
    action = triage_out.action.strip().upper()
    if action not in ("ALLOW", "ALLOW_WITH_GUARDRAILS", "BLOCK"):
        action = "ALLOW"

    # Create case directory early so we can always write debug even if model fails later
    case_dir = make_case_dir(out_dir, user_prompt, intent, action)

    # Save intent + triage
    (case_dir / "intent.json").write_text(intent_out.model_dump_json(indent=2), encoding="utf-8")
    (case_dir / "triage.json").write_text(triage_out.model_dump_json(indent=2), encoding="utf-8")

    # 3) Enforcement
    if action == "BLOCK":
        safe_md = (
            "# Request Blocked\n\n"
            "Your request appears to be unsafe or attempts to bypass system controls. "
            "I can help with a safer alternative (e.g., explaining concepts at a high level, "
            "or providing policy-compliant guidance).\n"
        )
        (case_dir / "answer.md").write_text(safe_md, encoding="utf-8")
        md_to_pdf(safe_md, case_dir / "answer.pdf", title="Blocked Request")
        (case_dir / "answer.json").write_text(json.dumps({"blocked": True}, indent=2), encoding="utf-8")
        return case_dir, intent_out, triage_out, safe_md

    # 4) Generation (split prompts by intent; capstone only for assessment)
    if intent == "ASSESSMENT_GEN":
        messages = build_assessment_messages(
            user_prompt,
            rag_snippets=rag_snippets,
            capstone_context=(CAPSTONE_CONTEXT if capstone else ""),
        )
        max_tokens = 1800
        temperature = 0.2
    else:
        messages = build_generic_qa_messages(user_prompt, rag_snippets=rag_snippets)
        max_tokens = 650  # allow more detail than short paragraph
        temperature = 0.2

    resp = llm.chat(messages=messages, model=model, temperature=temperature, max_tokens=max_tokens)
    answer_text = resp.text.strip()

    # Save raw answer JSON debug
    (case_dir / "answer.json").write_text(json.dumps({"text": answer_text}, indent=2), encoding="utf-8")

    # 5) Postprocess: ensure sources line for GENERIC_QA if retrieval exists
    if intent == "GENERIC_QA":
        extra_sources = answer_generic(user_prompt, rag_snippets)
        if extra_sources and "Sources:" not in answer_text:
            answer_text = answer_text.rstrip() + extra_sources

        md = "# Answer\n\n" + answer_text + "\n"
        (case_dir / "answer.md").write_text(md, encoding="utf-8")
        md_to_pdf(md, case_dir / "answer.pdf", title="Answer")

    # 6) Assessment artifacts
    if intent == "ASSESSMENT_GEN":
        artifacts = generate_assessment_artifacts(answer_text)
        # Always also write an overall answer.md/pdf
        overall_md = "# Assessment Output\n\n" + answer_text + "\n"
        (case_dir / "answer.md").write_text(overall_md, encoding="utf-8")
        md_to_pdf(overall_md, case_dir / "answer.pdf", title="Assessment Output")

        for fname, content in artifacts.items():
            p = case_dir / fname
            p.write_text(content.strip() + "\n", encoding="utf-8")
            md_to_pdf(content, p.with_suffix(".pdf"), title=fname.replace("_", " ").replace(".md", ""))

    return case_dir, intent_out, triage_out, answer_text


def main() -> None:
    ap = argparse.ArgumentParser(description="Secure Guarded LLM Demo (Option B: Intent Routing)")
    ap.add_argument("--interactive", action="store_true", help="Interactive mode")
    ap.add_argument("--rag", type=str, default="", help="Knowledge base directory (markdown files)")
    ap.add_argument("--out", type=str, default="out", help="Output directory")
    ap.add_argument("--model", type=str, default="llama3.1", help="Ollama model name")
    ap.add_argument("--capstone", action="store_true", help="Enable capstone requirements (ASSESSMENT_GEN only)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    llm = OllamaClient()

    rag = None
    if args.rag:
        kb = Path(args.rag)
        if kb.exists() and kb.is_dir():
            rag = LocalBM25RAG(kb)
            rag.build()

    title = "Secure Guarded LLM Demo (Option B: Intent Routing)"
    print(title)
    print("Type 'exit' to quit.\n")

    if args.interactive:
        while True:
            try:
                user_prompt = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break
            if not user_prompt:
                continue
            if user_prompt.lower() in ("exit", "quit"):
                break

            try:
                case_dir, intent_out, triage_out, answer_text = run_one(
                    llm=llm,
                    rag=rag,
                    out_dir=out_dir,
                    model=args.model,
                    user_prompt=user_prompt,
                    capstone=args.capstone,
                )
                print(f"\nIntent: {intent_out.intent} (conf={intent_out.confidence:.2f})")
                print(f"Decision: {triage_out.action} | Risk: {triage_out.risk}\n")
                # Print a short preview
                preview = answer_text.strip().splitlines()
                preview_text = "\n".join(preview[:8]).strip()
                if preview_text:
                    print(preview_text)
                print(f"\nArtifacts saved in:\n  {case_dir}\n")

            except ValidationError as ve:
                print("\nThe model returned JSON but it did not match schema.")
                print(str(ve))
            except Exception as e:
                print("\nThe assistant could not produce a valid output. Please retry or simplify the request.")
                print(f"Error: {e}\n")
    else:
        print("Run with --interactive for interactive demo.")


if __name__ == "__main__":
    main()
