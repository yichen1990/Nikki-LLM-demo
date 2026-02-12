from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from app.rag import LocalRAG, RAGResult


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
    exploit_path: str


class TriageOut(BaseModel):
    action: str = Field(..., description="ALLOW, ALLOW_WITH_GUARDRAILS, BLOCK")
    risk_score: int = Field(..., ge=0, le=100)
    risk_rationale: str
    threats: List[Threat] = Field(default_factory=list)
    safe_response: str = ""
    recommended_controls: List[str] = Field(default_factory=list)


# ---------------------------
# Terminal color helpers (ANSI)
# ---------------------------

ANSI_RESET = "\033[0m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_RED = "\033[31m"
ANSI_DIM = "\033[2m"


def _supports_color() -> bool:
    # Basic heuristic: most UNIX terminals + VSCode support ANSI.
    # Windows Terminal / new shells usually support it too, but some legacy shells may not.
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") in (None, "", "dumb"):
        return False
    return True


def colorize(text: str, color: str) -> str:
    if not _supports_color():
        return text
    return f"{color}{text}{ANSI_RESET}"


def color_for_action(action: str) -> str:
    a = action.strip().upper()
    if a == "ALLOW":
        return ANSI_GREEN
    if a == "ALLOW_WITH_GUARDRAILS":
        return ANSI_YELLOW
    if a == "BLOCK":
        return ANSI_RED
    return ANSI_RESET


def color_for_risk(risk_score: int) -> str:
    if risk_score <= 25:
        return ANSI_GREEN
    if risk_score <= 70:
        return ANSI_YELLOW
    return ANSI_RED


# ---------------------------
# Simple PDF exporter (wrap lines)
# ---------------------------

def md_to_pdf(md_text: str, pdf_path: Path, title: str = "") -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    width, height = A4  # noqa: F841
    margin = 50
    y = height - margin

    def draw_line(line: str, y_pos: float) -> float:
        max_chars = 95
        parts = [line[i:i + max_chars] for i in range(0, len(line), max_chars)] or [""]
        for p in parts:
            c.drawString(margin, y_pos, p)
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
# JSON helpers
# ---------------------------

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> str:
    text = text.strip()
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
    last = ""
    for _attempt in range(repair_attempts + 1):
        resp = llm.chat(messages=messages, model=model, temperature=temperature, max_tokens=max_tokens)
        last = resp.text
        candidate = extract_json(resp.text)
        try:
            return json.loads(candidate)
        except Exception:
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
# Postprocess helpers
# ---------------------------

def append_sources_if_missing(answer_text: str, sources: List[str]) -> str:
    if not sources:
        return answer_text
    if "Sources:" in answer_text:
        return answer_text
    uniq = list(dict.fromkeys(sources).keys())
    return answer_text.rstrip() + "\n\nSources: " + ", ".join(uniq)


def generate_assessment_artifacts(llm_answer: str) -> Dict[str, str]:
    text = llm_answer.strip()

    def find_section(title: str) -> str:
        pat = re.compile(rf"(?im)^\s*#+\s*{re.escape(title)}\s*$")
        m = pat.search(text)
        if not m:
            return ""
        start = m.end()
        m2 = re.compile(r"(?im)^\s*#+\s*").search(text, start)
        end = m2.start() if m2 else len(text)
        return text[start:end].strip()

    brief = find_section("Assessment Brief") or ""
    rubric = find_section("Marking Rubric") or ""
    checklist = find_section("Submission Checklist") or ""

    if not brief:
        brief = text
    if not rubric:
        rubric = (
            "## Marking Rubric\n\n"
            "- HD / D / C / P criteria (to be refined)\n"
            "- Alignment to learning outcomes\n"
            "- Security testing quality\n"
            "- Reporting and analysis quality\n"
            "- Ethics/compliance coverage\n"
        )
    if not checklist:
        checklist = (
            "## Submission Checklist\n\n"
            "- Report submitted (PDF)\n"
            "- Code repository link\n"
            "- Prompt-injection test suite included\n"
            "- Metrics reported (ASR, FBR, JSON validity, citation coverage)\n"
            "- Ethics/compliance section included\n"
        )

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
    rag: Optional[LocalRAG],
    out_dir: Path,
    model: str,
    user_prompt: str,
    capstone: bool,
) -> Tuple[Path, IntentOut, TriageOut, str]:
    # 1) Intent routing FIRST (so retrieval can be intent-aware)
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

    # 2) Retrieval SECOND (intent-aware + confidence gating)
    rag_meta: RAGResult = RAGResult([], [], "low", 0.0)
    rag_snippets = ""
    if rag:
        rag_meta = rag.retrieve(user_prompt, k=3, intent=intent, return_meta=True)  # type: ignore
        if rag_meta.confidence == "high":
            rag_snippets = "\n\n".join(rag_meta.snippets)
        else:
            rag_snippets = ""

    # 3) Security triage (schema must match app/prompts.py triage schema)
    triage_schema = (
        '{ "action":"ALLOW|ALLOW_WITH_GUARDRAILS|BLOCK", '
        '"risk_score":0-100, '
        '"risk_rationale":"...", '
        '"threats":[{"type":"...","severity":"LOW|MEDIUM|HIGH|CRITICAL","evidence":"...","exploit_path":"..."}], '
        '"safe_response":"...", '
        '"recommended_controls":["..."] }'
    )
    triage_json = llm_json(
        llm=llm,
        model=model,
        messages=build_triage_messages(user_prompt, rag_snippets=rag_snippets),
        schema_text=triage_schema,
        max_tokens=420,
        temperature=0.0,
    )
    triage_out = TriageOut(**triage_json)
    action = triage_out.action.strip().upper()
    if action not in ("ALLOW", "ALLOW_WITH_GUARDRAILS", "BLOCK"):
        action = "ALLOW"

    case_dir = make_case_dir(out_dir, user_prompt, intent, action)

    (case_dir / "intent.json").write_text(intent_out.model_dump_json(indent=2), encoding="utf-8")
    (case_dir / "triage.json").write_text(triage_out.model_dump_json(indent=2), encoding="utf-8")
    (case_dir / "retrieval.json").write_text(
        json.dumps(
            {
                "confidence": rag_meta.confidence,
                "top_score": rag_meta.top_score,
                "sources": rag_meta.sources,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # 4) Enforcement
    if action == "BLOCK":
        safe = triage_out.safe_response.strip()
        if not safe:
            safe = (
                "Your request appears to be unsafe or attempts to bypass system controls. "
                "I can help with a safer alternative (e.g., explaining concepts at a high level, "
                "or providing policy-compliant guidance)."
            )
        safe_md = "# Request Blocked\n\n" + safe + "\n"
        (case_dir / "answer.md").write_text(safe_md, encoding="utf-8")
        md_to_pdf(safe_md, case_dir / "answer.pdf", title="Blocked Request")
        (case_dir / "answer.json").write_text(
            json.dumps({"blocked": True, "safe_response": safe}, indent=2),
            encoding="utf-8",
        )
        return case_dir, intent_out, triage_out, safe_md

    # 5) Generation
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
        max_tokens = 650
        temperature = 0.2

    resp = llm.chat(messages=messages, model=model, temperature=temperature, max_tokens=max_tokens)
    answer_text = resp.text.strip()
    (case_dir / "answer.json").write_text(json.dumps({"text": answer_text}, indent=2), encoding="utf-8")

    # 6) Postprocess
    if intent == "GENERIC_QA":
        answer_text = append_sources_if_missing(answer_text, rag_meta.sources if rag_meta.confidence == "high" else [])
        md = "# Answer\n\n" + answer_text + "\n"
        (case_dir / "answer.md").write_text(md, encoding="utf-8")
        md_to_pdf(md, case_dir / "answer.pdf", title="Answer")

    if intent == "ASSESSMENT_GEN":
        artifacts = generate_assessment_artifacts(answer_text)
        overall_md = "# Assessment Output\n\n" + answer_text + "\n"
        (case_dir / "answer.md").write_text(overall_md, encoding="utf-8")
        md_to_pdf(overall_md, case_dir / "answer.pdf", title="Assessment Output")

        for fname, content in artifacts.items():
            p = case_dir / fname
            p.write_text(content.strip() + "\n", encoding="utf-8")
            md_to_pdf(content, p.with_suffix(".pdf"), title=fname.replace("_", " ").replace(".md", ""))

    return case_dir, intent_out, triage_out, answer_text


def main() -> None:
    ap = argparse.ArgumentParser(description="Secure Guarded LLM Demo")
    ap.add_argument("--interactive", action="store_true", help="Interactive mode")
    ap.add_argument("--rag", type=str, default="", help="Knowledge base directory (markdown files)")
    ap.add_argument("--out", type=str, default="out", help="Output directory")
    ap.add_argument("--model", type=str, default="llama3.1", help="Ollama model name")
    ap.add_argument("--capstone", action="store_true", help="Enable capstone requirements (ASSESSMENT_GEN only)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    llm = OllamaClient()

    rag: Optional[LocalRAG] = None
    if args.rag:
        kb = Path(args.rag)
        if kb.exists() and kb.is_dir():
            rag = LocalRAG(kb_dir=kb, max_chars=2000)

    print("Secure Guarded LLM Demo")
    print(colorize("Type 'exit' to quit.\n", ANSI_DIM))

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

                enquiry_label = "Enquiry Type"
                decision_color = color_for_action(triage_out.action)
                risk_color = color_for_risk(triage_out.risk_score)

                print(f"\n{enquiry_label}: {intent_out.intent} (conf={intent_out.confidence:.2f})")
                print(
                    "Decision: "
                    + colorize(triage_out.action, decision_color)
                    + " | Risk: "
                    + colorize(str(triage_out.risk_score), risk_color)
                )

                # Retrieval debug
                try:
                    rj = json.loads((case_dir / "retrieval.json").read_text(encoding="utf-8"))
                    retrieval_line = f"Retrieval: {rj.get('confidence')} | sources={rj.get('sources')}"
                    print(colorize(retrieval_line, ANSI_DIM))
                except Exception:
                    pass

                print()
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
