import json
import os
import re
from pathlib import Path
from typing import List, Optional
import numpy as np
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT_DIR = Path(__file__).resolve().parent.parent
RUNBOOKS_DIR = ROOT_DIR / "data" / "runbooks"
load_dotenv(ROOT_DIR / ".env")


class CopilotAdvice(BaseModel):
    root_cause_hypothesis: str = Field(..., description="Likely root cause based on ticket and SOP")
    immediate_action_steps: List[str] = Field(..., description="3 to 5 concrete technical resolution steps")
    recommended_escalation_team: str = Field(..., description="Target L2/L3 escalation group")
    manager_alert_draft: str = Field(..., description="Short executive escalation email/Slack alert")
    cited_runbook: str = Field(..., description="Source runbook filename used for grounding")
    engine_mode: str = Field(..., description="Active LLM backend used")


def _get_secret(key: str, default: str = "") -> str:
    val = os.getenv(key)
    if val:
        return val.strip()
    try:
        import streamlit as st
        if key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass
    return default


class RunbookRetriever:
    """Lightweight TF-IDF Vector Retriever optimized for low-RAM cloud deployment."""

    def __init__(self, runbooks_dir: Path = RUNBOOKS_DIR):
        self.docs: List[dict] = []
        if runbooks_dir.exists():
            for fp in sorted(runbooks_dir.glob("*.txt")):
                self.docs.append({"source": fp.name, "content": fp.read_text(encoding="utf-8")})

        if self.docs:
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            self.matrix = self.vectorizer.fit_transform([d["content"] for d in self.docs])
        else:
            self.vectorizer = None
            self.matrix = None

    def retrieve(self, query: str, top_k: int = 2) -> List[dict]:
        if not self.docs or self.vectorizer is None:
            return []
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix).flatten()
        ranked_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in ranked_idx:
            results.append(
                {
                    "source": self.docs[idx]["source"],
                    "score": round(float(scores[idx]), 3),
                    "content": self.docs[idx]["content"],
                }
            )
        return results


class SLACopilot:
    def __init__(self):
        self.retriever = RunbookRetriever()
        self.provider = _get_secret("LLM_PROVIDER", "github_models").lower()
        self.client, self.model_name, self.mode_label = self._init_client()

    def _init_client(self):
        azure_key = _get_secret("AZURE_OPENAI_API_KEY")
        azure_endpoint = _get_secret("AZURE_OPENAI_ENDPOINT")
        azure_deployment = _get_secret("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

        if self.provider == "azure" and azure_key and azure_endpoint:
            client = AzureOpenAI(
                api_key=azure_key,
                api_version="2024-06-01",
                azure_endpoint=azure_endpoint,
            )
            return client, azure_deployment, "Azure OpenAI (Enterprise)"

        github_token = _get_secret("GITHUB_TOKEN")
        if github_token:
            client = OpenAI(
                base_url="https://models.inference.ai.azure.com",
                api_key=github_token,
            )
            return client, "gpt-4o-mini", "Azure-Hosted GitHub Models (gpt-4o-mini)"

        groq_key = _get_secret("GROQ_API_KEY")
        if groq_key:
            client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=groq_key,
            )
            return client, "llama-3.3-70b-versatile", "Groq Cloud (Llama 3.3 70B)"

        return None, "", "Deterministic SOP Engine (Add GITHUB_TOKEN for Live LLM)"

    def _extract_json(self, text: str) -> dict:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)
        return json.loads(cleaned)

    def _deterministic_fallback(
        self, ticket: dict, breach_prob: float, retrieved: List[dict], reason_note: Optional[str] = None
    ) -> CopilotAdvice:
        top_doc = retrieved[0] if retrieved else {"source": "general_ops_sop.txt", "content": ""}
        lines = [ln.strip() for ln in top_doc["content"].splitlines() if ln.strip()]

        steps = [re.sub(r"^\d+\.\s*", "", ln) for ln in lines if re.match(r"^\d+\.", ln)][:4]
        if not steps:
            steps = [
                "Verify recent changes or deployments linked to the affected service.",
                "Inspect system metrics (CPU, latency, error logs) for bottlenecks.",
                "Escalate immediately to L2/L3 specialist team before SLA window closes.",
            ]

        esc_line = next((ln for ln in lines if ln.startswith("ESCALATION TEAM:")), "ESCALATION TEAM: L2-InfraOps")
        esc_team = esc_line.replace("ESCALATION TEAM:", "").strip()

        mode = self.mode_label if not reason_note else f"SOP Fallback ({reason_note})"
        return CopilotAdvice(
            root_cause_hypothesis=(
                f"High SLA breach exposure ({breach_prob * 100:.1f}%) driven by {ticket['priority']} severity, "
                f"{ticket['category']} complexity, queue load ({ticket['open_tickets_in_queue']} open tickets), "
                f"and {ticket['reassignment_count']} reassignments."
            ),
            immediate_action_steps=steps,
            recommended_escalation_team=esc_team,
            manager_alert_draft=(
                f"Subject: [SLA Risk Alert - {breach_prob * 100:.0f}%] {ticket['priority']} {ticket['category']} Ticket\n\n"
                f"Hi Ops Lead, an incoming {ticket['category']} incident assigned to {ticket['assigned_group']} is flagged "
                f"at {breach_prob * 100:.1f}% SLA breach risk (Target: {ticket['sla_target_hours']}h). "
                f"Issue summary: {ticket.get('description', 'N/A')}. "
                f"Recommending immediate parallel triage with {esc_team} per {top_doc['source']}."
            ),
            cited_runbook=top_doc["source"],
            engine_mode=mode,
        )

    def generate_advice(self, ticket: dict, breach_prob: float) -> CopilotAdvice:
        query = f"{ticket['category']} {ticket['priority']} {ticket.get('description', '')}"
        retrieved = self.retriever.retrieve(query, top_k=2)

        if not self.client:
            return self._deterministic_fallback(ticket, breach_prob, retrieved)

        context_text = "\n\n---\n\n".join(
            [f"[{d['source']}]\n{d['content']}" for d in retrieved]
        )

        system_prompt = (
            "You are SLA-Guard Copilot, an enterprise IT Operations SRE & Decision AI. "
            "Use ONLY the provided IT SOP runbooks to produce actionable mitigation guidance. "
            "Respond strictly in valid JSON with exact keys: "
            "root_cause_hypothesis (string), immediate_action_steps (list of 4 concise strings), "
            "recommended_escalation_team (string), manager_alert_draft (string), cited_runbook (string)."
        )

        user_prompt = f"""
TICKET DETAILS:
- Priority: {ticket['priority']}
- Category: {ticket['category']}
- Current Assigned Group: {ticket['assigned_group']}
- Queue Load: {ticket['open_tickets_in_queue']} open tickets
- Reassignments: {ticket['reassignment_count']}
- SLA Target: {ticket['sla_target_hours']} hours
- Predicted SLA Breach Probability: {breach_prob * 100:.1f}%
- Description: {ticket.get('description', 'Production degradation reported.')}

RETRIEVED ENTERPRISE RUNBOOKS:
{context_text}
"""

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = resp.choices[0].message.content
            parsed = self._extract_json(raw)
            parsed["engine_mode"] = self.mode_label
            if not parsed.get("cited_runbook") and retrieved:
                parsed["cited_runbook"] = retrieved[0]["source"]
            return CopilotAdvice(**parsed)
        except Exception as exc:
            return self._deterministic_fallback(ticket, breach_prob, retrieved, reason_note=type(exc).__name__)

    def ask_runbooks(self, question: str) -> dict:
        retrieved = self.retriever.retrieve(question, top_k=2)
        if not retrieved:
            return {"answer": "No runbooks indexed yet.", "sources": []}

        sources = [d["source"] for d in retrieved]
        if not self.client:
            top = retrieved[0]
            return {
                "answer": f"**Matched SOP ({top['source']}):**\n\n{top['content']}",
                "sources": sources,
            }

        context_text = "\n\n---\n\n".join([f"[{d['source']}]\n{d['content']}" for d in retrieved])
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": "Answer the IT Operations engineer's question concisely using ONLY the supplied runbooks. Cite the runbook filename.",
                    },
                    {"role": "user", "content": f"Question: {question}\n\nRunbooks:\n{context_text}"},
                ],
            )
            return {"answer": resp.choices[0].message.content, "sources": sources}
        except Exception:
            top = retrieved[0]
            return {"answer": f"**Matched SOP ({top['source']}):**\n\n{top['content']}", "sources": sources}
