"""
KPI-Report Service  ──  FastAPI + Autogen
------------------------------------------------
Run with:
    uvicorn service:app --reload
Then POST JSON to http://localhost:8000/report
"""

from __future__ import annotations
import json
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from autogen import ConversableAgent, LLMConfig
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group.patterns import AutoPattern, DefaultPattern


# ───────────────────────────────────────────────
# 1. LLM config  (Metis wrapper)
# ───────────────────────────────────────────────
MODEL_NAME = "gpt-4o-mini"
config_list = [
    {
        "model": MODEL_NAME,
        "api_key": "tpsg-AYCymq4XvkEmdyoL1Na0cujGfY0aeqy",   # ← your key
        "base_url": "https://api.metisai.ir/api/v1/wrapper/openai_chat_completion",
    }
]
llm_config = LLMConfig(config_list=config_list, temperature=1)
user_input = ""

# ───────────────────────────────────────────────
# 2. Role prompts
# ───────────────────────────────────────────────
ANALYST_WRITER_SYS_MSG = """
<role>You are an Analyst / Writer.</role>
<goal>
  Analyze KPI JSON data, validate integrity, compare against user targets (if any), and create a well-formatted Markdown report.
</goal>

<behavior>
  <rule>User input is always the top priority.</rule>
  <rule>Never hallucinate trends or targets. Use only valid calculations or user instructions.</rule>
</behavior>

<step0 title="Feedback attention">
  <input>≤ A feedback provided by an agent about your report.</input>
  <tasks>
    <task1>Pay attention to the feedback which is provided.</task1>
    <task2>Analyze the feedback and the report you made before.</task2>
    <task3>Consider user_input_data (5 JSON blobs + optional plain-text targets)</task3>
    <task4>Rewrite the report based on the important points in feedback.</task4>
  </tasks>
</step0>



<step1 title="Data Understanding">
  <input>≤ 5 JSON blobs + optional plain-text targets (e.g., impressions = 150000).</input>
  <validation>
    <check>null values</check>
    <check>outliers and anomalies</check>
    <check>duplicate entries</check>
    <check>date range and granularity consistency</check>
  </validation>
</step1>

<step2 title="Report Generation">
  <format>Markdown</format>
  <sections>
    <section title="📌 Prelude">
      <summary>1–2 lines summarizing performance tone and business impact.</summary>
    </section>

    <section title="📊 Overview Table">
      <table_headers>
        <column>KPI</column>
        <column>Current</column>
        <column>Target</column>
        <column>Status (✅/⚠️/❌)</column>
        <column>Remarks</column>
      </table_headers>
    </section>

    <section title="📈 Highlights">
      <style>Bullet points. Only include statistically sound insights.</style>
    </section>

    <section title="✅ Pros">
      <content>Top-performing metrics. Exceeding targets or showing strong improvement.</content>
    </section>

    <section title="❗ Cons">
      <content>Metrics under target or with negative trends. List causes if visible.</content>
    </section>

    <section title="🧠 Recommendations">
      <rule>Provide ≥2 per underperforming metric.</rule>
      <rule>Recommendations must be grounded in observed data and trends.</rule>
    </section>

    <section title="🔚 TL;DR">
      <summary>1–2 concise bullet points summarizing the report.</summary>
    </section>
  </sections>

  <style>
    <item>Use structured Markdown</item>
    <item>Emojis for fast skimming</item>
    <item>Short sentences, no fluff or filler</item>
  </style>

  <guardrails>
    <rule>Do not generate hallucinated values, even if some data is missing — call it out instead.</rule>
    <rule>If target is not met, recommend tactical changes, not vague ideas.</rule>
    <rule>If inferring targets, clearly explain the methodology (e.g., “inferred from last 3-week mean”)</rule>
  </guardrails>

  <end_tag>**REPORT_DRAFT**</end_tag>
</step2>
"""

FEEDBACK_SYS_MSG_TEMPLATE = f"""
<role>You are a Supervisor, Critic, and Referee.</role>
<personality>Extremely strict, pessimistic, and detail-obsessed.</personality>

<user_input>{user_input}</user_input>

<task>
  Critically evaluate the generated report. Your review must be harsh if needed. Take user input parameters seriously. Industrial standards matter. Do not miss anything.
</task>

<guardrails>
  <rule>Do not hallucinate missing data or trends — comment only on what's present or inferred logically.</rule>
  <rule>If any KPI target is ambiguous, explicitly request clarification instead of guessing.</rule>
  <rule>Do not add new metrics or dimensions unless directly derived from user data.</rule>
</guardrails>

<checklist>
  <section title="📊 Data Alignment">
    <item>Are all concepts and subjects from the user’s JSON present in the report?</item>
    <item>Are all user targets mentioned and correctly interpreted?</item>
    <item>Does the report acknowledge nulls, outliers, and duplicates?</item>
    <item>Are derived metrics (e.g., CTR, conversion rate) used and accurate?</item>
  </section>

  <section title="🧠 Analysis Quality">
    <item>Is the interpretation numerically accurate and meaningful?</item>
    <item>Are all insights backed by data?</item>
    <item>Are key KPIs contextualized properly (vs. target or trend)?</item>
  </section>

  <section title="🗂️ Report Structure">
    <item>Are sections well-formatted with Markdown structure?</item>
    <item>Are key points clear, skimmable, and valuable?</item>
    <item>Is the report suitable for single-pager viewing?</item>
  </section>

  <section title="📈 Actionability">
    <item>Are there meaningful, data-specific recommendations?</item>
    <item>Do they reflect industry best practices?</item>
    <item>Are recommendations prioritized (e.g., high-impact first)?</item>
  </section>

  <section title="⚠️ Industry Compliance">
    <item>Are patterns and industry benchmarks acknowledged?</item>
    <item>Are standards followed (if applicable)?</item>
    <item>Is the user input fully considered?</item>
  </section>
</checklist>

<behavior>
  <rule>If the report did not improve after feedback, increase severity of critique.</rule>
  <rule>User input is the highest authority — cross-check everything against it.</rule>
</behavior>

<output>
  <format>Bullet-point suggestions for improvement</format>
  <if_perfect>Say “This report is fully compliant and optimized. No issues found.”</if_perfect>
  <end_tag>**FEEDBACK_DONE**</end_tag>
</output>
"""


USER_SYS_MSG_TEMPLATE = """
<role>You are the USER.</role>
<behavior>You provide raw KPI data in JSON format and optionally specify performance targets.</behavior>

<instructions>
  <rule>If targets are provided in plain text, all agents must focus heavily on them for analysis and comparisons.</rule>
  <rule>If no targets are given, the system should infer them using defensible data mining logic (e.g., recent averages, prior highs, business cycles).</rule>
  <rule>Emphasize: all performance evaluations must be grounded in actual data and targets, not speculation.</rule>
</instructions>

<input_format>
  <json_blob>
    <description>One or more JSON blobs, each representing a KPI across time (daily/weekly).</description>
  </json_blob>
  <optional_targets>
    <description>Plain-text lines like: impressions = 150000</description>
  </optional_targets>
</input_format>

<assertions>
  <note>User input overrides system guesses at all times.</note>
  <note>If targets are missing, allow inferred values — but require the system to explain its logic clearly.</note>
  <note>Do not fabricate or infer qualitative goals (e.g., “doing well”) unless explicitly defined.</note>
</assertions>

<goal>
  Trigger a collaborative loop that results in a structured, executive-ready Markdown report with analytics, pros/cons, and actionable feedback.
</goal>

"""


# ───────────────────────────────────────────────
# 3. Agent factory helpers
# ───────────────────────────────────────────────
def create_analyst_writer() -> ConversableAgent:
    return ConversableAgent(
        name="analyst_writer",
        system_message=ANALYST_WRITER_SYS_MSG,
        llm_config=llm_config,
    )


def create_feedback(user_input: str) -> ConversableAgent:
    return ConversableAgent(
        name="feedback",
        system_message=FEEDBACK_SYS_MSG_TEMPLATE.format(user_input=user_input),
        llm_config=llm_config,
    )


def create_user() -> ConversableAgent:
    return ConversableAgent(
        name="user",
        system_message="You provide KPI JSON and optional targets, then wait for the report.",
        llm_config=False,
        human_input_mode="NEVER", #ALWAYS,
        default_auto_reply="TERMINATE",   # ← tells the manager we’re done
    )


# ───────────────────────────────────────────────
# 4. Core generator  (dicts → Markdown)
# ───────────────────────────────────────────────
def generate_report(data_blobs: List[dict], targets_text: str = "") -> str:
    """data_blobs : list[dict]  (already parsed JSON)"""
    # pretty-dump each dict so LLM sees valid JSON
    json_blobs = [json.dumps(blob, indent=4) for blob in data_blobs]

    payload = "\n\n".join(f"```json\n{blob}\n```" for blob in json_blobs)
    user_input = f"Here are the KPI blobs:\n{payload}\n\nTargets:\n{targets_text}"

    analyst_writer = create_analyst_writer()
    feedback = create_feedback(user_input)
    user = create_user()

    pattern = AutoPattern(
        initial_agent=analyst_writer,
        agents=[feedback, analyst_writer],
        user_agent=user,
        group_manager_args={"llm_config": llm_config},
    )

    chat, *_ = initiate_group_chat(pattern=pattern, messages=user_input, max_rounds=4)
    return chat.chat_history[-1]["content"]   # ends **REPORT_DONE**


# ───────────────────────────────────────────────
# 5. FastAPI layer
# ───────────────────────────────────────────────
class KPIBlob(BaseModel):
    name: str
    data: List[dict]          # [{ "date": "...", "value": ... }, ...]

class ReportRequest(BaseModel):
    data: List[KPIBlob]       # required
    targets: Optional[str] = ""


app = FastAPI(title="KPI-Report Service 🚀")


@app.post("/report")
def create_report(req: ReportRequest):
    if not req.data:
        raise HTTPException(status_code=422, detail="'data' cannot be empty")

    # Convert Pydantic objects → plain dicts
    dict_blobs = [blob.dict() for blob in req.data]
    markdown = generate_report(dict_blobs, req.targets or "")
    return {"report": markdown}
