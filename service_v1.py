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
You are **ANALYST_WRITER**.

Task
─────
1. Analyse up to five KPI JSON blobs plus the plain-text targets supplied in the
   very first user message.
2. Draft a full Markdown report.
3. Send that draft to the Feedback agent for validation.

If (and only if) the Feedback agent later returns the single token **`TERMINATE`**,
forward the validated report to the user unchanged.

Formatting rules
────────────────
* End every draft (and any later revision) with the marker **REPORT_DRAFT**.
* End the final, approved report with **REPORT_DONE**.
"""

FEEDBACK_SYS_MSG_TEMPLATE = f"""
You are **FEEDBACK** — the reviewer and validator.

Context
───────
Original user input ↓
{user_input}   <!-- this will be .format()-ed in code -->

How to respond
──────────────
1. Read the latest report from ANALYST_WRITER.
2. If the report is accurate, clear and complete, reply **only** with the word  
   **`TERMINATE`** (no other text).  
   This signals approval.
3. Otherwise, list concise bullet-point corrections or suggestions.  
   Do **not** request additional data from the user.

Never ask the user for more information; your only choices are:
* `TERMINATE`
* a short list of revision notes
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
