# inference.py

import os
import json
import time
from datetime import datetime
from typing import Dict, Any

from openai import OpenAI

from env import SupportDeskEnv, Action
from tasks import get_task


# ---------------------------------------------------------------------------
# Config from Environment Variables
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN", "")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or os.getenv("OPENAI_API_KEY"),
)


# ---------------------------------------------------------------------------
# Logging Helpers (STRICT FORMAT)
# ---------------------------------------------------------------------------

def log_start(task_id: str):
    print(
        f"[START] task_id={task_id} model={MODEL_NAME} "
        f"timestamp={datetime.utcnow().isoformat()}"
    )


def log_step(turn: int, action: Dict[str, Any], result: Dict[str, Any], reward):
    print(
        f"[STEP] turn={turn} "
        f"action={action['type']} "
        f"input={json.dumps(action)} "
        f"output={json.dumps(result)} "
        f"reward={reward.step_reward} "
        f"cumulative={reward.cumulative_reward}"
    )


def log_end(task_id: str, final_score: float, turns: int, status: str):
    print(
        f"[END] task_id={task_id} "
        f"final_score={final_score} "
        f"turns={turns} "
        f"status={status}"
    )


# ---------------------------------------------------------------------------
# Prompt Builder
# ---------------------------------------------------------------------------

def build_prompt(observation) -> str:
    """
    Convert environment observation into LLM prompt.
    """
    convo = "\n".join(
        [f"{t.role.upper()}: {t.content}" for t in observation.conversation]
    )

    kb = "\n".join(
        [
            f"[{r.id}] {r.title} (score={r.relevance_score})\n{r.policy}"
            for r in observation.kb_results
        ]
    )

    prompt = f"""
You are a Tier-1 support agent.

AVAILABLE ACTIONS:
1. search_kb(query)
2. respond_to_user(message)
3. apply_resolution(resolution_code, summary)
4. escalate(reason)

RULES:
- ALWAYS search KB before applying resolution
- NEVER hallucinate policies or codes
- Use KB results when available
- Be concise and professional

OUTPUT FORMAT (STRICT JSON):
{{
  "type": "<action_type>",
  "query": "...",
  "message": "...",
  "resolution_code": "...",
  "summary": "...",
  "reason": "..."
}}

CURRENT STATE:
Turn: {observation.turn}

TICKET:
{observation.ticket.body}

KB RESULTS:
{kb if kb else "None"}

CONVERSATION:
{convo}
"""
    return prompt


# ---------------------------------------------------------------------------
# LLM Action Generator
# ---------------------------------------------------------------------------

def get_action_from_model(prompt: str) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    text = response.choices[0].message.content.strip()

    try:
        action = json.loads(text)
    except Exception:
        # Fallback safe action
        action = {
            "type": "search_kb",
            "query": "general support issue"
        }

    return action


# ---------------------------------------------------------------------------
# Run Episode
# ---------------------------------------------------------------------------

def run_task(task_id: str):
    task = get_task(task_id)
    env = SupportDeskEnv(task)

    obs = env.reset()

    log_start(task_id)

    done = False

    while not done:
        prompt = build_prompt(obs)
        action_dict = get_action_from_model(prompt)

        action = Action(**action_dict)
        step_result = env.step(action)

        log_step(
            turn=obs.turn,
            action=action_dict,
            result=step_result.observation.model_dump(),
            reward=step_result.reward
        )

        obs = step_result.observation
        done = step_result.done

    # Final grading
    state = env.state()
    grade = task.grade(state)

    status = "resolved" if state["resolved"] else (
        "escalated" if state["escalated"] else "failed"
    )

    log_end(
        task_id=task_id,
        final_score=grade.score,
        turns=state["turn"],
        status=status
    )

    print("\n=== FINAL GRADE ===")
    print(json.dumps(grade.model_dump(), indent=2))


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="task_easy")
    args = parser.parse_args()

    run_task(args.task)
