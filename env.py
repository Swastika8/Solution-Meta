# env.py
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Knowledge Base (simulated JSON database)
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE: List[Dict[str, Any]] = [
    {
        "id": "KB001",
        "title": "Password Reset Procedure",
        "keywords": ["password", "reset", "forgot", "login", "access"],
        "policy": "Users may reset their password via the self-service portal at /account/reset. "
                  "If the portal is unavailable, agents may trigger a manual reset after verifying "
                  "the account via registered email + last 4 digits of phone number. "
                  "Resolution code: RES-PWD-01.",
        "resolution_code": "RES-PWD-01",
        "escalate_if": "Account is locked due to 10+ failed attempts — escalate to security team.",
    },
    {
        "id": "KB002",
        "title": "Billing Dispute — Partial Credit Policy",
        "keywords": ["billing", "charge", "refund", "invoice", "credit", "dispute", "overcharge"],
        "policy": "Agents may issue a one-time courtesy credit of up to $25 without manager approval. "
                  "For disputes between $25–$100, agents must collect proof (screenshot/invoice) and "
                  "submit via the billing portal. Disputes above $100 require Tier-2 escalation. "
                  "Resolution code: RES-BILL-02.",
        "resolution_code": "RES-BILL-02",
        "escalate_if": "Dispute amount exceeds $100 or customer requests legal escalation.",
    },
    {
        "id": "KB003",
        "title": "Service Outage — Communication Protocol",
        "keywords": ["outage", "down", "unavailable", "not working", "service", "offline"],
        "policy": "During a known outage, agents must: (1) Acknowledge the issue empathetically, "
                  "(2) Direct users to the status page at status.example.com, "
                  "(3) Do NOT promise an ETA unless one is posted on the status page, "
                  "(4) Offer a $10 service credit if the outage exceeds 4 hours. "
                  "Resolution code: RES-OUT-03.",
        "resolution_code": "RES-OUT-03",
        "escalate_if": "User reports data loss during the outage — escalate to engineering.",
    },
    {
        "id": "KB004",
        "title": "Account Cancellation and Retention Policy",
        "keywords": ["cancel", "cancellation", "close account", "terminate", "unsubscribe", "leave"],
        "policy": "Before processing a cancellation, agents must: (1) Ask for the reason, "
                  "(2) Offer one of three retention offers: 1-month free, 20% discount for 3 months, "
                  "or a plan downgrade. If the user still wishes to cancel after the offer, "
                  "process it immediately without further friction. "
                  "Resolution code: RES-CXL-04.",
        "resolution_code": "RES-CXL-04",
        "escalate_if": "User cites a legal or regulatory reason for cancellation.",
    },
    {
        "id": "KB005",
        "title": "Technical Troubleshooting — Connectivity Issues",
        "keywords": ["connect", "connectivity", "slow", "latency", "timeout", "network", "vpn"],
        "policy": "Standard troubleshooting steps: (1) Confirm the user's browser/app version is current, "
                  "(2) Ask user to clear cache and cookies, (3) Try incognito/private mode, "
                  "(4) Check if issue is reproduced on a second device. "
                  "If all four steps fail, collect a HAR file and escalate to Tier-2. "
                  "Resolution code: RES-TECH-05.",
        "resolution_code": "RES-TECH-05",
        "escalate_if": "All four standard steps fail — collect HAR file and escalate.",
    },
    {
        "id": "KB006",
        "title": "Data Export and GDPR / Privacy Requests",
        "keywords": ["data", "export", "gdpr", "privacy", "personal data", "delete", "right to erasure"],
        "policy": "GDPR / privacy requests must be acknowledged within 24 hours. "
                  "Data export requests are fulfilled within 30 days via the privacy portal. "
                  "Right-to-erasure requests must be escalated to the Data Protection Officer (DPO). "
                  "Agents must NOT attempt to process erasure requests manually. "
                  "Resolution code: RES-PRIV-06.",
        "resolution_code": "RES-PRIV-06",
        "escalate_if": "Any right-to-erasure or DPO-level request — always escalate.",
    },
    {
        "id": "KB007",
        "title": "Subscription Upgrade / Downgrade",
        "keywords": ["upgrade", "downgrade", "plan", "subscription", "tier", "premium", "basic"],
        "policy": "Agents may process plan changes effective immediately or at next billing cycle "
                  "(user's choice). Pro-rated credits apply for mid-cycle downgrades. "
                  "Upgrades take effect immediately with pro-rated charges. "
                  "Resolution code: RES-SUB-07.",
        "resolution_code": "RES-SUB-07",
        "escalate_if": "Enterprise plan changes require account manager involvement.",
    },
]

VALID_RESOLUTION_CODES = {kb["resolution_code"] for kb in KNOWLEDGE_BASE}
VALID_KB_IDS = {kb["id"] for kb in KNOWLEDGE_BASE}


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class Ticket(BaseModel):
    id: str
    subject: str
    body: str
    user_name: str
    account_id: str
    user_history: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KBResult(BaseModel):
    id: str
    title: str
    policy: str
    resolution_code: str
    escalate_if: str
    relevance_score: float


class ConversationTurn(BaseModel):
    role: Literal["agent", "user", "system"]
    content: str
    turn: int


class Observation(BaseModel):
    ticket: Ticket
    kb_results: List[KBResult] = Field(default_factory=list)
    conversation: List[ConversationTurn] = Field(default_factory=list)
    turn: int = 0
    resolved: bool = False
    escalated: bool = False
    available_actions: List[str] = Field(
        default_factory=lambda: [
            "search_kb", "respond_to_user", "apply_resolution", "escalate"
        ]
    )
    info: str = ""


class Action(BaseModel):
    type: Literal["search_kb", "respond_to_user", "apply_resolution", "escalate"]
    # search_kb
    query: Optional[str] = None
    # respond_to_user
    message: Optional[str] = None
    # apply_resolution
    resolution_code: Optional[str] = None
    summary: Optional[str] = None
    # escalate
    reason: Optional[str] = None


class Reward(BaseModel):
    step_reward: float = 0.0
    cumulative_reward: float = 0.0
    milestone_hit: Optional[str] = None
    penalty_hit: Optional[str] = None
    breakdown: Dict[str, float] = Field(default_factory=dict)


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Knowledge Base Search Engine
# ---------------------------------------------------------------------------

def _search_kb(query: str, top_k: int = 3) -> List[KBResult]:
    """
    Simple keyword-overlap search over the knowledge base.
    Returns up to top_k results sorted by relevance score.
    """
    query_tokens = set(re.sub(r"[^\w\s]", "", query.lower()).split())
    scored: List[Tuple[float, Dict]] = []

    for article in KNOWLEDGE_BASE:
        kw_tokens = set(article["keywords"])
        title_tokens = set(article["title"].lower().split())
        policy_tokens = set(article["policy"].lower().split())

        kw_overlap = len(query_tokens & kw_tokens) / max(len(kw_tokens), 1)
        title_overlap = len(query_tokens & title_tokens) / max(len(title_tokens), 1)
        policy_overlap = len(query_tokens & policy_tokens) / max(len(policy_tokens), 1)

        score = 0.6 * kw_overlap + 0.3 * title_overlap + 0.1 * policy_overlap
        if score > 0:
            scored.append((score, article))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [
        KBResult(
            id=art["id"],
            title=art["title"],
            policy=art["policy"],
            resolution_code=art["resolution_code"],
            escalate_if=art["escalate_if"],
            relevance_score=round(score, 4),
        )
        for score, art in scored[:top_k]
    ]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SupportDeskEnv:
    """
    OpenEnv-compatible Tier-1 Support Desk environment.

    Lifecycle:
        env = SupportDeskEnv(task)
        obs = env.reset()
        while not done:
            action = agent.act(obs)
            step_result = env.step(action)
            obs = step_result.observation
            done = step_result.done
    """

    MAX_TURNS: int = 10

    def __init__(self, task: "BaseTask"):  # noqa: F821
        self.task = task
        self._ticket: Optional[Ticket] = None
        self._conversation: List[ConversationTurn] = []
        self._turn: int = 0
        self._resolved: bool = False
        self._escalated: bool = False
        self._kb_searched: bool = False
        self._correct_policy_identified: bool = False
        self._cumulative_reward: float = 0.0
        self._reward_breakdown: Dict[str, float] = {}
        self._last_kb_results: List[KBResult] = []
        self._repeated_info_queries: int = 0
        self._info_already_in_ticket: set = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset the environment and return the initial observation."""
        self._ticket = self.task.get_ticket()
        self._conversation = []
        self._turn = 0
        self._resolved = False
        self._escalated = False
        self._kb_searched = False
        self._correct_policy_identified = False
        self._cumulative_reward = 0.0
        self._reward_breakdown = {}
        self._last_kb_results = []
        self._repeated_info_queries = 0
        self._info_already_in_ticket = self._extract_ticket_info(self._ticket)

        # System context turn
        self._conversation.append(ConversationTurn(
            role="system",
            content=(
                "You are a Tier-1 support agent. Resolve the customer's ticket "
                "using the knowledge base. Do not hallucinate policies. "
                f"You have a maximum of {self.MAX_TURNS} turns."
            ),
            turn=0,
        ))
        # User's opening message
        self._conversation.append(ConversationTurn(
            role="user",
            content=self._ticket.body,
            turn=0,
        ))

        return self._build_observation()

    def step(self, action: Action) -> StepResult:
        """Execute one action and return (observation, reward, done, info)."""
        if self._resolved or self._escalated:
            return StepResult(
                observation=self._build_observation(),
                reward=Reward(
                    step_reward=0.0,
                    cumulative_reward=self._cumulative_reward,
                    breakdown=self._reward_breakdown,
                ),
                done=True,
                info={"reason": "already_terminated"},
            )

        self._turn += 1
        step_reward = 0.0
        milestone_hit = None
        penalty_hit = None

        # ---- Dispatch action ----
        if action.type == "search_kb":
            step_reward, milestone_hit, penalty_hit = self._handle_search_kb(action)

        elif action.type == "respond_to_user":
            step_reward, milestone_hit, penalty_hit = self._handle_respond(action)

        elif action.type == "apply_resolution":
            step_reward, milestone_hit, penalty_hit = self._handle_resolution(action)

        elif action.type == "escalate":
            step_reward, milestone_hit, penalty_hit = self._handle_escalate(action)

        # ---- Turn limit penalty ----
        if self._turn > self.MAX_TURNS and not (self._resolved or self._escalated):
            penalty = -0.1
            step_reward += penalty
            self._reward_breakdown["exceeded_turns"] = (
                self._reward_breakdown.get("exceeded_turns", 0.0) + penalty
            )
            penalty_hit = "exceeded_turns"

        # ---- Clamp cumulative reward to [0.0, 1.0] ----
        self._cumulative_reward = max(
            0.0, min(1.0, self._cumulative_reward + step_reward)
        )

        done = (
            self._resolved
            or self._escalated
            or self._turn >= self.MAX_TURNS
        )

        reward = Reward(
            step_reward=round(step_reward, 4),
            cumulative_reward=round(self._cumulative_reward, 4),
            milestone_hit=milestone_hit,
            penalty_hit=penalty_hit,
            breakdown=dict(self._reward_breakdown),
        )

        return StepResult(
            observation=self._build_observation(),
            reward=reward,
            done=done,
            info={
                "turn": self._turn,
                "resolved": self._resolved,
                "escalated": self._escalated,
            },
        )

    def state(self) -> Dict[str, Any]:
        """Return full serializable environment state (for checkpointing)."""
        return {
            "ticket": self._ticket.model_dump() if self._ticket else None,
            "conversation": [c.model_dump() for c in self._conversation],
            "turn": self._turn,
            "resolved": self._resolved,
            "escalated": self._escalated,
            "kb_searched": self._kb_searched,
            "correct_policy_identified": self._correct_policy_identified,
            "cumulative_reward": self._cumulative_reward,
            "reward_breakdown": self._reward_breakdown,
        }

    # ------------------------------------------------------------------
    # Action Handlers
    # ------------------------------------------------------------------

    def _handle_search_kb(
        self, action: Action
    ) -> Tuple[float, Optional[str], Optional[str]]:
        if not action.query:
            return 0.0, None, None

        # Penalty: asking for info already in the ticket
        query_lower = action.query.lower()
        overlap = any(token in query_lower for token in self._info_already_in_ticket)
        if overlap and self._repeated_info_queries > 0:
            self._reward_breakdown["repeated_info_request"] = (
                self._reward_breakdown.get("repeated_info_request", 0.0) - 0.1
            )
            return -0.1, None, "repeated_info_request"

        self._last_kb_results = _search_kb(action.query)

        # Milestone: first KB search
        reward = 0.0
        milestone = None
        if not self._kb_searched and self._last_kb_results:
            self._kb_searched = True
            reward += 0.2
            self._reward_breakdown["kb_searched"] = 0.2
            milestone = "kb_searched"

        # Check if a correct policy was surfaced
        expected_codes = self.task.expected_resolution_codes()
        for result in self._last_kb_results:
            if result.resolution_code in expected_codes:
                if not self._correct_policy_identified:
                    self._correct_policy_identified = True
                    reward += 0.5
                    self._reward_breakdown["correct_policy_identified"] = 0.5
                    milestone = "correct_policy_identified"
                break

        return reward, milestone, None

    def _handle_respond(
        self, action: Action
    ) -> Tuple[float, Optional[str], Optional[str]]:
        if not action.message:
            return 0.0, None, None

        # Hallucination check: agent mentions a resolution code not in KB
        mentioned_codes = re.findall(r"RES-[A-Z]+-\d+", action.message.upper())
        for code in mentioned_codes:
            if code not in VALID_RESOLUTION_CODES:
                self._reward_breakdown["hallucinated_policy"] = (
                    self._reward_breakdown.get("hallucinated_policy", 0.0) - 0.3
                )
                self._conversation.append(ConversationTurn(
                    role="agent", content=action.message, turn=self._turn
                ))
                return -0.3, None, "hallucinated_policy"

        # Contradiction check: agent contradicts KB policy keywords
        penalty = self._check_contradiction(action.message)
        if penalty < 0:
            self._reward_breakdown["contradicts_kb"] = (
                self._reward_breakdown.get("contradicts_kb", 0.0) + penalty
            )
            self._conversation.append(ConversationTurn(
                role="agent", content=action.message, turn=self._turn
            ))
            return penalty, None, "contradicts_kb"

        self._conversation.append(ConversationTurn(
            role="agent", content=action.message, turn=self._turn
        ))

        # Simulate user follow-up from task
        user_reply = self.task.get_user_reply(self._turn, action.message)
        if user_reply:
            self._conversation.append(ConversationTurn(
                role="user", content=user_reply, turn=self._turn
            ))

        return 0.0, None, None

    def _handle_resolution(
        self, action: Action
    ) -> Tuple[float, Optional[str], Optional[str]]:
        if not action.resolution_code:
            return 0.0, None, None

        code = action.resolution_code.upper().strip()

        # Hallucination: code doesn't exist in KB at all
        if code not in VALID_RESOLUTION_CODES:
            self._reward_breakdown["hallucinated_policy"] = (
                self._reward_breakdown.get("hallucinated_policy", 0.0) - 0.3
            )
            return -0.3, None, "hallucinated_policy"

        expected_codes = self.task.expected_resolution_codes()

        if code in expected_codes:
            # Correct resolution
            self._resolved = True
            base = 1.0
            # Efficiency bonus: resolved faster earns higher reward
            turn_fraction = self._turn / self.MAX_TURNS
            efficiency_bonus = round(0.3 * (1.0 - turn_fraction), 4)
            total = base + efficiency_bonus
            self._reward_breakdown["resolved_correctly"] = base
            self._reward_breakdown["efficiency_bonus"] = efficiency_bonus
            self._conversation.append(ConversationTurn(
                role="system",
                content=f"Ticket resolved with code {code}. Summary: {action.summary}",
                turn=self._turn,
            ))
            return total, "resolved_correctly", None
        else:
            # Wrong resolution code applied
            self._reward_breakdown["wrong_resolution"] = (
                self._reward_breakdown.get("wrong_resolution", 0.0) - 0.2
            )
            return -0.2, None, "wrong_resolution"

    def _handle_escalate(
        self, action: Action
    ) -> Tuple[float, Optional[str], Optional[str]]:
        if not action.reason:
            return 0.0, None, None

        self._escalated = True
        should_escalate = self.task.should_escalate()

        if should_escalate:
            # Correct escalation
            self._reward_breakdown["correct_escalation"] = 0.8
            self._conversation.append(ConversationTurn(
                role="system",
                content=f"Ticket escalated to Tier-2. Reason: {action.reason}",
                turn=self._turn,
            ))
            return 0.8, "correct_escalation", None
        else:
            # Unnecessary escalation
            self._reward_breakdown["unnecessary_escalation"] = -0.2
            self._conversation.append(ConversationTurn(
                role="system",
                content=f"Ticket incorrectly escalated. Reason: {action.reason}",
                turn=self._turn,
            ))
            return -0.2, None, "unnecessary_escalation"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        return Observation(
            ticket=self._ticket,
            kb_results=self._last_kb_results,
            conversation=list(self._conversation),
            turn=self._turn,
            resolved=self._resolved,
            escalated=self._escalated,
            info=(
                "Ticket resolved." if self._resolved
                else "Ticket escalated." if self._escalated
                else f"Turn {self._turn}/{self.MAX_TURNS}. Use search_kb, respond_to_user, "
                     "apply_resolution, or escalate."
            ),
        )

    def _extract_ticket_info(self, ticket: Ticket) -> set:
        """Extract tokens from ticket body for repeated-info-request detection."""
        tokens = set(re.sub(r"[^\w\s]", "", ticket.body.lower()).split())
        tokens |= set(re.sub(r"[^\w\s]", "", ticket.subject.lower()).split())
        return tokens

    def _check_contradiction(self, message: str) -> float:
        """
        Lightweight contradiction check:
        Flags if agent says things like 'no refund possible' when KB allows refunds,
        or 'you must pay' when KB says credit is available.
        """
        contradiction_patterns = [
            (r"\bno refund\b", ["RES-BILL-02"]),
            (r"\bcannot reset\b", ["RES-PWD-01"]),
            (r"\bno credit\b", ["RES-OUT-03", "RES-BILL-02"]),
            (r"\bwe cannot cancel\b", ["RES-CXL-04"]),
            (r"\bno data export\b", ["RES-PRIV-06"]),
        ]
        msg_lower = message.lower()
        relevant_codes = self.task.expected_resolution_codes()
        for pattern, associated_codes in contradiction_patterns:
            if re.search(pattern, msg_lower):
                if any(c in relevant_codes for c in associated_codes):
                    return -0.4
        return 0.0
