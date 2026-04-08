# tasks.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from env import Ticket


# ---------------------------------------------------------------------------
# Base Task Interface
# ---------------------------------------------------------------------------

class BaseTask(ABC):
    """All tasks must implement this interface."""

    @abstractmethod
    def get_ticket(self) -> Ticket:
        """Return the support ticket for this task."""
        ...

    @abstractmethod
    def expected_resolution_codes(self) -> List[str]:
        """Return the list of valid resolution codes for this task."""
        ...

    @abstractmethod
    def should_escalate(self) -> bool:
        """Return True if correct behavior is to escalate this ticket."""
        ...

    @abstractmethod
    def get_user_reply(self, turn: int, agent_message: str) -> Optional[str]:
        """
        Simulate a user reply given the agent's message at a given turn.
        Return None if no user reply is needed at this turn.
        """
        ...

    @abstractmethod
    def grade(self, env_state: Dict[str, Any]) -> "GradeResult":
        """Score the final environment state. Returns a GradeResult."""
        ...


# ---------------------------------------------------------------------------
# Grade Result
# ---------------------------------------------------------------------------

class GradeResult(BaseModel):
    score: float                          # 0.0 – 1.0
    passed: bool
    breakdown: Dict[str, float]
    feedback: str


# ---------------------------------------------------------------------------
# Grader Base
# ---------------------------------------------------------------------------

class BaseGrader:
    """
    Shared grading logic. Each task's Grader subclasses this and
    overrides `expected_resolution_codes` and `must_escalate`.
    """

    PASS_THRESHOLD: float = 0.5

    def __init__(
        self,
        expected_codes: List[str],
        must_escalate: bool = False,
        max_turns: int = 10,
    ):
        self.expected_codes = expected_codes
        self.must_escalate = must_escalate
        self.max_turns = max_turns

    def grade(self, env_state: Dict[str, Any]) -> GradeResult:
        breakdown: Dict[str, float] = {}
        feedback_lines: List[str] = []

        resolved: bool = env_state.get("resolved", False)
        escalated: bool = env_state.get("escalated", False)
        kb_searched: bool = env_state.get("kb_searched", False)
        correct_policy: bool = env_state.get("correct_policy_identified", False)
        turns: int = env_state.get("turn", self.max_turns)
        reward_breakdown: Dict[str, float] = env_state.get("reward_breakdown", {})

        # ── Milestone 1: KB was searched ──────────────────────────────
        if kb_searched:
            breakdown["kb_searched"] = 0.10
            feedback_lines.append("✓ Knowledge base was queried.")
        else:
            breakdown["kb_searched"] = 0.0
            feedback_lines.append("✗ Knowledge base was never searched.")

        # ── Milestone 2: Correct policy identified ────────────────────
        if correct_policy:
            breakdown["correct_policy_identified"] = 0.20
            feedback_lines.append("✓ Correct policy was identified.")
        else:
            breakdown["correct_policy_identified"] = 0.0
            feedback_lines.append("✗ Correct policy was NOT identified.")

        # ── Milestone 3: Resolution / Escalation outcome ──────────────
        if self.must_escalate:
            if escalated:
                breakdown["correct_escalation"] = 0.50
                feedback_lines.append("✓ Ticket correctly escalated to Tier-2.")
            else:
                breakdown["correct_escalation"] = 0.0
                feedback_lines.append("✗ Ticket should have been escalated but was not.")
            if resolved:
                breakdown["incorrect_resolution_penalty"] = -0.20
                feedback_lines.append("✗ Agent applied resolution instead of escalating.")
        else:
            if resolved:
                breakdown["resolved_correctly"] = 0.50
                feedback_lines.append("✓ Ticket resolved with a valid resolution code.")
            else:
                breakdown["resolved_correctly"] = 0.0
                feedback_lines.append("✗ Ticket was not resolved.")
            if escalated:
                breakdown["unnecessary_escalation_penalty"] = -0.20
                feedback_lines.append("✗ Ticket was unnecessarily escalated.")

        # ── Milestone 4: Efficiency (turn usage) ──────────────────────
        if resolved or escalated:
            turn_ratio = turns / self.max_turns
            efficiency = round(0.20 * (1.0 - turn_ratio), 4)
            breakdown["efficiency"] = efficiency
            feedback_lines.append(
                f"✓ Efficiency bonus: {efficiency:.2f} "
                f"(resolved in {turns}/{self.max_turns} turns)."
            )
        else:
            breakdown["efficiency"] = 0.0

        # ── Penalties from reward breakdown ───────────────────────────
        penalty_keys = [
            "hallucinated_policy",
            "contradicts_kb",
            "repeated_info_request",
            "exceeded_turns",
            "wrong_resolution",
        ]
        for key in penalty_keys:
            if key in reward_breakdown and reward_breakdown[key] < 0:
                breakdown[f"penalty_{key}"] = reward_breakdown[key]
                feedback_lines.append(
                    f"✗ Penalty applied — {key.replace('_', ' ')}: "
                    f"{reward_breakdown[key]:.2f}"
                )

        # ── Final score (clamped) ──────────────────────────────────────
        raw_score = sum(breakdown.values())
        final_score = round(max(0.0, min(1.0, raw_score)), 4)
        passed = final_score >= self.PASS_THRESHOLD

        return GradeResult(
            score=final_score,
            passed=passed,
            breakdown=breakdown,
            feedback="\n".join(feedback_lines),
        )


# ---------------------------------------------------------------------------
# TASK 1 — Easy: Password Reset
# ---------------------------------------------------------------------------

class EasyTaskGrader(BaseGrader):
    def __init__(self):
        super().__init__(
            expected_codes=["RES-PWD-01"],
            must_escalate=False,
            max_turns=10,
        )


class EasyTask(BaseTask):
    """
    Scenario: A user forgot their password and cannot log in.
    The agent should search the KB, find the password reset policy,
    verify identity (email + last 4 digits), and apply RES-PWD-01.
    No escalation needed.
    """

    TICKET = Ticket(
        id="TKT-001",
        subject="Cannot log in — forgot password",
        body=(
            "Hi, I've forgotten my password and can't log into my account. "
            "My registered email is alice@example.com and my account ID is ACC-4821. "
            "I've tried the 'Forgot Password' link but I'm not receiving the reset email. "
            "Can you please help me reset it manually?"
        ),
        user_name="Alice Johnson",
        account_id="ACC-4821",
        user_history=[
            "2024-11-10: Contacted support about billing query — resolved.",
            "2024-12-02: Password reset email resent successfully.",
        ],
        metadata={"plan": "basic", "registered_email": "alice@example.com"},
    )

    # Scripted user replies keyed by turn number
    USER_REPLIES: Dict[int, str] = {
        1: (
            "Yes, my registered email is alice@example.com "
            "and the last 4 digits of my phone are 7823."
        ),
        2: "I still haven't received the reset email. Can you try again?",
        3: "Got it! The reset link worked. Thank you so much!",
    }

    def get_ticket(self) -> Ticket:
        return self.TICKET.model_copy(deep=True)

    def expected_resolution_codes(self) -> List[str]:
        return ["RES-PWD-01"]

    def should_escalate(self) -> bool:
        return False

    def get_user_reply(self, turn: int, agent_message: str) -> Optional[str]:
        return self.USER_REPLIES.get(turn)

    def grade(self, env_state: Dict[str, Any]) -> GradeResult:
        return EasyTaskGrader().grade(env_state)


# ---------------------------------------------------------------------------
# TASK 2 — Medium: Billing Dispute with Partial Credit
# ---------------------------------------------------------------------------

class MediumTaskGrader(BaseGrader):
    def __init__(self):
        super().__init__(
            expected_codes=["RES-BILL-02"],
            must_escalate=False,
            max_turns=10,
        )

    def grade(self, env_state: Dict[str, Any]) -> GradeResult:
        result = super().grade(env_state)

        # Medium-specific bonus: agent offered credit within policy limits
        conversation = env_state.get("conversation", [])
        agent_messages = [
            t["content"] for t in conversation
            if t.get("role") == "agent"
        ]
        full_agent_text = " ".join(agent_messages).lower()

        extra_breakdown: Dict[str, float] = {}
        extra_feedback: List[str] = []

        # Bonus: agent mentioned the $25 courtesy credit
        if "25" in full_agent_text or "courtesy credit" in full_agent_text:
            extra_breakdown["offered_courtesy_credit"] = 0.10
            extra_feedback.append("✓ Agent correctly offered courtesy credit up to $25.")

        # Penalty: agent promised a refund over $25 without asking for proof
        if "refund" in full_agent_text and "proof" not in full_agent_text:
            over_promise = any(
                str(amt) in full_agent_text
                for amt in range(26, 200)
            )
            if over_promise:
                extra_breakdown["overpromised_refund"] = -0.10
                extra_feedback.append(
                    "✗ Agent promised a refund over $25 without requesting proof."
                )

        new_breakdown = {**result.breakdown, **extra_breakdown}
        raw = sum(new_breakdown.values())
        final_score = round(max(0.0, min(1.0, raw)), 4)

        return GradeResult(
            score=final_score,
            passed=final_score >= self.PASS_THRESHOLD,
            breakdown=new_breakdown,
            feedback=result.feedback + (
                "\n" + "\n".join(extra_feedback) if extra_feedback else ""
            ),
        )


class MediumTask(BaseTask):
    """
    Scenario: A user was charged $45 for a plan they claim to have cancelled.
    The agent must: identify the billing dispute policy, verify the charge,
    offer up to $25 courtesy credit (or request proof for the remainder),
    and apply RES-BILL-02.
    Escalation is NOT required (amount is $45, within Tier-1 range with proof).
    """

    TICKET = Ticket(
        id="TKT-002",
        subject="Incorrect charge of $45 on my account",
        body=(
            "Hello, I was charged $45 on March 3rd for the Premium plan, "
            "but I cancelled my subscription on February 28th. "
            "My account ID is ACC-9034. I have a screenshot of the cancellation confirmation. "
            "I'd like this charge reversed immediately. "
            "This is really frustrating — I've been a customer for 3 years!"
        ),
        user_name="Bob Martinez",
        account_id="ACC-9034",
        user_history=[
            "2024-01-15: Upgraded to Premium plan.",
            "2024-02-28: Cancellation request submitted by user.",
            "2024-03-03: $45 charge processed (system error — cancellation not propagated).",
        ],
        metadata={
            "plan": "premium_cancelled",
            "dispute_amount": 45.00,
            "cancellation_date": "2024-02-28",
            "charge_date": "2024-03-03",
        },
    )

    USER_REPLIES: Dict[int, str] = {
        1: (
            "Yes, I have a screenshot of the cancellation confirmation email "
            "dated February 28th. How do I send it to you?"
        ),
        2: (
            "I've uploaded the screenshot to the billing portal as you suggested. "
            "Ticket reference is BP-77234."
        ),
        3: (
            "Thank you. Will the full $45 be refunded or just the $25 courtesy credit?"
        ),
        4: "Okay, I understand. Thanks for processing this.",
    }

    def get_ticket(self) -> Ticket:
        return self.TICKET.model_copy(deep=True)

    def expected_resolution_codes(self) -> List[str]:
        return ["RES-BILL-02"]

    def should_escalate(self) -> bool:
        return False

    def get_user_reply(self, turn: int, agent_message: str) -> Optional[str]:
        return self.USER_REPLIES.get(turn)

    def grade(self, env_state: Dict[str, Any]) -> GradeResult:
        return MediumTaskGrader().grade(env_state)


# ---------------------------------------------------------------------------
# TASK 3 — Hard: Multi-Issue Technical Complaint + Escalation Decision
# ---------------------------------------------------------------------------

class HardTaskGrader(BaseGrader):
    def __init__(self):
        super().__init__(
            expected_codes=["RES-TECH-05"],
            must_escalate=True,   # All 4 steps fail → must escalate
            max_turns=10,
        )

    def grade(self, env_state: Dict[str, Any]) -> GradeResult:
        result = super().grade(env_state)

        conversation = env_state.get("conversation", [])
        agent_messages = [
            t["content"] for t in conversation
            if t.get("role") == "agent"
        ]
        full_agent_text = " ".join(agent_messages).lower()

        extra_breakdown: Dict[str, float] = {}
        extra_feedback: List[str] = []

        # Bonus: agent walked through all 4 troubleshooting steps before escalating
        steps_mentioned = sum([
            "browser" in full_agent_text or "app version" in full_agent_text,
            "cache" in full_agent_text or "cookies" in full_agent_text,
            "incognito" in full_agent_text or "private mode" in full_agent_text,
            "second device" in full_agent_text or "another device" in full_agent_text,
        ])
        if steps_mentioned >= 3:
            extra_breakdown["followed_troubleshooting_steps"] = 0.15
            extra_feedback.append(
                f"✓ Agent covered {steps_mentioned}/4 troubleshooting steps before escalating."
            )
        else:
            extra_breakdown["followed_troubleshooting_steps"] = 0.0
            extra_feedback.append(
                f"✗ Agent only covered {steps_mentioned}/4 troubleshooting steps."
            )

        # Bonus: agent mentioned HAR file collection
        if "har" in full_agent_text:
            extra_breakdown["har_file_requested"] = 0.10
            extra_feedback.append("✓ Agent requested a HAR file before escalating.")
        else:
            extra_breakdown["har_file_requested"] = 0.0
            extra_feedback.append("✗ Agent did not request a HAR file before escalating.")

        # Penalty: agent promised an ETA for resolution (not allowed per KB003)
        if any(phrase in full_agent_text for phrase in ["eta", "fixed by", "resolved by", "within 24"]):
            extra_breakdown["promised_eta_penalty"] = -0.10
            extra_feedback.append("✗ Agent promised an ETA, which violates KB003 policy.")

        new_breakdown = {**result.breakdown, **extra_breakdown}
        raw = sum(new_breakdown.values())
        final_score = round(max(0.0, min(1.0, raw)), 4)

        return GradeResult(
            score=final_score,
            passed=final_score >= self.PASS_THRESHOLD,
            breakdown=new_breakdown,
            feedback=result.feedback + (
                "\n" + "\n".join(extra_feedback) if extra_feedback else ""
            ),
        )


class HardTask(BaseTask):
    """
    Scenario: A user reports severe connectivity issues AND a potential
    data loss event during a recent outage. The agent must:
      1. Follow all 4 KB005 troubleshooting steps.
      2. Recognise the data loss mention triggers KB003 mandatory escalation.
      3. Collect a HAR file.
      4. Escalate to Tier-2 — applying a resolution code instead is WRONG.
    This tests the agent's ability to handle multi-policy tickets and
    make the correct escalation decision under complexity.
    """

    TICKET = Ticket(
        id="TKT-003",
        subject="Severe connectivity issues + possible data loss after outage",
        body=(
            "Hi, I've been experiencing severe connectivity issues for the past 48 hours. "
            "Pages time out, the app crashes on load, and my VPN connection keeps dropping. "
            "I've already tried clearing my cache and restarting my router. "
            "To make things worse, during last Tuesday's outage I think some of my saved "
            "project files may have been lost or corrupted. I really need this sorted urgently. "
            "Account ID: ACC-1177."
        ),
        user_name="Carol Nguyen",
        account_id="ACC-1177",
        user_history=[
            "2024-03-05: Service outage reported — lasted 5.5 hours.",
            "2024-03-05: $10 service credit automatically applied.",
            "2024-03-06: User logged in — no complaint at that time.",
            "2024-03-07: Current ticket opened.",
        ],
        metadata={
            "plan": "premium",
            "outage_date": "2024-03-05",
            "outage_duration_hours": 5.5,
            "data_loss_reported": True,
        },
    )

    USER_REPLIES: Dict[int, str] = {
        1: (
            "I've already tried clearing cache and cookies. "
            "Still timing out on both Chrome and Firefox."
        ),
        2: (
            "Yes, I tried incognito mode too — same issue. "
            "Also tried on my phone and it's happening there as well."
        ),
        3: (
            "I can try to generate a HAR file. Can you walk me through it? "
            "Also, what about my lost project files? That's actually the more urgent issue."
        ),
        4: (
            "Okay, I've captured the HAR file. "
            "And yes — at least 3 project files I was working on are now gone. "
            "This is a serious problem."
        ),
        5: (
            "Thank you for escalating. "
            "Please make sure the data loss is flagged as the priority issue."
        ),
    }

    def get_ticket(self) -> Ticket:
        return self.TICKET.model_copy(deep=True)

    def expected_resolution_codes(self) -> List[str]:
        # RES-TECH-05 is the connectivity code, but since data loss is present,
        # the correct action is escalation, not resolution.
        return ["RES-TECH-05"]

    def should_escalate(self) -> bool:
        return True  # Data loss during outage → mandatory Tier-2

    def get_user_reply(self, turn: int, agent_message: str) -> Optional[str]:
        return self.USER_REPLIES.get(turn)

    def grade(self, env_state: Dict[str, Any]) -> GradeResult:
        return HardTaskGrader().grade(env_state)


# ---------------------------------------------------------------------------
# Task Registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: Dict[str, BaseTask] = {
    "task_easy": EasyTask(),
    "task_medium": MediumTask(),
    "task_hard": HardTask(),
}


def get_task(task_id: str) -> BaseTask:
    if task_id not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task_id '{task_id}'. "
            f"Available: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[task_id]
