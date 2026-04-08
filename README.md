# Solution-Meta

# 🧠 Automated Tier-1 Support Desk (OpenEnv)

A realistic OpenEnv environment simulating a **Tier-1 Customer Support Agent** that must resolve tickets using a structured **Knowledge Base (KB)**.

---

## 🚀 Overview

This environment evaluates an AI agent’s ability to:

- Search a knowledge base
- Identify correct policies
- Communicate with users
- Apply correct resolutions
- Decide when to escalate

---

## 📦 Repository

GitHub: https://github.com/Swastika8/Solution-Meta.git  
Hugging Face Space: https://huggingface.co/spaces/Swastika137/SupportDesk

---

## 🧩 Environment Structure

/
├── openenv.yaml
├── env.py
├── tasks.py
├── inference.py
├── requirements.txt
├── Dockerfile
└── README.md


---

## 🧠 Action Space

| Action Type        | Description |
|-------------------|------------|
| `search_kb`        | Query the knowledge base |
| `respond_to_user`  | Send message to customer |
| `apply_resolution` | Apply resolution code |
| `escalate`         | Escalate to Tier-2 |

---

## 👁 Observation Space

- `ticket`: Full ticket info
- `kb_results`: Top KB matches
- `conversation`: Chat history
- `turn`: Current turn
- `resolved`: Boolean
- `escalated`: Boolean

---

## 🏆 Reward Design

### ✅ Positive Rewards

| Milestone | Reward |
|----------|--------|
| KB searched | +0.2 |
| Correct policy identified | +0.5 |
| Correct resolution | +1.0 |
| Efficiency bonus | up to +0.3 |

---

### ❌ Penalties

| Event | Penalty |
|------|--------|
| Hallucinated policy | −0.3 |
| Contradicts KB | −0.4 |
| Unnecessary escalation | −0.2 |
| Repeated info request | −0.1 |
| Exceeded turns | −0.1 |

👉 Reward is **clamped between 0.0 and 1.0**

---

## 📚 Knowledge Base

Simulated JSON database inside `env.py` with realistic policies:

- Password reset
- Billing disputes
- Outages
- Technical troubleshooting
- GDPR/privacy
- Subscription management

---

## 🧪 Tasks

### 🟢 Easy
- Password reset
- No escalation
- Expected: `RES-PWD-01`

### 🟡 Medium
- Billing dispute ($45)
- Partial credit logic
- Expected: `RES-BILL-02`

### 🔴 Hard
- Connectivity + data loss
- MUST escalate
- Multi-policy reasoning

---

## ⚙️ Running Locally

### 1. Install dependencies
```bash
pip install -r requirements.txt
