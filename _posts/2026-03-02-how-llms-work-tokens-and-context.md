---
title: "How LLMs Work: Tokens, Context Windows, and Sampling Parameters"
date: 2026-03-02 08:00:00 +0530
categories: [AI, LLMs]
tags: [llm, tokens, context-window, sampling, roadmap]
---

Before you can engineer prompts or build reliable AI systems, you need to understand what's actually happening under the hood of a large language model. This post covers the core mechanics every AI Engineer must know.

## What is a Token?

LLMs don't process text character by character or word by word — they process **tokens**. A token is roughly:

- ~4 characters of English text
- Or about 0.75 words on average

Examples:
- `"Hello"` → 1 token
- `"Hello, world!"` → 4 tokens
- `"unbelievable"` → 3 tokens (`un`, `believ`, `able`)

Why does this matter? Because:
- **Cost** is billed per token (input + output)
- **Context limits** are measured in tokens
- **Response length** is capped in tokens

### Counting Tokens

Use `tiktoken` (OpenAI's library) to count tokens before sending a request:

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")
tokens = enc.encode("Hello, how are you?")
print(len(tokens))  # 5
```

## The Context Window

The **context window** is the total number of tokens the model can "see" at once — including your system prompt, conversation history, tool outputs, and the response being generated.

| Model             | Context Window   |
| ----------------- | ---------------- |
| GPT-4o            | 128,000 tokens   |
| Claude 3.5 Sonnet | 200,000 tokens   |
| Gemini 1.5 Pro    | 1,000,000 tokens |
| Llama 3.1 (70B)   | 128,000 tokens   |

Key implications:
- Everything outside the context window is **invisible** to the model
- Long conversations need context management strategies (summarization, trimming)
- Larger contexts cost more and can increase latency

### What Goes into the Context?

```
[System Prompt]
[Conversation History]
[Retrieved Documents (RAG)]
[Tool Call Results]
[Current User Message]
──────────────────────────
[Model Response ←── generated here]
```

## Sampling Parameters

When a model generates a response, it predicts the next token probabilistically — it doesn't just pick the single "best" token every time. Sampling parameters control this process.

### Temperature

Temperature scales the probability distribution over possible next tokens.

- **Temperature = 0** → Always pick the highest-probability token (deterministic, repetitive)
- **Temperature = 1** → Sample according to the raw probability distribution (default)
- **Temperature > 1** → Amplify lower-probability tokens (more creative, more random)

```python
# Factual Q&A — low temperature for consistency
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    temperature=0.1
)

# Creative writing — higher temperature
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a poem about rain."}],
    temperature=0.9
)
```

**Rule of thumb:**
- Factual tasks: `0.0–0.3`
- Balanced: `0.5–0.7`
- Creative: `0.8–1.2`

### Top-K Sampling

At each generation step, only consider the **K most likely** next tokens and sample from them.

- `top_k=1` → Always pick the most likely token (greedy)
- `top_k=50` → Sample from the top 50 candidates
- Higher K = more diversity, lower K = more focused

### Top-P (Nucleus Sampling)

Instead of a fixed K, Top-P samples from the smallest set of tokens whose cumulative probability exceeds **P**.

- `top_p=0.9` → Consider tokens that together account for 90% of the probability mass
- If a model is very confident (one token has 95% probability), Top-P=0.9 might select only that one token
- If the model is uncertain, many tokens may be included

Top-P is generally preferred over Top-K because it adapts to the model's confidence level.

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    temperature=0.7,
    top_p=0.9  # nucleus sampling
)
```

### Repetition Penalties

LLMs have a natural tendency to repeat themselves, especially on longer generations. Repetition penalties discourage this.

- **Frequency penalty** (OpenAI) — Reduces the probability of tokens that have already appeared, proportional to how many times they've appeared
- **Presence penalty** (OpenAI) — Flat penalty for any token that has appeared at all

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    frequency_penalty=0.5,  # -2.0 to 2.0
    presence_penalty=0.3    # -2.0 to 2.0
)
```

## Practical Defaults

For most production use cases, start with:

```python
{
    "temperature": 0.7,
    "top_p": 0.9,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "max_tokens": 1024
}
```

Then tune based on your use case and eval results.

## Key Takeaways

- **Tokens** are the unit of cost and context — understand them
- **Context window** = everything the model can see at once
- **Temperature** controls creativity vs. determinism
- **Top-P** is generally better than Top-K for controlling randomness
- **Repetition penalties** help with longer generations

---

*Part of the [AI Engineer Roadmap series](/tags/roadmap/) — following the structure from [roadmap.sh/ai-engineer](https://roadmap.sh/ai-engineer).*
