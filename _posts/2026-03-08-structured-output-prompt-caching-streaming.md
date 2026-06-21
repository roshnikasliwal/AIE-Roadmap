---
title: "Structured Output, Prompt Caching, and Streaming Responses"
date: 2026-03-08 08:00:00 +0530
categories: [AI, Prompt Engineering]
tags: [structured-output, prompt-caching, streaming, llm, production, roadmap]
---

Three techniques that are essential for production-quality AI applications: structured output for reliable parsing, prompt caching for cost reduction, and streaming for better user experience. This post covers all three.

## Structured Output

By default, LLM responses are free-form text. In applications, you usually need structured data that your code can parse reliably. Structured output guarantees the model responds in a specific format — typically JSON.

### Why You Need It

Without structured output, you might try to parse a natural language response like:

> "The sentiment is positive, with a confidence score of around 0.92. The key phrases I identified are 'great service' and 'fast delivery'."

This is fragile — the wording varies, parsing fails easily, and your application breaks unpredictably.

### JSON Mode

The simplest approach — instruct the model to output JSON and enable JSON mode:

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": "Extract sentiment data. Respond with valid JSON only: {\"sentiment\": string, \"confidence\": float, \"key_phrases\": array}"
        },
        {"role": "user", "content": "Great service and fast delivery!"}
    ],
    response_format={"type": "json_object"}
)

data = json.loads(response.choices[0].message.content)
# {"sentiment": "positive", "confidence": 0.95, "key_phrases": ["great service", "fast delivery"]}
```

### Structured Outputs with Schema (OpenAI)

OpenAI's Structured Outputs feature enforces a JSON Schema at the API level — the model is *guaranteed* to match the schema:

```python
from pydantic import BaseModel
from openai import OpenAI

class SentimentResult(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    key_phrases: list[str]

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Extract sentiment data from the text."},
        {"role": "user", "content": "Great service and fast delivery!"}
    ],
    response_format=SentimentResult
)

result = response.choices[0].message.parsed
print(result.sentiment)    # "positive"
print(result.confidence)   # 0.95
```

### When to Use Structured Output

- Any time your code processes the model's response programmatically
- Data extraction pipelines
- Classification tasks
- Generating structured data for databases
- API responses that must match a schema

---

## Prompt Caching

**Prompt caching** dramatically reduces cost and latency for applications where the beginning of the prompt stays the same across many requests.

### How It Works

When you send the same prefix (system prompt, few-shot examples, documents) repeatedly, the inference provider caches the computed attention states for that prefix. Subsequent requests that share the same prefix skip recomputation.

**Cost savings:** Cached tokens are typically 50–90% cheaper than uncached tokens.

**Latency improvement:** First token time decreases significantly for long prompts.

### Anthropic (Claude) — Explicit Caching

```python
import anthropic
client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "You are a legal document analyst...",
        },
        {
            "type": "text",
            "text": "[Very long legal document — 50,000 tokens]",
            "cache_control": {"type": "ephemeral"}  # Cache this part
        }
    ],
    messages=[{"role": "user", "content": "What are the termination clauses?"}]
)
```

The long document is cached after the first request. Subsequent questions about the same document hit the cache and cost significantly less.

### OpenAI — Automatic Caching

OpenAI automatically caches prompt prefixes without any code changes. The first 1024+ tokens of your prompt prefix are eligible. You see cache hit/miss stats in the API response:

```python
usage = response.usage
print(usage.prompt_tokens_details.cached_tokens)  # How many were cached
```

### Best Practices for Caching

1. **Put static content first** — System prompt, instructions, documents, few-shot examples
2. **Put dynamic content last** — User messages, timestamps, per-request data
3. **Keep the prefix stable** — Even small changes to the cached portion invalidate the cache
4. **Use with large prompts** — Caching has minimal impact on short prompts; the benefit grows with prefix length

---

## Streaming Responses

**Streaming** sends tokens to the client as they're generated, instead of waiting for the full response. This is critical for user-facing applications — a streaming response feels 3–10x faster to users even if total generation time is the same.

### Without Streaming

```python
# User waits 8 seconds for the full response to appear all at once
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Explain quantum computing."}]
)
print(response.choices[0].message.content)
```

### With Streaming

```python
# Tokens appear as they're generated — feels much faster
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Explain quantum computing."}],
    stream=True
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
```

### Streaming in Web Applications

For web apps, use Server-Sent Events (SSE) to stream tokens from your backend to the browser:

```python
# FastAPI example
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/chat")
async def chat(message: str):
    async def generate():
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": message}],
            stream=True
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield f"data: {delta.content}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

### Streaming with Tool Calls

When streaming with function calling, you need to accumulate tool call arguments as they stream in:

```python
tool_calls = []
current_tool_call = None

for chunk in stream:
    delta = chunk.choices[0].delta
    
    if delta.tool_calls:
        for tc_chunk in delta.tool_calls:
            if tc_chunk.index == len(tool_calls):
                tool_calls.append({"id": "", "function": {"name": "", "arguments": ""}})
            
            if tc_chunk.id:
                tool_calls[tc_chunk.index]["id"] = tc_chunk.id
            if tc_chunk.function.name:
                tool_calls[tc_chunk.index]["function"]["name"] += tc_chunk.function.name
            if tc_chunk.function.arguments:
                tool_calls[tc_chunk.index]["function"]["arguments"] += tc_chunk.function.arguments
```

## Combining All Three

A production-quality chat application uses all three:

```python
# 1. Cached system prompt + documents
# 2. Structured output for extraction tasks  
# 3. Streaming for chat responses

stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": CACHED_SYSTEM_PROMPT},  # auto-cached
        {"role": "user", "content": user_message}
    ],
    stream=True  # stream to UI
    # For extraction tasks: add response_format={"type": "json_object"}
)
```

## Key Takeaways

- **Structured output** makes LLM responses parseable and reliable — use it whenever your code processes responses
- **Prompt caching** cuts costs by 50–90% on repeated prefixes — put static content first
- **Streaming** dramatically improves perceived performance — enable it for all user-facing chat

---

*Part of the [AI Engineer Roadmap series]({{ site.baseurl }}/tags/roadmap/) — following the structure from [roadmap.sh/ai-engineer](https://roadmap.sh/ai-engineer).*
