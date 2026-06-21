---
title: "Chain of Thought and ReAct: Advanced Reasoning Techniques for LLMs"
date: 2026-03-05 08:00:00 +0530
categories: [AI, Prompt Engineering]
tags: [prompt-engineering, chain-of-thought, react, reasoning, agents, roadmap]
---

LLMs make more mistakes when forced to answer immediately — they perform significantly better when prompted to reason step by step. Chain of Thought (CoT) and ReAct are two of the most impactful techniques for improving LLM reasoning quality.

## Chain of Thought (CoT) Prompting

**Chain of Thought prompting** instructs the model to show its reasoning process before arriving at a final answer. This mirrors how humans solve complex problems — by thinking through intermediate steps.

### Without CoT

```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 balls. How many tennis balls does he have now?

A: 11
```

### With CoT

```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 balls. How many tennis balls does he have now?

A: Let me think step by step.
Roger starts with 5 tennis balls.
He buys 2 cans × 3 balls per can = 6 new balls.
Total = 5 + 6 = 11 tennis balls.

The answer is 11.
```

The final answer is the same — but CoT dramatically improves accuracy on harder problems where the model would otherwise short-circuit to a wrong answer.

### How to Trigger CoT

**Explicit instruction:**
```
Think through this step by step before giving your final answer.
```

**Magic phrase (works surprisingly well):**
```
Let's think step by step.
```

**Few-shot CoT (most reliable):**
Provide examples where the correct reasoning chain is demonstrated, then let the model follow the pattern.

### When CoT Helps Most

- Math and logic problems
- Multi-step reasoning (legal analysis, medical diagnosis flows)
- Tasks requiring fact synthesis from multiple sources
- Any task where the model consistently gets the wrong answer without reasoning

### Zero-Shot CoT vs Few-Shot CoT

| | Zero-Shot CoT | Few-Shot CoT |
|---|---|---|
| **Setup** | Add "think step by step" | Provide examples with reasoning chains |
| **Effort** | Minimal | Requires crafting examples |
| **Reliability** | Good | Better |
| **Best for** | Quick improvement | Consistent, specific reasoning patterns |

## ReAct Prompting

**ReAct** (Reasoning + Acting) is a framework where the model alternates between reasoning about what to do and taking actions (tool calls), observing the results, and repeating until the task is complete.

ReAct is the conceptual foundation for most AI agent systems.

### The ReAct Loop

```
Thought: [Reason about the current state and what to do next]
Action: [Call a tool or take an action]
Observation: [Result of the action]
Thought: [Reason about the observation and next step]
Action: [Next tool call]
...
Answer: [Final response once task is complete]
```

### ReAct Example — Research Agent

```
Task: What is the current population of Tokyo?

Thought: I need to find the current population of Tokyo. I should search for up-to-date information.
Action: search("Tokyo population 2026")
Observation: Tokyo's population is approximately 13.96 million in the city proper, and 37.4 million in the greater metropolitan area as of 2025.

Thought: I now have the information needed to answer the question.
Answer: Tokyo's city proper population is approximately 13.96 million, while the greater metropolitan area has about 37.4 million people, making it the world's most populous metropolitan area.
```

### Implementing ReAct with Tool Calling

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful research assistant. Use tools to find accurate, current information."},
        {"role": "user", "content": "What is the current population of Tokyo?"}
    ],
    tools=tools,
    tool_choice="auto"
)

# Check if model wants to call a tool
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    # Execute the tool, get result, add to messages, call again
```

### CoT vs ReAct

| | Chain of Thought | ReAct |
|---|---|---|
| **Actions** | None — reasoning only | Calls external tools |
| **Grounding** | Model's internal knowledge | External observations |
| **Best for** | Reasoning, math, analysis | Research, multi-step tasks requiring real data |
| **Use in agents?** | As reasoning component | As the overall agent loop |

## Combining CoT and ReAct

In practice, the best agents use CoT *within* the Thought steps of a ReAct loop. Each Thought is a mini chain-of-thought that reasons about the observations before deciding the next action.

```
Thought: The search results mention two different population figures — one for the city proper and one for the metro area. I should clarify which one the user is asking about and present both.
Action: answer_user(...)
```

## Practical Tips

1. **Use CoT for any task that requires multi-step reasoning** — it's almost always an improvement
2. **Start with zero-shot CoT** ("think step by step") before building elaborate few-shot examples
3. **Use ReAct when your agent needs external data** — internal reasoning alone isn't enough
4. **Log Thought steps in production** — they're invaluable for debugging agent failures
5. **Combine both** in complex agents: ReAct for the outer loop, CoT for each Thought step

---

*Part of the [AI Engineer Roadmap series]({{ site.baseurl }}/tags/roadmap/) — following the structure from [roadmap.sh/ai-engineer](https://roadmap.sh/ai-engineer).*
