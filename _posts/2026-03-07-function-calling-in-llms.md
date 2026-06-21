---
title: "Function Calling in LLMs: Giving Models the Ability to Use Tools"
date: 2026-03-07 08:00:00 +0530
categories: [AI, Prompt Engineering]
tags: [function-calling, tool-use, llm, agents, openai, roadmap]
---

Function calling (also called "tool use") is the capability that transforms an LLM from a text generator into an agent. It lets models decide when to call external functions, what arguments to pass, and how to use the results — unlocking real-world actions like searching the web, querying databases, or sending emails.

## What is Function Calling?

Without function calling, LLMs are limited to their training data and the context window. With function calling, they can:

- Fetch live data from APIs
- Execute code
- Read and write files
- Query databases
- Call any external service you expose as a function

The model doesn't actually execute the function — it generates a structured function call specification, which your application code executes. The result is then fed back to the model.

## How It Works

### Step 1: Define Tools

You define the functions the model can call using JSON Schema:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g. 'Mumbai'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["city"]
            }
        }
    }
]
```

### Step 2: First API Call

```python
messages = [{"role": "user", "content": "What's the weather in Mumbai?"}]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice="auto"  # model decides when to call tools
)
```

### Step 3: Check for Tool Call

```python
message = response.choices[0].message

if message.tool_calls:
    tool_call = message.tool_calls[0]
    function_name = tool_call.function.name   # "get_weather"
    arguments = json.loads(tool_call.function.arguments)  # {"city": "Mumbai"}
```

### Step 4: Execute the Function

```python
def get_weather(city, unit="celsius"):
    # Your actual implementation — call a weather API
    return {"city": city, "temperature": 32, "condition": "Sunny", "unit": unit}

result = get_weather(**arguments)
```

### Step 5: Second API Call with Result

```python
messages.append(message)  # Add assistant's tool call message
messages.append({
    "role": "tool",
    "tool_call_id": tool_call.id,
    "content": json.dumps(result)
})

final_response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools
)

print(final_response.choices[0].message.content)
# "The current weather in Mumbai is 32°C and Sunny."
```

## Parallel Tool Calls

Modern models can call multiple tools simultaneously when the tasks are independent:

```
User: "What's the weather in Mumbai and Delhi?"

Model calls:
- get_weather(city="Mumbai")
- get_weather(city="Delhi")
[both at the same time]
```

```python
if message.tool_calls:
    results = []
    for tool_call in message.tool_calls:
        args = json.loads(tool_call.function.arguments)
        result = get_weather(**args)
        results.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        })
    
    messages.append(message)
    messages.extend(results)
```

## `tool_choice` Options

| Option                                                | Behavior                                   |
| ----------------------------------------------------- | ------------------------------------------ |
| `"auto"`                                              | Model decides when to call tools (default) |
| `"required"`                                          | Model must call at least one tool          |
| `"none"`                                              | Model cannot call tools this turn          |
| `{"type": "function", "function": {"name": "my_fn"}}` | Force a specific tool call                 |

## Writing Good Tool Descriptions

The model decides which tool to call and what arguments to pass based on the **description** you write. Good descriptions are critical.

**Bad:**
```json
{
  "name": "db_query",
  "description": "Query the database"
}
```

**Good:**
```json
{
  "name": "search_products",
  "description": "Search the product catalog by name, category, or price range. Use this when the user asks about available products, prices, or stock. Returns up to 10 matching products."
}
```

## Error Handling

Always handle the case where the function fails:

```python
try:
    result = execute_tool(tool_call)
except Exception as e:
    result = {"error": str(e), "success": False}

# Feed the error back to the model — it can handle it gracefully
messages.append({
    "role": "tool",
    "tool_call_id": tool_call.id,
    "content": json.dumps(result)
})
```

The model will read the error and either retry with different arguments, explain the failure to the user, or try an alternative approach.

## Function Calling vs RAG

|                    | Function Calling         | RAG                      |
| ------------------ | ------------------------ | ------------------------ |
| **Data freshness** | Real-time                | Indexed (may be stale)   |
| **Data type**      | Structured (APIs, DBs)   | Unstructured (documents) |
| **Latency**        | Depends on external call | Fast (vector search)     |
| **Best for**       | Actions, live data       | Knowledge retrieval      |

## Real-World Tool Examples

- `search_web(query)` — Real-time web search
- `run_python(code)` — Code execution sandbox
- `query_database(sql)` — Structured data access
- `send_email(to, subject, body)` — Email integration
- `create_calendar_event(title, time, attendees)` — Calendar actions
- `get_stock_price(ticker)` — Financial data
- `read_file(path)` — File system access

## Key Takeaways

- Function calling = the bridge between LLMs and the real world
- The model generates the call spec, your code executes it, you feed results back
- Write clear, specific tool descriptions — they directly affect which tools get called
- Always handle errors and feed them back to the model
- Parallel tool calls improve latency for independent operations

---

*Part of the [AI Engineer Roadmap series](/tags/roadmap/) — following the structure from [roadmap.sh/ai-engineer](https://roadmap.sh/ai-engineer).*
