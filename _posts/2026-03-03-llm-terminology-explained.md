---
title: "LLM Terminology Explained: Embeddings, RAG, Fine-tuning, Inference, and More"
date: 2026-03-03 08:00:00 +0530
categories: [AI, LLMs]
tags: [llm, embeddings, rag, fine-tuning, inference, terminology, roadmap]
---

The AI space is full of jargon that gets thrown around loosely. This post cuts through the noise and gives clear, practical definitions for the core terminology every AI Engineer needs to know.

## AI vs AGI

**AI (Artificial Intelligence)** — Systems designed to perform specific tasks intelligently: image recognition, language translation, text generation. All current commercial AI is narrow AI.

**AGI (Artificial General Intelligence)** — A hypothetical system with human-level (or beyond) intelligence across all domains. Does not exist yet. Every time someone says "we're close to AGI," treat it with healthy skepticism.

For AI Engineers, the practical distinction is simple: you're working with narrow AI tools that are powerful but brittle outside their training distribution.

## Large Language Model (LLM)

A **Large Language Model** is a neural network trained on massive amounts of text data to predict the next token in a sequence. Through this simple training objective applied at enormous scale, LLMs develop surprisingly broad capabilities.

Key characteristics:
- Trained on internet-scale text (trillions of tokens)
- Billions to hundreds of billions of parameters
- Accessed via API or run locally
- Examples: GPT-4o, Claude 3.5, Gemini 1.5, Llama 3, Mistral

## Training vs Inference

**Training** — The computationally expensive process of updating a model's weights using gradient descent on a large dataset. This is what ML Engineers do. It requires thousands of GPUs running for weeks or months.

**Inference** — Running a trained model on new inputs to generate outputs. This is what AI Engineers do. Much cheaper than training, can run on a single GPU or even a laptop for smaller models.

> When you call the OpenAI API, you're doing inference. You're not training anything.

## Embeddings

An **embedding** is a dense vector (array of floating-point numbers) that represents the semantic meaning of text (or images, audio, etc.).

```python
from openai import OpenAI
client = OpenAI()

response = client.embeddings.create(
    model="text-embedding-3-small",
    input="The quick brown fox"
)
vector = response.data[0].embedding
# [0.0023, -0.0045, 0.0189, ...]  — 1536 dimensions
```

Why are embeddings useful? Texts with similar meaning produce vectors that are close together in vector space. This enables semantic search — finding relevant content by meaning rather than keyword matching.

## Vector Databases

A **vector database** stores embeddings and enables fast similarity search (finding the K nearest vectors to a query vector). This is the backbone of RAG systems.

Popular options:
- **Pinecone** — Managed, easy to start with
- **Qdrant** — Open source, fast, self-hostable
- **Weaviate** — Combines vector search with structured filtering
- **Chroma** — Lightweight, great for prototyping
- **pgvector** — Postgres extension for vector search

## RAG (Retrieval-Augmented Generation)

**RAG** is a technique for grounding LLM responses in external, up-to-date knowledge by:

1. Converting documents into embeddings and storing them in a vector DB
2. At query time, embedding the user's question
3. Retrieving the most similar document chunks
4. Injecting those chunks into the LLM's context before generating a response

RAG solves two major LLM problems:
- **Hallucination** — The model makes up facts it doesn't know
- **Knowledge cutoff** — The model's training data has a cutoff date

## AI Agents

An **AI Agent** is an LLM that can perceive its environment, reason about it, and take actions — usually through tool use — in a loop until a goal is achieved.

Unlike a simple chatbot that responds to a single message, an agent:
- Plans multi-step approaches to problems
- Calls external tools (search, code execution, APIs)
- Observes results and adjusts its approach
- Iterates until the task is complete

## Fine-tuning

**Fine-tuning** is the process of taking a pre-trained model and continuing to train it on a smaller, domain-specific dataset to adapt its behavior.

Use fine-tuning when you need to:
- Teach the model a specific output format it struggles with
- Adapt the model to specialized vocabulary or domain knowledge
- Reduce the need for lengthy system prompts
- Improve consistency on a narrow, well-defined task

Fine-tuning is **not** the first thing to reach for. Most problems can be solved with better prompts, RAG, or few-shot examples — which are cheaper and faster to iterate on.

## Prompt Engineering

**Prompt engineering** is the practice of designing and optimizing the text inputs given to an LLM to produce better outputs. It's part art, part science, and heavily empirical.

Good prompt engineering includes:
- Clear task description
- Relevant examples (few-shot)
- Output format specification
- Constraints and guardrails

## Context Engineering

**Context engineering** is the broader discipline of managing *what information* goes into the model's context window, and when. It goes beyond individual prompt design to encompass:

- What documents to retrieve and inject
- How to summarize long conversation history
- When to reset context vs. continue
- How to structure multi-turn conversations

## Inference Parameters Recap

| Term               | Meaning                                       |
| ------------------ | --------------------------------------------- |
| **Temperature**    | Randomness of output (0 = deterministic)      |
| **Top-P**          | Nucleus sampling — probability mass cutoff    |
| **Max tokens**     | Maximum length of the generated response      |
| **Stop sequences** | Strings that halt generation when encountered |
| **System prompt**  | Instructions that frame the model's behavior  |

## Summary

Understanding this terminology is the foundation for everything that follows in the AI Engineer roadmap. When someone says "run inference on a fine-tuned model with RAG using embeddings from a vector DB" — you now know exactly what all of that means.

---

*Part of the [AI Engineer Roadmap series](/tags/roadmap/) — following the structure from [roadmap.sh/ai-engineer](https://roadmap.sh/ai-engineer).*
