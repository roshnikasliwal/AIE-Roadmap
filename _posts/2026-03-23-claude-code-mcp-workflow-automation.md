---
title: "Claude Code + MCP: Automating Your Development Workflow"
date: 2026-03-23 09:00:00 +0530
categories: [AI, MCP]
tags: [mcp, claude-code, developer-tools, python, automation, mcp-series]
mermaid: true
---

Claude Code becomes significantly more powerful when you connect it to MCP servers. Out of the box it can read files, run commands, and search the codebase. Add MCP servers and it can also query your database, check your monitoring dashboards, create GitHub issues, search internal documentation, and interact with any system that has an MCP interface.

This post covers how to configure MCP servers for Claude Code and how to build custom servers that automate the repetitive parts of a development workflow.

## Configuring MCP Servers in Claude Code

Claude Code reads MCP server configuration from `.claude/settings.json` in your project or from `~/.claude/settings.json` globally. The format is the same as Claude Desktop:

```json
{
  "mcpServers": {
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "POSTGRES_CONNECTION_STRING": "postgresql://user:pass@localhost:5432/mydb"
      }
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
      }
    },
    "project-tools": {
      "command": "python",
      "args": [".claude/mcp/project_server.py"]
    }
  }
}
```

Three servers configured:
1. **postgres** — official PostgreSQL MCP server, allows Claude Code to query your DB schema and run read-only queries
2. **github** — official GitHub MCP server, for reading issues, PRs, and code
3. **project-tools** — a custom server in `.claude/mcp/` for project-specific tooling

Use `${ENV_VAR}` syntax to reference environment variables without embedding secrets in the config file.

## Community MCP Servers Worth Configuring

Before building a custom server, check if one already exists:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/dir"]
    },
    "fetch": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-fetch"]
    },
    "sqlite": {
      "command": "uvx",
      "args": ["mcp-server-sqlite", "--db-path", "data/local.db"]
    },
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "${BRAVE_API_KEY}"
      }
    },
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    }
  }
}
```

With these configured, Claude Code can browse allowed directories, fetch URLs, query your SQLite database, search the web, and maintain a persistent knowledge graph across sessions — all without writing any custom code.

## Building a Project-Specific MCP Server

For tooling specific to your project, a custom server is the right approach. Here's a complete server that automates common development tasks:

```python
# .claude/mcp/project_server.py
import subprocess
import json
import os
import re
from pathlib import Path
from datetime import datetime
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, Resource
import asyncio

PROJECT_ROOT = Path(__file__).parent.parent.parent
app = Server("project-tools")

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="run_tests",
            description="Run the project test suite (or a specific test file/pattern)",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Test file or pattern to run (e.g., 'tests/test_auth.py', 'test_login'). Runs all tests if omitted."
                    },
                    "verbose": {
                        "type": "boolean",
                        "default": False
                    }
                }
            }
        ),
        Tool(
            name="check_code_quality",
            description="Run linting and type checking on a file or the entire project",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File or directory to check. Defaults to entire project."
                    },
                    "checks": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["ruff", "mypy", "black"]},
                        "default": ["ruff", "mypy"]
                    }
                }
            }
        ),
        Tool(
            name="get_git_context",
            description="Get current branch, recent commits, and changed files",
            inputSchema={
                "type": "object",
                "properties": {
                    "num_commits": {"type": "integer", "default": 10}
                }
            }
        ),
        Tool(
            name="search_codebase",
            description="Search for a pattern across all Python files in the project",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for"
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "Glob pattern for files to search (default: **/*.py)",
                        "default": "**/*.py"
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Lines of context around each match",
                        "default": 2
                    }
                },
                "required": ["pattern"]
            }
        ),
        Tool(
            name="create_github_issue",
            description="Create a GitHub issue for a bug or feature request",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "e.g., ['bug', 'priority:high']"
                    }
                },
                "required": ["title", "body"]
            }
        ),
        Tool(
            name="get_dependency_info",
            description="List installed packages and check for outdated dependencies",
            inputSchema={
                "type": "object",
                "properties": {
                    "check_outdated": {"type": "boolean", "default": False}
                }
            }
        )
    ]

def _run_command(cmd: list[str], cwd: Path = None) -> dict:
    """Run a subprocess and return structured output."""
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(cwd or PROJECT_ROOT),
        timeout=120
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0
    }

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "run_tests":
            pattern = arguments.get("pattern")
            verbose = arguments.get("verbose", False)
            
            cmd = ["python", "-m", "pytest"]
            if verbose:
                cmd.append("-v")
            if pattern:
                cmd.append(pattern)
            cmd.extend(["--tb=short", "--no-header", "-q"])
            
            result = _run_command(cmd)
            
            output = result["stdout"] + (result["stderr"] if result["stderr"] else "")
            status = "PASSED" if result["success"] else "FAILED"
            
            return [TextContent(
                type="text",
                text=f"Test run: {status}\n\n{output}"
            )]
        
        elif name == "check_code_quality":
            path = arguments.get("path", ".")
            checks = arguments.get("checks", ["ruff", "mypy"])
            
            results = {}
            
            if "ruff" in checks:
                r = _run_command(["ruff", "check", path, "--output-format=text"])
                results["ruff"] = {
                    "passed": r["success"],
                    "output": r["stdout"] or r["stderr"] or "No issues found"
                }
            
            if "mypy" in checks:
                r = _run_command(["mypy", path, "--ignore-missing-imports"])
                results["mypy"] = {
                    "passed": r["success"],
                    "output": r["stdout"] or r["stderr"] or "No type errors found"
                }
            
            if "black" in checks:
                r = _run_command(["black", "--check", path])
                results["black"] = {
                    "passed": r["success"],
                    "output": r["stdout"] or r["stderr"] or "Formatting OK"
                }
            
            all_passed = all(v["passed"] for v in results.values())
            
            output_lines = [f"Code quality check: {'PASSED' if all_passed else 'FAILED'}"]
            for tool_name, result in results.items():
                status = "✓" if result["passed"] else "✗"
                output_lines.append(f"\n{status} {tool_name}:\n{result['output']}")
            
            return [TextContent(type="text", text="\n".join(output_lines))]
        
        elif name == "get_git_context":
            num_commits = arguments.get("num_commits", 10)
            
            branch_result = _run_command(["git", "branch", "--show-current"])
            branch = branch_result["stdout"].strip()
            
            log_result = _run_command([
                "git", "log", f"-{num_commits}",
                "--oneline", "--format=%h %s (%an, %ar)"
            ])
            
            diff_result = _run_command(["git", "diff", "--stat", "HEAD"])
            staged_result = _run_command(["git", "diff", "--stat", "--cached"])
            status_result = _run_command(["git", "status", "--short"])
            
            output = f"""Branch: {branch}

Recent commits:
{log_result['stdout']}

Unstaged changes:
{diff_result['stdout'] or '(none)'}

Staged changes:
{staged_result['stdout'] or '(none)'}

Working tree status:
{status_result['stdout'] or '(clean)'}"""
            
            return [TextContent(type="text", text=output)]
        
        elif name == "search_codebase":
            pattern = arguments["pattern"]
            file_pattern = arguments.get("file_pattern", "**/*.py")
            context_lines = arguments.get("context_lines", 2)
            
            matches = []
            for filepath in PROJECT_ROOT.glob(file_pattern):
                if ".git" in str(filepath) or "__pycache__" in str(filepath):
                    continue
                
                try:
                    content = filepath.read_text()
                    lines = content.splitlines()
                    
                    for i, line in enumerate(lines):
                        if re.search(pattern, line, re.IGNORECASE):
                            start = max(0, i - context_lines)
                            end = min(len(lines), i + context_lines + 1)
                            context = lines[start:end]
                            
                            matches.append({
                                "file": str(filepath.relative_to(PROJECT_ROOT)),
                                "line": i + 1,
                                "match": line.strip(),
                                "context": "\n".join(context)
                            })
                except (UnicodeDecodeError, PermissionError):
                    continue
            
            if not matches:
                return [TextContent(type="text", text=f"No matches found for pattern: {pattern}")]
            
            output_lines = [f"Found {len(matches)} match(es) for '{pattern}':\n"]
            for m in matches[:50]:
                output_lines.append(f"── {m['file']}:{m['line']}")
                output_lines.append(m["context"])
                output_lines.append("")
            
            return [TextContent(type="text", text="\n".join(output_lines))]
        
        elif name == "create_github_issue":
            title = arguments["title"]
            body = arguments["body"]
            labels = arguments.get("labels", [])
            
            cmd = ["gh", "issue", "create", "--title", title, "--body", body]
            for label in labels:
                cmd.extend(["--label", label])
            
            result = _run_command(cmd)
            
            if result["success"]:
                return [TextContent(
                    type="text",
                    text=f"Issue created: {result['stdout'].strip()}"
                )]
            else:
                return [TextContent(
                    type="text",
                    text=f"Failed to create issue: {result['stderr']}"
                )]
        
        elif name == "get_dependency_info":
            check_outdated = arguments.get("check_outdated", False)
            
            list_result = _run_command(["pip", "list", "--format=json"])
            packages = json.loads(list_result["stdout"])
            
            output = f"Installed packages: {len(packages)}\n\n"
            output += "\n".join(f"{p['name']}=={p['version']}" for p in sorted(packages, key=lambda x: x['name'].lower()))
            
            if check_outdated:
                outdated_result = _run_command(["pip", "list", "--outdated", "--format=json"])
                if outdated_result["stdout"]:
                    outdated = json.loads(outdated_result["stdout"])
                    if outdated:
                        output += f"\n\nOutdated packages ({len(outdated)}):\n"
                        output += "\n".join(
                            f"{p['name']} {p['version']} → {p['latest_version']}"
                            for p in outdated
                        )
            
            return [TextContent(type="text", text=output)]
        
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    except subprocess.TimeoutExpired:
        return [TextContent(type="text", text="Error: Command timed out after 120 seconds")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

# ─── Resources: Project documentation ─────────────────────────────────────────

@app.list_resources()
async def list_resources() -> list[Resource]:
    return [
        Resource(
            uri="project://readme",
            name="Project README",
            description="Project overview, setup instructions, and architecture notes",
            mimeType="text/markdown"
        ),
        Resource(
            uri="project://api-routes",
            name="API Routes",
            description="All registered API endpoints",
            mimeType="text/plain"
        ),
    ]

@app.read_resource()
async def read_resource(uri: str):
    from mcp.types import TextResourceContents
    
    if uri == "project://readme":
        readme_path = PROJECT_ROOT / "README.md"
        if readme_path.exists():
            return [TextResourceContents(
                uri=uri,
                mimeType="text/markdown",
                text=readme_path.read_text()
            )]
        return [TextResourceContents(uri=uri, mimeType="text/plain", text="README.md not found")]
    
    elif uri == "project://api-routes":
        result = _run_command([
            "grep", "-r", "@app.route\|@router\.", "src/", "--include=*.py", "-n"
        ])
        return [TextResourceContents(
            uri=uri,
            mimeType="text/plain",
            text=result["stdout"] or "No routes found"
        )]
    
    raise ValueError(f"Unknown resource: {uri}")

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
```

## The Developer Workflow This Enables

With this server connected, Claude Code can handle complex multi-step tasks autonomously:

**"Fix the failing tests and make sure linting passes"**
1. `run_tests` → sees which tests fail and what the errors are
2. Reads the relevant source files
3. Edits the code to fix the failures
4. `run_tests` again → confirms tests pass
5. `check_code_quality` → fixes any linting issues introduced

**"Find everywhere we use the deprecated `get_user_by_email` function and migrate to `fetch_user`"**
1. `search_codebase(pattern="get_user_by_email")` → finds all call sites
2. Reads and edits each file
3. `run_tests` → confirms no regressions

**"Create a GitHub issue for the bug I just found"**
1. `get_git_context` → gets current branch and recent changes for context
2. `create_github_issue(title=..., body=..., labels=["bug"])` → creates the issue with full context

## Per-Project vs Global Configuration

Put project-specific servers in `.claude/settings.json` (committed to the repo) and shared infrastructure servers in `~/.claude/settings.json` (your personal global config):

```json
// ~/.claude/settings.json — global, applies to all projects
{
  "mcpServers": {
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {"BRAVE_API_KEY": "${BRAVE_API_KEY}"}
    },
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    }
  }
}
```

```json
// .claude/settings.json — project-specific, committed to git
{
  "mcpServers": {
    "project-tools": {
      "command": "python",
      "args": [".claude/mcp/project_server.py"]
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "POSTGRES_CONNECTION_STRING": "${DATABASE_URL}"
      }
    }
  }
}
```

Team members clone the repo and immediately get access to the project's MCP servers — no manual setup beyond setting the environment variables.

## Security Considerations

**Only expose read operations by default.** The `run_tests`, `search_codebase`, `get_git_context`, and `check_code_quality` tools above are all read-only or confined to the project directory. Write operations (`create_github_issue`) are fine for tools that interact with external systems where the change is visible and reversible.

**Be careful with `subprocess`** — don't accept arbitrary shell commands as tool inputs. The `run_tests` tool above accepts a pytest pattern, not an arbitrary command string. Validate and restrict inputs before passing them to `subprocess.run`.

**Scope filesystem access** — if you're exposing file system tools, restrict them to specific directories. The built-in MCP filesystem server requires you to specify allowed directories explicitly.

## Key Takeaways

1. **`.claude/settings.json` wires MCP servers to a project** — commit it to give the whole team access
2. **Use community servers before building custom ones** — postgres, github, sqlite, brave-search, memory are ready to use
3. **Custom servers live in `.claude/mcp/`** — project-specific tooling as Python files alongside the codebase
4. **Return errors as text, never raise** — the agent needs to see the error to recover from it
5. **Read-only by default** — expose write tools only where writes are reversible and visible

---

*Part of the [MCP Deep Dive series]({{ site.baseurl }}/tags/mcp-series/) — building production-grade integrations with Model Context Protocol.*


## ## Claude Code MCP Workflow

```mermaid
flowchart LR
    A[Claude Code CLI] --> B[Built-in MCP
Client]
    B --> C[File System
MCP Server]
    B --> D[GitHub
MCP Server]
    B --> E[Custom
MCP Servers]
    C --> F[Read/write files
list directories]
    D --> G[PR reviews
issue management
CI status]
    E --> H[Databases
Internal APIs
Slack/Jira]
    F & G & H --> I[Automated
Dev Workflow]
```
