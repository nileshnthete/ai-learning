"""
Flipp agent as an MCP HOST.

Same agent loop as flipp.py, with one big difference: it does NOT import the
tools. It starts flipp_server.py, asks it which tools it has (discovery), and
calls them through the MCP protocol. This is the role Claude Desktop plays.

Usage:
    python mcp_agent.py "what's the best value strawberries this week?"
"""

import asyncio
import json
import sys
from pathlib import Path

from anthropic import AsyncAnthropic
from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp_types import TextContent

from flipp import MODEL, SYSTEM   # reuse the same model and system prompt

SERVER_SCRIPT = Path(__file__).parent / "flipp_server.py"
MAX_TURNS = 10                    # hard stop so a confused agent can't loop forever

anthropic = AsyncAnthropic()      # async version of the client, since MCP is async


async def call_mcp_tool(mcp_client, block):
    """Execute one tool_use block through the MCP server and build the tool_result."""
    print(f"[mcp] -> {block.name}({block.input})")
    try:
        result = await mcp_client.call_tool(block.name, block.input)
        text = "\n".join(b.text for b in result.content if isinstance(b, TextContent))
        is_error = bool(result.is_error)
    except Exception as e:                       # e.g. the server crashed
        text, is_error = f"Error: {e}", True
    return {"type": "tool_result", "tool_use_id": block.id, "content": text, "is_error": is_error}


async def run_agent(mcp_client, tools, user_message):
    messages = [{"role": "user", "content": user_message}]

    for turn in range(MAX_TURNS):
        response = await anthropic.messages.create(
            model=MODEL, max_tokens=2000, system=SYSTEM, tools=tools, messages=messages
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            if response.stop_reason == "max_tokens":
                print("[warning] answer was cut off by max_tokens")
            return "".join(b.text for b in response.content if b.type == "text")

        tool_results = [
            await call_mcp_tool(mcp_client, block)
            for block in response.content if block.type == "tool_use"
        ]
        messages.append({"role": "user", "content": tool_results})

    return f"Stopped after {MAX_TURNS} turns without a final answer."


async def main():
    question = " ".join(sys.argv[1:]) or "what's the best value milk this week?"

    # How to launch the server: same Python (this venv) + the server script.
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER_SCRIPT)])

    # Entering this block starts flipp_server.py and connects to it;
    # leaving it shuts the server down.
    async with Client(stdio_client(params)) as mcp_client:

        # 1. DISCOVERY: ask the server what tools it offers
        listing = await mcp_client.list_tools()
        tools = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in listing.tools
        ]

        print("Tools discovered from the MCP server:")
        for t in tools:
            print(f"  - {t['name']}")
        print("\nGenerated schema for price_per_unit (built from the type hints + docstring):")
        print(json.dumps(next(t for t in tools if t["name"] == "price_per_unit"), indent=2))
        print()

        # 2. Same agent loop as before - but tools now run on the server
        answer = await run_agent(mcp_client, tools, question)
        print("\n" + answer)


if __name__ == "__main__":
    asyncio.run(main())