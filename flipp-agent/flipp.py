"""
Flipp grocery deal agent - a first Claude agent with one tool.

Usage:
    python agent.py "milk"
    python agent.py "cheapest eggs near N2L 3G1"

Note: Flipp has no official public API. This uses the unofficial endpoint
behind its web app, so it may change or break. Keep it to personal,
low-volume learning use.
"""

import json
import os
import sys
from tools import search_flipp, price_per_unit, DEFAULT_POSTAL_CODE

import anthropic


MODEL = "claude-sonnet-5-5"        # cheaper for learning; swap to claude-opus-5-5 if you like
DEFAULT_POSTAL_CODE = "N2L3G1"     # Waterloo, ON - change to your own
FLIPP_URL = "https://backflipp.wishabi.com/flipp/items/search"

client = anthropic.Anthropic()     # reads ANTHROPIC_API_KEY from the environment



# ---------- 2. Describe the tool to Claude ----------

TOOLS = [{
    "name": "search_flipp",
    "description": (
        "Search current Canadian grocery flyer deals on Flipp for an item near a postal code. "
        "Returns up to `limit` items with product name, store, price text and when the deal ends."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Item to search for, e.g. 'milk'"},
            "postal_code": {"type": "string", "description": "Canadian postal code, e.g. 'N2L 3G1'"},
            "limit": {"type": "integer", "description": "How many results to return", "default": 5},
        },
        "required": ["query"],
    },
},
{
    "name": "price_per_unit",
    "description": (
        "Calculate price per unit so products in different pack sizes can be compared. "
        "Sizes in L/ml give price per L, g/kg/lb give price per kg, and counts "
        "(e.g. '12 Count', '18pcs', '1 dozen') give price per item. "
        "Use this when the user asks for best value or cheapest per unit. Call it once per product. "
        "Only compare results that have the same unit."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "price": {"type": "number", "description": "Price to convert in price per unit, e.g. '4.99'"},
            "size": {"type": "string", "description": "Size with unit, e.g. '400ml or 400 ml'"},
        },
        "required": ["price", "size"],
    },
}]

SYSTEM = f"""<role>
You are a grocery deal finder for shoppers in Canada. You find current flyer deals using the search_flipp tool.
</role>

<defaults>
If no postal code is given, use {DEFAULT_POSTAL_CODE}.
If the question is not about groceries, answer it briefly without calling any tools.
</defaults>

<how_to_use_tools>
- For any grocery item, call search_flipp first.
- For best-value questions, then call price_per_unit once for each result that has a size, so prices are in the same unit and can be compared.
</how_to_use_tools>

<rules>
1. Only state prices and sizes that appear in tool results. If nothing is found, say so and do not estimate. Users act on these prices, so a guessed price means a wasted trip.
2. Never guess package sizes or calculate prices for hypothetical sizes. If a size is missing, say it can't be compared. A guessed size makes the comparison meaningless.
3. Only compare results that have the same unit ($/L with $/L, $/kg with $/kg).
4. Flag unclear listings: no size, no price ("see flyer"), or multi-buy prices like "2/$7". Users need to know which deals to check in the flyer.
</rules>

<output_format>
A short numbered list: product, size (if known), store, price, deal end date.
For best-value questions, start with one line naming the best value and its unit price, then the list.
</output_format>

<example>
These examples show the FORMAT only. The stores and prices are not real; never reuse them.

Question: milk
1. Brand A Milk, 4 L, Store X, $5.49, ends Oct 21
2. Brand B Milk, size not listed, Store Y, $4.99, ends Oct 21

Question: best value milk
Best value: Brand A Milk 4 L at Store X, $1.37/L.
1. Brand A Milk, 4 L, Store X, $5.49 ($1.37/L), ends Oct 21
2. Brand C Milk, 2 L, Store Z, $3.29 ($1.65/L), ends Oct 21
3. Brand B Milk, size not listed, Store Y, $4.99. Can't compare: no size given.
</example>"""

TOOL_FUNCTIONS = {"search_flipp": search_flipp, "price_per_unit": price_per_unit}
# ---------- 3. The agent loop ----------

def run_agent(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model=MODEL, max_tokens=1000, system=SYSTEM, tools=TOOLS, messages=messages, thinking={"type": "adaptive", "display": "summarized"}
        )
        messages.append({"role": "assistant", "content": response.content})
        print(f"My print {response.content}")
        # No tool requested -> Claude is done, return its text
        # --- add this ---
        for b in response.content:
            if b.type == "thinking":
                print(f"\n[{b.thinking}]\n", )
        # ----------------
        if response.stop_reason != "tool_use":
            print(f"post if {response.stop_reason}")
            return "".join(b.text for b in response.content if b.type == "text")
        
        # Run every tool Claude asked for and send the results back
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"[tool] {block.name}({block.input})")
                try:
                    output = TOOL_FUNCTIONS[block.name](**block.input)
                    tool_results.append({
                        "type": "tool_result", "tool_use_id": block.id, "content": json.dumps(output)
                    })
                except Exception as e:
                    tool_results.append({
                        "type": "tool_result", "tool_use_id": block.id,
                        "content": f"Error: {e}", "is_error": True,
                    })
        messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set.")
    item = " ".join(sys.argv[1:]) or "milk"
    print(run_agent(f"{item}"))