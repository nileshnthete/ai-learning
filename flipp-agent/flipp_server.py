"""
Flipp MCP server - exposes the Flipp agent's tools to any MCP client
(Claude Desktop, Claude Code, other agents).

The tool logic lives in tools.py and is shared with flipp.py.
This file only wraps it in the MCP standard.

IMPORTANT: this is a stdio server - never print() anything here.
stdout carries the MCP messages; printing would corrupt them.
Use logging instead (it writes to stderr).
"""

import logging

from mcp.server import MCPServer

import tools
from typing import Annotated
from pydantic import Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = MCPServer("flipp")


@mcp.tool()
def search_flipp(query: str, postal_code: str = tools.DEFAULT_POSTAL_CODE, limit: int = 5) -> list[dict]:
    """Search current Canadian grocery flyer deals on Flipp near a postal code.

    Returns up to `limit` unique items with product name, store, price text and the
    date the deal ends. Only report prices and sizes that appear in these results;
    never estimate or guess missing ones.

    Args:
        query: Item to search for, e.g. 'milk'
        postal_code: Canadian postal code, e.g. 'N2L 3G1'. Defaults to Waterloo, ON.
        limit: How many results to return
    """
    logger.info("search_flipp query=%r postal_code=%r", query, postal_code)
    return tools.search_flipp(query, postal_code, limit)


@mcp.tool()
def price_per_unit(
    price: Annotated[float, Field(description="A single price in dollars, e.g. 4.99")],
    size: Annotated[str, Field(description="Text containing the size, e.g. 'Neilson 2% Milk 4 L'. "
                                           "The full product name can be passed as-is.")],) -> dict:
    """Calculate price per unit so products in different pack sizes can be compared.

    Sizes in L/ml give price per L, g/kg/lb give price per kg, and counts
    (e.g. '12 Count', '18pcs', '1 dozen') give price per item. Use this for
    best-value questions, once per product. Only compare results with the same unit.
    If a product has no size, don't guess one - say it can't be compared.

    """
    logger.info("price_per_unit price=%r size=%r", price, size)
    return tools.price_per_unit(price, size)


if __name__ == "__main__":
    mcp.run(transport="stdio")