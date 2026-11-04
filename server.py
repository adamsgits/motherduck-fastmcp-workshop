"""Your MCP server for San Francisco 311 data.

FastMCP Gateway runs this file. Its entrypoint setting is `server.py:mcp`.
"""

import re
from collections import Counter, defaultdict
from statistics import median
from typing import Annotated, Literal

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.utilities.logging import get_logger
from pydantic import Field

from sf311.data import BY_ID, LOCAL_NAMES, NEIGHBORHOODS, REQUESTS, SERVICES, Request, closest
from sf311.kit import caller_email

mcp = FastMCP("sf-311")
log = get_logger("sf311")

MAX_ROWS = 50
MIN_CLOSED_FOR_MEDIAN = 10

GROUPS = {
    "neighborhood": lambda r: r.neighborhood,
    "service": lambda r: r.service,
    "agency": lambda r: r.agency,
    "source": lambda r: r.source,
    "weekday": lambda r: r.opened.strftime("%A"),
    "status": lambda r: r.status,
}

CountGroup = Literal["neighborhood", "service", "agency", "source", "weekday", "status"]
MedianGroup = Literal["neighborhood", "agency"]
ServiceArg = Annotated[str | None, Field(description="Exact service name from dataset_info, e.g. 'Graffiti Public'.")]
NeighborhoodArg = Annotated[
    str | None, Field(description="Exact official neighborhood name from dataset_info, e.g. 'Mission'.")
]


HOUSE_NUMBER = re.compile(r"^(\d+)\s+([^,]+)")


def mask_address(address: str) -> str:
    """Reduce an address to its block: "994 HARRISON ST, ..." -> "900 block of HARRISON ST".

    Addresses without a leading house number (intersections, etc.) are returned unchanged.
    """
    match = HOUSE_NUMBER.match(address)
    if not match:
        return address
    block = int(match.group(1)) // 100 * 100
    return f"{block} block of {match.group(2).strip()}"


def _public(request: Request) -> dict:
    """A request as a dict, safe to return to the model (address masked to the block)."""
    row = request.as_dict()
    row["address"] = mask_address(row["address"])
    return row


def _check_name(value: str | None, choices: list[str], kind: str) -> None:
    if value is None or value in choices:
        return
    suggestions = closest(value, choices)
    hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    raise ToolError(f"Unknown {kind} '{value}'.{hint} Call dataset_info for the full list of {kind} names.")


def _matching(service: str | None = None, neighborhood: str | None = None, status: str | None = None) -> list[Request]:
    _check_name(service, SERVICES, "service")
    _check_name(neighborhood, NEIGHBORHOODS, "neighborhood")
    return [
        r
        for r in REQUESTS
        if (service is None or r.service == service)
        and (neighborhood is None or r.neighborhood == neighborhood)
        and (status is None or r.status == status)
    ]


@mcp.tool(annotations={"readOnlyHint": True})
def dataset_info() -> dict:
    """Describe the 311 dataset this server answers questions about.

    Call this first to learn the date range and the exact service and
    neighborhood names that other tools accept.
    """
    return {
        "description": "San Francisco 311 service requests filed in one week.",
        "first_opened": min(request.opened for request in REQUESTS).isoformat(),
        "last_opened": max(request.opened for request in REQUESTS).isoformat(),
        "requests": len(REQUESTS),
        "services": SERVICES,
        "neighborhoods": NEIGHBORHOODS,
        "local_neighborhood_names": len(LOCAL_NAMES),
    }


@mcp.tool(annotations={"readOnlyHint": True})
def count_requests(
    service: ServiceArg = None,
    neighborhood: NeighborhoodArg = None,
    group_by: CountGroup | None = None,
) -> int | dict[str, int]:
    """Count this week's 311 requests, optionally broken down by one field.

    Without group_by, returns a single number. With group_by, returns a dict
    of group -> count, largest first, at most 50 groups (blank values skipped).
    Use this for any "how many" or "which X has the most" question, such as
    "which agency handled the most Graffiti Public requests" (service +
    group_by="agency"). Do not page through search_requests to count.
    For how long requests take to close, use median_close_hours instead.
    """
    rows = _matching(service, neighborhood)
    if group_by is None:
        return len(rows)
    key = GROUPS[group_by]
    counts = Counter(key(r) for r in rows if key(r))
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:MAX_ROWS])


@mcp.tool(annotations={"readOnlyHint": True})
def median_close_hours(
    service: ServiceArg = None,
    group_by: MedianGroup | None = None,
) -> float | None | dict[str, float]:
    """Median hours from opened to closed for closed requests, rounded to 0.1.

    Without group_by, returns one number (or null if nothing has closed).
    With group_by "neighborhood" or "agency", returns a dict of group -> median
    hours, fastest first, leaving out groups with fewer than 10 closed requests.
    Use this for "how fast" / "which is slowest" questions. Open requests are
    ignored; use count_requests with group_by="status" to see how many are open.
    """
    rows = [r for r in _matching(service) if r.hours_to_close is not None]
    if group_by is None:
        return round(median(r.hours_to_close for r in rows), 1) if rows else None
    key = GROUPS[group_by]
    hours: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if key(r):
            hours[key(r)].append(r.hours_to_close)
    medians = {g: round(median(v), 1) for g, v in hours.items() if len(v) >= MIN_CLOSED_FOR_MEDIAN}
    return dict(sorted(medians.items(), key=lambda item: (item[1], item[0]))[:MAX_ROWS])


@mcp.tool(annotations={"readOnlyHint": True})
def search_requests(
    service: ServiceArg = None,
    neighborhood: NeighborhoodArg = None,
    status: Literal["Open", "Closed"] | None = None,
    limit: Annotated[int, Field(ge=1, le=MAX_ROWS, description="Rows to return, 1 to 50.")] = 20,
) -> list[dict]:
    """List individual 311 requests matching the filters, newest first.

    Returns up to `limit` (max 50) requests, each with request_id, opened,
    closed, status, service, subtype, agency, neighborhood, local_name,
    address (to the block only, e.g. '900 block of HARRISON ST') and
    source. Use this to look at examples or details. It is a
    sample, not a full result: for totals or rankings use count_requests,
    and for close times use median_close_hours. To fetch one known request,
    use get_request.
    """
    rows = _matching(service, neighborhood, status)
    rows.sort(key=lambda r: r.opened, reverse=True)
    return [_public(r) for r in rows[:limit]]


@mcp.tool(annotations={"readOnlyHint": True})
def get_request(request_id: str) -> dict:
    """Return one 311 request by its request_id, with all its fields.

    The address is given only to the block (e.g. '900 block of HARRISON ST').
    Use this when you already have an ID (for example from search_requests).
    To find requests by service or neighborhood, use search_requests.
    """
    log.info("get_request request_id=%s caller=%s", request_id, caller_email() or "local")
    request = BY_ID.get(request_id.strip())
    if request is None:
        raise ToolError(f"No request with id '{request_id}'. Use search_requests to find valid request IDs.")
    return _public(request)


if __name__ == "__main__":
    mcp.run()
