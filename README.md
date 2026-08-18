# llama-index-tools-livetennisapi

A [LlamaIndex](https://www.llamaindex.ai/) tool spec for the
[Live Tennis API](https://livetennisapi.com) — real-time tennis for LLM agents:
live and upcoming matches, current scores with a derived break-point flag,
player search, a player's profile (current ranking **and** current Elo), and
upcoming fixtures across ATP, WTA, Challenger, ITF and juniors.

[![PyPI](https://img.shields.io/pypi/v/llama-index-tools-livetennisapi.svg)](https://pypi.org/project/llama-index-tools-livetennisapi/)
[![CI](https://github.com/livetennisapi/llama-index-tools-livetennisapi/actions/workflows/ci.yml/badge.svg)](https://github.com/livetennisapi/llama-index-tools-livetennisapi/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://spdx.org/licenses/MIT.html)

> **Vendor-authored.** This package is published by Live Tennis API, the operator
> of the API it calls. It is not affiliated with LlamaIndex. It is a thin,
> read-only wrapper over the public REST API — judge it accordingly.

## What it is

`LiveTennisAPIToolSpec` is a `BaseToolSpec` that exposes five FREE-tier tools an
LLM agent can call:

| Tool | What it does |
|------|--------------|
| `get_live_matches` | Live or upcoming matches with players and the latest score |
| `get_match_score` | The current score of one match, plus a derived `break_point` flag |
| `search_players` | Find players by name (returns their current ranking) |
| `get_player` | One player's profile, current ranking, and current Elo |
| `get_fixtures` | Upcoming scheduled fixtures, earliest first |

## Installation

```bash
pip install llama-index-tools-livetennisapi
```

## API key

Grab a free key (no card) at <https://livetennisapi.com/subscribe/free> and
export it — the tool reads `LIVETENNIS_API_KEY` by default (and falls back to
`LIVETENNISAPI_KEY` if that is what you already have set for the other Live
Tennis API libraries):

```bash
export LIVETENNIS_API_KEY="your_key_here"
```

You can also pass it directly: `LiveTennisAPIToolSpec(api_key="...")`.

## Usage

```python
from llama_index.tools.livetennisapi import LiveTennisAPIToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

tool_spec = LiveTennisAPIToolSpec()  # reads LIVETENNIS_API_KEY

agent = FunctionAgent(
    tools=tool_spec.to_tool_list(),
    llm=OpenAI(model="gpt-4.1"),
)

print(await agent.run("Which ATP matches are live right now, and is anyone facing a break point?"))
print(await agent.run("What is Carlos Alcaraz's current ranking and Elo?"))
```

Or call a tool directly:

```python
tool_spec = LiveTennisAPIToolSpec()
matches = tool_spec.get_live_matches(tour="atp")
score = tool_spec.get_match_score(matches[0]["id"])
print(score["break_point"])  # True / False / None
```

### The `break_point` flag

`get_match_score` (and the score on each live match) derives a `break_point`
flag from the current game: it is `True` when the receiver is one point from
breaking serve (receiver at `AD`, or at `40` while the server is at
`0`/`15`/`30`), `False` when they are not, and `None` when it cannot be
determined. A tiebreak is always `False`; a null server or null points is
`None` — the flag is derived, never guessed.

## Free-tier limits (and what needs a paid tier)

The FREE tier allows **30 requests/minute and 100 requests/day**. That suits
development, testing, and periodic (~15-minute) checks — **not** continuous fast
polling. The tools clamp `limit` to 100 and reject `status="completed"` on
`get_live_matches` because paging completed results is a paid surface.

Everything this package returns is FREE-tier. The following are honestly out of
scope here and require a higher tier (see <https://livetennisapi.com> for the
current pricing):

- **BASIC** — completed-match history, point-by-point tapes, head-to-head, and
  the 1968–2022 results archive.
- **PRO** — match events, market prices, bulk history packages, and the
  rank-ordered rankings *listing* (a single player's own ranking stays FREE and
  is returned by `get_player`).
- **ULTRA** — model win-probability and in-play statistics on the live score,
  per-player as-of rankings, the as-of Elo tape, rally charting, and the
  WebSocket feed.

## Development

```bash
pip install -e ".[dev]"
pytest -q
ruff check .
```

The test suite is fully mocked (httpx `MockTransport`) and makes no network
calls.

## License

MIT — see [LICENSE](LICENSE).
