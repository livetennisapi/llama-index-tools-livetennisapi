"""Live Tennis API tool spec for LlamaIndex.

Vendor-authored. This package is published by Live Tennis API
(https://livetennisapi.com), the operator of the API it calls. It wraps the
FREE tier of the public REST API as a set of read-only tools an LLM agent can
call through LlamaIndex.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx
from llama_index.core.tools.tool_spec.base import BaseToolSpec

DEFAULT_BASE_URL = "https://api.livetennisapi.com/api/public/v1"

# The env vars checked, in order, when no key is passed to the constructor.
# LIVETENNIS_API_KEY is the documented name for this package; LIVETENNISAPI_KEY
# is accepted as a fallback so a key already exported for the other Live Tennis
# API libraries keeps working.
_ENV_KEYS = ("LIVETENNIS_API_KEY", "LIVETENNISAPI_KEY")


class LiveTennisAPIError(RuntimeError):
    """Raised when the Live Tennis API returns a non-2xx response.

    Attributes:
        status_code: The HTTP status code, or ``None`` for a transport error.
        code: The stable machine-readable ``error`` code from the API body,
            when one was present (e.g. ``upgrade_required``, ``rate_limited``).
    """

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        code: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class LiveTennisAPIToolSpec(BaseToolSpec):
    """Live Tennis API tool spec — FREE-tier tennis data for LlamaIndex agents.

    Vendor-authored by Live Tennis API (https://livetennisapi.com). Exposes the
    self-serve FREE tier: live and upcoming matches, the current score of a
    match (with a derived break-point flag), player search, a single player's
    profile including their current ranking and Elo, and upcoming fixtures.

    The FREE tier allows 30 requests/minute and 100 requests/day, which suits
    development, testing, and periodic (~15-minute) checks rather than
    continuous fast polling. Historical results, point-by-point tapes, market
    prices, and model win-probability live on the paid tiers (BASIC, PRO,
    ULTRA) and are intentionally not implemented here; the per-method docstrings
    note where a richer answer needs one of those tiers.

    Get a free key (no card) at https://livetennisapi.com/subscribe/free and
    either pass it as ``api_key`` or export ``LIVETENNIS_API_KEY``.
    """

    spec_functions = [
        "get_live_matches",
        "get_match_score",
        "search_players",
        "get_player",
        "get_fixtures",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 15.0,
        default_tour: Optional[str] = None,
        client: Optional[httpx.Client] = None,
    ) -> None:
        """Initialize the tool spec.

        Args:
            api_key: A Live Tennis API key. If omitted, the ``LIVETENNIS_API_KEY``
                (or ``LIVETENNISAPI_KEY``) environment variable is read instead.
            base_url: API base URL. Defaults to the public v1 base.
            timeout: Per-request timeout in seconds.
            default_tour: Optional default tour filter (``atp``, ``wta``,
                ``challenger``, ``itf``, ``juniors``) applied to the listing
                tools when they are called without an explicit ``tour``.
            client: An optional pre-built ``httpx.Client`` (used mainly for
                testing with a mock transport). When supplied it is used as-is;
                the API key header is still attached per request.
        """
        key = api_key or _key_from_env()
        if not key:
            raise ValueError(
                "A Live Tennis API key is required. Pass api_key=... or set the "
                "LIVETENNIS_API_KEY environment variable. Get a free key at "
                "https://livetennisapi.com/subscribe/free"
            )
        self.api_key = key
        self._base_url = base_url.rstrip("/")
        self.default_tour = default_tour
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(timeout=timeout)

    # -- HTTP plumbing ----------------------------------------------------

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """GET ``path`` (relative to the base URL) and return parsed JSON."""
        url = f"{self._base_url}/{path.lstrip('/')}"
        clean = {k: v for k, v in (params or {}).items() if v is not None}
        try:
            resp = self._client.get(
                url, params=clean, headers={"X-API-Key": self.api_key}
            )
        except httpx.HTTPError as exc:  # transport / timeout / connection
            raise LiveTennisAPIError(f"request to {path} failed: {exc}") from exc

        if resp.status_code == 200:
            try:
                return resp.json()
            except ValueError as exc:
                raise LiveTennisAPIError(
                    f"could not decode JSON from {path}: {exc}", resp.status_code
                ) from exc

        self._raise_for_status(resp, path)
        return None  # pragma: no cover - _raise_for_status always raises

    @staticmethod
    def _raise_for_status(resp: httpx.Response, path: str) -> None:
        code: Optional[str] = None
        detail: Optional[str] = None
        try:
            body = resp.json()
            if isinstance(body, dict):
                code = body.get("error")
                detail = body.get("detail")
        except ValueError:
            pass

        status = resp.status_code
        base = f"{status} from {path}"
        if code:
            base += f" ({code})"
        if detail:
            base += f": {detail}"

        if status == 401:
            base = (
                f"{base}. The API key is missing or invalid — check "
                "LIVETENNIS_API_KEY or the api_key argument."
            )
        elif status == 403:
            base = (
                f"{base}. This data requires a higher tier than FREE. See the "
                "tiers at https://livetennisapi.com."
            )
        elif status == 404:
            base = f"{base}. No such resource."
        elif status == 429:
            base = (
                f"{base}. Rate limit hit — the FREE tier allows 30 req/min and "
                "100 req/day. Slow down or upgrade."
            )
        raise LiveTennisAPIError(base, status_code=status, code=code)

    def __del__(self) -> None:  # best-effort cleanup
        client = getattr(self, "_client", None)
        if client is not None and getattr(self, "_owns_client", False):
            try:
                client.close()
            except Exception:  # noqa: BLE001 - never raise from __del__
                pass

    # -- derivations ------------------------------------------------------

    @staticmethod
    def _derive_break_point(score: Optional[Dict[str, Any]]) -> Optional[bool]:
        """Return whether the receiver currently holds a break point.

        Break-point rule: the receiver is one point from taking the game —
        receiver at ``AD``, or receiver at ``40`` while the server is at
        ``0``/``15``/``30``. Never true inside a tiebreak. Returns ``None``
        (unknown) when the server or points are null, because the state cannot
        be derived — never guessed.
        """
        if not score:
            return None
        if score.get("is_tiebreak"):
            return False
        server = score.get("server")
        points = score.get("points")
        if server not in (1, 2) or not points or len(points) < 2:
            return None
        receiver = 2 if server == 1 else 1
        receiver_point = points[receiver - 1]
        server_point = points[server - 1]
        if receiver_point is None or server_point is None:
            return None
        if receiver_point == "AD":
            return True
        if receiver_point == "40" and server_point in ("0", "15", "30"):
            return True
        return False

    @classmethod
    def _shape_score(cls, score: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not score:
            return None
        return {
            "sets": score.get("sets"),
            "games": score.get("games"),
            "points": score.get("points"),
            "server": score.get("server"),
            "is_tiebreak": bool(score.get("is_tiebreak", False)),
            "break_point": cls._derive_break_point(score),
            "timestamp": score.get("timestamp"),
        }

    @staticmethod
    def _shape_player(player: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": player.get("id"),
            "name": player.get("name"),
            "country": player.get("country"),
            "tour": player.get("tour"),
            "ranking": player.get("ranking"),
            "ranking_points": player.get("ranking_points"),
            "ranking_movement": player.get("ranking_movement"),
            "hand": player.get("hand"),
        }

    @classmethod
    def _shape_match(cls, match: Dict[str, Any]) -> Dict[str, Any]:
        players = match.get("players") or {}
        return {
            "id": match.get("id"),
            "tournament": match.get("tournament"),
            "tour": match.get("tour"),
            "round": match.get("round"),
            "surface": match.get("surface"),
            "status": match.get("status"),
            "scheduled_time": match.get("scheduled_time"),
            "player1": cls._shape_player(players["p1"]) if players.get("p1") else None,
            "player2": cls._shape_player(players["p2"]) if players.get("p2") else None,
            "score": cls._shape_score(match.get("score")),
            "winner": match.get("winner"),
        }

    @staticmethod
    def _shape_fixture(fixture: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": fixture.get("id"),
            "event_date": fixture.get("event_date"),
            "start_time": fixture.get("start_time"),
            "tournament": fixture.get("tournament"),
            "tour": fixture.get("tour"),
            "round": fixture.get("round"),
            "surface": fixture.get("surface"),
            "player1_id": fixture.get("player1_id"),
            "player1_name": fixture.get("player1_name"),
            "player2_id": fixture.get("player2_id"),
            "player2_name": fixture.get("player2_name"),
            "status": fixture.get("status"),
        }

    # -- tools ------------------------------------------------------------

    def get_live_matches(
        self,
        status: str = "live",
        tour: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """List currently live or upcoming tennis matches (FREE tier).

        Returns the current-state picture: each match with its players and the
        latest score. For live matches the score carries a derived
        ``break_point`` flag (see ``get_match_score``).

        Args:
            status (str): ``live`` (default) or ``upcoming``. Only these two are
                available on the FREE tier. ``completed`` is rejected here
                because paging completed results is the paid History surface
                (BASIC and up).
            tour (Optional[str]): Optional tour filter — one of ``atp``, ``wta``,
                ``challenger``, ``itf``, ``juniors``. Defaults to the spec's
                ``default_tour`` if one was set.
            limit (int): Maximum number of matches to return (1-100).

        Returns:
            A list of match objects (id, tournament, tour, round, surface,
            status, both players with their current ranking, and the score).
        """
        if status not in ("live", "upcoming"):
            raise ValueError(
                "status must be 'live' or 'upcoming' on the FREE tier. Completed "
                "results are part of the paid History product (BASIC and up)."
            )
        params = {
            "status": status,
            "tour": tour or self.default_tour,
            "limit": _clamp(limit, 1, 100),
        }
        body = self._get("matches", params)
        data = body.get("data", []) if isinstance(body, dict) else []
        return [self._shape_match(m) for m in data]

    def get_match_score(self, match_id: int) -> Dict[str, Any]:
        """Get the current score of one match, with a derived break-point flag (FREE tier).

        This is a point-in-time snapshot — the single current state, overwritten
        on every score commit. It carries no history and no accumulated
        statistics.

        The returned ``break_point`` is derived from the score: ``True`` when the
        receiver is one point from breaking serve (receiver at ``AD``, or at
        ``40`` while the server is at ``0``/``15``/``30``), ``False`` when they
        are not, and ``None`` when it cannot be determined (a tiebreak returns
        ``False``; a null server or null points returns ``None``). In-play
        statistics and live model win-probability are ULTRA-tier and are not
        included on this object.

        Args:
            match_id (int): The match id (from ``get_live_matches``).

        Returns:
            The current score: sets, per-set games, in-game points, the serving
            player (1 or 2, or null), ``is_tiebreak``, the derived
            ``break_point``, and the score ``timestamp``.
        """
        score = self._get(f"matches/{int(match_id)}/score")
        shaped = self._shape_score(score if isinstance(score, dict) else None)
        return shaped if shaped is not None else {}

    def search_players(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search players by name (FREE tier).

        Args:
            query (str): A full or partial player name, e.g. ``"alcaraz"``.
            limit (int): Maximum number of players to return (1-100).

        Returns:
            A list of players (ranked first) with id, name, country, tour, and
            current ranking. Use ``get_player`` with an id for the full profile
            including Elo. The listed rows do not carry the ``stats`` object.
        """
        params = {"search": query, "limit": _clamp(limit, 1, 100)}
        body = self._get("players", params)
        data = body.get("data", []) if isinstance(body, dict) else []
        return [self._shape_player(p) for p in data]

    def get_player(self, player_id: int) -> Dict[str, Any]:
        """Get one player's profile, current ranking, and current Elo (FREE tier).

        Args:
            player_id (int): The player id (from ``search_players``).

        Returns:
            The player's bio, their current ranking and ranking points, and a
            ``ratings`` object carrying their current Elo (the current Elo is
            free on this object). The point-in-time as-of Elo tape and the
            rank-ordered rankings LISTING are separate paid surfaces (ULTRA and
            PRO respectively) and are not returned here.
        """
        player = self._get(f"players/{int(player_id)}")
        if not isinstance(player, dict):
            return {}
        shaped = self._shape_player(player)
        stats = player.get("stats") or {}
        shaped["ratings"] = stats.get("ratings")
        shaped["season"] = stats.get("season")
        shaped["hand"] = player.get("hand")
        shaped["birthday"] = player.get("birthday")
        return shaped

    def get_fixtures(
        self,
        tour: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """List upcoming scheduled fixtures, earliest first (FREE tier).

        Args:
            tour (Optional[str]): Optional tour filter — one of ``atp``, ``wta``,
                ``challenger``, ``itf``, ``juniors``. Defaults to the spec's
                ``default_tour`` if one was set.
            limit (int): Maximum number of fixtures to return (1-100).

        Returns:
            A list of fixtures with start time (null until the order of play
            assigns one), both player names (always present) and their ids where
            resolved, tournament, round, and surface.
        """
        params = {"tour": tour or self.default_tour, "limit": _clamp(limit, 1, 100)}
        body = self._get("fixtures", params)
        data = body.get("data", []) if isinstance(body, dict) else []
        return [self._shape_fixture(f) for f in data]


def _key_from_env() -> Optional[str]:
    for name in _ENV_KEYS:
        value = os.getenv(name)
        if value:
            return value
    return None


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))
