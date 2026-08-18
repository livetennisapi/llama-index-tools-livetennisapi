"""Fully mocked tests for LiveTennisAPIToolSpec — no network.

Every request is served by an httpx.MockTransport, so the suite exercises the
tool outputs, the auth header, error handling, and the break-point derivation
without ever touching the real API.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import httpx
import pytest

from llama_index.tools.livetennisapi import LiveTennisAPIError, LiveTennisAPIToolSpec

API_KEY = "twjp_test_key"


class Recorder:
    """A MockTransport handler that records requests and returns canned replies."""

    def __init__(self, routes: Dict[str, Tuple[int, Any]]) -> None:
        # routes maps a URL path -> (status_code, json_body)
        self.routes = routes
        self.requests: List[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        # Requests carry the full base path (/api/public/v1/...); routes are
        # keyed by the endpoint suffix, so match on endswith.
        for suffix, (status, body) in self.routes.items():
            if request.url.path.endswith(suffix):
                return httpx.Response(status, content=json.dumps(body).encode())
        return httpx.Response(404, content=json.dumps({"error": "not_found"}).encode())


def make_spec(
    routes: Dict[str, Tuple[int, Any]],
    **kwargs: Any,
) -> Tuple[LiveTennisAPIToolSpec, Recorder]:
    recorder = Recorder(routes)
    client = httpx.Client(transport=httpx.MockTransport(recorder))
    spec = LiveTennisAPIToolSpec(api_key=API_KEY, client=client, **kwargs)
    return spec, recorder


# -- sample payloads ------------------------------------------------------

SCORE_LIVE = {
    "sets": [1, 0],
    "games": [[4], [3]],
    "points": ["30", "40"],
    "server": 1,
    "is_tiebreak": False,
    "timestamp": "2026-08-18T12:00:00Z",
}

MATCH = {
    "id": 555,
    "tournament": "Cincinnati",
    "tour": "atp",
    "round": "QF",
    "surface": "hard",
    "status": "live",
    "scheduled_time": "2026-08-18T11:30:00Z",
    "players": {
        "p1": {"id": 1, "name": "C. Alcaraz", "country": "ESP", "ranking": 2},
        "p2": {"id": 2, "name": "J. Sinner", "country": "ITA", "ranking": 1},
    },
    "score": SCORE_LIVE,
    "winner": None,
}

PLAYER = {
    "id": 1,
    "name": "C. Alcaraz",
    "country": "ESP",
    "tour": "atp",
    "ranking": 2,
    "ranking_points": 8500,
    "ranking_movement": "same",
    "hand": "R",
    "birthday": "2003-05-05",
    "stats": {"ratings": {"elo": 2185.0}, "season": [{"year": 2026, "wins": 40}]},
}

FIXTURE = {
    "id": 900,
    "event_date": "2026-08-19",
    "start_time": None,
    "tournament": "Cincinnati",
    "tour": "atp",
    "round": "SF",
    "surface": "hard",
    "player1_id": 1,
    "player1_name": "C. Alcaraz",
    "player2_id": None,
    "player2_name": "Qualifier",
    "status": "scheduled",
}


# -- construction & auth --------------------------------------------------


def test_missing_key_raises():
    with pytest.raises(ValueError, match="API key is required"):
        LiveTennisAPIToolSpec(api_key=None, client=httpx.Client())


def test_key_from_primary_env(monkeypatch):
    monkeypatch.delenv("LIVETENNISAPI_KEY", raising=False)
    monkeypatch.setenv("LIVETENNIS_API_KEY", "env_primary")
    spec = LiveTennisAPIToolSpec(client=httpx.Client())
    assert spec.api_key == "env_primary"


def test_key_from_fallback_env(monkeypatch):
    monkeypatch.delenv("LIVETENNIS_API_KEY", raising=False)
    monkeypatch.setenv("LIVETENNISAPI_KEY", "env_fallback")
    spec = LiveTennisAPIToolSpec(client=httpx.Client())
    assert spec.api_key == "env_fallback"


def test_auth_header_sent():
    spec, rec = make_spec({"/matches": (200, {"data": [MATCH]})})
    spec.get_live_matches()
    assert rec.requests[0].headers.get("X-API-Key") == API_KEY


def test_base_url_and_path():
    spec, rec = make_spec({"/matches": (200, {"data": []})})
    spec.get_live_matches()
    url = rec.requests[0].url
    assert url.host == "api.livetennisapi.com"
    assert url.path == "/api/public/v1/matches"


# -- get_live_matches -----------------------------------------------------


def test_get_live_matches_shapes_output():
    spec, rec = make_spec({"/matches": (200, {"data": [MATCH]})})
    out = spec.get_live_matches()
    assert len(out) == 1
    m = out[0]
    assert m["id"] == 555
    assert m["player1"]["name"] == "C. Alcaraz"
    assert m["player2"]["ranking"] == 1
    assert m["score"]["break_point"] is True  # receiver p2 at 40, server p1 at 30
    assert rec.requests[0].url.params["status"] == "live"


def test_get_live_matches_upcoming_status_passed():
    spec, rec = make_spec({"/matches": (200, {"data": []})})
    spec.get_live_matches(status="upcoming")
    assert rec.requests[0].url.params["status"] == "upcoming"


def test_get_live_matches_completed_rejected():
    spec, _ = make_spec({"/matches": (200, {"data": []})})
    with pytest.raises(ValueError, match="FREE tier"):
        spec.get_live_matches(status="completed")


def test_default_tour_applied():
    spec, rec = make_spec({"/matches": (200, {"data": []})}, default_tour="wta")
    spec.get_live_matches()
    assert rec.requests[0].url.params["tour"] == "wta"


def test_explicit_tour_overrides_default():
    spec, rec = make_spec({"/matches": (200, {"data": []})}, default_tour="wta")
    spec.get_live_matches(tour="atp")
    assert rec.requests[0].url.params["tour"] == "atp"


def test_limit_clamped():
    spec, rec = make_spec({"/matches": (200, {"data": []})})
    spec.get_live_matches(limit=9999)
    assert rec.requests[0].url.params["limit"] == "100"


# -- get_match_score & break-point derivation -----------------------------


def test_get_match_score_snapshot():
    spec, rec = make_spec({"/matches/555/score": (200, SCORE_LIVE)})
    out = spec.get_match_score(555)
    assert out["server"] == 1
    assert out["break_point"] is True
    assert rec.requests[0].url.path == "/api/public/v1/matches/555/score"


@pytest.mark.parametrize(
    "score,expected",
    [
        # receiver (p2) at AD -> break point
        ({"points": ["40", "AD"], "server": 1, "is_tiebreak": False}, True),
        # receiver (p2) at 40 while server (p1) at 30 -> break point
        ({"points": ["30", "40"], "server": 1, "is_tiebreak": False}, True),
        # receiver (p1) at 40 while server (p2) at 15 -> break point
        ({"points": ["40", "15"], "server": 2, "is_tiebreak": False}, True),
        # both at 40 (deuce), no AD -> not a break point
        ({"points": ["40", "40"], "server": 1, "is_tiebreak": False}, False),
        # server at 40, receiver at 30 -> not a break point
        ({"points": ["40", "30"], "server": 1, "is_tiebreak": False}, False),
        # server holds AD -> not a break point
        ({"points": ["AD", "40"], "server": 1, "is_tiebreak": False}, False),
        # tiebreak -> always False regardless of points
        ({"points": ["6", "5"], "server": 1, "is_tiebreak": True}, False),
        # null server -> unknown
        ({"points": ["30", "40"], "server": None, "is_tiebreak": False}, None),
        # null points entry -> unknown
        ({"points": [None, None], "server": 1, "is_tiebreak": False}, None),
        # missing points -> unknown
        ({"server": 1, "is_tiebreak": False}, None),
    ],
)
def test_break_point_derivation(score, expected):
    assert LiveTennisAPIToolSpec._derive_break_point(score) is expected


def test_break_point_none_score():
    assert LiveTennisAPIToolSpec._derive_break_point(None) is None
    assert LiveTennisAPIToolSpec._derive_break_point({}) is None  # empty/falsy -> unknown


def test_get_match_score_empty_body():
    spec, _ = make_spec({"/matches/1/score": (200, None)})
    assert spec.get_match_score(1) == {}


# -- search_players & get_player ------------------------------------------


def test_search_players():
    spec, rec = make_spec({"/players": (200, {"data": [PLAYER]})})
    out = spec.search_players("alcaraz")
    assert out[0]["name"] == "C. Alcaraz"
    assert out[0]["ranking"] == 2
    assert "ratings" not in out[0]  # list rows carry no stats
    assert rec.requests[0].url.params["search"] == "alcaraz"


def test_get_player_includes_ranking_and_elo():
    spec, rec = make_spec({"/players/1": (200, PLAYER)})
    out = spec.get_player(1)
    assert out["ranking"] == 2
    assert out["ratings"] == {"elo": 2185.0}
    assert out["season"][0]["year"] == 2026
    assert rec.requests[0].url.path == "/api/public/v1/players/1"


def test_get_player_missing_stats():
    thin = {"id": 7, "name": "Q. Player", "ranking": None}
    spec, _ = make_spec({"/players/7": (200, thin)})
    out = spec.get_player(7)
    assert out["ratings"] is None


# -- get_fixtures ---------------------------------------------------------


def test_get_fixtures():
    spec, rec = make_spec({"/fixtures": (200, {"data": [FIXTURE]})})
    out = spec.get_fixtures(tour="atp")
    assert out[0]["player2_name"] == "Qualifier"
    assert out[0]["player2_id"] is None
    assert out[0]["start_time"] is None
    assert rec.requests[0].url.params["tour"] == "atp"


# -- error handling -------------------------------------------------------


@pytest.mark.parametrize(
    "status,code,needle",
    [
        (401, "unauthorized", "missing or invalid"),
        (403, "upgrade_required", "higher tier"),
        (404, "not_found", "No such resource"),
        (429, "rate_limited", "Rate limit"),
    ],
)
def test_http_errors_raise(status, code, needle):
    spec, _ = make_spec({"/matches": (status, {"error": code})})
    with pytest.raises(LiveTennisAPIError) as exc:
        spec.get_live_matches()
    assert exc.value.status_code == status
    assert exc.value.code == code
    assert needle in str(exc.value)


def test_transport_error_wrapped():
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    client = httpx.Client(transport=httpx.MockTransport(boom))
    spec = LiveTennisAPIToolSpec(api_key=API_KEY, client=client)
    with pytest.raises(LiveTennisAPIError, match="failed"):
        spec.get_live_matches()


def test_500_error_raises_with_status():
    spec, _ = make_spec({"/players": (500, {"error": "server_error"})})
    with pytest.raises(LiveTennisAPIError) as exc:
        spec.search_players("x")
    assert exc.value.status_code == 500


# -- LlamaIndex integration ----------------------------------------------


def test_to_tool_list_exposes_all_functions():
    spec, _ = make_spec({})
    tools = spec.to_tool_list()
    names = {t.metadata.name for t in tools}
    assert names == {
        "get_live_matches",
        "get_match_score",
        "search_players",
        "get_player",
        "get_fixtures",
    }


def test_spec_functions_are_all_methods():
    for fn in LiveTennisAPIToolSpec.spec_functions:
        assert callable(getattr(LiveTennisAPIToolSpec, fn))
