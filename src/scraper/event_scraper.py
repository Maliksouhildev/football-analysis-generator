"""
src/scraper/event_scraper.py
Multi-Provider Football Match Event Ingestion & Scraping Engine.

Ingests live match telemetry from Sofascore, API-Football, ESPN Soccer Public API,
or Synthetic Offline Fixtures. Normalizes raw provider payloads into CanonicalMatchEvent
records, manages per-provider token bucket rate limiters, and executes a fault-tolerant
circuit breaker failover cascade.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import threading
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("scraper.event_scraper")


# =============================================================================
# 1. Canonical Schema & Enumerations
# =============================================================================

class CanonicalEventType(str, Enum):
    """Standardized football event classifications."""
    GOAL = "goal"
    PENALTY_GOAL = "penalty_goal"
    BIG_CHANCE = "big_chance"
    SHOT = "shot"
    TURNOVER = "turnover"
    BROKEN_PRESS = "broken_press"
    KEY_RECOVERY = "key_recovery"
    PROGRESSIVE_CARRY = "progressive_carry"
    TACTICAL_SUB = "tactical_sub"
    VAR_OVERTURN = "var_overturn"
    YELLOW_CARD = "yellow_card"
    RED_CARD = "red_card"
    FOUL = "foul"
    PASS = "pass"
    CLEARANCE = "clearance"
    THROW_IN = "throw_in"
    CORNER_KICK = "corner_kick"
    KICKOFF = "kickoff"
    HALFTIME = "halftime"
    FULLTIME = "fulltime"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PlayerInfo:
    """Standardized representation of a player participating in an event."""
    name: str
    id: Optional[str] = None
    jersey_number: Optional[int] = None
    position: Optional[str] = None
    role: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(frozen=True)
class TeamInfo:
    """Standardized representation of a football club involved in an event."""
    name: str
    id: Optional[str] = None
    is_home: bool = True
    formation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class CanonicalMatchEvent:
    """
    Authoritative Canonical Match Event Contract.
    Conforms to spec_r1_r2.md §3.3.1, PROJECT.md §Interface Contracts,
    and tests/e2e/fixtures/match_feed_fixtures.py.
    """
    id: str
    match_id: str
    period: int = 1
    minute: int = 0
    second: int = 0
    stoppage_minute: int = 0
    clock_display: str = ""

    event_type: CanonicalEventType = CanonicalEventType.UNKNOWN
    team: Optional[TeamInfo] = None
    primary_player: Optional[PlayerInfo] = None
    secondary_player: Optional[PlayerInfo] = None

    pitch_x: float = 50.0  # 0.0 (defending goal) to 100.0 (attacking goal)
    pitch_y: float = 50.0  # 0.0 (left touchline) to 100.0 (right touchline)

    xg: float = 0.0
    is_big_chance: bool = False
    description: str = ""
    formation_delta: Optional[str] = None
    tactical_score: Optional[float] = None

    provider: str = "synthetic"
    raw_payload: Optional[Dict[str, Any]] = field(default=None, repr=False)

    def __post_init__(self):
        px = self.pitch_x if self.pitch_x is not None else 50.0
        py = self.pitch_y if self.pitch_y is not None else 50.0
        self.pitch_x = min(100.0, max(0.0, float(px)))
        self.pitch_y = min(100.0, max(0.0, float(py)))
        self.minute = int(self.minute or 0)
        self.second = int(self.second or 0)
        self.stoppage_minute = int(self.stoppage_minute or 0)
        self.xg = float(self.xg or 0.0)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes to standard dictionary consumed by tests and tactical filter."""
        res: Dict[str, Any] = {
            "id": self.id,
            "match_id": self.match_id,
            "period": self.period,
            "minute": self.minute,
            "second": self.second,
            "stoppage_minute": self.stoppage_minute,
            "clock_display": self.clock_display or self._format_clock_display(),
            "type": self.event_type.value if isinstance(self.event_type, CanonicalEventType) else str(self.event_type),
            "pitch_x": round(self.pitch_x, 2),
            "pitch_y": round(self.pitch_y, 2),
            "xg": round(self.xg, 4),
            "is_big_chance": self.is_big_chance,
            "description": self.description,
            "provider": self.provider,
        }
        if self.team:
            res["team"] = self.team.name
            res["team_info"] = self.team.to_dict()
        if self.primary_player:
            res["player"] = self.primary_player.name
            res["primary_player"] = self.primary_player.to_dict()
        if self.secondary_player:
            res["secondary_player"] = self.secondary_player.to_dict()
            if self.event_type == CanonicalEventType.TACTICAL_SUB:
                res["player_in"] = self.primary_player.name if self.primary_player else None
                res["player_out"] = self.secondary_player.name
        if self.formation_delta:
            res["formation_delta"] = self.formation_delta
        if self.tactical_score is not None:
            res["significance_score"] = self.tactical_score
        return res

    def _format_clock_display(self) -> str:
        if self.stoppage_minute > 0:
            return f"{self.minute}'+{self.stoppage_minute}'"
        return f"{self.minute:02d}:{self.second:02d}"


# =============================================================================
# 2. Resilient Network Infrastructure: Rate Limiter & Circuit Breaker
# =============================================================================

class CircuitState(str, Enum):
    CLOSED = "CLOSED"        # Normal operation: requests flow through
    OPEN = "OPEN"            # Tripped: requests fail fast
    HALF_OPEN = "HALF_OPEN"  # Testing recovery


class CircuitBreaker:
    """Thread-safe circuit breaker preventing tight retry loops against failing providers."""
    def __init__(self, name: str, failure_threshold: int = 2, recovery_timeout_sec: float = 60.0):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout_sec = recovery_timeout_sec
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self._lock = threading.Lock()

    def can_execute(self) -> bool:
        with self._lock:
            now = time.time()
            if self.state == CircuitState.OPEN:
                if now - self.last_failure_time >= self.recovery_timeout_sec:
                    self.state = CircuitState.HALF_OPEN
                    logger.info("CircuitBreaker [%s] transitioned OPEN -> HALF_OPEN", self.name)
                    return True
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            self.failure_count = 0
            if self.state != CircuitState.CLOSED:
                logger.info("CircuitBreaker [%s] recovered -> CLOSED", self.name)
            self.state = CircuitState.CLOSED

    def record_failure(self, force_open: bool = False) -> None:
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if force_open or self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(
                    "CircuitBreaker [%s] TRIPPED -> OPEN (failures: %d, force: %s)",
                    self.name, self.failure_count, force_open
                )


class TokenBucketRateLimiter:
    """Thread-safe token bucket rate limiter enforcing provider request caps."""
    def __init__(self, capacity: float, refill_rate: float):
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate)  # Tokens per second
        self.tokens = float(capacity)
        self.last_refill = time.time()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0, timeout: float = 5.0) -> bool:
        start_time = time.time()
        while True:
            with self._lock:
                now = time.time()
                elapsed = now - self.last_refill
                self.last_refill = now
                self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True

                remaining = timeout - (now - start_time)
                if remaining <= 0:
                    return False

                needed = tokens - self.tokens
                wait_time = min(needed / max(self.refill_rate, 0.001), 0.5, remaining)
            time.sleep(max(wait_time, 0.001))


# =============================================================================
# 3. Provider Clients
# =============================================================================

class BaseProviderClient:
    """Abstract interface for football telemetry providers."""
    def __init__(self, name: str, rate_limit_rps: float, simulated_status: int = 200):
        self.name = name
        self.simulated_status = simulated_status
        self.circuit_breaker = CircuitBreaker(name=name, failure_threshold=2, recovery_timeout_sec=60.0)
        self.rate_limiter = TokenBucketRateLimiter(capacity=max(rate_limit_rps * 5, 2.0), refill_rate=rate_limit_rps)

    def fetch_events(self, fixture_id: str) -> Dict[str, Any]:
        """Fetches raw events payload or raises error if provider fails."""
        raise NotImplementedError


class SofascoreClient(BaseProviderClient):
    """
    Primary Provider: Sofascore API.
    Handles Cloudflare WAF anti-bot protection via Playwright response interception,
    REST requests, and status simulation.
    """
    def __init__(self, simulated_status: int = 200, use_playwright: bool = True, live_fetch: bool = False):
        super().__init__(name="sofascore", rate_limit_rps=1.0, simulated_status=simulated_status)
        self.use_playwright = use_playwright
        self.live_fetch = live_fetch

    def fetch_events(self, fixture_id: str) -> Dict[str, Any]:
        if not self.circuit_breaker.can_execute():
            raise ConnectionError("Sofascore circuit breaker is OPEN")

        rl_timeout = 0.5 if self.live_fetch else 0.0
        if not self.rate_limiter.acquire(timeout=rl_timeout):
            raise ConnectionError("Sofascore rate limit exceeded")

        if self.simulated_status != 200:
            if self.simulated_status == 403:
                self.circuit_breaker.record_failure(force_open=True)
                raise PermissionError("Sofascore blocked by Cloudflare (HTTP 403)")
            self.circuit_breaker.record_failure(force_open=False)
            raise ConnectionError(f"Sofascore HTTP {self.simulated_status}")

        if not self.live_fetch:
            self.circuit_breaker.record_success()
            if any(kw in str(fixture_id).lower() for kw in ("unplayed", "not_started", "fixture_empty", "empty")):
                return {"status": {"type": "notstarted"}, "incidents": []}
            return {
                "incidents": [
                    {
                        "id": "sofa_101",
                        "time": 24,
                        "second": 0,
                        "incidentType": "goal",
                        "isHome": True,
                        "player": {"name": "Vinicius Jr"},
                        "playerCoordinates": {"x": 88.0, "y": 55.0},
                        "xg": 0.42,
                    }
                ]
            }

        # Live scrape execution
        try:
            return self._fetch_live(fixture_id)
        except Exception as e:
            is_cf_block = "403" in str(e) or "Forbidden" in str(e) or "Cloudflare" in str(e)
            self.circuit_breaker.record_failure(force_open=is_cf_block)
            raise

    def _fetch_live(self, fixture_id: str) -> Dict[str, Any]:
        if self.use_playwright:
            try:
                from playwright.sync_api import sync_playwright
                with sync_playwright() as p:
                    browser = p.chromium.launch(
                        headless=True,
                        args=["--disable-blink-features=AutomationControlled"]
                    )
                    context = browser.new_context(
                        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                        viewport={"width": 1280, "height": 800}
                    )
                    page = context.new_page()
                    captured_incidents: Optional[Dict[str, Any]] = None

                    def handle_response(response):
                        nonlocal captured_incidents
                        if f"/event/{fixture_id}/incidents" in response.url and response.status == 200:
                            try:
                                captured_incidents = response.json()
                            except Exception:
                                pass

                    page.on("response", handle_response)
                    page.goto(f"https://www.sofascore.com/event/{fixture_id}", timeout=10000)
                    page.wait_for_timeout(2000)
                    browser.close()

                    if captured_incidents:
                        self.circuit_breaker.record_success()
                        return captured_incidents
            except Exception as pe:
                logger.warning("Sofascore Playwright capture failed: %s", pe)
                raise PermissionError(f"Sofascore Playwright failed: {pe}")

        raise PermissionError("Sofascore Cloudflare 403: Playwright unavailable or blocked")


class ApiFootballClient(BaseProviderClient):
    """
    Secondary Provider: API-Football (API-Sports).
    Enforces header authentication and detects 429 quota exhaustion.
    """
    def __init__(self, api_key: Optional[str] = None, simulated_status: int = 200, live_fetch: bool = False):
        super().__init__(name="api_football", rate_limit_rps=0.166, simulated_status=simulated_status)
        self.api_key = api_key
        self.live_fetch = live_fetch

    def fetch_events(self, fixture_id: str) -> Dict[str, Any]:
        if not self.circuit_breaker.can_execute():
            raise ConnectionError("API-Football circuit breaker is OPEN")

        rl_timeout = 0.5 if self.live_fetch else 0.0
        if not self.rate_limiter.acquire(timeout=rl_timeout):
            raise ConnectionError("API-Football rate limit exceeded")

        if self.simulated_status != 200:
            if self.simulated_status == 429:
                self.circuit_breaker.record_failure(force_open=True)
                raise ConnectionError("API-Football quota exhausted (HTTP 429)")
            self.circuit_breaker.record_failure(force_open=False)
            raise ConnectionError(f"API-Football HTTP {self.simulated_status}")

        if not self.live_fetch:
            self.circuit_breaker.record_success()
            if any(kw in str(fixture_id).lower() for kw in ("unplayed", "not_started", "fixture_empty", "empty")):
                return {"response": [], "parameters": {"fixture": fixture_id}}
            return {
                "response": [
                    {
                        "time": {"elapsed": 24, "extra": None},
                        "type": "Goal",
                        "player": {"name": "Vinicius Jr"},
                        "team": {"name": "Home"},
                    }
                ]
            }

        url = f"https://v3.football.api-sports.io/fixtures/events?fixture={fixture_id}"
        headers = {
            "x-apisports-key": self.api_key or "demo_key",
            "User-Agent": "FootballAutomation/1.0"
        }
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                errors = data.get("errors", {})
                if errors and any(k in errors for k in ("rateLimit", "requests")):
                    self.circuit_breaker.record_failure(force_open=True)
                    raise ResourceWarning(f"API-Football quota error: {errors}")
                self.circuit_breaker.record_success()
                return data
        except Exception as e:
            is_429 = "429" in str(e)
            self.circuit_breaker.record_failure(force_open=is_429)
            raise


class EspnPublicClient(BaseProviderClient):
    """
    Fallback Provider: Zero-Authentication ESPN Soccer Public API.
    Unblocked public REST endpoints with full keyEvents and pitch coordinates.
    """
    def __init__(self, simulated_status: int = 200, live_fetch: bool = False):
        super().__init__(name="espn", rate_limit_rps=2.0, simulated_status=simulated_status)
        self.live_fetch = live_fetch

    def fetch_events(self, fixture_id: str, league: str = "eng.1") -> Dict[str, Any]:
        if not self.circuit_breaker.can_execute():
            raise ConnectionError("ESPN Public API circuit breaker is OPEN")

        rl_timeout = 0.5 if self.live_fetch else 0.0
        if not self.rate_limiter.acquire(timeout=rl_timeout):
            raise ConnectionError("ESPN Public API rate limit exceeded")

        if self.simulated_status != 200:
            self.circuit_breaker.record_failure(force_open=False)
            raise ConnectionError(f"ESPN Public API HTTP {self.simulated_status}")

        if not self.live_fetch:
            self.circuit_breaker.record_success()
            if any(kw in str(fixture_id).lower() for kw in ("unplayed", "not_started", "fixture_empty", "empty")):
                return {
                    "header": {"competitions": [{"status": {"type": {"state": "pre", "completed": False}}}]},
                    "keyEvents": [],
                    "rosters": []
                }
            return {
                "header": {"competitions": [{"status": {"type": {"completed": True}}}]},
                "keyEvents": [
                    {
                        "id": "espn_001",
                        "clock": {"displayValue": "24'", "value": 1440.0},
                        "type": {"text": "Goal - Header", "type": "goal"},
                        "text": "Vinicius Jr Goal",
                        "team": {"displayName": "Home"},
                        "participants": [{"athlete": {"displayName": "Vinicius Jr"}}],
                        "fieldPositionX": 0.88,
                        "fieldPositionY": 0.55,
                    }
                ],
                "rosters": [{"team": {"name": "Home"}, "formation": "4-3-3"}],
            }

        url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/summary?event={fixture_id}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                self.circuit_breaker.record_success()
                return data
        except Exception as e:
            self.circuit_breaker.record_failure(force_open=False)
            raise ConnectionError(f"ESPN Public API fetch failed: {e}")


class SyntheticFixtureClient(BaseProviderClient):
    """
    Offline & CI Provider: Self-Contained Deterministic Match Fixtures.
    Guarantees instant, zero-network execution for unit and E2E test suites.
    """
    def __init__(self, fixture_catalog: Optional[Dict[str, List[Dict[str, Any]]]] = None):
        super().__init__(name="synthetic", rate_limit_rps=1000.0, simulated_status=200)
        self.fixture_catalog = fixture_catalog or {}

    def fetch_events(self, fixture_id: str) -> List[Dict[str, Any]]:
        if fixture_id in self.fixture_catalog:
            return self.fixture_catalog[fixture_id]
        return [
            {
                "id": "s_01",
                "minute": 24,
                "second": 0,
                "type": "goal",
                "team": "Real Madrid",
                "player": "Vinicius Jr",
                "pitch_x": 88.0,
                "pitch_y": 55.0,
                "xg": 0.42,
                "is_big_chance": True,
            }
        ]


# =============================================================================
# 4. Multi-Provider Orchestrator & Normalizer Engine
# =============================================================================

class MultiProviderScraper:
    """
    High-Resilience Match Event Scraper & Ingestion Orchestrator.
    Controls the provider priority cascade, circuit breaker failover,
    and canonical event normalization.
    """
    def __init__(
        self,
        sofascore_status: int = 200,
        api_football_status: int = 200,
        espn_status: int = 200,
        api_football_key: Optional[str] = None,
        use_synthetic_fallback: bool = False,
        fixture_catalog: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        live_fetch: bool = False,
    ):
        self.sofascore_status = sofascore_status
        self.api_football_status = api_football_status
        self.espn_status = espn_status
        self.live_fetch = live_fetch
        self.fixture_catalog = fixture_catalog or {}

        self.sofascore = SofascoreClient(simulated_status=sofascore_status, live_fetch=live_fetch)
        self.api_football = ApiFootballClient(api_key=api_football_key, simulated_status=api_football_status, live_fetch=live_fetch)
        self.espn = EspnPublicClient(simulated_status=espn_status, live_fetch=live_fetch)
        self.synthetic = SyntheticFixtureClient(fixture_catalog=fixture_catalog)
        self.use_synthetic_fallback = use_synthetic_fallback

        self.provider_used: Optional[str] = None
        self._last_status: int = 200

    def fetch_match_events(self, fixture_id: str, league: str = "eng.1") -> Dict[str, Any]:
        """
        Executes failover cascade: Sofascore -> API-Football -> ESPN -> Synthetic.
        Matches interface required by tests:
        Returns: {"provider": str, "status": 200, "events": List[Dict[str, Any]]}
        """
        errors: List[str] = []

        # 1. Attempt Primary: Sofascore
        try:
            raw = self.sofascore.fetch_events(fixture_id)
            self.provider_used = "sofascore"
            self._last_status = self.sofascore.simulated_status
            if self.fixture_catalog and fixture_id in self.fixture_catalog:
                events = [self._dict_to_canonical(d, fixture_id) for d in self.fixture_catalog[fixture_id]]
            else:
                events = self._normalize_sofascore(raw, fixture_id)
            return {"provider": "sofascore", "status": self._last_status, "events": [e.to_dict() for e in events]}
        except Exception as e:
            errors.append(f"Sofascore failed: {e}")
            logger.info("Sofascore primary unavailable: %s. Initiating failover.", e)

        # 2. Attempt Secondary: API-Football
        try:
            raw = self.api_football.fetch_events(fixture_id)
            self.provider_used = "api_football"
            self._last_status = self.api_football.simulated_status
            if self.fixture_catalog and fixture_id in self.fixture_catalog:
                events = [self._dict_to_canonical(d, fixture_id) for d in self.fixture_catalog[fixture_id]]
            else:
                events = self._normalize_api_football(raw, fixture_id)
            return {"provider": "api_football", "status": self._last_status, "events": [e.to_dict() for e in events]}
        except Exception as e:
            errors.append(f"API-Football failed: {e}")
            logger.info("API-Football secondary unavailable: %s. Initiating failover.", e)

        # 3. Attempt Fallback: ESPN Public API
        try:
            raw = self.espn.fetch_events(fixture_id, league=league)
            self.provider_used = "espn"
            self._last_status = self.espn.simulated_status
            if self.fixture_catalog and fixture_id in self.fixture_catalog:
                events = [self._dict_to_canonical(d, fixture_id) for d in self.fixture_catalog[fixture_id]]
            else:
                events = self._normalize_espn(raw, fixture_id)
            return {"provider": "espn", "status": self._last_status, "events": [e.to_dict() for e in events]}
        except Exception as e:
            errors.append(f"ESPN Public API failed: {e}")
            logger.info("ESPN Public API fallback unavailable: %s.", e)

        # 4. Attempt Offline Synthetic Fixture
        if self.use_synthetic_fallback or (self.fixture_catalog and fixture_id in self.fixture_catalog):
            self.provider_used = "synthetic"
            self._last_status = 200
            raw_synth = self.synthetic.fetch_events(fixture_id)
            events = [self._dict_to_canonical(d, fixture_id) for d in raw_synth]
            return {"provider": "synthetic", "status": 200, "events": [e.to_dict() for e in events]}

        # If all providers failed and offline fallback not enabled, raise ConnectionError
        error_msg = f"All match feed providers failed for fixture {fixture_id}: {'; '.join(errors)}"
        logger.error(error_msg)
        raise ConnectionError(error_msg)

    def fetch_canonical_events(self, fixture_id: str, league: str = "eng.1") -> List[CanonicalMatchEvent]:
        """Fetches events and returns strongly-typed CanonicalMatchEvent objects."""
        res = self.fetch_match_events(fixture_id, league=league)
        return [self._dict_to_canonical(d, fixture_id) for d in res.get("events", [])]

    # -------------------------------------------------------------------------
    # Provider Normalizers
    # -------------------------------------------------------------------------

    def _normalize_sofascore(self, raw: Dict[str, Any], fixture_id: str) -> List[CanonicalMatchEvent]:
        """Normalizes Sofascore incidents into CanonicalMatchEvents."""
        incidents = raw.get("incidents", [])
        if not incidents and "events" in raw:
            incidents = raw["events"]

        normalized: List[CanonicalMatchEvent] = []
        for inc in incidents:
            inc_type = str(inc.get("incidentType", inc.get("type", ""))).lower()
            minute = int(inc.get("time") or inc.get("minute") or 0)
            stoppage = int(inc.get("addedTime") or inc.get("stoppage_minute") or 0)
            sec = int(inc.get("timeSeconds") or inc.get("second") or 0) % 60

            event_type = CanonicalEventType.UNKNOWN
            is_big_chance = False
            xg = float(inc.get("xg") or 0.0)

            if inc_type in ("goal", "ingamepenalty"):
                event_type = CanonicalEventType.GOAL
                is_big_chance = True
            elif inc_type == "vardecision":
                event_type = CanonicalEventType.VAR_OVERTURN
            elif inc_type == "substitution":
                event_type = CanonicalEventType.TACTICAL_SUB
            elif inc_type == "card":
                event_type = CanonicalEventType.YELLOW_CARD
            else:
                for member in CanonicalEventType:
                    if member.value == inc_type:
                        event_type = member
                        break

            player_name = inc.get("player", {}).get("name") if isinstance(inc.get("player"), dict) else str(inc.get("player", ""))
            team_name = "Home" if inc.get("isHome") else "Away"

            coords = inc.get("playerCoordinates", {})
            raw_x = coords.get("x") if isinstance(coords, dict) else None
            raw_y = coords.get("y") if isinstance(coords, dict) else None
            px = min(100.0, max(0.0, float(raw_x if raw_x is not None else 50.0)))
            py = min(100.0, max(0.0, float(raw_y if raw_y is not None else 50.0)))

            evt = CanonicalMatchEvent(
                id=str(inc.get("id", f"sofa_{minute}_{len(normalized)}")),
                match_id=fixture_id,
                period=1 if minute <= 45 else 2,
                minute=minute,
                second=sec,
                stoppage_minute=stoppage,
                event_type=event_type,
                team=TeamInfo(name=team_name),
                primary_player=PlayerInfo(name=player_name) if player_name else None,
                pitch_x=px,
                pitch_y=py,
                xg=xg,
                is_big_chance=is_big_chance or xg >= 0.15,
                description=str(inc.get("description", "")),
                provider="sofascore",
                raw_payload=inc,
            )
            normalized.append(evt)
        return normalized

    def _normalize_api_football(self, raw: Dict[str, Any], fixture_id: str) -> List[CanonicalMatchEvent]:
        """Normalizes API-Football fixture events into CanonicalMatchEvents."""
        items = raw.get("response", [])
        if not items and "events" in raw:
            items = raw["events"]

        normalized: List[CanonicalMatchEvent] = []
        for item in items:
            t_info = item.get("time") or {}
            minute = int(t_info.get("elapsed") or item.get("minute") or 0)
            stoppage = int(t_info.get("extra") or item.get("stoppage_minute") or 0)

            raw_type = str(item.get("type", "")).lower()
            detail = str(item.get("detail", "")).lower()

            event_type = CanonicalEventType.UNKNOWN
            is_big_chance = False
            pitch_x = 50.0

            if raw_type == "goal":
                event_type = CanonicalEventType.GOAL
                is_big_chance = True
                pitch_x = 90.0
            elif raw_type == "var" and "cancelled" in detail:
                event_type = CanonicalEventType.VAR_OVERTURN
            elif raw_type in ("subst", "substitution"):
                event_type = CanonicalEventType.TACTICAL_SUB
            elif raw_type == "card":
                event_type = CanonicalEventType.YELLOW_CARD
            else:
                for member in CanonicalEventType:
                    if member.value == raw_type:
                        event_type = member
                        break

            player_dict = item.get("player", {})
            player_name = player_dict.get("name") if isinstance(player_dict, dict) else str(player_dict or "")
            team_dict = item.get("team", {})
            team_name = team_dict.get("name") if isinstance(team_dict, dict) else str(team_dict or "")

            evt = CanonicalMatchEvent(
                id=str(item.get("id", f"apifb_{minute}_{len(normalized)}")),
                match_id=fixture_id,
                period=1 if minute <= 45 else 2,
                minute=minute,
                second=0,
                stoppage_minute=stoppage,
                event_type=event_type,
                team=TeamInfo(name=team_name),
                primary_player=PlayerInfo(name=player_name) if player_name else None,
                pitch_x=pitch_x,
                pitch_y=50.0,
                is_big_chance=is_big_chance,
                description=detail or raw_type,
                provider="api_football",
                raw_payload=item,
            )
            normalized.append(evt)
        return normalized

    def _normalize_espn(self, raw: Dict[str, Any], fixture_id: str) -> List[CanonicalMatchEvent]:
        """Normalizes ESPN Soccer Public API summary into CanonicalMatchEvents."""
        key_events = raw.get("keyEvents", [])
        if not key_events and "events" in raw:
            key_events = raw["events"]

        normalized: List[CanonicalMatchEvent] = []
        for ke in key_events:
            clock_info = ke.get("clock", {})
            display_val = str(clock_info.get("displayValue", ""))
            clock_val = float(clock_info.get("value", 0.0))

            minute = int(clock_val // 60) if clock_val > 0 else 0
            stoppage = 0
            sec = int(clock_val % 60)

            match = re.search(r"(\d+)'(?:\+(\d+)')?", display_val)
            if match:
                minute = int(match.group(1))
                if match.group(2):
                    stoppage = int(match.group(2))

            type_info = ke.get("type", {})
            type_text = str(type_info.get("text", type_info.get("type", ""))).lower()

            event_type = CanonicalEventType.UNKNOWN
            is_big_chance = False

            if "goal" in type_text:
                event_type = CanonicalEventType.GOAL
                is_big_chance = True
            elif "substitution" in type_text:
                event_type = CanonicalEventType.TACTICAL_SUB
            elif "yellow" in type_text:
                event_type = CanonicalEventType.YELLOW_CARD
            elif "red" in type_text:
                event_type = CanonicalEventType.RED_CARD
            elif "kickoff" in type_text:
                event_type = CanonicalEventType.KICKOFF
            else:
                for member in CanonicalEventType:
                    if member.value in type_text:
                        event_type = member
                        break

            raw_x = ke.get("fieldPositionX")
            raw_y = ke.get("fieldPositionY")
            if raw_x is not None:
                fx = float(raw_x)
                px = fx * 100.0 if fx <= 1.0 else fx
            else:
                px = 50.0
            if raw_y is not None:
                fy = float(raw_y)
                py = fy * 100.0 if fy <= 1.0 else fy
            else:
                py = 50.0
            px = min(100.0, max(0.0, px))
            py = min(100.0, max(0.0, py))

            participants = ke.get("participants", [])
            primary_name = participants[0].get("athlete", {}).get("displayName", "") if len(participants) > 0 and isinstance(participants[0], dict) else ""
            secondary_name = participants[1].get("athlete", {}).get("displayName", "") if len(participants) > 1 and isinstance(participants[1], dict) else ""
            team_info = ke.get("team", {})
            team_name = team_info.get("displayName", team_info.get("name", ""))

            p_raw = ke.get("period")
            if isinstance(p_raw, dict):
                period = int(p_raw.get("number", 1 if minute <= 45 else 2))
            elif isinstance(p_raw, (int, float)):
                period = int(p_raw)
            else:
                period = 1 if minute <= 45 else 2

            evt = CanonicalMatchEvent(
                id=str(ke.get("id", f"espn_{minute}_{len(normalized)}")),
                match_id=fixture_id,
                period=period,
                minute=minute,
                second=sec,
                stoppage_minute=stoppage,
                clock_display=display_val,
                event_type=event_type,
                team=TeamInfo(name=team_name),
                primary_player=PlayerInfo(name=primary_name) if primary_name else None,
                secondary_player=PlayerInfo(name=secondary_name) if secondary_name else None,
                pitch_x=px,
                pitch_y=py,
                is_big_chance=is_big_chance,
                description=str(ke.get("text", "")),
                provider="espn",
                raw_payload=ke,
            )
            normalized.append(evt)
        return normalized

    def _dict_to_canonical(self, d: Dict[str, Any], fixture_id: str) -> CanonicalMatchEvent:
        """Converts generic dictionary into CanonicalMatchEvent."""
        raw_type = str(d.get("type", "")).lower()
        evt_type = CanonicalEventType.UNKNOWN
        for member in CanonicalEventType:
            if member.value == raw_type:
                evt_type = member
                break

        minute = int(d.get("minute") or 0)
        second = int(d.get("second") or 0)
        stoppage = int(d.get("stoppage_minute") or 0)
        p_val = d.get("period")
        period = int(p_val) if p_val is not None else (1 if minute <= 45 else 2)

        raw_px = d.get("pitch_x")
        raw_py = d.get("pitch_y")
        px = min(100.0, max(0.0, float(raw_px if raw_px is not None else 50.0)))
        py = min(100.0, max(0.0, float(raw_py if raw_py is not None else 50.0)))

        xg = float(d.get("xg") or 0.0)
        is_bc = bool(d.get("is_big_chance") or False)
        sig_score = float(d["significance_score"]) if d.get("significance_score") is not None else None

        player_in = d.get("player_in")
        player_out = d.get("player_out")
        primary_player = PlayerInfo(name=str(d.get("player") or player_in or "")) if (d.get("player") or player_in) else None
        secondary_player = PlayerInfo(name=str(player_out)) if player_out else None

        return CanonicalMatchEvent(
            id=str(d.get("id", f"syn_{minute}")),
            match_id=fixture_id,
            period=period,
            minute=minute,
            second=second,
            stoppage_minute=stoppage,
            event_type=evt_type,
            team=TeamInfo(name=str(d.get("team") or "Home")),
            primary_player=primary_player,
            secondary_player=secondary_player,
            pitch_x=px,
            pitch_y=py,
            xg=xg,
            is_big_chance=is_bc,
            description=str(d.get("description") or ""),
            formation_delta=d.get("formation_delta"),
            tactical_score=sig_score,
            provider=str(d.get("provider") or "synthetic"),
            raw_payload=d,
        )


# Backward-compatible and test simulator aliases
MultiProviderEventScraper = MultiProviderScraper
MultiProviderScraperSimulator = MultiProviderScraper


# =============================================================================
# 5. Timestamp-Synchronized Match Event Telemetry Engine (Opta / Sofascore / FotMob)
# =============================================================================

AUTHENTIC_MATCH_TELEMETRY: List[Dict[str, Any]] = []


class MatchEventTelemetryEngine:
    """
    Ingests structured Opta / Sofascore / FotMob match event telemetry:
    {minute, second, team, playerName, shirtNo, type, x, y, endX, endY, receiverName}
    and provides deterministic lookup of the exact passer, receiver, and pass coordinates
    for any match timestamp MM:SS.
    """

    _event_registry: List[Dict[str, Any]] = []

    @classmethod
    def register_event_logs(cls, event_logs: List[Dict[str, Any]]) -> None:
        """Appends external structured event logs (from Opta, Sofascore, or FotMob)."""
        if isinstance(event_logs, list):
            cls._event_registry.extend(event_logs)

    @classmethod
    def clear_event_logs(cls) -> None:
        """Clears registered event logs."""
        cls._event_registry = []

    @classmethod
    def lookup_event_at_timestamp(
        cls,
        minute: int,
        second: int = 0,
        clip_id: str = "",
        window_sec: float = 60.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Locates the authoritative event telemetry record matching the given timestamp or clip_id.
        """
        cid = str(clip_id or "").lower()

        # Check exact clip_id first
        if cid:
            for ev in cls._event_registry:
                ev_cid = str(ev.get("clip_id", "")).lower()
                if ev_cid and (ev_cid in cid or cid in ev_cid):
                    return copy.deepcopy(ev)

        # Match by timestamp within window
        target_t = float(minute * 60 + second)
        best_ev = None
        min_diff = float("inf")

        for ev in cls._event_registry:
            ev_m = int(ev.get("minute", 0))
            ev_s = int(ev.get("second", 0))
            ev_t = float(ev_m * 60 + ev_s)
            diff = abs(target_t - ev_t)
            if diff <= window_sec and diff < min_diff:
                min_diff = diff
                best_ev = ev

        return copy.deepcopy(best_ev) if best_ev else None

