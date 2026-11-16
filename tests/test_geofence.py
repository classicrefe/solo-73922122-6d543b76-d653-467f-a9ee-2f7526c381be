"""Geofence events: geometry, interval semantics and snapshot isolation."""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.errors import ValidationError
from app.geofence import compute_stays, contains_point, validate_polygon
from app.main import create_app
from app.registry import Registry

# Square fence: lat in [40, 41], lon in [10, 11] (vertices are (lat, lon)).
SQUARE = [(40.0, 10.0), (40.0, 11.0), (41.0, 11.0), (41.0, 10.0)]
BASE = datetime(2026, 4, 1, tzinfo=UTC)


def at(seconds: float) -> datetime:
    return BASE + timedelta(seconds=seconds)


# ------------------------------------------------------------ geometry unit
def test_boundary_points_count_as_inside() -> None:
    assert contains_point(SQUARE, 40.5, 10.5)  # strictly inside
    assert not contains_point(SQUARE, 41.5, 10.5)  # strictly outside
    assert contains_point(SQUARE, 40.0, 10.5)  # on an edge
    assert contains_point(SQUARE, 40.0, 10.0)  # exactly on a vertex
    assert contains_point(SQUARE, 40.25, 11.0)  # on the eastern edge
    assert not contains_point(SQUARE, 40.0 - 1e-6, 10.5)  # just outside


def test_concave_polygon() -> None:
    # L-shaped ring; the notch at (40.5, 10.8) is outside.
    ring = [
        (40.0, 10.0),
        (40.0, 11.0),
        (40.6, 11.0),
        (40.6, 10.6),
        (41.0, 10.6),
        (41.0, 10.0),
    ]
    assert contains_point(ring, 40.3, 10.8)
    assert not contains_point(ring, 40.8, 10.8)
    assert contains_point(ring, 40.8, 10.3)


def test_validate_polygon_normalises_and_rejects_degenerate() -> None:
    closed = SQUARE + [SQUARE[0]]
    assert validate_polygon(closed) == SQUARE  # closing vertex dropped
    duplicated = [(40.0, 10.0), (40.0, 10.0)] + SQUARE
    assert validate_polygon(duplicated) == SQUARE  # consecutive dup dropped
    with pytest.raises(ValidationError):
        validate_polygon([(40.0, 10.0), (40.0, 11.0)])  # too few vertices
    with pytest.raises(ValidationError):
        validate_polygon([(40.0, 10.0), (40.5, 10.5), (41.0, 11.0)])  # collinear


# ------------------------------------------------------------ interval unit
def track(*points: tuple[float, float, float]) -> list[tuple]:
    """(seconds, lat, lon) -> sorted sample list."""
    return sorted((at(s), lat, lon) for s, lat, lon in points)


def test_continuous_stay_is_one_interval() -> None:
    samples = track(
        (0, 39.5, 10.5),  # out
        (10, 40.2, 10.5),  # in  \
        (20, 40.4, 10.5),  # in   | one stay
        (30, 40.6, 10.5),  # in  /
        (40, 41.5, 10.5),  # out
    )
    (stay,) = compute_stays(samples, SQUARE)
    assert stay.entered_at == at(10)
    assert stay.left_at == at(40)
    assert stay.inside_points == 3
    assert stay.closed


def test_unclosed_intervals_at_window_edges() -> None:
    samples = track(
        (0, 40.2, 10.5),  # already inside at window start
        (10, 41.5, 10.5),  # out
        (20, 40.5, 10.5),  # in again, still inside at window end
    )
    first, second = compute_stays(samples, SQUARE)
    assert first.entered_at is None and first.left_at == at(10)
    assert not first.closed
    assert second.entered_at == at(20) and second.left_at is None
    assert not second.closed


def test_single_inside_point_is_unclosed_both_sides() -> None:
    (stay,) = compute_stays(track((0, 40.5, 10.5)), SQUARE)
    assert stay.entered_at is None and stay.left_at is None
    assert stay.first_inside_at == stay.last_inside_at == at(0)
    assert not stay.closed


def test_no_samples_or_all_outside() -> None:
    assert compute_stays([], SQUARE) == []
    assert compute_stays(track((0, 0.0, 0.0), (10, 1.0, 1.0)), SQUARE) == []


# ------------------------------------------------------------------ API e2e
@pytest.fixture()
def client(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LAKE_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("LAKE_MAINTENANCE_INTERVAL_S", "3600")
    app = create_app(start_maintenance=False)
    app.state.registry.start()
    with TestClient(app) as test_client:
        yield test_client
    app.state.registry.stop()


def csv_rows(rows: list[tuple[str, int, float, float]]) -> bytes:
    lines = ["vehicle_id,timestamp,latitude,longitude"]
    for vehicle, seconds, lat, lon in rows:
        moment = (BASE + timedelta(seconds=seconds)).isoformat().replace(
            "+00:00", "Z"
        )
        lines.append(f"{vehicle},{moment},{lat},{lon}")
    return ("\n".join(lines) + "\n").encode()


def upload(client: TestClient, batch_id: str, payload: bytes) -> dict:
    response = client.post(
        "/v1/batches",
        data={"batch_id": batch_id},
        files={"file": ("batch.csv", payload, "text/csv")},
    )
    assert response.status_code == 201, response.text
    return response.json()


def geo_payload(vehicle: str = "GEO1", **overrides) -> dict:
    body = {
        "vehicle_id": vehicle,
        "start": BASE.isoformat().replace("+00:00", "Z"),
        "end": (BASE + timedelta(seconds=3600)).isoformat().replace(
            "+00:00", "Z"
        ),
        "polygon": [{"lat": lat, "lon": lon} for lat, lon in SQUARE]
        + [{"lat": 40.0, "lon": 10.0}],
    }
    body.update(overrides)
    return body


def seed_track(client: TestClient) -> list[str]:
    """Two batches whose points interleave in time (cross-file stitching)."""
    first = upload(
        client,
        "geo-a",
        csv_rows(
            [
                ("GEO1", 0, 39.5, 10.5),  # out
                ("GEO1", 20, 40.2, 10.5),  # in
                ("GEO1", 40, 41.5, 10.5),  # out
                ("GEO1", 60, 40.5, 10.5),  # in (stays inside at window end)
                ("OTHER", 10, 40.5, 10.5),  # different vehicle, ignored
            ]
        ),
    )
    second = upload(
        client,
        "geo-b",
        csv_rows(
            [
                ("GEO1", 10, 40.0, 10.5),  # on the boundary -> inside
                ("GEO1", 30, 40.4, 10.5),  # in
                ("GEO1", 50, 41.2, 10.5),  # out
            ]
        ),
    )
    return [first["file"]["file_id"], second["file"]["file_id"]]


def test_geofence_events_stitch_across_files(client: TestClient) -> None:
    file_ids = seed_track(client)
    response = client.post("/v1/geofence/events", json=geo_payload())
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["file_ids"] == file_ids  # the exact snapshot that was read
    assert body["snapshot_file_count"] == 2
    assert body["matched_points"] == 7  # OTHER's point excluded
    assert body["inside_points"] == 4  # boundary point counts as inside
    assert body["unclosed_count"] == 1
    stays = body["events"]
    assert len(stays) == 2
    # First stay: boundary point at t=10 opens it, t=40 outside closes it.
    assert stays[0]["entered_at"] == "2026-04-01T00:00:10.000000Z"
    assert stays[0]["left_at"] == "2026-04-01T00:00:40.000000Z"
    assert stays[0]["inside_points"] == 3
    assert stays[0]["closed"] is True
    # Second stay: re-entry at t=60, never leaves inside the window.
    assert stays[1]["entered_at"] == "2026-04-01T00:01:00.000000Z"
    assert stays[1]["left_at"] is None
    assert stays[1]["inside_points"] == 1
    assert stays[1]["closed"] is False


def test_geofence_window_filters_points(client: TestClient) -> None:
    seed_track(client)
    narrow = geo_payload(end=(BASE + timedelta(seconds=25)).isoformat().replace("+00:00", "Z"))
    body = client.post("/v1/geofence/events", json=narrow).json()
    assert body["matched_points"] == 3  # t = 0, 10, 20
    (stay,) = body["events"]
    assert stay["entered_at"] == "2026-04-01T00:00:10.000000Z"
    assert stay["left_at"] is None  # still inside when the window closes
    assert body["unclosed_count"] == 1


def test_geofence_empty_lake_and_unknown_vehicle(client: TestClient) -> None:
    empty = client.post("/v1/geofence/events", json=geo_payload()).json()
    assert empty["file_ids"] == [] and empty["events"] == []
    assert empty["matched_points"] == 0
    seed_track(client)
    nobody = client.post("/v1/geofence/events", json=geo_payload("NOPE")).json()
    assert nobody["events"] == [] and nobody["matched_points"] == 0
    assert nobody["snapshot_file_count"] == 2  # snapshot still reported


def test_geofence_validation_errors(client: TestClient) -> None:
    seed_track(client)
    degenerate = geo_payload(
        polygon=[{"lat": 40.0, "lon": 10.0}, {"lat": 41.0, "lon": 11.0}]
    )
    assert client.post("/v1/geofence/events", json=degenerate).status_code == 400
    collinear = geo_payload(
        polygon=[
            {"lat": 40.0, "lon": 10.0},
            {"lat": 40.5, "lon": 10.5},
            {"lat": 41.0, "lon": 11.0},
        ]
    )
    assert client.post("/v1/geofence/events", json=collinear).status_code == 400
    inverted = geo_payload(
        end=(BASE - timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
    )
    assert client.post("/v1/geofence/events", json=inverted).status_code == 422
    naive = geo_payload(start="2026-04-01 00:00:00")
    assert client.post("/v1/geofence/events", json=naive).status_code == 422
    blank = geo_payload(vehicle_id="   ")
    assert client.post("/v1/geofence/events", json=blank).status_code == 400


def test_geofence_endpoint_is_documented(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    assert "/v1/geofence/events" in spec["paths"]
    assert client.get("/docs").status_code == 200


# ------------------------------------------- snapshot isolation under churn
@pytest.fixture()
def registry(tmp_path: Path, monkeypatch) -> Registry:
    monkeypatch.setenv("LAKE_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("LAKE_MAINTENANCE_INTERVAL_S", "3600")
    from app.config import load_settings

    instance = Registry(load_settings())
    instance.start()
    yield instance
    instance.stop()


def test_query_never_mixes_snapshots_during_merges(registry: Registry) -> None:
    """Concurrent merges must never leak into a query's file set or rows."""
    points = [(0, 39.5, 10.5), (10, 40.5, 10.5), (20, 41.5, 10.5)]
    batch_ids: list[str] = []
    for index in range(4):
        payload = csv_rows(
            [("GEO", s + index, lat, lon) for s, lat, lon in points]
        )
        batch_ids.append(
            registry.ingest_batch(f"geo-{index}", payload)["file"].file_id
        )
    total_points = 4 * len(points)
    polygon = validate_polygon(SQUARE)
    start, end = BASE - timedelta(seconds=60), BASE + timedelta(seconds=3600)
    failures: list[BaseException] = []
    stop = threading.Event()

    def merger() -> None:
        from app.errors import ConflictError

        while not stop.is_set():
            try:
                registry.merge()
            except ConflictError:
                pass  # another merge holds the lock or nothing to merge
            except BaseException as exc:  # pragma: no cover - surfaced below
                failures.append(exc)
                return

    def querier() -> None:
        while not stop.is_set():
            try:
                result = registry.geofence_events(
                    vehicle_id="GEO", start=start, end=end, polygon=polygon
                )
                file_ids = result["file_ids"]
                # The snapshot is either the pre-merge set or one merged file
                # (merges of a single file are also possible).  Never a mix.
                assert set(file_ids) == set(batch_ids) or all(
                    file_id.startswith("m_") for file_id in file_ids
                ), file_ids
                # Every row of the vehicle is present exactly once: no file
                # was half-read, dropped, or counted from two snapshots.
                assert len(result["samples"]) == total_points
                assert sum(stay.inside_points for stay in result["stays"]) == 4
            except BaseException as exc:  # pragma: no cover - surfaced below
                failures.append(exc)
                return

    workers = [threading.Thread(target=merger) for _ in range(2)] + [
        threading.Thread(target=querier) for _ in range(3)
    ]
    for thread in workers:
        thread.start()
    deadline = threading.Event()
    deadline.wait(2.0)
    stop.set()
    for thread in workers:
        thread.join()
    assert not failures
    # Every internal manifest was released: no leaked pins block recycling.
    assert registry.stats()["open_manifests"] == 0
