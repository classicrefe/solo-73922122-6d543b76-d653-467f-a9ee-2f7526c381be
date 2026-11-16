"""Geofence event extraction over a consistent snapshot of trajectory files.

The fence is a closed ring of WGS-84 vertices.  Points exactly on an edge or
a vertex count as *inside*.  The vehicle's in-window samples are sorted by
timestamp (stably, so stitching points that come from different Parquet files
is deterministic) and every maximal run of inside samples becomes one stay
interval — a continuous stay never produces more than one interval:

* ``entered_at`` is the first inside sample of the run: the moment the entry
  was observed.  If the vehicle is already inside at the first in-window
  sample, the entry itself was not observed and ``entered_at`` is ``None``.
* ``left_at`` is the first *outside* sample after the run: the moment the
  exit was observed.  If the vehicle is still inside at the last in-window
  sample, the exit was not observed and ``left_at`` is ``None``.

A stay with either side unobserved is an unclosed interval (``closed`` is
false).  Containment uses ray casting on the (lon, lat) plane, so the fence
is treated as planar: fences spanning the antimeridian or hugging the poles
are out of scope (documented in the README).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pyarrow as pa

from .errors import ValidationError

# A vertex is a (lat, lon) pair in degrees.
Vertex = tuple[float, float]

# Tolerances in degrees.  1e-9 deg is about 0.1 mm, far below GPS noise, so
# only points that are geometrically on the boundary are treated as such.
_VERTEX_TOL = 1e-12
_EDGE_TOL = 1e-9

MAX_POLYGON_VERTICES = 512


@dataclass(frozen=True)
class Stay:
    """One continuous in-fence stay of a vehicle."""

    entered_at: datetime | None  # None: already inside at the window's first sample
    left_at: datetime | None  # None: still inside at the window's last sample
    first_inside_at: datetime
    last_inside_at: datetime
    inside_points: int

    @property
    def closed(self) -> bool:
        return self.entered_at is not None and self.left_at is not None


def _same_vertex(a: Vertex, b: Vertex) -> bool:
    return abs(a[0] - b[0]) <= _VERTEX_TOL and abs(a[1] - b[1]) <= _VERTEX_TOL


def validate_polygon(points: list[Vertex]) -> list[Vertex]:
    """Normalise and validate a fence ring.

    Accepts a ring with or without the repeated closing vertex, drops
    consecutive duplicate vertices, and requires at least three distinct,
    non-collinear vertices.  Returns the vertices in order (not explicitly
    closed).  Raises ``ValidationError`` (HTTP 400) on degenerate input.
    """
    if len(points) > MAX_POLYGON_VERTICES:
        raise ValidationError(
            f"围栏多边形顶点过多：{len(points)} 个，最多 {MAX_POLYGON_VERTICES} 个"
        )
    deduped: list[Vertex] = []
    for point in points:
        if not deduped or not _same_vertex(deduped[-1], point):
            deduped.append(point)
    while len(deduped) >= 2 and _same_vertex(deduped[0], deduped[-1]):
        deduped.pop()
    if len(deduped) < 3:
        raise ValidationError(
            f"围栏多边形去重后只剩 {len(deduped)} 个不同顶点，至少需要 3 个"
        )
    # Signed area (planar, degrees^2): reject collinear / zero-area rings.
    area2 = 0.0
    for index, (lat_a, lon_a) in enumerate(deduped):
        lat_b, lon_b = deduped[(index + 1) % len(deduped)]
        area2 += lon_a * lat_b - lon_b * lat_a
    scale = max(
        1.0,
        max(abs(lat) for lat, _ in deduped),
        max(abs(lon) for _, lon in deduped),
    )
    if abs(area2) <= _EDGE_TOL * scale * scale:
        raise ValidationError("围栏多边形面积为零（顶点共线或重合），无法构成围栏")
    return deduped


def _on_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> bool:
    """True if P lies on segment AB (within a tight tolerance)."""
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    scale = max(abs(bx - ax), abs(by - ay), 1.0)
    tol = _EDGE_TOL * scale * scale
    if abs(cross) > tol:
        return False
    dot = (px - ax) * (bx - ax) + (py - ay) * (by - ay)
    if dot < -tol:
        return False
    seg_len2 = (bx - ax) ** 2 + (by - ay) ** 2
    return dot - seg_len2 <= tol


def contains_point(polygon: list[Vertex], lat: float, lon: float) -> bool:
    """Ray-casting containment; boundary points count as inside."""
    x, y = lon, lat
    inside = False
    count = len(polygon)
    for index in range(count):
        ay, ax = polygon[index]
        by, bx = polygon[(index + 1) % count]
        if _on_segment(x, y, ax, ay, bx, by):
            return True
        # Half-open edge rule: exactly one of the endpoints is strictly above
        # the ray, so vertices on the ray are counted once, not zero or twice.
        if (ay > y) != (by > y):
            x_cross = ax + (y - ay) * (bx - ax) / (by - ay)
            if x_cross > x:
                inside = not inside
    return inside


def collect_samples(
    table: pa.Table,
    vehicle_id: str,
    start: datetime,
    end: datetime,
) -> list[tuple[datetime, float, float]]:
    """In-window samples of one vehicle from one Parquet table.

    The window is inclusive on both ends: ``start <= timestamp <= end``.
    """
    vehicles = table.column("vehicle_id").to_pylist()
    timestamps = table.column("timestamp").to_pylist()
    latitudes = table.column("latitude").to_pylist()
    longitudes = table.column("longitude").to_pylist()
    samples: list[tuple[datetime, float, float]] = []
    for vehicle, moment, lat, lon in zip(
        vehicles, timestamps, latitudes, longitudes, strict=True
    ):
        if vehicle == vehicle_id and start <= moment <= end:
            samples.append((moment, lat, lon))
    return samples


def compute_stays(
    samples: list[tuple[datetime, float, float]],
    polygon: list[Vertex],
) -> list[Stay]:
    """Turn time-sorted samples into maximal in-fence stay intervals.

    ``samples`` must already be sorted by timestamp (a stable sort over the
    snapshot's file order makes cross-file stitching deterministic).
    """
    stays: list[Stay] = []
    entered_at: datetime | None = None
    first_inside_at: datetime | None = None
    last_inside_at: datetime | None = None
    inside_points = 0
    inside_now = False
    seen_any = False
    for moment, lat, lon in samples:
        if contains_point(polygon, lat, lon):
            if not inside_now:
                # The entry was observed only if an earlier in-window sample
                # was outside; otherwise the vehicle was already inside.
                entered_at = moment if seen_any else None
                first_inside_at = moment
                inside_points = 0
                inside_now = True
            last_inside_at = moment
            inside_points += 1
        elif inside_now:
            stays.append(
                Stay(
                    entered_at=entered_at,
                    left_at=moment,
                    first_inside_at=first_inside_at,  # type: ignore[arg-type]
                    last_inside_at=last_inside_at,  # type: ignore[arg-type]
                    inside_points=inside_points,
                )
            )
            inside_now = False
            entered_at = None
        seen_any = True
    if inside_now:
        stays.append(
            Stay(
                entered_at=entered_at,
                left_at=None,
                first_inside_at=first_inside_at,  # type: ignore[arg-type]
                last_inside_at=last_inside_at,  # type: ignore[arg-type]
                inside_points=inside_points,
            )
        )
    return stays
