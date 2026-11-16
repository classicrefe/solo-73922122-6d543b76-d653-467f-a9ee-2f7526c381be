"""API request/response models (also used by the generated OpenAPI docs)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import AwareDatetime, BaseModel, Field, field_validator, model_validator


class FileInfo(BaseModel):
    file_id: str = Field(description="不可变 Parquet 文件 ID")
    seq: int = Field(description="登记顺序号，用于稳定的现行清单排序")
    kind: Literal["batch", "merged"] = Field(description="原始上传批次或合并结果")
    rows: int = Field(description="文件内数据行数（重复行按原样保留）")
    bytes: int = Field(description="Parquet 文件字节数")
    sha256: str = Field(description="Parquet 文件内容的 SHA-256")
    created_at: str = Field(description="UTC 登记时间（ISO 8601）")
    source_batch_id: str | None = Field(default=None, description="上传时使用的批号")
    merge_id: str | None = Field(default=None, description="生成该文件时的合并作业 ID")


class LiveFilesResponse(BaseModel):
    files: list[FileInfo]
    file_count: int
    row_count: int


class BatchUploadResponse(BaseModel):
    batch_id: str = Field(description="本次上传的批号")
    file: FileInfo
    message: str = "批次已登记为现行文件"


class MergeResponse(BaseModel):
    merge_id: str
    output_file: FileInfo
    input_file_ids: list[str] = Field(description="本次合并开始时快照到的现行文件")
    input_count: int
    row_count: int
    committed_at: str


class ManifestFile(BaseModel):
    ordinal: int
    file_id: str
    rows: int
    bytes: int
    sha256: str
    download_path: str = Field(description="相对 API 根路径的逐件下载地址")


class ManifestResponse(BaseModel):
    manifest_id: str
    label: str | None
    state: Literal["open", "released"]
    created_at: str
    released_at: str | None
    file_count: int
    row_count: int
    bytes: int
    files: list[ManifestFile]
    message: str | None = None


class ClaimRequest(BaseModel):
    label: str | None = Field(
        default=None,
        max_length=200,
        description="可选的作业标签，仅用于识别，不参与一致性判断",
    )


class ReleaseResponse(BaseModel):
    manifest_id: str
    state: Literal["released"]
    released_at: str
    already_released: bool
    message: str


class StatsResponse(BaseModel):
    live_files: int
    live_rows: int
    retired_files: int
    recycled_files: int
    open_manifests: int
    released_manifests: int
    total_files: int


class HealthResponse(BaseModel):
    status: str
    instance_id: str
    root: str


# ------------------------------------------------------------------ geofence
class GeoPoint(BaseModel):
    lat: float = Field(ge=-90.0, le=90.0, description="纬度（WGS-84）")
    lon: float = Field(ge=-180.0, le=180.0, description="经度（WGS-84）")


class GeofenceEventsRequest(BaseModel):
    """One geofence query: fence polygon + vehicle + inclusive UTC window."""

    vehicle_id: str = Field(
        min_length=1, max_length=256, description="要查询的车辆号（精确匹配）"
    )
    start: AwareDatetime = Field(description="时间窗起点（UTC，含边界）")
    end: AwareDatetime = Field(description="时间窗终点（UTC，含边界）")
    polygon: list[GeoPoint] = Field(
        description=(
            "闭合围栏多边形的顶点（按顺序）。首尾顶点相同时会自动去重；"
            "边界上的点按在围栏内处理"
        ),
    )

    @field_validator("start", "end")
    @classmethod
    def _normalize_to_utc(cls, value: datetime) -> datetime:
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _check_window(self) -> GeofenceEventsRequest:
        if self.end < self.start:
            raise ValueError("end 必须不早于 start")
        return self


class GeofenceEvent(BaseModel):
    entered_at: str | None = Field(
        description="观测到进入的时刻（该区间首个围栏内点）；null 表示时间窗开始时车辆已在围栏内"
    )
    left_at: str | None = Field(
        description="观测到离开的时刻（该区间之后首个围栏外点）；null 表示时间窗结束时车辆仍在围栏内"
    )
    first_inside_at: str = Field(description="该区间首个围栏内观测点")
    last_inside_at: str = Field(description="该区间最后一个围栏内观测点")
    inside_points: int = Field(description="该区间内的围栏内观测点数")
    closed: bool = Field(
        description="进入与离开都被观测到；false 即为未闭合区间"
    )


class GeofenceEventsResponse(BaseModel):
    vehicle_id: str
    start: str = Field(description="实际使用的查询窗口起点（UTC）")
    end: str = Field(description="实际使用的查询窗口终点（UTC）")
    file_ids: list[str] = Field(
        description="本次查询读取的一致性快照文件 ID（内部领取的固定清单，查询结束已释放）"
    )
    snapshot_file_count: int = Field(description="快照中的文件件数")
    matched_points: int = Field(description="窗口内该车辆的观测点总数")
    inside_points: int = Field(description="窗口内该车辆位于围栏内的观测点数")
    events: list[GeofenceEvent] = Field(
        description="按时间顺序的围栏内停留区间；连续停留只生成一个区间"
    )
    unclosed_count: int = Field(description="未闭合区间（closed=false）的数量")
