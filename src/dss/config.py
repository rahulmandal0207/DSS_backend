"""Typed runtime configuration, read from ``DSS_*`` environment variables or a ``.env`` file."""

from __future__ import annotations

import ipaddress
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class DetectionThresholds(BaseSettings):
    """Decision thresholds. Ratios are unitless, angles in degrees, durations in seconds."""

    model_config = SettingsConfigDict(env_prefix="DSS_", env_file=".env", extra="ignore")

    ear_closed: float = Field(default=0.20, gt=0, lt=1)
    mar_yawn: float = Field(default=0.60, gt=0, lt=2)

    eye_closure_critical_s: float = Field(default=1.5, gt=0)
    perclos_window_s: float = Field(default=60.0, gt=0)
    perclos_warning: float = Field(default=0.15, gt=0, le=1)
    perclos_min_coverage: float = Field(default=0.5, gt=0, le=1)

    yawn_min_duration_s: float = Field(default=1.5, gt=0)
    yawn_window_s: float = Field(default=300.0, gt=0)
    yawn_warning_count: int = Field(default=3, ge=1)

    pose_calibration_s: float = Field(default=5.0, ge=0)
    head_pitch_limit_deg: float = Field(default=20.0, gt=0, le=90)
    head_yaw_limit_deg: float = Field(default=30.0, gt=0, le=90)
    head_roll_limit_deg: float = Field(default=25.0, gt=0, le=90)
    head_pose_hold_s: float = Field(default=2.0, gt=0)

    face_lost_hold_s: float = Field(default=2.0, gt=0)

    camera_frozen_hold_s: float = Field(default=3.0, gt=0)
    camera_min_change: float = Field(default=0.05, gt=0)


class Settings(BaseSettings):
    """Service configuration. Unset values fall back to safe local-only defaults."""

    model_config = SettingsConfigDict(env_prefix="DSS_", env_file=".env", extra="ignore")

    video_source: str = "0"
    video_loop: bool = False
    camera_backend: Literal["auto", "dshow", "msmf", "v4l2"] = "auto"
    camera_retry_s: float = Field(default=2.0, gt=0)

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    server_threads: int = Field(default=8, ge=2)
    api_key: str | None = None
    allow_insecure_bind: bool = False
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    stream_enabled: bool = True
    jpeg_quality: int = Field(default=80, ge=10, le=100)

    model_path: Path | None = None
    model_drowsy_probability: float = Field(default=0.7, gt=0, lt=1)
    model_smoothing_s: float = Field(default=3.0, gt=0)

    sound_enabled: bool = False
    alert_history_size: int = Field(default=200, ge=1)

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["text", "json"] = "text"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("api_key", mode="before")
    @classmethod
    def _blank_key_is_none(cls, value: object) -> object:
        return None if value == "" else value

    @model_validator(mode="after")
    def _refuse_unauthenticated_network_bind(self) -> Settings:
        if self.api_key is None and not self.allow_insecure_bind and not self.is_loopback:
            raise ValueError(
                f"Refusing to serve a driver video feed on {self.host!r} without "
                "authentication. Set DSS_API_KEY or bind to 127.0.0.1."
            )
        return self

    @property
    def is_loopback(self) -> bool:
        if self.host == "localhost":
            return True
        try:
            return ipaddress.ip_address(self.host).is_loopback
        except ValueError:
            return False

    @property
    def camera_index(self) -> int | None:
        """The webcam index, or None when ``video_source`` is a file path or URL."""
        return int(self.video_source) if self.video_source.isdigit() else None
