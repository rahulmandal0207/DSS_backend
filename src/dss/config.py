"""Typed runtime configuration, read from ``DSS_*`` environment variables or a ``.env`` file.

This uses only the standard library. Windows Smart App Control blocks unsigned compiled
libraries, and a settings loader is the last place DSS can afford a dependency that might
refuse to load.
"""

from __future__ import annotations

import ipaddress
import os
import types
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, NamedTuple, Union, get_args, get_origin, get_type_hints

PREFIX = "DSS_"
ENV_FILE = Path(".env")

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


class ConfigError(ValueError):
    """Every problem found in the configuration, so they can all be fixed in one go."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


def env_name(field_name: str) -> str:
    return PREFIX + field_name.upper()


def _bounded(
    default: Any,
    *,
    gt: float | None = None,
    ge: float | None = None,
    lt: float | None = None,
    le: float | None = None,
) -> Any:
    return field(default=default, metadata={"bounds": (gt, ge, lt, le)})


def _problems(instance: object) -> list[str]:
    """Range and choice problems with an already-typed instance."""
    problems = []
    hints = get_type_hints(type(instance))
    for item in fields(instance):  # type: ignore[arg-type]
        value = getattr(instance, item.name)
        name = env_name(item.name)

        choices = get_args(hints[item.name]) if get_origin(hints[item.name]) is Literal else ()
        if choices and value not in choices:
            problems.append(f"{name} must be one of {', '.join(choices)}, got {value!r}")

        gt, ge, lt, le = item.metadata.get("bounds", (None, None, None, None))
        if gt is not None and not value > gt:
            problems.append(f"{name} must be greater than {gt}, got {value}")
        if ge is not None and not value >= ge:
            problems.append(f"{name} must be at least {ge}, got {value}")
        if lt is not None and not value < lt:
            problems.append(f"{name} must be less than {lt}, got {value}")
        if le is not None and not value <= le:
            problems.append(f"{name} must be at most {le}, got {value}")
    return problems


@dataclass(frozen=True)
class DetectionThresholds:
    """Decision thresholds. Ratios are unitless, angles in degrees, durations in seconds."""

    ear_closed: float = _bounded(0.20, gt=0, lt=1)
    mar_yawn: float = _bounded(0.60, gt=0, lt=2)

    eye_closure_critical_s: float = _bounded(1.5, gt=0)
    perclos_window_s: float = _bounded(60.0, gt=0)
    perclos_warning: float = _bounded(0.15, gt=0, le=1)
    perclos_min_coverage: float = _bounded(0.5, gt=0, le=1)

    yawn_min_duration_s: float = _bounded(1.5, gt=0)
    yawn_window_s: float = _bounded(300.0, gt=0)
    yawn_warning_count: int = _bounded(3, ge=1)

    pose_calibration_s: float = _bounded(5.0, ge=0)
    head_pitch_limit_deg: float = _bounded(20.0, gt=0, le=90)
    head_yaw_limit_deg: float = _bounded(30.0, gt=0, le=90)
    head_roll_limit_deg: float = _bounded(25.0, gt=0, le=90)
    head_pose_hold_s: float = _bounded(2.0, gt=0)

    face_lost_hold_s: float = _bounded(2.0, gt=0)

    camera_frozen_hold_s: float = _bounded(3.0, gt=0)
    camera_min_change: float = _bounded(0.05, gt=0)

    def __post_init__(self) -> None:
        problems = _problems(self)
        if problems:
            raise ConfigError(problems)


@dataclass(frozen=True)
class Settings:
    """Service configuration. Unset values fall back to safe local-only defaults."""

    video_source: str = "0"
    video_loop: bool = False
    camera_backend: Literal["auto", "dshow", "msmf", "v4l2"] = "auto"
    camera_retry_s: float = _bounded(2.0, gt=0)

    host: str = "127.0.0.1"
    port: int = _bounded(8000, ge=1, le=65535)
    server_threads: int = _bounded(8, ge=2)
    api_key: str | None = None
    allow_insecure_bind: bool = False
    cors_origins: list[str] = field(default_factory=list)
    stream_enabled: bool = True
    jpeg_quality: int = _bounded(80, ge=10, le=100)

    model_path: Path | None = None
    model_drowsy_probability: float = _bounded(0.7, gt=0, lt=1)
    model_smoothing_s: float = _bounded(3.0, gt=0)

    sound_enabled: bool = False
    alert_history_size: int = _bounded(200, ge=1)

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["text", "json"] = "text"

    def __post_init__(self) -> None:
        if self.api_key == "":
            object.__setattr__(self, "api_key", None)

        problems = _problems(self)
        if self.api_key is None and not self.allow_insecure_bind and not self.is_loopback:
            problems.append(
                f"Refusing to serve a driver video feed on {self.host!r} without "
                "authentication. Set DSS_API_KEY or bind to 127.0.0.1."
            )
        if problems:
            raise ConfigError(problems)

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


class LoadedConfig(NamedTuple):
    settings: Settings
    thresholds: DetectionThresholds
    unrecognised: tuple[str, ...]
    """``DSS_*`` names that match no setting, which are probably typos."""


def read_env_file(path: Path) -> dict[str, str]:
    """Parse ``KEY=value`` lines. Blank lines and ``#`` comments are skipped."""
    values = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").lstrip()
        key, separator, value = line.partition("=")
        if not separator or not key.strip():
            raise ConfigError([f"{path}, line {number}: expected KEY=value"])

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        values[key.strip()] = value
    return values


def _convert(raw: str, hint: Any) -> object:
    """Turn an environment string into ``hint``'s type, raising ValueError with a reason."""
    origin = get_origin(hint)
    if origin in (Union, types.UnionType):
        if not raw.strip():
            return None
        (inner,) = (arg for arg in get_args(hint) if arg is not type(None))
        return _convert(raw, inner)
    if origin is Literal:
        # Matched without regard to case, so DSS_LOG_LEVEL=debug works.
        for choice in get_args(hint):
            if raw.strip().lower() == str(choice).lower():
                return choice
        return raw.strip()
    if origin is list:
        return [item.strip() for item in raw.split(",") if item.strip()]
    if hint is bool:
        word = raw.strip().lower()
        if word in _TRUE | _FALSE:
            return word in _TRUE
        raise ValueError("must be true or false")
    if hint is int:
        try:
            return int(raw)
        except ValueError:
            raise ValueError("must be a whole number") from None
    if hint is float:
        try:
            return float(raw)
        except ValueError:
            raise ValueError("must be a number") from None
    if hint is Path:
        return Path(raw.strip())
    return raw


def _build(cls: type[Any], values: Mapping[str, str], problems: list[str]) -> Any:
    hints = get_type_hints(cls)
    arguments = {}
    for item in fields(cls):
        name = env_name(item.name)
        if name not in values:
            continue
        try:
            arguments[item.name] = _convert(values[name], hints[item.name])
        except ValueError as error:
            problems.append(f"{name} {error}, got {values[name]!r}")
    try:
        return cls(**arguments)
    except ConfigError as error:
        problems.extend(error.problems)
        return None


def load_config(
    environ: Mapping[str, str] | None = None, env_file: Path | None = ENV_FILE
) -> LoadedConfig:
    """Read settings from ``env_file``, then the environment, which takes precedence.

    Every problem is collected before raising, so a broken configuration is fixed in one pass
    rather than one error at a time.
    """
    environ = os.environ if environ is None else environ
    values: dict[str, str] = {}
    if env_file is not None and env_file.is_file():
        values.update(read_env_file(env_file))
    values.update(environ)
    values = {key.upper(): value for key, value in values.items() if key.upper().startswith(PREFIX)}

    problems: list[str] = []
    settings = _build(Settings, values, problems)
    thresholds = _build(DetectionThresholds, values, problems)
    if problems:
        raise ConfigError(problems)

    known = {env_name(item.name) for cls in (Settings, DetectionThresholds) for item in fields(cls)}
    return LoadedConfig(settings, thresholds, tuple(sorted(set(values) - known)))
