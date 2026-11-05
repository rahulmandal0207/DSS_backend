"""Configuration must fail safely, before anything is exposed."""

import pytest
from pydantic import ValidationError

from dss.cli import build_parser
from dss.config import DetectionThresholds, Settings


def make(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type,call-arg]


def test_defaults_are_local_only() -> None:
    settings = make()
    assert settings.host == "127.0.0.1"
    assert settings.api_key is None
    assert settings.cors_origins == []


def test_refuses_network_bind_without_a_key() -> None:
    with pytest.raises(ValidationError, match="Refusing to serve"):
        make(host="0.0.0.0")


def test_network_bind_is_allowed_with_a_key() -> None:
    assert make(host="0.0.0.0", api_key="secret").host == "0.0.0.0"


def test_blank_key_counts_as_no_key() -> None:
    with pytest.raises(ValidationError, match="Refusing to serve"):
        make(host="0.0.0.0", api_key="")


def test_cors_origins_can_be_a_comma_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DSS_CORS_ORIGINS", "https://a.example, https://b.example")
    assert make().cors_origins == ["https://a.example", "https://b.example"]


def test_camera_index_and_file_sources() -> None:
    assert make(video_source="2").camera_index == 2
    assert make(video_source="drive.mp4").camera_index is None


def test_invalid_threshold_is_rejected() -> None:
    with pytest.raises(ValidationError):
        DetectionThresholds(_env_file=None, ear_closed=-1)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "argv",
    [
        ["serve"],
        ["monitor"],
        ["extract", "--input", "in", "--output", "out.csv"],
        ["train", "--data", "d.csv", "--output", "m.joblib"],
    ],
)
def test_cli_commands_parse(argv: list[str]) -> None:
    assert build_parser().parse_args(argv).command == argv[0]
