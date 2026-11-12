"""Configuration must fail safely, before anything is exposed."""

from pathlib import Path

import pytest

from dss.cli import build_parser
from dss.config import ConfigError, DetectionThresholds, Settings, load_config, read_env_file


def load(env: dict[str, str], env_file: Path | None = None) -> tuple[Settings, DetectionThresholds]:
    config = load_config(environ=env, env_file=env_file)
    return config.settings, config.thresholds


def test_defaults_are_local_only() -> None:
    settings, _ = load({})
    assert settings.host == "127.0.0.1"
    assert settings.api_key is None
    assert settings.cors_origins == []


def test_refuses_network_bind_without_a_key() -> None:
    with pytest.raises(ConfigError, match="Refusing to serve"):
        Settings(host="0.0.0.0")


def test_network_bind_is_allowed_with_a_key() -> None:
    assert Settings(host="0.0.0.0", api_key="secret").host == "0.0.0.0"


def test_blank_key_counts_as_no_key() -> None:
    with pytest.raises(ConfigError, match="Refusing to serve"):
        load({"DSS_HOST": "0.0.0.0", "DSS_API_KEY": ""})


def test_insecure_bind_can_be_allowed_explicitly() -> None:
    settings, _ = load({"DSS_HOST": "0.0.0.0", "DSS_ALLOW_INSECURE_BIND": "true"})
    assert settings.allow_insecure_bind


def test_cors_origins_can_be_a_comma_list() -> None:
    settings, _ = load({"DSS_CORS_ORIGINS": "https://a.example, https://b.example,"})
    assert settings.cors_origins == ["https://a.example", "https://b.example"]


def test_camera_index_and_file_sources() -> None:
    assert Settings(video_source="2").camera_index == 2
    assert Settings(video_source="drive.mp4").camera_index is None


def test_unknown_camera_backend_is_rejected() -> None:
    with pytest.raises(ConfigError, match="DSS_CAMERA_BACKEND must be one of auto, dshow"):
        Settings(camera_backend="webcam")  # type: ignore[arg-type]


def test_invalid_threshold_is_rejected() -> None:
    with pytest.raises(ConfigError, match="DSS_EAR_CLOSED must be greater than 0"):
        DetectionThresholds(ear_closed=-1)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("true", True), ("1", True), ("Yes", True), ("ON", True), ("false", False), ("0", False)],
)
def test_booleans(raw: str, expected: bool) -> None:
    settings, _ = load({"DSS_VIDEO_LOOP": raw})
    assert settings.video_loop is expected


def test_every_type_is_converted() -> None:
    settings, thresholds = load(
        {
            "DSS_PORT": "9000",
            "DSS_CAMERA_RETRY_S": "0.5",
            "DSS_MODEL_PATH": "models/m.joblib",
            "DSS_LOG_LEVEL": "debug",
            "DSS_CAMERA_BACKEND": "DShow",
            "DSS_YAWN_WARNING_COUNT": "4",
            "DSS_EAR_CLOSED": "0.18",
        }
    )
    assert settings.port == 9000
    assert settings.camera_retry_s == 0.5
    assert settings.model_path == Path("models/m.joblib")
    assert settings.log_level == "DEBUG"
    assert settings.camera_backend == "dshow"
    assert thresholds.yawn_warning_count == 4
    assert thresholds.ear_closed == 0.18


def test_blank_optional_values_mean_unset() -> None:
    settings, _ = load({"DSS_MODEL_PATH": "", "DSS_API_KEY": "  "})
    assert settings.model_path is None


def test_every_problem_is_reported_at_once() -> None:
    with pytest.raises(ConfigError) as caught:
        load(
            {
                "DSS_PORT": "eighty",
                "DSS_VIDEO_LOOP": "maybe",
                "DSS_JPEG_QUALITY": "500",
                "DSS_EAR_CLOSED": "x",
                "DSS_HEAD_YAW_LIMIT_DEG": "120",
            }
        )
    problems = caught.value.problems
    assert "DSS_PORT must be a whole number, got 'eighty'" in problems
    assert "DSS_VIDEO_LOOP must be true or false, got 'maybe'" in problems
    assert "DSS_JPEG_QUALITY must be at most 100, got 500" in problems
    assert "DSS_EAR_CLOSED must be a number, got 'x'" in problems
    assert "DSS_HEAD_YAW_LIMIT_DEG must be at most 90, got 120.0" in problems


def test_env_file_is_read(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# webcam settings\n"
        "\n"
        "DSS_VIDEO_SOURCE=1\n"
        "export DSS_CAMERA_BACKEND=dshow\n"
        'DSS_HOST="127.0.0.1"\n'
        "DSS_PORT=8001  # a comment\n"
        "OTHER_TOOL=ignored\n",
        encoding="utf-8",
    )
    settings, _ = load({}, env_file)
    assert settings.video_source == "1"
    assert settings.camera_backend == "dshow"
    assert settings.port == 8001


def test_environment_overrides_the_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("DSS_PORT=8001\nDSS_VIDEO_SOURCE=1\n", encoding="utf-8")
    settings, _ = load({"DSS_PORT": "8002"}, env_file)
    assert settings.port == 8002
    assert settings.video_source == "1"


def test_env_file_written_by_notepad_with_a_bom(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("DSS_PORT=8003\n", encoding="utf-8-sig")
    assert load({}, env_file)[0].port == 8003


def test_missing_env_file_is_fine(tmp_path: Path) -> None:
    assert load({}, tmp_path / "missing.env")[0].port == 8000


def test_malformed_env_file_line_is_reported(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("DSS_PORT=8000\nnot a setting\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="line 2: expected KEY=value"):
        read_env_file(env_file)


def test_names_are_not_case_sensitive() -> None:
    assert load({"dss_port": "8004"})[0].port == 8004


def test_typos_are_reported_not_ignored() -> None:
    config = load_config(environ={"DSS_EAR_CLOSE": "0.1", "PATH": "x"}, env_file=None)
    assert config.unrecognised == ("DSS_EAR_CLOSE",)


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
