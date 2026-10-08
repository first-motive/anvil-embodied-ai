"""Behaviour tests for the per-episode MCAP recorder."""

from __future__ import annotations

import dataclasses
import signal
import subprocess
from pathlib import Path

import pytest
import yaml
from classical_control.recorder import (
    BYTES_PER_GB,
    EpisodeRecorder,
    RecordingConfig,
    free_bytes,
    load_recording_config,
    record_command,
)

SHIPPED_CONFIG = Path(__file__).resolve().parent.parent / "config" / "collect.yaml"

VALID = {
    "recording": {
        "topics": ["/joint_states", "/tf"],
        "topic_regex": "^/gripper/tactile/.*",
        "storage": "mcap",
        "stop_timeout_s": 3.0,
        "settle_s": 0.5,
    },
    "disk_guard": {"min_free_gb": 2},
    "loop": {"cycles": 5},  # another reader's section; must be ignored
}


class FakeProcess:
    """Stands in for the `ros2 bag record` child process.

    Args:
        argv: The command it was started with.
        exits_on_sigint: Whether SIGINT makes it exit within the wait timeout.
        exited_with: Exit code if it has already died before stop().
    """

    def __init__(self, argv, exits_on_sigint=True, exited_with=None, **kwargs):
        self.argv = argv
        self.kwargs = kwargs
        self.exits_on_sigint = exits_on_sigint
        self.returncode = exited_with
        self.signals: list[int] = []
        self.wait_timeouts: list[float | None] = []
        self.killed = False

    def poll(self):
        return self.returncode

    def send_signal(self, sig):
        self.signals.append(sig)
        if sig == signal.SIGINT and self.exits_on_sigint:
            self.returncode = 0

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        if self.returncode is None:
            raise subprocess.TimeoutExpired(self.argv, timeout)
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -signal.SIGKILL


class FakeFactory:
    """Records every process the recorder launches."""

    def __init__(self, **process_kwargs):
        self.process_kwargs = process_kwargs
        self.launched: list[FakeProcess] = []

    def __call__(self, argv, **kwargs):
        process = FakeProcess(argv, **self.process_kwargs, **kwargs)
        self.launched.append(process)
        return process


def write_config(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "collect.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.fixture
def config(tmp_path) -> RecordingConfig:
    return load_recording_config(write_config(tmp_path, VALID))


# --- Config ---


def test_shipped_config_records_loader_and_tactile_topics():
    config = load_recording_config(SHIPPED_CONFIG)
    assert "/joint_states" in config.topics
    assert "/cam_wrist_r/image_raw/compressed" in config.topics
    assert "/tf_static" in config.topics
    assert len(config.topics) == 12
    assert config.topic_regex == "^/gripper/tactile/.*"
    assert config.storage == "mcap"
    assert config.stop_timeout_s > 0.0
    assert config.min_free_bytes == 50 * BYTES_PER_GB


def test_config_loads_from_file_and_ignores_other_sections(config):
    assert config.topics == ("/joint_states", "/tf")
    assert config.stop_timeout_s == 3.0
    assert config.settle_s == 0.5
    assert config.min_free_bytes == 2 * BYTES_PER_GB


def test_regex_alone_is_enough(tmp_path):
    data = {**VALID, "recording": {**VALID["recording"], "topics": []}}
    assert load_recording_config(write_config(tmp_path, data)).topics == ()


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("recording", "stop_timeout_s", 0.0),
        ("recording", "stop_timeout_s", -1.0),
        ("recording", "settle_s", -0.1),
        ("recording", "storage", ""),
        ("recording", "topics", ["joint_states"]),
        ("recording", "stop_timout_s", 3.0),
        ("disk_guard", "min_free_gb", -1),
    ],
)
def test_invalid_config_raises(tmp_path, section, key, value):
    data = {**VALID, section: {**VALID[section], key: value}}
    with pytest.raises(ValueError):
        load_recording_config(write_config(tmp_path, data))


def test_config_without_topics_or_regex_raises(tmp_path):
    data = {**VALID, "recording": {**VALID["recording"], "topics": [], "topic_regex": ""}}
    with pytest.raises(ValueError, match="topic"):
        load_recording_config(write_config(tmp_path, data))


def test_config_missing_a_section_raises(tmp_path):
    with pytest.raises(ValueError, match="disk_guard"):
        load_recording_config(write_config(tmp_path, {"recording": VALID["recording"]}))


# --- Command ---


def test_command_records_every_topic_and_the_regex_as_mcap(config, tmp_path):
    out_dir = tmp_path / "run" / "0001"
    command = record_command(config, out_dir)

    assert command[:3] == ["ros2", "bag", "record"]
    assert command[command.index("-s") + 1] == "mcap"
    assert command[command.index("-o") + 1] == str(out_dir)
    topics_at = command.index("--topics")
    assert command[topics_at + 1 : topics_at + 3] == ["/joint_states", "/tf"]
    assert "--regex=^/gripper/tactile/.*" in command


def test_command_omits_empty_regex(config, tmp_path):
    no_regex = dataclasses.replace(config, topic_regex="")
    assert not any(arg.startswith("--regex") for arg in record_command(no_regex, tmp_path / "0001"))


# --- Recorder ---


def test_disabled_recorder_creates_episode_dir_and_launches_nothing(config, tmp_path):
    factory = FakeFactory()
    recorder = EpisodeRecorder(config, enabled=False, process_factory=factory)
    out_dir = tmp_path / "run" / "0001"

    recorder.start(out_dir)

    assert out_dir.is_dir()
    assert factory.launched == []
    assert not recorder.running
    assert recorder.stop() is None


def test_enabled_recorder_launches_bag_in_own_session(config, tmp_path):
    factory = FakeFactory()
    recorder = EpisodeRecorder(config, process_factory=factory)
    out_dir = tmp_path / "run" / "0001"

    recorder.start(out_dir)

    (process,) = factory.launched
    assert process.argv == record_command(config, out_dir)
    assert process.kwargs == {"start_new_session": True}
    # ros2 bag creates the episode dir itself and refuses one that exists.
    assert not out_dir.exists()
    assert out_dir.parent.is_dir()
    assert recorder.running


def test_enabled_recorder_refuses_existing_dir(config, tmp_path):
    factory = FakeFactory()
    recorder = EpisodeRecorder(config, process_factory=factory)
    out_dir = tmp_path / "0001"
    out_dir.mkdir()

    with pytest.raises(FileExistsError):
        recorder.start(out_dir)
    assert factory.launched == []


def test_second_start_without_stop_raises(config, tmp_path):
    recorder = EpisodeRecorder(config, process_factory=FakeFactory())
    recorder.start(tmp_path / "0001")
    with pytest.raises(RuntimeError):
        recorder.start(tmp_path / "0002")


def test_stop_sends_sigint_and_waits_for_clean_exit(config, tmp_path):
    factory = FakeFactory()
    recorder = EpisodeRecorder(config, process_factory=factory)
    recorder.start(tmp_path / "0001")

    result = recorder.stop()

    (process,) = factory.launched
    assert process.signals == [signal.SIGINT]
    assert process.wait_timeouts == [config.stop_timeout_s]
    assert not process.killed
    assert result.clean
    assert not recorder.running


def test_stop_kills_recorder_that_ignores_sigint(config, tmp_path):
    factory = FakeFactory(exits_on_sigint=False)
    recorder = EpisodeRecorder(config, process_factory=factory)
    recorder.start(tmp_path / "0001")

    result = recorder.stop()

    (process,) = factory.launched
    assert process.signals == [signal.SIGINT]
    assert process.killed
    assert result.killed
    assert not result.clean


def test_recorder_that_died_early_is_reported(config, tmp_path):
    factory = FakeFactory(exited_with=1)
    recorder = EpisodeRecorder(config, process_factory=factory)
    recorder.start(tmp_path / "0001")

    assert not recorder.running
    result = recorder.stop()

    assert result.died_early
    assert result.returncode == 1
    assert not result.clean
    assert factory.launched[0].signals == []


def test_double_stop_is_safe(config, tmp_path):
    factory = FakeFactory()
    recorder = EpisodeRecorder(config, process_factory=factory)
    recorder.start(tmp_path / "0001")

    assert recorder.stop() is not None
    assert recorder.stop() is None
    assert factory.launched[0].signals == [signal.SIGINT]


def test_stop_without_start_is_noop(config):
    assert EpisodeRecorder(config, process_factory=FakeFactory()).stop() is None


def test_recorder_starts_next_episode_after_stop(config, tmp_path):
    factory = FakeFactory()
    recorder = EpisodeRecorder(config, process_factory=factory)
    recorder.start(tmp_path / "0001")
    recorder.stop()
    recorder.start(tmp_path / "0002")
    assert len(factory.launched) == 2


# --- Disk ---


def test_free_bytes_reports_free_space(tmp_path):
    free = free_bytes(tmp_path)
    assert isinstance(free, int)
    assert free > 0


def test_recorder_that_exits_as_it_is_signalled_counts_as_died_early(config, tmp_path):
    class ExitsBeforeSignal(FakeProcess):
        def send_signal(self, _sig):
            self.returncode = 1
            raise ProcessLookupError

    recorder = EpisodeRecorder(config, process_factory=lambda argv, **_: ExitsBeforeSignal(argv))
    recorder.start(tmp_path / "0001")
    result = recorder.stop()

    assert result.died_early
    assert result.returncode == 1


def test_bare_storage_key_is_refused(tmp_path):
    data = {**VALID, "recording": {**VALID["recording"], "storage": None}}
    path = tmp_path / "collect.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        load_recording_config(path)
