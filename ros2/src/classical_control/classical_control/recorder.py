"""Per-episode MCAP recording for the unattended collection loop.

Each pick-and-place cycle is recorded as its own episode by running
`ros2 bag record` as a child process around the cycle. A subprocess, rather
than rosbag2_py inside the node, keeps a crashing or slow writer out of the
node that commands the arm, and `ros2 bag record` is exactly what produced the
loader's episodes, so collected bags share their layout.

Three facts about `ros2 bag record` (Jazzy) shape this module:

- `-o <run>/0001` writes `<run>/0001/0001_0.mcap` plus `metadata.yaml`, and it
  refuses to start if that directory already exists. The recorder therefore
  never creates the episode directory in record mode, and checks for it first
  so the failure is a clear exception instead of a child that exits at once.
- The MCAP summary and index are written on shutdown. The recorder is stopped
  with SIGINT and given time to finish; only after that does it get SIGKILL.
- A recorder that cannot subscribe or open storage exits on its own. The node
  must treat that as a failed episode, so `running` and `stop()` report it.

There are no ROS imports here: the process is injected, so the whole lifecycle
is unit-tested off the robot.
"""

from __future__ import annotations

import shutil
import signal
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

#: Bytes in one gigabyte as `min_free_gb` counts them (decimal, like `df -H`).
BYTES_PER_GB = 1_000_000_000


@dataclass(frozen=True)
class RecordingConfig:
    """What one episode records and how its recorder is started and stopped.

    Attributes:
        topics: Topics recorded by exact name.
        topic_regex: Pattern for further topics, or "" for none.
        storage: rosbag2 storage plugin id, e.g. "mcap".
        stop_timeout_s: Time allowed after SIGINT before the recorder is killed.
        settle_s: Wait after start before the cycle begins, for discovery.
        min_free_bytes: Free disk below which the run stops before a new cycle.
    """

    topics: tuple[str, ...]
    topic_regex: str
    storage: str
    stop_timeout_s: float
    settle_s: float
    min_free_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "topics", tuple(self.topics))
        if not self.topics and not self.topic_regex:
            raise ValueError("recording needs at least one topic or a topic_regex")
        if any(not topic.startswith("/") for topic in self.topics):
            raise ValueError("recorded topics must be absolute names starting with '/'")
        if not self.storage:
            raise ValueError("storage must name a rosbag2 storage plugin")
        if not self.stop_timeout_s > 0.0:
            raise ValueError("stop_timeout_s must be positive")
        if not self.settle_s >= 0.0:
            raise ValueError("settle_s must not be negative")
        if self.min_free_bytes < 0:
            raise ValueError("min_free_gb must not be negative")


_RECORDING_KEYS = {"topics", "topic_regex", "storage", "stop_timeout_s", "settle_s"}


def load_recording_config(path: str | Path) -> RecordingConfig:
    """Load the `recording` and `disk_guard` sections of a collect config.

    Other top-level sections (the loop's own settings) are left to their
    readers; unknown keys inside these two sections are an error, so a typo
    cannot silently fall back to a default.

    Args:
        path: Path to a file shaped like ``config/collect.yaml``.

    Returns:
        The validated recording config.

    Raises:
        ValueError: If a section is missing, has unknown keys, or holds an invalid value.
    """
    with open(Path(path).expanduser(), encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    recording = _section(data, "recording")
    disk_guard = _section(data, "disk_guard")

    unknown = set(recording) - _RECORDING_KEYS
    if unknown:
        raise ValueError(f"unknown recording keys: {sorted(unknown)}")
    unknown = set(disk_guard) - {"min_free_gb"}
    if unknown:
        raise ValueError(f"unknown disk_guard keys: {sorted(unknown)}")

    return RecordingConfig(
        topics=tuple(str(topic) for topic in recording.get("topics") or ()),
        topic_regex=str(recording.get("topic_regex") or ""),
        storage=str(recording.get("storage") or ""),
        stop_timeout_s=float(recording["stop_timeout_s"]),
        settle_s=float(recording["settle_s"]),
        min_free_bytes=int(float(disk_guard["min_free_gb"]) * BYTES_PER_GB),
    )


def _section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = data.get(name)
    if not isinstance(section, Mapping):
        raise ValueError(f"collect config needs a '{name}' section")
    return section


def free_bytes(path: str | Path) -> int:
    """Return the free bytes on the filesystem holding `path`."""
    return shutil.disk_usage(path).free


def record_command(config: RecordingConfig, out_dir: str | Path) -> list[str]:
    """Build the `ros2 bag record` argv for one episode.

    In Jazzy `--topics` takes a list and `-e/--regex` a pattern, and a topic is
    recorded if it matches either.

    Args:
        config: What to record.
        out_dir: Episode directory; must not exist yet.

    Returns:
        The argv, ready for subprocess.
    """
    command = ["ros2", "bag", "record", "-s", config.storage, "-o", str(out_dir)]
    if config.topics:
        command += ["--topics", *config.topics]
    if config.topic_regex:
        # One argv element, so a pattern that starts with "-" is never read as an option.
        command.append(f"--regex={config.topic_regex}")
    return command


class Process(Protocol):
    """The part of subprocess.Popen the recorder uses."""

    def poll(self) -> int | None: ...

    def send_signal(self, sig: int) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def kill(self) -> None: ...


#: Starts a process from an argv; subprocess.Popen in production.
ProcessFactory = Callable[..., Process]


@dataclass(frozen=True)
class StopResult:
    """How an episode's recorder ended.

    Attributes:
        returncode: Exit status of `ros2 bag record`; negative for a signal.
        died_early: The recorder had already exited before stop() was called.
        killed: SIGINT did not stop it within stop_timeout_s, so it was killed.
    """

    returncode: int
    died_early: bool
    killed: bool

    @property
    def clean(self) -> bool:
        """True if the bag was stopped on request and closed normally."""
        return self.returncode == 0 and not self.died_early and not self.killed


class EpisodeRecorder:
    """Starts and stops `ros2 bag record` around one episode at a time.

    When disabled (`--no-record`) it records nothing but still creates the
    episode directory, so the episode's metadata has somewhere to go.
    """

    def __init__(
        self,
        config: RecordingConfig,
        enabled: bool = True,
        process_factory: ProcessFactory = subprocess.Popen,
    ) -> None:
        self._config = config
        self._enabled = enabled
        self._process_factory = process_factory
        self._process: Process | None = None

    @property
    def enabled(self) -> bool:
        """True if episodes are recorded, False in `--no-record` mode."""
        return self._enabled

    @property
    def running(self) -> bool:
        """True while a started recorder is still alive.

        False after the recorder exited on its own, which the node should treat
        as a failed episode before trusting the cycle's data.
        """
        return self._process is not None and self._process.poll() is None

    def start(self, out_dir: str | Path) -> None:
        """Start recording one episode into `out_dir`.

        Args:
            out_dir: Episode directory, e.g. ``<run>/0001``. Must not exist in
                record mode; ros2 bag creates it.

        Raises:
            RuntimeError: If a recording is already in progress.
            FileExistsError: If `out_dir` exists in record mode.
        """
        if self._process is not None:
            raise RuntimeError("a recording is already in progress; stop it first")
        out_dir = Path(out_dir)
        if not self._enabled:
            out_dir.mkdir(parents=True, exist_ok=True)
            return
        if out_dir.exists():
            raise FileExistsError(f"episode directory already exists: {out_dir}")
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        # tradeoff: a new session keeps the recorder out of the node's process
        # group, so a Ctrl-C at the terminal reaches only the node, which then
        # stops the bag once through stop(). Without it the bag gets SIGINT
        # twice (terminal, then stop()), and rosbag2 can abort its shutdown on
        # the second. The cost: if the node is SIGKILLed the recorder is
        # orphaned and keeps recording until killed by hand or by the container.
        self._process = self._process_factory(
            record_command(self._config, out_dir), start_new_session=True
        )

    def stop(self) -> StopResult | None:
        """Stop the current recording, letting rosbag2 close the bag.

        Sends SIGINT, waits up to `stop_timeout_s`, then kills. Safe to call
        repeatedly and when nothing was started.

        Returns:
            How the recorder ended, or None if no recording was running
            (disabled mode, never started, or already stopped).
        """
        process = self._process
        if process is None:
            return None
        self._process = None

        returncode = process.poll()
        if returncode is not None:
            return StopResult(returncode=returncode, died_early=True, killed=False)

        try:
            process.send_signal(signal.SIGINT)
        except ProcessLookupError:
            # Exited between poll() and the signal: as early as a death before stop().
            return StopResult(returncode=process.wait(), died_early=True, killed=False)
        try:
            returncode = process.wait(timeout=self._config.stop_timeout_s)
            return StopResult(returncode=returncode, died_early=False, killed=False)
        except subprocess.TimeoutExpired:
            process.kill()
            return StopResult(returncode=process.wait(), died_early=False, killed=True)
