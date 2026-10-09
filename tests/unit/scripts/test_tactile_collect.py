"""Behaviour tests for scripts/run/tactile-collect.sh, the `fm tactile-collect` verb.

The script runs against a throwaway checkout: a copy of itself, a stub run_classical.sh
and a collect.yaml, with stub docker, df and timeout on PATH. The stubs stand in for the
robot (system boundary), so every refusal, the JSON envelope and the exit-code contract
are checked without docker or ROS.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "run" / "tactile-collect.sh"

DOCKER = """#!/bin/sh
case "$1" in
  inspect)
    if [ "$4" = classical-control ]; then echo "$NODES_UP"
    elif [ -f "$LOOP_FLAG" ]; then echo true
    else echo "$LOOP_UP"; fi ;;
  exec) echo "$ROS_VERDICT" ;;
esac
"""
DF = """#!/bin/sh
echo "Filesystem 1-blocks Used Available Capacity Mounted"
echo "x 1 1 $FREE_GB 1% /"
"""
TIMEOUT = """#!/bin/sh
shift
exec "$@"
"""
RUN_CLASSICAL = """#!/bin/sh
echo "$@" >> "$(dirname "$0")/run_classical.calls"
[ -n "$STARTS_LOOP" ] && touch "$LOOP_FLAG"
exit "${RUN_CLASSICAL_RC:-0}"
"""


def executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """A fake robot checkout that passes every precondition until a test breaks one."""
    root = tmp_path / "anvil-embodied-ai"
    (root / "scripts" / "run").mkdir(parents=True)
    shutil.copy(SCRIPT, root / "scripts" / "run" / SCRIPT.name)
    executable(root / "scripts" / "run_classical.sh", RUN_CLASSICAL)
    config = root / "ros2" / "src" / "classical_control" / "config"
    config.mkdir(parents=True)
    (config / "collect.yaml").write_text("disk_guard:\n  min_free_gb: 50\n")
    collect = root / "data" / "classical" / "collect"
    collect.mkdir(parents=True)
    (collect / ".g2_passed").write_text("2026-10-08 20261008T120000Z\n")
    (root / "data" / "classical" / "chest_background.png").write_bytes(b"png")
    bin_dir = tmp_path / "bin"
    executable(bin_dir / "docker", DOCKER)
    executable(bin_dir / "df", DF)
    executable(bin_dir / "timeout", TIMEOUT)
    return root


def run(checkout: Path, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
    environment = {
        **os.environ,
        "PATH": f"{checkout.parent / 'bin'}:{os.environ['PATH']}",
        "NODES_UP": "true",
        "LOOP_UP": "false",
        "ROS_VERDICT": "tactile 4",
        "FREE_GB": "120",
        "LOOP_FLAG": str(checkout.parent / "loop.up"),
        "STARTS_LOOP": "",
        **env,
    }
    script = checkout / "scripts" / "run" / SCRIPT.name
    return subprocess.run(
        ["bash", str(script), *args], env=environment, capture_output=True, text=True, timeout=30
    )


def refusal(result: subprocess.CompletedProcess[str]) -> str:
    return json.loads(result.stdout)["error"]["code"]


@pytest.mark.parametrize(
    "args",
    [
        ["bogus"],
        ["start"],
        ["start", "--object", "--detach"],
        ["start", "--object", "../can"],
        ["start", "--object", "can", "--hours", "x"],
        ["start", "--object", "can", "--cycles", "1.5"],
        ["status", "--object", "can"],
        ["status", "--host", "-oProxyCommand=x"],
    ],
)
def test_usage_errors_exit_2(checkout, args):
    assert run(checkout, *args).returncode == 2


def test_help_prints_the_header_and_no_code(checkout):
    result = run(checkout, "--help")
    assert result.returncode == 0
    assert result.stdout.startswith("tactile-collect.sh")
    assert "set -uo" not in result.stdout


def test_status_carries_an_empty_warnings_list(checkout):
    assert json.loads(run(checkout, "status", "--json").stdout)["warnings"] == []


def test_status_reports_the_latest_run(checkout):
    run_dir = checkout / "data" / "classical" / "collect" / "20261008T130000Z"
    for name, status in (("0001", "success"), ("0002", "failure"), ("0003", "success")):
        (run_dir / name).mkdir(parents=True)
        (run_dir / name / "metadata.json").write_text(json.dumps({"status": status}, indent=2))
    (run_dir / "summary.json").write_text(json.dumps({"stop_reason": "operator"}, indent=2))

    result = run(checkout, "status", "--json")

    assert result.returncode == 0
    data = json.loads(result.stdout)["data"]
    assert data == {
        "active": False,
        "run": "20261008T130000Z",
        "cycles": 3,
        "successes": 2,
        "stop_reason": "operator",
        "free_gb": 120,
    }


@pytest.mark.parametrize(
    ("removed", "env", "code"),
    [
        (None, {"LOOP_UP": "true"}, "already_running"),
        (None, {"NODES_UP": "false"}, "nodes_down"),
        ("data/classical/chest_background.png", {}, "no_background"),
        (None, {"FREE_GB": "40"}, "disk"),
        (None, {"ROS_VERDICT": "hardware_inactive"}, "hardware_inactive"),
        (None, {"ROS_VERDICT": "not_commanded_ee"}, "not_commanded_ee"),
    ],
)
def test_each_precondition_refuses_start_with_exit_3(checkout, removed, env, code):
    if removed is not None:
        (checkout / removed).unlink()
    result = run(checkout, "start", "--object", "can", "--json", **env)

    assert result.returncode == 3
    assert refusal(result) == code
    assert not (checkout / "scripts" / "run_classical.calls").exists()


def test_clean_start_reports_the_run_and_no_warnings(checkout):
    result = run(checkout, "start", "--object", "can", "--json", STARTS_LOOP="1")

    assert result.returncode == 0
    reply = json.loads(result.stdout)
    assert reply["ok"] is True
    assert reply["warnings"] == []
    assert reply["data"]["active"] is True


def test_missing_tactile_and_g2_warn_but_start(checkout):
    (checkout / "data/classical/collect/.g2_passed").unlink()
    result = run(
        checkout, "start", "--object", "can", "--json", STARTS_LOOP="1", ROS_VERDICT="tactile 3"
    )

    assert result.returncode == 0
    codes = [warning["code"] for warning in json.loads(result.stdout)["warnings"]]
    assert codes == ["no_g2", "tactile_silent"]


def test_warnings_go_to_stderr_in_text_mode(checkout):
    result = run(checkout, "start", "--object", "can", STARTS_LOOP="1", ROS_VERDICT="tactile 0")

    assert result.returncode == 0
    assert "warning: 0 of 4 /gripper/tactile/*" in result.stderr


def test_unreachable_ros_is_unhealthy_not_a_precondition(checkout):
    result = run(checkout, "start", "--object", "can", "--json", ROS_VERDICT="")
    assert result.returncode == 1
    assert refusal(result) == "ros_unreachable"


def test_start_launches_a_detached_loop_with_a_run_id(checkout):
    result = run(checkout, "start", "--object", "can", "--hours", "2", LOOP_UP="")
    # The stub never brings the loop up, so the post-start check reports it exited.
    assert result.returncode == 1
    calls = (checkout / "scripts" / "run_classical.calls").read_text().split()
    assert calls[:5] == ["collect", "--object", "can", "--hours", "2"]
    assert "--detach" in calls
    assert calls[calls.index("--run-id") + 1].endswith("Z")


def test_failed_launch_reports_the_reason(checkout):
    result = run(checkout, "start", "--object", "can", "--json", RUN_CLASSICAL_RC="1")
    assert result.returncode == 1
    assert refusal(result) == "start_failed"


def test_stop_with_no_loop_running_is_done(checkout):
    result = run(checkout, "stop")
    assert result.returncode == 0
    assert "no loop was running" in result.stdout
