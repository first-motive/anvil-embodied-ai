#!/usr/bin/env bash
# tactile-collect.sh — start, stop, or report the unattended visuo-tactile collection loop.
#
# The loop is the classical baseline picking the can and placing it at a random free point,
# one MCAP episode per cycle with every loader and /gripper/tactile/* topic. It runs in the
# detached container classical-collect on the robot PC; see scripts/run_classical.sh collect.
#
#   scripts/run/tactile-collect.sh status [--json]            # on the robot
#   scripts/run/tactile-collect.sh start --object can [--hours H] [--cycles N] [--speed S] [--no-record]
#   scripts/run/tactile-collect.sh stop                        # Ctrl-C the loop: HOME, summary.json
#   scripts/run/tactile-collect.sh status --host fm-rob-01     # from any machine, over ssh
#
# start runs a preflight and refuses (exit 3) unless: the classical nodes are up, the
# loader is up with hardware active in commanded-EE mode, all four /gripper/tactile/*
# topics have a publisher, the empty-table background is captured, free disk is above the
# guard, no loop is already running, and data/classical/collect/.g2_passed exists (the
# attended one-hour run has passed). It returns once the container is up, with the run id.
#
# Over --host the script runs on the robot against the checkout at $ANVIL_EMBODIED_AI_DIR
# (default ~/anvil-embodied-ai). Exit 0 done, 1 unhealthy, 2 usage, 3 precondition.
set -uo pipefail

usage() { sed -n '2,20p' "${BASH_SOURCE[0]:-$0}" | sed 's/^# \{0,1\}//'; }

HOST="" JSON=false ACTION="status" START_ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    start|stop|status) ACTION="$1"; shift ;;
    --host)
      [[ "${2:-}" =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.@:-]*$ ]] || { echo "error: --host needs one SSH host or alias" >&2; exit 2; }
      HOST="$2"; shift 2 ;;
    --json) JSON=true; shift ;;
    --object)
      [[ "${2:-}" =~ ^[a-z0-9_][a-z0-9_-]*$ ]] || { echo "error: --object needs a config/objects name" >&2; exit 2; }
      START_ARGS+=("$1" "$2"); shift 2 ;;
    --hours|--speed)
      [[ "${2:-}" =~ ^[0-9]*\.?[0-9]+$ ]] || { echo "error: $1 needs a number" >&2; exit 2; }
      START_ARGS+=("$1" "$2"); shift 2 ;;
    --cycles)
      [[ "${2:-}" =~ ^[0-9]+$ ]] || { echo "error: --cycles needs a whole number" >&2; exit 2; }
      START_ARGS+=("$1" "$2"); shift 2 ;;
    --no-record) START_ARGS+=("$1"); shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
  esac
done
if [ "${#START_ARGS[@]}" -gt 0 ] && [ "$ACTION" != start ]; then
  echo "error: --object/--hours/--cycles/--speed/--no-record only go with start" >&2; exit 2
fi

if [ -n "$HOST" ]; then
  remote_args=("$ACTION"); $JSON && remote_args+=(--json)
  remote_args+=("${START_ARGS[@]+"${START_ARGS[@]}"}")
  exec ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 -- "$HOST" bash -s -- "${remote_args[@]}" < "${BASH_SOURCE[0]}"
fi

# Run from a checkout this script sits in, or, piped over ssh, the robot's checkout.
if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
  REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
else
  REPO="${ANVIL_EMBODIED_AI_DIR:-$HOME/anvil-embodied-ai}"
fi
RUN_CLASSICAL="$REPO/scripts/run_classical.sh"
COLLECT_DIR="$REPO/data/classical/collect"
NODES=classical-control LOOP=classical-collect
G2_MARKER="$COLLECT_DIR/.g2_passed"
BACKGROUND="$REPO/data/classical/chest_background.png"
COLLECT_YAML="$REPO/ros2/src/classical_control/config/collect.yaml"
COMMANDED_EE_MODE=openarm_v2_quest_teleop_commanded_ee.yaml

[ -x "$RUN_CLASSICAL" ] || { echo "error: no anvil-embodied-ai checkout at $REPO (set ANVIL_EMBODIED_AI_DIR)" >&2; exit 3; }

json_escape() { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }
running() { [ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" = true ]; }
free_gb() { mkdir -p "$COLLECT_DIR"; df -P -B 1000000000 "$COLLECT_DIR" | awk 'NR == 2 { print $4 }' | grep . || echo 0; }
min_free_gb() { awk '$1 == "min_free_gb:" { print $2 }' "$COLLECT_YAML"; }
latest_run() { find "$COLLECT_DIR" -mindepth 1 -maxdepth 1 -type d ! -name '.*' 2>/dev/null | sort | tail -n 1; }
# A command inside the running classical nodes' container, with ROS sourced. Bounded, so a
# hung ROS call cannot hold a Slack-triggered start past its budget.
in_nodes() {
  timeout 12 docker exec "$NODES" bash -c "source /opt/ros/jazzy/setup.bash && source /workspace/install/setup.bash && $1"
}

# Print the result and exit. Refusals carry a stable code for Desktop and Slack.
finish() {  # status(0|1|3)  code(ok|<refusal>)  detail
  local status="$1" code="$2" detail="$3" run active=false cycles=0 successes=0 reason=""
  run="$(latest_run)"
  running "$LOOP" && active=true
  if [ -n "$run" ]; then
    cycles="$(find "$run" -mindepth 2 -maxdepth 2 -name metadata.json | wc -l | tr -d ' ')"
    successes="$(grep -l '"status": "success"' "$run"/*/metadata.json 2>/dev/null | wc -l | tr -d ' ')"
    [ -f "$run/summary.json" ] && reason="$(sed -n 's/.*"stop_reason": "\{0,1\}\([^",]*\)"\{0,1\},*$/\1/p' "$run/summary.json")"
  fi
  if $JSON; then
    printf '{"schema_version":1,"verb":"tactile-collect","host":"%s","ok":%s,' "$(json_escape "$(hostname)")" "$([ "$status" = 0 ] && echo true || echo false)"
    [ "$code" = ok ] || printf '"error":{"code":"%s","detail":"%s"},' "$code" "$(json_escape "$detail")"
    printf '"data":{"active":%s,"run":%s,"cycles":%s,"successes":%s,"stop_reason":%s,"free_gb":%s}}\n' \
      "$active" "$([ -n "$run" ] && echo "\"$(json_escape "$(basename "$run")")\"" || echo null)" "$cycles" "$successes" \
      "$([ -n "$reason" ] && [ "$reason" != null ] && echo "\"$(json_escape "$reason")\"" || echo null)" "$(free_gb)"
  else
    echo "loop: $($active && echo running || echo stopped)  run: ${run:+$(basename "$run")}  cycles: $cycles  successes: $successes  stop: ${reason:-–}  free: $(free_gb) GB"
    case "$status" in
      0) echo "tactile-collect: $detail" ;;
      3) echo "tactile-collect: refused: $detail" >&2 ;;
      *) echo "tactile-collect: failed: $detail" >&2 ;;
    esac
  fi
  exit "$status"
}

preflight() {
  running "$LOOP" && finish 3 already_running "a collection loop is already running; stop it first"
  [ -f "$G2_MARKER" ] || finish 3 no_g2 "no $G2_MARKER: pass the attended one-hour run (G2) before unattended starts"
  running "$NODES" || finish 3 nodes_down "classical nodes are not up: $RUN_CLASSICAL up"
  [ -f "$BACKGROUND" ] || finish 3 no_background "no empty-table background: clear the table, then $RUN_CLASSICAL background"
  local free min
  free="$(free_gb)" min="$(min_free_gb)"
  [ "${free:-0}" -gt "${min:-50}" ] || finish 3 disk "${free:-?} GB free, the guard is ${min:-50} GB: move recordings off the robot"
  # One exec for every ROS check, so start stays inside its 15 s budget. The hardware probe
  # is the one task_node makes: a no-op when active, refused when stopped or missing. A
  # /commanded_ee_right subscriber shows the loader is in commanded-EE mode.
  local verdict
  verdict="$(in_nodes '
    ros2 service call /hardware_state_controller/set_state anvil_msgs/srv/SetHardwareState "{state: active}" \
      | grep -q "accepted=True" || { echo hardware_inactive; exit 0; }
    ros2 topic info /commanded_ee_right | grep -q "Subscription count: [1-9]" || { echo not_commanded_ee; exit 0; }
    n=0
    for topic in $(ros2 topic list | grep "^/gripper/tactile/"); do
      ros2 topic info "$topic" | grep -q "Publisher count: [1-9]" && n=$((n + 1))
    done
    echo "tactile $n"' 2>/dev/null)"
  case "$verdict" in
    hardware_inactive) finish 3 hardware_inactive "hardware is not active: restart the loader (fm robot fm-rob-01 up), then the classical nodes" ;;
    not_commanded_ee) finish 3 not_commanded_ee "loader is not in commanded-EE mode: fm robot fm-rob-01 mode $COMMANDED_EE_MODE" ;;
    "tactile "[4-9]*) ;;
    "tactile "*) finish 3 tactile_silent "${verdict#tactile } of 4 /gripper/tactile/* topics have a publisher: bring the tactile container up" ;;
    *) finish 1 ros_unreachable "could not query ROS inside $NODES" ;;
  esac
}

case "$ACTION" in
  status) finish 0 ok "status" ;;
  stop)
    running "$LOOP" || finish 0 ok "no loop was running"
    # </dev/null: over --host this script arrives on stdin, which a child must not read.
    "$RUN_CLASSICAL" collect-stop </dev/null >/dev/null || finish 1 stop_timeout "loop did not finish after SIGINT; check the arm and docker logs $LOOP"
    finish 0 ok "stopped" ;;
  start)
    [[ " ${START_ARGS[*]-} " == *" --object "* ]] || { echo "error: start needs --object" >&2; exit 2; }
    preflight
    run_id="$(date -u +%Y%m%dT%H%M%SZ)"
    if ! output="$("$RUN_CLASSICAL" collect "${START_ARGS[@]}" --run-id "$run_id" --detach </dev/null 2>&1)"; then
      finish 1 start_failed "the loop container did not start: $(printf '%s\n' "$output" | tail -n 1)"
    fi
    sleep 2
    running "$LOOP" || finish 1 start_failed "the loop exited at once; rerun attended with $RUN_CLASSICAL collect to see why"
    finish 0 ok "started run $run_id" ;;
esac
