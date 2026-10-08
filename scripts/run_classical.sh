#!/usr/bin/env bash
# Runs the classical can-on-paper baseline on the robot PC.
#
# Usage:
#   ./scripts/run_classical.sh <verb> [args]
#
# Verbs:
#   build                  Vendor anvil_msgs if missing, then build the image
#   up | down | logs       Start, stop, or follow the perception + task nodes
#   mine [args]            Mine grasp/place params from the demos → data/classical/
#   eval [args]            Offline perception eval vs mined ground truth → data/classical/eval/
#   calibrate [args]       Sweep the arm with the hand marker and fit the chest camera
#                          → data/classical/calibration/ (--dry-run moves nothing,
#                          --fit-only re-fits saved views, --marker-id/--marker-size)
#   overlay [args]         Project the live TCP and a table grid onto a chest frame → jpg
#   background             Capture the empty-table reference (clear the table first)
#   trial [--dry-run] [--n N] [--speed S]
#                          Run N pick-place goals, prompting between each, and append
#                          one row per goal to data/classical/trials.csv
#   collect --object NAME [--hours H] [--cycles N] [--speed S] [--no-record]
#           [--dry-run] [--run-id ID] [--detach]
#                          Unattended pick and random re-place loop, one MCAP episode
#                          per cycle → data/classical/collect/<run>/. Ctrl-C (or
#                          collect-stop) ends it at HOME with summary.json written.
#                          --detach runs it in the background container classical-collect
#   collect-stop           SIGINT the detached loop and wait for it to finish
#   -h | --help            Show this message
#
# Extra args after mine/eval/calibrate/overlay pass straight to the tool.
#
# Environment variables:
#   RECORDINGS_DIR   Demo episodes (default: ~/anvil-loader/data/recordings/pick-and-place-can)
#   ROS_DOMAIN_ID    ROS domain (default: 1, matching the loader)
#
# The task node and calibrate move the arm through commanded EE, which bypasses the
# loader's own limiter. Keep the webapp e-stop in reach for every run that is not --dry-run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE=(docker compose -f "${REPO_ROOT}/docker-compose.classical.yml")
DATA_DIR="${REPO_ROOT}/data/classical"
TRIALS_CSV="${DATA_DIR}/trials.csv"
COLLECT_CONTAINER=classical-collect
# Time the loop gets after SIGINT to cancel, drive HOME and write its summary.
COLLECT_STOP_TIMEOUT_S=180

usage() {
    sed -n '2,/^set -/p' "$0" | sed -e '$d' -e 's/^# \{0,1\}//'
}

# One-shot tool in a fresh container; works whether or not the nodes are up.
run_tool() {
    "${COMPOSE[@]}" run --rm --no-deps classical "$@"
}

# Command inside the running node container. `exec` skips the entrypoint, so the
# workspace has to be sourced here.
in_running() {
    "${COMPOSE[@]}" exec -T classical bash -c \
        "source /opt/ros/jazzy/setup.bash && source /workspace/install/setup.bash && $*"
}

# First value of `field:` after the line matching `section:` in ros2 CLI YAML output.
field_after() {
    awk -v section="$1:" -v field="$2:" '
        $1 == section { found = 1; next }
        found && $1 == field { print $2; exit }
    '
}

trial() {
    local dry_run=false n=1 speed=0.5
    while [ $# -gt 0 ]; do
        case "$1" in
            --dry-run) dry_run=true ;;
            --n) n="$2"; shift ;;
            --speed) speed="$2"; shift ;;
            *) echo "unknown trial option: $1" >&2; exit 2 ;;
        esac
        shift
    done

    # Both end up inside a `bash -c` string, so only plain numbers get through.
    [[ "$speed" =~ ^[0-9]*\.?[0-9]+$ ]] || { echo "--speed must be a number, got: $speed" >&2; exit 2; }
    [[ "$n" =~ ^[0-9]+$ ]] || { echo "--n must be a whole number, got: $n" >&2; exit 2; }

    if [ ! -f "$TRIALS_CSV" ]; then
        echo "timestamp,dry_run,speed_scale,error_code,last_phase,duration_s,can_x,can_y,paper_x,paper_y,success,note" \
            > "$TRIALS_CSV"
    fi

    local i output error_code last_phase duration can_x can_y paper_x paper_y success note
    for ((i = 1; i <= n; i++)); do
        read -r -p "Trial ${i}/${n}: place can + paper, then Enter (q to stop) " reply
        [ "$reply" = "q" ] && break

        output=$(in_running ros2 action send_goal --feedback /classical/pick_place \
            classical_control_msgs/action/PickPlace \
            "'{dry_run: ${dry_run}, speed_scale: ${speed}}'" 2>&1 || true)
        echo "$output" | tail -n 40

        error_code=$(echo "$output" | awk '$1 == "error_code:" { print $2 }' | tail -n 1)
        last_phase=$(echo "$output" | awk '$1 == "phase:" { print $2 }' | tail -n 1)
        duration=$(echo "$output" | awk '$1 == "duration_s:" { print $2 }' | tail -n 1)
        can_x=$(echo "$output" | sed -n '/can_pose:/,$p' | field_after position x)
        can_y=$(echo "$output" | sed -n '/can_pose:/,$p' | field_after position y)
        paper_x=$(echo "$output" | sed -n '/paper_pose:/,$p' | field_after position x)
        paper_y=$(echo "$output" | sed -n '/paper_pose:/,$p' | field_after position y)

        success=""
        note=""
        if [ "$dry_run" = false ]; then
            read -r -p "Can upright on paper? [y/n] " success
            read -r -p "Note (optional): " note
        fi

        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"%s"\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$dry_run" "$speed" "${error_code:-}" \
            "${last_phase:-}" "${duration:-}" "${can_x:-}" "${can_y:-}" \
            "${paper_x:-}" "${paper_y:-}" "$success" "${note//\"/\"\"}" >> "$TRIALS_CSV"
        echo "→ error_code=${error_code:-?} phase=${last_phase:-?} logged to ${TRIALS_CSV}"
    done
}

collect() {
    local object="" hours="" cycles="" speed=0.5 run_id="" detach=false
    local flags=()
    while [ $# -gt 0 ]; do
        case "$1" in
            --object) object="$2"; shift ;;
            --hours) hours="$2"; shift ;;
            --cycles) cycles="$2"; shift ;;
            --speed) speed="$2"; shift ;;
            --run-id) run_id="$2"; shift ;;
            --no-record) flags+=(--no-record) ;;
            --dry-run) flags+=(--dry-run) ;;
            --detach) detach=true ;;
            *) echo "unknown collect option: $1" >&2; exit 2 ;;
        esac
        shift
    done

    [[ "$object" =~ ^[a-z0-9_-]+$ ]] || { echo "--object must name a config/objects/*.yaml, got: ${object:-nothing}" >&2; exit 2; }
    [ -f "${REPO_ROOT}/ros2/src/classical_control/config/objects/${object}.yaml" ] \
        || { echo "no object config: config/objects/${object}.yaml" >&2; exit 2; }
    [[ "$speed" =~ ^[0-9]*\.?[0-9]+$ ]] || { echo "--speed must be a number, got: $speed" >&2; exit 2; }
    [ -z "$hours" ] || [[ "$hours" =~ ^[0-9]*\.?[0-9]+$ ]] || { echo "--hours must be a number, got: $hours" >&2; exit 2; }
    [ -z "$cycles" ] || [[ "$cycles" =~ ^[0-9]+$ ]] || { echo "--cycles must be a whole number, got: $cycles" >&2; exit 2; }
    [ -z "$run_id" ] || [[ "$run_id" =~ ^[0-9A-Za-z_-]+$ ]] || { echo "--run-id must be letters, digits, _ or -" >&2; exit 2; }

    local args=(--object "$object" --speed "$speed" "${flags[@]+"${flags[@]}"}")
    [ -n "$hours" ] && args+=(--hours "$hours")
    [ -n "$cycles" ] && args+=(--cycles "$cycles")
    [ -n "$run_id" ] && args+=(--run-id "$run_id")

    mkdir -p "${DATA_DIR}/collect"
    local run=("${COMPOSE[@]}" run --rm --no-deps --name "$COLLECT_CONTAINER")
    $detach && run+=(-d)
    # The image holds no .git, so the commit is read here and stamped into the run metadata.
    local sha
    sha="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    # python -m rather than `ros2 run`, so the node is the container's PID 1 and gets the
    # collect-stop SIGINT itself, with no wrapper in between.
    "${run[@]}" classical python3 -m classical_control.collect_node \
        --objects-dir /workspace/config/objects \
        --collect-config /workspace/config/collect.yaml \
        --perception-config /workspace/config/perception.yaml \
        --out-root /data/collect --git-sha "$sha" "${args[@]}"
}

collect_stop() {
    docker kill --signal=SIGINT "$COLLECT_CONTAINER" >/dev/null 2>&1 \
        || { echo "no collection loop is running"; return 0; }
    local waited=0
    while docker inspect "$COLLECT_CONTAINER" >/dev/null 2>&1; do
        if [ "$waited" -ge "$COLLECT_STOP_TIMEOUT_S" ]; then
            echo "loop still running after ${COLLECT_STOP_TIMEOUT_S}s; check the arm and 'docker logs ${COLLECT_CONTAINER}'" >&2
            return 1
        fi
        sleep 1
        waited=$((waited + 1))
    done
    echo "collection loop stopped after ${waited}s"
}

VERB="${1:-}"
[ $# -gt 0 ] && shift
mkdir -p "$DATA_DIR"

case "$VERB" in
    build)
        [ -d "${REPO_ROOT}/vendor/anvil_msgs" ] || "${SCRIPT_DIR}/vendor_anvil_msgs.sh"
        "${COMPOSE[@]}" build
        ;;
    up) "${COMPOSE[@]}" up -d ;;
    down) "${COMPOSE[@]}" down ;;
    logs) "${COMPOSE[@]}" logs -f ;;
    mine)
        run_tool ros2 run classical_control mine_episodes \
            --recordings /recordings --out-dir /data "$@"
        ;;
    eval)
        run_tool ros2 run classical_control eval_offline \
            --recordings /recordings --ground-truth /data/ground_truth.csv \
            --camera-yaml /workspace/config/camera_chest.yaml --out-dir /data/eval "$@"
        ;;
    calibrate)
        run_tool ros2 run classical_control calibrate_chest \
            --camera-yaml /workspace/config/camera_chest.yaml \
            --task-params /workspace/config/task.yaml \
            --out-dir /data/calibration "$@"
        ;;
    overlay)
        run_tool ros2 run classical_control overlay_check \
            --camera-yaml /workspace/config/camera_chest.yaml \
            --output "/data/overlay_$(date +%Y%m%d_%H%M%S).jpg" "$@"
        ;;
    background)
        in_running ros2 service call /classical/capture_background std_srvs/srv/Trigger
        ;;
    trial) trial "$@" ;;
    collect) collect "$@" ;;
    collect-stop) collect_stop ;;
    -h | --help | "") usage ;;
    *)
        echo "unknown verb: ${VERB}" >&2
        usage >&2
        exit 2
        ;;
esac
