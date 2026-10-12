#!/usr/bin/env bash
# Take or release this job's host leases: the whole server (`gpu`) and host memory (`cpu-heavy`).
#
#   gpu_job_lease.sh acquire|release
#
# The leases are taken through the host's own lease tool, mounted read-only at /gpu_lease_tool, so this job queues in
# the same fair order as every local caller. The tool is host infrastructure behind a seam: when it is absent the job
# takes the plain `flock` locks through scripts/gpu_lease_wrap.sh, gpu then cpu-heavy (the tool's own order), without
# fair ordering. When the lease directory itself is absent the host does not use leases, and the job proceeds.
#
# The agent runs one job at a time, so these leases serialize this job only against LOCAL work on the host. Waiting
# here for tens of minutes behind a local GPU job is the lease working, not a hung build.
set -uo pipefail

ACTION=${1:?acquire or release}
DIR=${GPU_JOB_LEASE_DIR:-/gpu_leases}         # overridable for local tests; the pipeline mounts these two paths
TOOL=${GPU_JOB_LEASE_TOOL:-/gpu_lease_tool}
WAIT=${GPU_JOB_LEASE_WAIT:-2400}
LABEL="azure-${BUILD_BUILDID:-local}"
PIDFILE=${GPU_JOB_LEASE_PIDFILE:-/tmp/azp_gpu_job_lease.pid}
MODE="${PIDFILE}.mode"

# shellcheck source=/dev/null
source "$(dirname "$0")/../../scripts/gpu_lease_wrap.sh"

tool_usable() { [[ -f "$TOOL" && -x "$TOOL" ]]; }

case "$ACTION" in
    acquire)
        if [[ ! -d "$DIR" ]]; then
            echo "gpu_job_lease: ${DIR} not mounted; this host does not use GPU leases. Proceeding unleased."
            exit 0
        fi
        if tool_usable; then
            # Wait for cpu-heavy BEFORE taking gpu: the tool takes gpu first and only then waits on cpu-heavy (the
            # fixed order that rules out deadlock), so without this a long local CPU-only job holding cpu-heavy would
            # leave this job holding GPUs it cannot use. Racy by nature, which only costs that wait again.
            GPU_LEASE_DIR="$DIR" "$TOOL" --wait-only --lease cpu-heavy --timeout "$WAIT" --project "$LABEL" || exit $?
            GPU_LEASE_DIR="$DIR" "$TOOL" --hold --pidfile "$PIDFILE" --gpus all --cpu-heavy --timeout "$WAIT" \
                --project "$LABEL" || exit $?
            echo tool > "$MODE"
        else
            echo "##vso[task.logissue type=warning]The host lease tool is not mounted at ${TOOL}; taking the plain" \
                "gpu and cpu-heavy locks, without fair ordering."
            ci_gpu_lease_acquire "$DIR" "$PIDFILE" "$WAIT" "$LABEL" gpu || exit $?
            echo plain > "$MODE"
            ci_gpu_lease_acquire "$DIR" "${PIDFILE}.cpu" "$WAIT" "$LABEL" cpu-heavy || exit $?
        fi
        ;;
    release)
        # Belt-and-braces for the always() cleanup: if this job dies or its container is torn down, every process
        # inside it dies too and the kernel frees the leases anyway.
        case "$(cat "$MODE" 2>/dev/null)" in
            tool) GPU_LEASE_DIR="$DIR" "$TOOL" --release --pidfile "$PIDFILE" ;;
            plain)
                ci_gpu_lease_release "$DIR" "${PIDFILE}.cpu" cpu-heavy
                ci_gpu_lease_release "$DIR" "$PIDFILE" gpu
                ;;
            *) echo "gpu_job_lease: nothing to release." ;;
        esac
        rm -f "$MODE"
        ;;
    *)
        echo "usage: $0 acquire|release" >&2
        exit 2
        ;;
esac
