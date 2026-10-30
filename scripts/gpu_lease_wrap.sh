#!/usr/bin/env bash
#
# Optional serialization of GPU-heavy work across concurrent sessions on a shared host.
#
# This host runs more than one project against the same two GPUs (and the same self-hosted CI agent).
# Nothing coordinated them, so one session could start a 25-minute multi-GPU suite while another was
# mid-run; the contention was silent, surfacing as OOM, flaky timing, or mutual slowdown rather than an
# obvious error.
#
# The lease implementation itself is intentionally NOT vendored here -- it is host/operator infrastructure,
# not part of this package, and other projects on the host share one copy of it. Point `GPU_LEASE_CMD` at
# it to opt in:
#
#   export GPU_LEASE_CMD=/path/to/gpu_lease.sh
#   ./scripts/gen_fts_coverage.sh --repo-home=${PWD} --target-env-name=fts_latest
#
# When `GPU_LEASE_CMD` is unset or not executable, `maybe_gpu_lease` runs the command directly, so the
# default behavior for contributors and CI is completely unchanged. This file is kept byte-identical
# across the projects that share this host so the contract does not drift between them.
#
# Optional knobs:
#   GPU_LEASE_CMD      path to the lease wrapper; unset => no-op passthrough
#   GPU_LEASE_ARGS     extra args for the lease (e.g. "--timeout 7200" or "--cpu-heavy")
#   GPU_LEASE_PROJECT  label recorded as the lease holder; defaults to the repo directory name
#   GPU_LEASE_HELD     set BY the lease for its descendants; presence means "already held, do not re-acquire"

# Run "$@" under the GPU lease when one is configured, otherwise run it directly.
#
# Re-entrancy matters: a coverage run acquires the lease and then runs the special tests, which would
# also try to acquire it. flock is not re-entrant across processes, so a nested acquisition would block on
# the lease the parent already holds -- a deadlock. The lease exports GPU_LEASE_HELD for exactly this
# reason, and we honor it here.

# True when GPU_LEASE_HELD is set AND its recorded holder pid is still alive.
#
# The pid tag matters. A bare "is the variable set?" check is unsafe: an exported GPU_LEASE_HELD can
# outlive its holder (a shell that inherited it, a `.bashrc` that set it by mistake, a resumed session).
# That failure is silent and total -- every consumer would skip acquiring and run unserialized, which is
# precisely the contention this exists to prevent. Validating the pid turns it into a no-op instead.
_gpu_lease_really_held() {
    [[ -n "${GPU_LEASE_HELD:-}" ]] || return 1
    local pid="${GPU_LEASE_HELD##*:}"
    [[ "$pid" =~ ^[0-9]+$ ]] || return 0          # untagged (older format): trust it
    if kill -0 "$pid" 2>/dev/null; then return 0; fi
    echo "gpu_lease_wrap: ignoring STALE GPU_LEASE_HELD='${GPU_LEASE_HELD}' (holder pid ${pid} is dead)." >&2
    unset GPU_LEASE_HELD
    return 1
}

maybe_gpu_lease() {
    if _gpu_lease_really_held; then
        "$@"                       # an ancestor already holds it
    elif [[ -n "${GPU_LEASE_CMD:-}" && -x "${GPU_LEASE_CMD}" ]]; then
        # shellcheck disable=SC2086  # GPU_LEASE_ARGS is intentionally word-split
        "${GPU_LEASE_CMD}" ${GPU_LEASE_ARGS:-} \
            --project "${GPU_LEASE_PROJECT:-$(basename "$(git rev-parse --show-toplevel 2>/dev/null || pwd)")}" \
            -- "$@"
    else
        "$@"                       # not configured: unchanged default behavior
    fi
}

# Re-exec the *calling script* under the lease, once, then continue normally.
#
# Preferred over wrapping individual commands with `maybe_gpu_lease` for scripts whose whole run is
# GPU-heavy (the standalone suite, a coverage run). Two reasons:
#   1. The lease runs in a separate process, so it can only exec external commands -- it cannot invoke a
#      shell function defined by the caller. Re-exec sidesteps that entirely.
#   2. Holding the lease once for the whole run is what we actually want. Acquiring and releasing per test
#      would let another session interleave partway through a suite, which is the contention we are
#      trying to prevent.
#
# Usage, as early as possible in the script (before any GPU work):
#   gpu_lease_reexec "$0" "$@"
gpu_lease_reexec() {
    _gpu_lease_really_held && return 0                              # already held by an ancestor
    [[ -n "${GPU_LEASE_CMD:-}" && -x "${GPU_LEASE_CMD}" ]] || return 0   # not configured: no-op
    echo "gpu_lease_wrap: acquiring GPU lease via ${GPU_LEASE_CMD} ..." >&2
    # shellcheck disable=SC2086  # GPU_LEASE_ARGS is intentionally word-split
    exec "${GPU_LEASE_CMD}" ${GPU_LEASE_ARGS:-} \
        --project "${GPU_LEASE_PROJECT:-$(basename "$(git rev-parse --show-toplevel 2>/dev/null || pwd)")}" \
        -- "$@"
}


# ---------------------------------------------------------------------------------------------------
# CI (multi-step job) primitives
# ---------------------------------------------------------------------------------------------------
# A lease lives only as long as the process holding the fd, so a pipeline job cannot acquire it in one
# step and still hold it in the next -- each step is a separate shell. These helpers park a detached
# holder process for the life of the job.
#
# Two routes, behind one pair of functions (ci_host_lease_hold / ci_host_lease_release, below):
#   - the host's lease tool, when an operator has installed it and the job mounts it read-only. That gives the
#     job the same fair queue, device placement and cpu-heavy handling as every local caller. This repo is
#     public, so it never depends on the tool: the path is a mount point the pipeline names, nothing more.
#   - plain `flock` against the shared lease directory (ci_gpu_lease_acquire / ci_gpu_lease_release), when the
#     tool is absent. The CONTRACT is the lock file, not the code -- `flock` works on the inode, so a
#     bind-mounted lock file interlocks between container and host, and the tool honors the same locks.
#
# Fail-open by design: if the lease directory is not mounted, CI proceeds unserialized rather than
# failing. A missing convenience must never break the build.
#
#   ci_gpu_lease_acquire <lease_dir> <pidfile> [timeout_secs] [label] [lease_name]
#   ci_gpu_lease_release <lease_dir> <pidfile> [lease_name]
#
# `lease_name` selects the lock (`<lease_dir>/<lease_name>.lock`, default `gpu`); the host tool keeps one lock per
# resource (`gpu`, `cpu-heavy`), and a job that needs both takes them in the host tool's order, gpu then cpu-heavy,
# so no caller can hold one while waiting for the other in the opposite order.

ci_gpu_lease_acquire() {
    local dir="${1:?lease dir}" pidfile="${2:?pidfile}" timeout="${3:-2700}" label="${4:-ci-job}" lease="${5:-gpu}"
    # Fail open ONLY when the host does not use leases at all (directory not mounted). Any other failure
    # below is a real misconfiguration and is reported as such -- see the note on the permission trap.
    if [[ ! -d "$dir" ]]; then
        echo "ci_gpu_lease: '${dir}' not mounted; this host does not use GPU leases. Proceeding." >&2
        return 0
    fi
    local lock="${dir}/${lease}.lock" ready="${pidfile}.ready"
    rm -f "$pidfile" "$ready"
    # The lock is opened READ-ONLY. Under rootless docker with userns remapping this container's uid is a
    # host subuid that does not own the lock file, so opening it read-write fails with EACCES. flock(2)
    # places a lock regardless of the fd's open mode, so a read-only fd gives full mutual exclusion and
    # sidesteps ownership entirely. Opening read-write here previously caused a silent fail-open that
    # defeated serialization completely -- the job ran GPU tests alongside a local suite.
    if [[ ! -e "$lock" ]]; then
        : > "$lock" 2>/dev/null || {
            echo "##vso[task.logissue type=error]ci_gpu_lease: ${lock} is missing and cannot be created." >&2
            return 1; }
    fi
    if [[ ! -r "$lock" ]]; then
        echo "##vso[task.logissue type=error]ci_gpu_lease: ${lock} is not readable; cannot serialize." >&2
        echo "  The lease directory is mounted, so this is a misconfiguration rather than an opt-out." >&2
        echo "  Un-mount ${dir} to disable GPU leasing deliberately." >&2
        return 1
    fi
    setsid bash -c '
        lock="$1"; pidfile="$2"; ready="$3"; tmo="$4"
        exec {fd}<"$lock" || exit 1
        flock -x -w "$tmo" "$fd" || exit 75
        echo $$ > "$pidfile"; : > "$ready"
        while :; do sleep 3600; done
    ' _ "$lock" "$pidfile" "$ready" "$timeout" </dev/null >/dev/null 2>&1 &
    disown
    local deadline=$(( $(date +%s) + timeout + 15 ))
    while [[ ! -e "$ready" ]]; do
        if (( $(date +%s) > deadline )); then
            echo "ci_gpu_lease: timed out after ${timeout}s waiting for the ${lease} lease." >&2
            [[ -r "${dir}/${lease}.holder" ]] && { echo "  current holder:" >&2; sed 's/^/    /' "${dir}/${lease}.holder" >&2; }
            return 75
        fi
        sleep 2
    done
    rm -f "$ready"
    { echo "pid=$(cat "$pidfile")"; echo "started=$(date -Iseconds)"; echo "epoch=$(date +%s)"
      echo "project=${label}"; echo "host=$(hostname)"; echo "container=yes"
      echo "cmd=azure pipeline job"; } | _ci_write_holder "${dir}/${lease}.holder"
    # Make the holder repairable by whoever takes the lease next. This container owns the file it just
    # created so the chmod succeeds HERE, whereas a local uid could never repair it later: chmod(2)
    # requires ownership, and unlink in the sticky lease dir does too, and a local user owns neither the
    # file (written as a rootless-docker subuid) nor the directory (az_pipeline_agent).
    #
    # Scope, so this is not read as fixing a recurring failure: the normal path never needs it, because
    # ci_gpu_lease_release below removes the holder outright and runs under `condition: always()`.
    # Measured clean in both directions across consecutive builds; handoffs are clean. It matters only when a
    # termination bypasses that cleanup (container teardown mid-step, agent kill, host crash), which
    # would otherwise leave a 0644 container-owned holder that the next LOCAL acquire cannot overwrite,
    # so `--status` would misattribute a local hold to a finished build. That is the attribution the
    # "never force-reset a CI-held lease" rule depends on. It self-heals at the next CI build, which
    # shares the subuid, so the exposure is one abnormal termination wide. At ACQUIRE rather than at
    # release precisely because a torn-down job never reaches its release step.
    chmod 0666 "${dir}/${lease}.holder" 2>/dev/null || true
    echo "ci_gpu_lease: acquired the ${lease} lease (holder $(cat "$pidfile"))." >&2
}

ci_gpu_lease_release() {
    local dir="${1:?lease dir}" pidfile="${2:?pidfile}" lease="${3:-gpu}"
    [[ -f "$pidfile" ]] || { echo "ci_gpu_lease: nothing to release for ${lease}." >&2; return 0; }
    local hpid; hpid=$(cat "$pidfile" 2>/dev/null); rm -f "$pidfile"
    # Remove the holder record BEFORE freeing the lock. In the other order the lock is free while this job's
    # record remains, and a waiter that acquires in that window cannot replace a record another uid owns (see
    # _ci_write_holder), so its whole hold shows as unattributed. A record missing while the lock is still held
    # reads as "held, unattributed", which is already to be treated as CI-held.
    rm -f "${dir}/${lease}.holder" 2>/dev/null || true
    # Kill the process GROUP: the holder's `sleep` child inherits the flock'd fd, so killing only the
    # holder shell would leave the lock held by an orphan.
    [[ -n "$hpid" ]] && { kill -TERM -- "-${hpid}" 2>/dev/null || kill -TERM "$hpid" 2>/dev/null; sleep 1
                          kill -KILL -- "-${hpid}" 2>/dev/null || kill -KILL "$hpid" 2>/dev/null; }
    echo "ci_gpu_lease: released the ${lease} lease." >&2
}

# Write stdin to the holder record at "$1". Where a record already exists, open it without O_CREAT: under
# `fs.protected_regular` (the default on current Ubuntu and systemd-based distributions) the kernel refuses a
# creating open of another uid's file in a sticky, world-writable directory whatever its mode, unless the
# directory's owner owns the file, so a stale record left by a local process could not otherwise be replaced.
# A plain open with O_TRUNC is governed by the file's mode alone, which the lease tools keep at 0666.
_ci_write_holder() {
    local path="$1"
    if [[ -e "$path" ]]; then
        dd of="$path" conv=nocreat status=none 2>/dev/null || true
    else
        cat > "$path" 2>/dev/null || true
    fi
}


# ---------------------------------------------------------------------------------------------------
# One CI hold, through the host's lease tool when it is mounted, else through the plain locks
# ---------------------------------------------------------------------------------------------------
#   ci_host_lease_hold <lease_dir> <tool> <pidfile> <timeout_secs> <label> [request...]
#   ci_host_lease_release <lease_dir> <tool> <pidfile>
#
# [request] is the lease tool's own request options: `--gpus all` (the default) or `--gpus 1 --min-vram <GB>`,
# and `--cpu-heavy` to take host memory too. The grant, "all" or a device UUID, is written to <pidfile>.device.
#
# With the tool (a regular, executable file at <tool>: a bind mount whose source is missing appears as a
# directory), the hold is the tool's `--hold`, which queues in fair order, places a device request on the
# smallest device that fits, and waits for cpu-heavy before taking the GPUs. Without it, the plain locks take the
# whole server (a device request cannot be placed, so it is granted "all"), plus cpu-heavy when requested, after
# the same wait for cpu-heavy; every wait shares <timeout_secs>. Exit status: 0 when held, or when <lease_dir> is
# absent (this host uses no leases; the grant reads "all"); 75 on a timeout, holding nothing; anything else is a
# misconfiguration, never a silent pass.
ci_host_lease_hold() {
    local dir="${1:?lease dir}" tool="${2:?tool path}" pidfile="${3:?pidfile}" timeout="${4:?timeout}"
    local label="${5:?label}" rc
    shift 5
    rm -f "${pidfile}.mode" "${pidfile}.device"
    if [[ ! -d "$dir" ]]; then
        echo "ci_host_lease: '${dir}' not mounted; this host does not use GPU leases. Proceeding." >&2
        echo all > "${pidfile}.device"
        return 0
    fi
    if [[ -f "$tool" && -x "$tool" ]]; then
        GPU_LEASE_DIR="$dir" "$tool" --hold --pidfile "$pidfile" --timeout "$timeout" --project "$label" "$@"
        rc=$?
        (( rc == 0 )) && echo tool > "${pidfile}.mode"
        return $rc
    fi
    # on stdout, where the pipeline agent reads its logging commands
    echo "##vso[task.logissue type=warning]ci_host_lease: the host lease tool is not mounted at ${tool}; taking" \
        "the plain locks for the whole server, without fair order or device placement."
    local cpu_heavy=0 arg deadline wait_s fd
    for arg in "$@"; do [[ "$arg" == --cpu-heavy ]] && cpu_heavy=1; done
    deadline=$(( SECONDS + timeout ))
    if (( cpu_heavy )) && [[ -r "${dir}/cpu-heavy.lock" ]]; then
        # wait, holding nothing, for cpu-heavy to be free before taking the GPUs (racy: costs only that wait again)
        exec {fd}<"${dir}/cpu-heavy.lock"
        if ! flock -x -w "$timeout" "$fd"; then
            exec {fd}<&-
            echo "ci_host_lease: timed out after ${timeout}s waiting for the cpu-heavy lease." >&2
            return 75
        fi
        exec {fd}<&-
    fi
    wait_s=$(( deadline - SECONDS > 0 ? deadline - SECONDS : 1 ))
    ci_gpu_lease_acquire "$dir" "$pidfile" "$wait_s" "$label" gpu || return $?
    echo plain > "${pidfile}.mode"
    echo all > "${pidfile}.device"
    if (( cpu_heavy )); then
        wait_s=$(( deadline - SECONDS > 0 ? deadline - SECONDS : 1 ))
        ci_gpu_lease_acquire "$dir" "${pidfile}.cpu" "$wait_s" "$label" cpu-heavy || {
            rc=$?; ci_host_lease_release "$dir" "$tool" "$pidfile"; return $rc; }
    fi
    return 0
}

ci_host_lease_release() {
    local dir="${1:?lease dir}" tool="${2:?tool path}" pidfile="${3:?pidfile}"
    case "$(cat "${pidfile}.mode" 2>/dev/null)" in
        tool) GPU_LEASE_DIR="$dir" "$tool" --release --pidfile "$pidfile" ;;
        plain)
            ci_gpu_lease_release "$dir" "${pidfile}.cpu" cpu-heavy
            ci_gpu_lease_release "$dir" "$pidfile" gpu
            ;;
        *) echo "ci_host_lease: nothing to release for ${pidfile}." >&2 ;;
    esac
    rm -f "${pidfile}.mode" "${pidfile}.device"
}
