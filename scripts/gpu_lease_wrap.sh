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
#   ./tests/special_tests.sh --mark_type=standalone
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
# Re-entrancy matters: a coverage run acquires the lease and then invokes special_tests.sh, which would
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
# Deliberately implemented with plain `flock` against a shared directory rather than by calling the
# host's lease tool: this repo is public and must not depend on operator-private paths, and the job runs
# inside a container where that tool is not present. The CONTRACT is the lock file, not the code --
# `flock` works on the inode, so a bind-mounted lock file interlocks between container and host.
#
# Fail-open by design: if the lease directory is not mounted, CI proceeds unserialized rather than
# failing. A missing convenience must never break the build.
#
#   ci_gpu_lease_acquire <lease_dir> <pidfile> [timeout_secs] [label]
#   ci_gpu_lease_release <lease_dir> <pidfile>

ci_gpu_lease_acquire() {
    local dir="${1:?lease dir}" pidfile="${2:?pidfile}" timeout="${3:-2700}" label="${4:-ci-job}"
    if [[ ! -d "$dir" ]]; then
        echo "ci_gpu_lease: '${dir}' not present; proceeding without a lease (fail-open)." >&2
        return 0
    fi
    local lock="${dir}/gpu.lock" ready="${pidfile}.ready"
    rm -f "$pidfile" "$ready"
    : > "$lock" 2>/dev/null || { echo "ci_gpu_lease: cannot write ${lock}; fail-open." >&2; return 0; }
    setsid bash -c '
        lock="$1"; pidfile="$2"; ready="$3"; tmo="$4"
        exec {fd}<>"$lock" || exit 1
        flock -x -w "$tmo" "$fd" || exit 75
        echo $$ > "$pidfile"; : > "$ready"
        while :; do sleep 3600; done
    ' _ "$lock" "$pidfile" "$ready" "$timeout" </dev/null >/dev/null 2>&1 &
    disown
    local deadline=$(( $(date +%s) + timeout + 15 ))
    while [[ ! -e "$ready" ]]; do
        if (( $(date +%s) > deadline )); then
            echo "ci_gpu_lease: timed out after ${timeout}s waiting for the GPU lease." >&2
            [[ -r "${dir}/gpu.holder" ]] && { echo "  current holder:" >&2; sed 's/^/    /' "${dir}/gpu.holder" >&2; }
            return 75
        fi
        sleep 2
    done
    rm -f "$ready"
    { echo "pid=$(cat "$pidfile")"; echo "started=$(date -Iseconds)"; echo "epoch=$(date +%s)"
      echo "project=${label}"; echo "host=$(hostname)"; echo "container=yes"
      echo "cmd=azure pipeline job"; } > "${dir}/gpu.holder" 2>/dev/null || true
    echo "ci_gpu_lease: acquired the GPU lease (holder $(cat "$pidfile"))." >&2
}

ci_gpu_lease_release() {
    local dir="${1:?lease dir}" pidfile="${2:?pidfile}"
    [[ -f "$pidfile" ]] || { echo "ci_gpu_lease: nothing to release." >&2; return 0; }
    local hpid; hpid=$(cat "$pidfile" 2>/dev/null); rm -f "$pidfile"
    # Kill the process GROUP: the holder's `sleep` child inherits the flock'd fd, so killing only the
    # holder shell would leave the lock held by an orphan.
    [[ -n "$hpid" ]] && { kill -TERM -- "-${hpid}" 2>/dev/null || kill -TERM "$hpid" 2>/dev/null; sleep 1
                          kill -KILL -- "-${hpid}" 2>/dev/null || kill -KILL "$hpid" 2>/dev/null; }
    rm -f "${dir}/gpu.holder" 2>/dev/null || true
    echo "ci_gpu_lease: released the GPU lease." >&2
}
