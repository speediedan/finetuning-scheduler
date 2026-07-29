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
maybe_gpu_lease() {
    if [[ -n "${GPU_LEASE_HELD:-}" ]]; then
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
    [[ -n "${GPU_LEASE_HELD:-}" ]] && return 0                      # already held by an ancestor
    [[ -n "${GPU_LEASE_CMD:-}" && -x "${GPU_LEASE_CMD}" ]] || return 0   # not configured: no-op
    echo "gpu_lease_wrap: acquiring GPU lease via ${GPU_LEASE_CMD} ..." >&2
    # shellcheck disable=SC2086  # GPU_LEASE_ARGS is intentionally word-split
    exec "${GPU_LEASE_CMD}" ${GPU_LEASE_ARGS:-} \
        --project "${GPU_LEASE_PROJECT:-$(basename "$(git rev-parse --show-toplevel 2>/dev/null || pwd)")}" \
        -- "$@"
}
