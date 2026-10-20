#!/bin/bash
#
# zenith podman environment
#
# rootless podman on this MI355X cluster (and on any site where the user has no
# /etc/subuid + /etc/subgid entries) needs three things that the stock config
# does not provide:
#
#  1. XDG_RUNTIME_DIR -- systemd is not creating /run/user/$UID for this login,
#     so podman's default runroot is unwritable and every command dies with
#     "mkdir /run/user/<uid>: permission denied"
#  2. a graph root on a big local filesystem. $HOME is a 32T NFS mount that is
#     82% full and does not support the overlay driver correctly; /scratch is
#     a 56T local ext4 with plenty of headroom
#  3. fuse-overlayfs + ignore_chown_errors. Without subuid/subgid ranges the
#     user namespace is a single uid, so image layers that chown to non-root
#     (e.g. /etc/shadow -> 0:42) fail with "lchown: invalid argument" under the
#     native overlay driver. fuse-overlayfs with ignore_chown_errors swallows
#     those, which is safe here because everything runs as the mapped user
#
# source this file, then use "$PODMAN" instead of "podman".

ZENITH_UID="$(id -u)"

# 1. runtime dir
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/podman-run-${ZENITH_UID}}"
mkdir -p "${XDG_RUNTIME_DIR}"
chmod 700 "${XDG_RUNTIME_DIR}"

# 1b. podman's *own* tmpdir, which is NOT --tmpdir and NOT --runroot: it is
# always $XDG_RUNTIME_DIR/libpod/tmp, and it holds pause.pid, the pid of the
# rootless pause process that owns the user namespace. Podman creates it on
# first use, but this cluster's /tmp reaper deletes empty directories under
# /tmp, and a node that has been idle a while loses it. Podman then cannot even
# read its own pause pid and every command dies with
#
#   error creating temporary file: No such file or directory
#   invalid internal status, try resetting the pause process with
#   "podman system migrate": open .../libpod/tmp/pause.pid: no such file
#
# which on a worker node aborts the launcher before it can start ray. Just make
# sure the directory is there.
mkdir -p "${XDG_RUNTIME_DIR}/libpod/tmp"

# 2. graph root -- prefer a local scratch fs, fall back to /tmp
if [ -d /scratch ] && [ -w /scratch ]; then
    ZENITH_STORE_BASE="/scratch/$(whoami)"
else
    ZENITH_STORE_BASE="/tmp/$(whoami)"
fi
export ZENITH_GRAPHROOT="${ZENITH_GRAPHROOT:-${ZENITH_STORE_BASE}/containers/storage}"
export ZENITH_TMPDIR="${ZENITH_TMPDIR:-${ZENITH_STORE_BASE}/containers/tmp}"

export ZENITH_RUNROOT="${ZENITH_RUNROOT:-${XDG_RUNTIME_DIR}/containers}"
mkdir -p "${ZENITH_GRAPHROOT}" "${ZENITH_RUNROOT}" "${ZENITH_TMPDIR}"

# 2b. repair a runroot left stale by a killed job.
#
# The runroot holds podman's transient per-container state, including the pid of
# the rootless "pause" process that owns the user namespace. Both slurm and flux
# kill every process the user owns on a node when a job ends, so the pause
# process dies but its bookkeeping under the runroot survives. The next job on
# that node then finds a runroot pointing at a dead pid and every podman command
# fails before it can even create a container:
#
#   cannot find UID/GID for user <user>: no subuid ranges found
#   cannot read ".../userdata/conmon.pid": EOF
#   invalid internal status, try resetting the pause process with
#   "podman system migrate"
#
# That is what killed the first 2-node Inkling attempt, on nodes where a
# cancelled job had run minutes earlier.
#
# The runroot cannot simply be made per-job: podman records the runroot path in
# the libpod database inside the *graph* root, and a mismatch is a hard error
# ("database run root ... does not match our run root"). The graph root has to
# stay shared -- it caches the 61 GB image, which is the expensive part. So keep
# one runroot and clear it when it is stale instead.
#
# Detect staleness by probing rather than by inspecting podman's internals. The
# damage can present several ways -- a dead pause pid, a missing pause.pid, a
# runroot whose per-container userdata was reaped while the libpod record in the
# graph root survived -- and each one wants the same remedy, so trying to
# enumerate them is both fragile and unnecessary. Just ask podman if it is well:
#
#   $ podman ps
#   invalid internal status, try resetting the pause process with
#   "podman system migrate": cannot read ".../userdata/conmon.pid": EOF
#
# `podman system migrate` is podman's own documented fix, and it is what the
# error message asks for: it recreates the pause process and marks orphaned
# containers stopped. Safe to run unconditionally-on-failure, because we only
# reach it when podman is already refusing to work.
#
# This matters here because both slurm and flux kill every user process on a
# node at job end, so the pause process dies without podman ever cleaning up.
# The next job on that node inherits the wreckage -- which is exactly how two
# consecutive 2-node Inkling runs died before reaching ray.
if ! podman --root "${ZENITH_GRAPHROOT}" --runroot "${ZENITH_RUNROOT}" \
        --tmpdir "${ZENITH_TMPDIR}" ps >/dev/null 2>&1; then
    # Never migrate automatically: `podman system migrate` stops EVERY container of this
    # user on the node, including other live servers. Refuse and fail instead; run it by
    # hand, or re-source with ZENITH_ALLOW_MIGRATE=1 once nothing else of yours is running.
    if [ "${ZENITH_ALLOW_MIGRATE:-0}" = 1 ]; then
        echo "lux: podman is in an invalid internal state on $(hostname);" \
             "ZENITH_ALLOW_MIGRATE=1 -> running system migrate (stops all containers)" >&2
        podman --root "${ZENITH_GRAPHROOT}" --runroot "${ZENITH_RUNROOT}" \
            --tmpdir "${ZENITH_TMPDIR}" system migrate >/dev/null 2>&1 || true
    else
        echo "lux: ERROR podman is in an invalid internal state on $(hostname)" \
             "(stale state from a previous job). REFUSING 'podman system migrate' (it stops all" \
             "containers on the node). Stop your other containers, then re-source with ZENITH_ALLOW_MIGRATE=1." >&2
        ZENITH_PODMAN_REFUSED=1
    fi
fi

# keep the (very large) image build scratch off of /tmp, which is a small tmpfs
export TMPDIR="${ZENITH_TMPDIR}"

# 3. assembled podman invocation
PODMAN="podman \
 --root ${ZENITH_GRAPHROOT} \
 --runroot ${ZENITH_RUNROOT} \
 --tmpdir ${ZENITH_TMPDIR} \
 --cgroup-manager cgroupfs \
 --storage-driver overlay \
 --storage-opt mount_program=/usr/bin/fuse-overlayfs \
 --storage-opt ignore_chown_errors=true"
export PODMAN

# refused migrate above: fail the sourcing shell's `source ... || exit` / set -e callers
if [ "${ZENITH_PODMAN_REFUSED:-0}" = 1 ]; then
    unset ZENITH_PODMAN_REFUSED
    return 1 2>/dev/null || exit 1
fi
