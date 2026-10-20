#!/usr/bin/env python3
"""
Let the NIXL region partitioner recognise a block-major KV pool whose backing
allocation was rounded up for alignment.

What breaks without this
------------------------
Disaggregated prefill of a model whose KV pool is block-major but padded by
the allocator cannot start. Every worker of both instances dies at KV
registration, before a token is served (measured case on 8x MI355X):

    AssertionError: KV cache view cannot be partitioned into NIXL regions:
      layer=model.layers.0.attn, cache_nbytes=47259616, num_blocks=40462,
      block_stride=1435968, physical_page_size=1728,
      cache_shape=(40462, 1, 2, 584), cache_stride=(1435968, 1168, 584, 1)

The actual cause, measured
--------------------------
base_worker.py classifies a layout as block-major with an exact equality:

    storage_is_block_major = num_blocks * block_stride == registration_len

where `registration_len` is `cache.untyped_storage().nbytes()`, i.e. the size
of the whole KV allocation. In the measured case the two sides are:

    registration_len        = 58,102,140,928
    num_blocks*block_stride = 58,102,137,216
    difference              =           3,712 bytes

3,712 bytes out of 54.1 GiB -- a relative error of 6.4e-8. The layout IS
block-major; the allocation is simply 4096-byte aligned and `num_blocks *
block_stride` is not a multiple of 4096 (it leaves 384). The allocator rounds
up, and an exact-equality test against a rounded-up size fails.

These numbers are measured, not inferred. A one-off diagnostic probe
printed every variable the branch reads, on all 8 workers:

    registration_len=58102140928 num_blocks=40462 block_stride=1435968
    block_major=False page_contiguous=False virtual_pages=False
    packed_storage=True offset_in_storage=0

`offset_in_storage=0` and `storage_ptr == data_ptr` say layer 0 starts at the
base of its own allocation, and `registration_len` is the pool -- the layer is
not a strided slice of a shared pool. A strided-region workaround built on
that wrong reading would have registered regions of 1,168 bytes where the
transfer needs 1,435,968 -- a connector that starts cleanly and moves 0.08% of
each block, i.e. silently wrong continuations rather than a crash.

The fix
-------
Replace the exact equality with the condition that the consuming branch
actually requires. That branch computes

    storage_block_len = registration_len // num_blocks

and uses the quotient as both the region length and the region stride. So the
layout is usable exactly when that quotient equals `block_stride` -- which is
the same thing as saying the slack is smaller than one block's worth of
rounding:

    storage_is_block_major = num_blocks > 0 and registration_len // num_blocks
                             == block_stride

Why this is the right predicate and not a loosened one
------------------------------------------------------
It is not a tolerance, a fudge factor, or an epsilon. It is the precondition of
the arithmetic downstream, stated directly. Everything it accepts produces
byte-identical regions to what the old test produced when the old test passed;
everything it rejects would have produced wrong regions.

Verified against the cases that matter (see the table in the module test
below, and `--self-test`):

  * exact fit (what already worked)          -> True, unchanged
  * the measured 4096B-aligned pool          -> True, newly accepted
  * two segments packed in one allocation    -> False, still rejected and still
                                                routed to the segmenting branch,
                                                which is what handles that case
  * allocation one byte too small            -> False
  * slack of num_blocks bytes or more        -> False, because at that point the
                                                quotient shifts and the derived
                                                block length would be wrong

The implicit tolerance is `num_blocks - 1` = 40,461 bytes; the measured slack
is 3,712, so there is 10.9x headroom. A genuine second segment is 54.1 GiB
past the boundary -- fifteen million times the tolerance -- so the two cases
cannot be confused by any allocator rounding policy.

Blast radius
------------
The guard only widens which layouts reach an already-existing branch; it adds
no branch and changes no region arithmetic. A layout that satisfied the old
equality satisfies the new condition identically (if `nb*bs == rl` then
`rl//nb == bs` for `nb>0`), so every configuration that registers today
registers to the same bytes. Non-disagg runs never call register_kv_caches at
all and are untouched.

The `num_blocks > 0` term guards the division. `num_blocks` is derived from
the cache config and is positive in every real configuration; it is there so a
degenerate config raises the informative assert below rather than
ZeroDivisionError from inside the predicate.

Upstream
--------
Report as: "NIXL block-major detection uses exact equality against
untyped_storage().nbytes(), which fails when the allocator pads the KV pool
for alignment (ROCm: 3,712 bytes of slack in 54.1 GiB)".
Build 0.2.1.dev32+gbc0f47cd0.
"""
import argparse
import os
import sys

TARGET = "vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py"
MARKER = "# zenith: tolerate allocator alignment slack"

OLD = """                storage_is_block_major = num_blocks * block_stride == registration_len"""

NEW = """                # zenith: tolerate allocator alignment slack.
                # The exact equality `num_blocks * block_stride ==
                # registration_len` fails when the backing allocation is
                # rounded up for alignment -- a measured gfx950 case leaves
                # 3,712 bytes of slack in a 54.1 GiB pool: the block span is
                # not a multiple of 4096. The branch below derives
                # `registration_len // num_blocks` and uses it as both the
                # region length and the region stride, so the condition it
                # actually needs is that this quotient equals block_stride.
                # That is implied by (and strictly weaker than) the old
                # equality, so layouts that registered before register to
                # identical bytes.
                storage_is_block_major = (
                    num_blocks > 0
                    and registration_len // num_blocks == block_stride
                )"""


def find_target(explicit):
    if explicit:
        return explicit if os.path.exists(explicit) else None
    for root in sys.path:
        cand = os.path.join(root, TARGET)
        if os.path.exists(cand):
            return cand
    return None


def self_test():
    """The predicate change, checked against the layouts it must separate."""
    def old(rl, nb, bs):
        return nb * bs == rl

    def new(rl, nb, bs):
        return nb > 0 and rl // nb == bs

    nb, bs = 40462, 1435968
    exact = nb * bs
    cases = [
        ("measured 4096B-aligned pool", 58102140928, True),
        ("exact fit (what already worked)", exact, True),
        ("slack of num_blocks-1 bytes (max safe)", exact + nb - 1, True),
        ("slack of num_blocks bytes (quotient shifts)", exact + nb, False),
        ("two segments in one allocation", 2 * exact, False),
        ("three segments", 3 * exact, False),
        ("allocation one byte too small", exact - 1, False),
        ("one full extra block", exact + bs, False),
    ]
    failures = 0
    print(f"{'case':44} {'old':>6} {'new':>6} {'want':>6}  verdict")
    for name, rl, want in cases:
        o, n = old(rl, nb, bs), new(rl, nb, bs)
        ok = n == want
        # Anything the old test accepted the new one must also accept, or the
        # patch would regress a working configuration.
        monotone = (not o) or n
        if not ok or not monotone:
            failures += 1
        print(f"{name:44} {str(o):>6} {str(n):>6} {str(want):>6}  "
              f"{'OK' if ok and monotone else 'FAIL'}")
    print()
    if failures:
        print(f"zenith: {failures} self-test failure(s)")
        return 1
    print("zenith: self-test passed (8/8), and the new predicate accepts "
          "everything the old one did")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None)
    # --check is what the Dockerfile's patch loop passes to every script.
    ap.add_argument("--dry-run", "--check", dest="dry_run", action="store_true")
    ap.add_argument("--self-test", action="store_true",
                    help="check the predicate against known layouts and exit")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    path = find_target(args.file)
    if not path:
        print(f"zenith: {TARGET} not found; nothing to patch")
        return 0

    with open(path) as fh:
        src = fh.read()

    if MARKER in src:
        print(f"zenith: {os.path.basename(path)} already tolerates allocator "
              f"alignment slack; no change")
        return 0

    # Upstream fixed this by 29468dde with an equivalent bounded-slack test:
    #   0 <= registration_len - num_blocks * block_stride < block_stride
    if "registration_len - num_blocks * block_stride < block_stride" in src:
        print(f"zenith: {os.path.basename(path)} already tolerates allocator "
              f"alignment slack upstream; no change")
        return 0

    n = src.count(OLD)
    if n != 1:
        print(f"zenith: WARNING -- anchor matched {n}x, expected 1; not applying")
        print("zenith: the block-major predicate moved; re-derive against the "
              "current source before forcing this patch")
        return 1

    if args.dry_run:
        print(f"zenith: would patch {path}")
        return 0

    with open(path, "w") as fh:
        fh.write(src.replace(OLD, NEW, 1))
    print(f"zenith: patched {path} -- block-major detection now tolerates "
          f"allocator alignment slack")
    return 0


if __name__ == "__main__":
    sys.exit(main())
