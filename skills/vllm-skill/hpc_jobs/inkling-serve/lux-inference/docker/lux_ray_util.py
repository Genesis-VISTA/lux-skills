"""
Shared helpers for the zenith ray health checks.

The one thing in here that matters: sizing a worker's CPU request against the
resources ray actually has, rather than against the node's nominal core count.

run_container.sh deliberately starts the raylet with `--num-cpus=$(nproc)-8` so
that the GCS/raylet/dashboard have cores of their own. A health check that then
asks for `cpus_per_node // gpus_per_node` (e.g. 256//8 = 32, times 8 workers =
256 CPUs) requests more than the 248 ray owns. Ray does not fail that request --
it parks the placement group and retries forever, so the check appears to hang
and then dies on the 60 s worker-group startup timeout. That failure mode looked
exactly like "vLLM won't start" from the outside.
"""

import ray


def cpu_budget_per_worker(cpus_per_node_hint, total_workers, reserve=2, minimum=1):
    """
    CPUs to request per training worker.

    Uses ray's own view of cluster CPUs when available (that is the number the
    scheduler actually validates against) and falls back to the caller's hint
    only if ray reports nothing. `reserve` leaves a little headroom so the
    trainer's own controller actor can still be placed.
    """
    try:
        cluster_cpus = int(ray.cluster_resources().get("CPU", 0))
    except Exception:                                        # noqa: BLE001
        cluster_cpus = 0

    if cluster_cpus <= 0:
        cluster_cpus = int(cpus_per_node_hint)

    usable = max(cluster_cpus - reserve, minimum)
    return max(usable // max(total_workers, 1), minimum)
