import sys
import torch
import torch.nn as nn
import ray
from ray.train.torch import TorchTrainer, TorchConfig
from ray.train import ScalingConfig, RunConfig, FailureConfig
from lux_ray_util import cpu_budget_per_worker
import socket

def health_check_func(config):
    torch.cuda.set_device(ray.train.torch.get_device())
    model = nn.Linear(10, 10).cuda()
    model = ray.train.torch.prepare_model(model)
    
    input_data = torch.randn(32, 10).cuda()
    loss = model(input_data).sum()
    loss.backward()

    ray.train.report({
        "node": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "status": "SUCCESS"
    })

def main():
    if len(sys.argv) < 4:
        print("Usage: python ray_health_overall.py <nnodes> <cpus_per_node> <gpus_per_node>")
        sys.exit(1)

    nnodes = int(sys.argv[1])
    cpus_per_node = int(sys.argv[2])
    gpus_per_node = int(sys.argv[3])
    
    total_workers = nnodes * gpus_per_node

    ray.init(address="auto")

    # size the per-worker CPU request against what ray *actually* owns, not the
    # node's nominal core count. run_container.sh starts the raylet with
    # --num-cpus=$(nproc)-8, so asking for cpus_per_node//gpus_per_node here
    # over-subscribes the cluster and the worker group never schedules -- it
    # just retries the placement group forever.
    cpus_per_worker = cpu_budget_per_worker(cpus_per_node, total_workers)

    trainer = TorchTrainer(
        train_loop_per_worker=health_check_func,
        scaling_config=ScalingConfig(
            num_workers=total_workers,
            use_gpu=True,
            resources_per_worker={"CPU": cpus_per_worker, "GPU": 1}
        ),
        # gloo, not nccl: this checks that Ray places a worker on every GPU and
        # that each can run a step. Over nccl, DDP's gradient all-reduce would
        # build one RCCL comm over the whole world -- a comm vLLM never creates,
        # and the one that fails at init on MI250X (ray_health_rccl.py). The
        # fabric is ray_health_rccl.py's job, on the comms vLLM actually uses.
        torch_config=TorchConfig(backend="gloo"),
        # a health check that retries forever is worse than one that fails: the
        # launcher blocks and the real cause scrolls away
        run_config=RunConfig(failure_config=FailureConfig(max_failures=0)),
    )

    print(f"Verifying {total_workers} workers on {nnodes} nodes "
          f"({cpus_per_worker} CPU + 1 GPU each)...")
    try:
        trainer.fit()
    except Exception as exc:                                 # noqa: BLE001
        print(f"RAY CLUSTER HEALTH CHECK FAILED: {exc}")
        print(f"cluster resources: {ray.cluster_resources()}")
        sys.exit(1)
    print("Ray Cluster Health Verified.")

if __name__ == "__main__":
    main()
