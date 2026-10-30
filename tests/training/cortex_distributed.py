r"""torchrun checks for DDP and real DeepSpeed ZeRO-0/1/2/3, including checkpoint resume.

FP32 gradients and AdamW updates are compared with the complete global batch. BF16 checks initialization, finite
gradients/updates, rank agreement and reload; it does not assume that DeepSpeed's FP32 optimizer masters equal BF16
AdamW.

Run the full matrix from the repository root with at least two CUDA GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 make tests_training_cortex
```

The target also runs under `make tests_training`. Set CORTEX_NPROC to change the worker count and
CORTEX_TRAINING_OUTPUT to choose a report directory; otherwise it prints a fresh temporary directory.

Select an engine and model combination directly:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
    --module tests.training.cortex_distributed --engine zero3 \
    --cases qwen3_moe:shared+routed --inits random --dtypes float32 \
    --output /tmp/cortex-zero3-validation
```

Each worker writes rank-<rank>.json in the output directory. Avoid glob metacharacters ([, ], *, ?) in output paths
because DeepSpeed uses whole-path glob matching to discover checkpoints.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import traceback
from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModelForCausalLM
from transformers.integrations.deepspeed import HfDeepSpeedConfig, unset_hf_deepspeed_config

from peft import CortexConfig, PeftModel, get_peft_model
from peft.tuners.cortex.layer import CortexLayer
from peft.tuners.cortex.structure import is_moe, read_components
from tests.testing_utils import create_tiny_causal_lm as tiny_model


DEVICE_CASES = [
    ("gemma3_text", "routed"),
    ("qwen3", "routed"),
    ("qwen3_moe", "shared"),
    ("qwen3_moe", "routed"),
    ("qwen3_moe", ["shared", "routed"]),
    ("gpt_oss", "shared"),
    ("gpt_oss", "routed"),
    ("gpt_oss", ["shared", "routed"]),
]

PUBLIC_ROUTING_CASES = [
    (kind, mode) for kind in ("olmoe", "qwen3_moe", "gpt_oss") for mode in ("shared", ["shared", "routed"])
]


def assert_initialization(model, stage, config):
    """Check output-preserving initialization against the requested bias policy."""
    init = config.init_weights
    context = nullcontext()
    if stage == 3:
        import deepspeed

        context = deepspeed.zero.GatheredParameters(list(model.parameters()), modifier_rank=None)
    with context:
        for layer in (module for module in model.modules() if isinstance(module, CortexLayer)):
            sources = read_components(layer.base_layer, include_down=False)
            branches = [("", layer.per_expert)] + ([("shared_", False)] if layer.role == "routed_public" else [])
            for prefix, per_expert in branches:
                down = getattr(layer, f"cortex_{prefix}down")["default"]
                assert torch.count_nonzero(down.weight) == 0
                assert down.bias is None or torch.count_nonzero(down.bias) == 0
                if init != "sample":
                    for kind in ("gate", "up"):
                        projection = getattr(layer, f"cortex_{prefix}{kind}")["default"]
                        if projection.bias is None:
                            continue
                        assert torch.isfinite(projection.bias).all()
                        if config.adapter_bias_init == "uniform":
                            # Packed [experts, out, in] projections use the logical input width.
                            bound = 1 / math.sqrt(projection.weight.shape[-1])
                            tolerance = torch.finfo(projection.bias.dtype).eps * bound
                            assert (projection.bias.detach().float().abs() <= bound + tolerance).all()
                        else:
                            assert torch.count_nonzero(projection.bias) == 0
                    continue
                for expert in range(layer.structure.num_experts if per_expert else 1):
                    gate_source, up_source = sources["gate"], sources["up"]
                    if per_expert:
                        gate_source, up_source = gate_source[expert], up_source[expert]
                    elif layer.structure.packed:
                        gate_source = gate_source.flatten(0, 1)
                        up_source = up_source.flatten(0, 1)
                    gate = getattr(layer, f"cortex_{prefix}gate")["default"].weight
                    up = getattr(layer, f"cortex_{prefix}up")["default"].weight
                    gate, up = (gate[expert], up[expert]) if per_expert else (gate, up)
                    matches = (gate[:, None] == gate_source[None]).all(-1)
                    assert (matches.sum(-1) == 1).all(), "sample must copy existing gate neurons"
                    indices = matches.int().argmax(-1)
                    torch.testing.assert_close(up, up_source[indices], rtol=0, atol=0)
                    for kind in ("gate", "up"):
                        projection = getattr(layer, f"cortex_{prefix}{kind}")["default"]
                        if projection.bias is None:
                            continue
                        bias = projection.bias[expert] if per_expert else projection.bias
                        if kind + "_bias" not in sources:
                            assert torch.count_nonzero(bias) == 0
                            continue
                        source = sources[kind + "_bias"]
                        if per_expert:
                            source = source[expert]
                        elif layer.structure.packed:
                            source = source.flatten(0, 1)
                        torch.testing.assert_close(bias, source[indices], rtol=0, atol=0)


MICRO_BATCH, SEQUENCE_LENGTH = 2, 5
OPTIMIZER_OPTIONS = {"lr": 0.01, "weight_decay": 0.02, "betas": (0.9, 0.999), "eps": 1e-8}
DISTRIBUTED_CASES = [(kind, mode, "unweighted") for kind, mode in DEVICE_CASES] + [
    (kind, mode, "routing") for kind, mode in PUBLIC_ROUTING_CASES
]


def case_name(kind, mode, public_weighting):
    selection = "+".join(mode) if isinstance(mode, list) else mode
    return f"{kind}:{selection}" + (":routing" if public_weighting == "routing" else "")


def run_name(kind, mode, init, dtype_name, public_weighting):
    # DeepSpeed globs the complete checkpoint path, including this directory.
    selection = "+".join(mode) if isinstance(mode, list) else mode
    name = f"{kind}-{selection}-{init}-{dtype_name}"
    return name + ("-routing" if public_weighting == "routing" else "")


def model_state(model, stage):
    context = nullcontext()
    if stage == 3:
        import deepspeed

        context = deepspeed.zero.GatheredParameters(list(model.parameters()), modifier_rank=None)
    with context:
        return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def attach_divergent_routing(model, rank, *, global_batch=False, routing_mass=False):
    """Force different expert subsets per rank while retaining native experts."""

    def hook(module, inputs, output):
        logits, weights, indices = output
        owners = (
            torch.arange(logits.shape[0], device=logits.device) // (MICRO_BATCH * SEQUENCE_LENGTH)
            if global_batch
            else torch.full((logits.shape[0],), rank, device=logits.device, dtype=torch.long)
        )
        slots = torch.arange(indices.shape[-1], device=indices.device)
        selected = (owners[:, None] + slots) % logits.shape[-1]
        scores = torch.full_like(weights, 1 / indices.shape[-1])
        if routing_mass:
            token_offsets = torch.arange(logits.shape[0], device=logits.device) % SEQUENCE_LENGTH
            mass = 0.35 + 0.05 * (owners % 4) + 0.02 * token_offsets
            scores = scores * mass[:, None].to(scores.dtype)
        return logits, scores, selected

    for module in model.modules():
        if type(module).__name__ in {"Qwen3MoeTopKRouter", "GptOssTopKRouter", "OlmoeTopKRouter"}:
            module.register_forward_hook(hook)


def ds_config(stage, dtype, world_size, offload="none"):
    if offload not in {"none", "optimizer", "parameters"}:
        raise ValueError(f"Unknown CPU offload mode: {offload}")
    if offload != "none" and stage != 3:
        raise ValueError("CPU offload validation requires ZeRO-3.")
    zero = {"stage": stage, "overlap_comm": False, "contiguous_gradients": True}
    if stage == 3:
        # Tiny models would otherwise fit under the persistence threshold and
        # fail to exercise real parameter sharding.
        zero.update(stage3_param_persistence_threshold=0, stage3_prefetch_bucket_size=100_000)
    if offload != "none":
        zero["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
    if offload == "parameters":
        zero["offload_param"] = {"device": "cpu", "pin_memory": True}
    return {
        "train_micro_batch_size_per_gpu": MICRO_BATCH,
        "train_batch_size": MICRO_BATCH * world_size,
        "gradient_accumulation_steps": 1,
        "gradient_clipping": 0.0,
        "fp16": {"enabled": False},
        "bf16": {"enabled": dtype == torch.bfloat16},
        "zero_allow_untested_optimizer": True,
        "zero_optimization": zero,
        "steps_per_print": 1_000_000,
        "wall_clock_breakdown": False,
    }


def initialize_deepspeed_communication():
    import deepspeed

    # zero.Init parses DeepSpeedConfig before it initializes deepspeed.comm.
    # Attach the existing torchrun process group first so config parsing sees
    # the real world size instead of its uninitialized-backend fallback of 1.
    deepspeed.init_distributed(dist_backend=dist.get_backend(), dist_init_required=False)
    assert deepspeed.comm.get_world_size() == dist.get_world_size()
    assert deepspeed.comm.get_rank() == dist.get_rank()


def make_runner(clean, config, engine, dtype, device, rank, offload="none"):
    if offload != "none" and (engine != "zero3" or device.type != "cuda"):
        raise ValueError("CPU offload validation requires a CUDA ZeRO-3 engine.")
    stage = None if engine == "ddp" else int(engine[-1])
    hf_config = None
    if stage == 3:
        import deepspeed

        initialize_deepspeed_communication()
        hf_config = HfDeepSpeedConfig(ds_config(stage, dtype, dist.get_world_size(), offload))
        base = AutoModelForCausalLM.from_config(deepcopy(clean.config), dtype=dtype)
        parameters = list(base.parameters())
        assert all(hasattr(parameter, "ds_id") for parameter in parameters)
        assert any(parameter.ds_tensor.numel() < parameter.ds_numel for parameter in parameters)
        if offload == "parameters":
            assert all(parameter.ds_tensor.device.type == "cpu" for parameter in parameters)
        with deepspeed.zero.GatheredParameters(parameters, modifier_rank=0):
            if rank == 0:
                base.load_state_dict(clean.state_dict(), strict=True)
    else:
        base = deepcopy(clean)
    model = get_peft_model(base, deepcopy(config), autocast_adapter_dtype=False)
    if stage == 3:
        # ZeRO-3 leaf reduction uses a full backward hook. A differentiable
        # embedding output makes it run after leaf parameter gradients even
        # though the original embedding weights remain frozen.
        model.enable_input_require_grads()
    # This occurs before engine.initialize: sample sees truly partitioned base
    # parameters under ZeRO-3 and must gather them rather than fall back.
    assert_initialization(model, stage, config)
    attach_divergent_routing(model, rank, routing_mass=config.public_weighting == "routing")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if offload != "none":
        from deepspeed.ops.adam import DeepSpeedCPUAdam

        optimizer = DeepSpeedCPUAdam(parameters, adamw_mode=True, **OPTIMIZER_OPTIONS)
    else:
        optimizer = torch.optim.AdamW(parameters, **OPTIMIZER_OPTIONS)
    if engine == "ddp":
        runner = DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=False,
        )
    else:
        import deepspeed
        from deepspeed.utils import set_z3_leaf_modules

        if stage == 3:
            classes = {type(module) for module in model.modules() if isinstance(module, CortexLayer) or is_moe(module)}
            set_z3_leaf_modules(model, list(classes))
        runner, optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config=ds_config(stage, dtype, dist.get_world_size(), offload),
            dist_init_required=False,
        )
        assert runner.zero_optimization_stage() == stage
        if stage in (1, 2):
            assert optimizer.partition_gradients == (stage == 2)
            assert all(count == dist.get_world_size() for count in optimizer.partition_count)
            for full, shard in zip(optimizer.bit16_groups_flat, optimizer.single_partition_of_fp32_groups):
                assert shard.numel() * dist.get_world_size() == full.numel()
                assert shard.numel() < full.numel(), f"ZeRO-{stage} optimizer masters must actually be partitioned."
        elif stage == 3:
            assert all(hasattr(parameter, "ds_id") for parameter in parameters)
            assert any(parameter.ds_tensor.numel() < parameter.ds_numel for parameter in parameters)
            assert_offload_state(runner, offload)
    return runner, optimizer, hf_config


def assert_offload_state(runner, offload, expected_step=0):
    """Verify CPU residency in the running engine, not just its requested configuration."""
    if offload == "none":
        return
    from deepspeed.ops.adam import DeepSpeedCPUAdam

    optimizer = runner.optimizer
    assert optimizer.offload_optimizer
    assert isinstance(optimizer.optimizer, DeepSpeedCPUAdam)
    masters = optimizer.fp32_partitioned_groups_flat
    assert masters and all(parameter.device.type == "cpu" for parameter in masters)
    if offload == "parameters":
        parameters = list(runner.module.parameters())
        assert parameters and all(parameter.ds_tensor.device.type == "cpu" for parameter in parameters)
    if expected_step:
        states = [optimizer.optimizer.state[parameter] for parameter in masters]
        assert states
        for state in states:
            assert state["step"] == expected_step
            for name in ("exp_avg", "exp_avg_sq"):
                assert state[name].device.type == "cpu" and torch.isfinite(state[name]).all()


def batch(step, world_size, device):
    return torch.cat(
        [
            torch.randint(
                3, 32, (MICRO_BATCH, SEQUENCE_LENGTH), generator=torch.Generator().manual_seed(700 + 11 * step + r)
            )
            for r in range(world_size)
        ]
    ).to(device)


def forward(runner, tokens, kind):
    kwargs = {"input_ids": tokens, "labels": tokens, "use_cache": False}
    if kind in {"qwen3_moe", "gpt_oss"}:
        kwargs["output_router_logits"] = False
    return runner(**kwargs)


def assert_rank_state(state, device):
    # Replicated engines must agree; ZeRO-3 has just assembled the same complete
    # logical tensors on all ranks. FP32 reference checks validate their values.
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        digest.update(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    fingerprints = [None] * dist.get_world_size()
    dist.all_gather_object(fingerprints, digest.hexdigest())
    assert len(set(fingerprints)) == 1, "Logical model parameters differ between ranks."


def training_step(runner, optimizer, reference, reference_optimizer, initial, engine, dtype, kind, step, device, rank):
    stage = None if engine == "ddp" else int(engine[-1])
    tokens = batch(step, dist.get_world_size(), device)
    local = tokens[rank * MICRO_BATCH : (rank + 1) * MICRO_BATCH]
    runner.train()
    reference.train()
    optimizer.zero_grad(set_to_none=True) if engine == "ddp" else runner.zero_grad()
    reference_optimizer.zero_grad(set_to_none=True)
    actual = forward(runner, local, kind)
    expected = forward(reference, tokens, kind)
    assert torch.isfinite(actual.loss) and torch.isfinite(actual.logits).all()
    reduced_loss = actual.loss.detach().float().clone()
    dist.all_reduce(reduced_loss)
    reduced_loss /= dist.get_world_size()
    tolerance = {"rtol": 2e-4, "atol": 2e-5} if dtype == torch.float32 else {"rtol": 0.06, "atol": 0.002}
    # BF16 optimizer masters can differ after the first update; its contract
    # here is finite/synchronized training and checkpoint recovery.
    if dtype == torch.float32 or step == 0:
        torch.testing.assert_close(reduced_loss, expected.loss.float(), **tolerance)
    actual.loss.backward() if engine == "ddp" else runner.backward(actual.loss)
    expected.loss.backward()
    expected_parameters = dict(reference.named_parameters())
    nonzero = False
    for name, parameter in runner.module.named_parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None
            continue
        if engine == "ddp":
            gradient = parameter.grad
        else:
            from deepspeed.utils import safe_get_full_grad

            gradient = safe_get_full_grad(parameter)
        assert gradient is not None and torch.isfinite(gradient).all(), f"Missing/non-finite gradient: {name}"
        nonzero |= bool(gradient.abs().sum() > 0)
        if dtype == torch.float32:
            reference_gradient = expected_parameters[name].grad.float()
            torch.testing.assert_close(gradient.float().to(reference_gradient.device), reference_gradient, **tolerance)
    assert nonzero
    optimizer.step() if engine == "ddp" else runner.step()
    reference_optimizer.step()
    current = model_state(runner.module, stage)
    expected_state = reference.state_dict()
    for name, tensor in current.items():
        assert torch.isfinite(tensor).all()
        if "cortex_" not in name:
            torch.testing.assert_close(tensor, initial[name], rtol=0, atol=0)
        elif dtype == torch.float32:
            torch.testing.assert_close(tensor, expected_state[name].detach().cpu(), **tolerance)
    assert any("cortex_" in name and not torch.equal(tensor, initial[name]) for name, tensor in current.items()), (
        "The optimizer must update adapter parameters."
    )
    assert_rank_state(current, device)
    return current


def release_runner(runner, engine):
    if engine != "ddp":
        runner.destroy()
    unset_hf_deepspeed_config()


def run_case(engine, kind, mode, init, dtype, device, rank, output, *, public_weighting="unweighted", offload="none"):
    stage = None if engine == "ddp" else int(engine[-1])
    unset_hf_deepspeed_config()
    torch.random.default_generator.manual_seed(123)
    if device.type == "cuda":
        torch.cuda.manual_seed(123)
    clean = tiny_model(kind).to(device=device, dtype=dtype).eval()
    with torch.no_grad():
        for name, parameter in clean.named_parameters():
            if name.endswith("gate_up_proj_bias"):
                # Nonzero native OSS biases make sample's paired bias copy a
                # meaningful check even before a training update.
                parameter.copy_(torch.linspace(-0.15, 0.15, parameter.numel(), device=device).reshape_as(parameter))
    config = CortexConfig(
        k=3,
        moe_target=mode,
        public_weighting=public_weighting,
        init_weights=init,
        init_noise_std=0,
        adapter_bias=kind == "gpt_oss",
        task_type="CAUSAL_LM",
    )
    reference = get_peft_model(deepcopy(clean), deepcopy(config), autocast_adapter_dtype=False)
    attach_divergent_routing(reference, rank, global_batch=True, routing_mass=public_weighting == "routing")
    reference_optimizer = torch.optim.AdamW(
        [parameter for parameter in reference.parameters() if parameter.requires_grad], **OPTIMIZER_OPTIONS
    )
    runner, optimizer, hf_config = make_runner(clean, config, engine, dtype, device, rank, offload)
    initial = model_state(runner.module, stage)
    reference.load_state_dict(initial, strict=True)
    assert_rank_state(initial, device)
    tokens = batch(0, dist.get_world_size(), device)[rank * MICRO_BATCH : (rank + 1) * MICRO_BATCH]
    runner.eval()
    with torch.no_grad():
        with runner.module.disable_adapter():
            baseline = forward(runner, tokens, kind).logits
        torch.testing.assert_close(forward(runner, tokens, kind).logits, baseline, rtol=0, atol=0)
    for step in range(2):
        saved = training_step(
            runner, optimizer, reference, reference_optimizer, initial, engine, dtype, kind, step, device, rank
        )
        assert_offload_state(runner, offload, expected_step=step + 1)
    runner.eval()
    with torch.no_grad():
        expected_logits = forward(runner, tokens, kind).logits.detach().clone()
    if rank == 0:
        runner.module.save_pretrained(output / "adapter", state_dict=saved, save_embedding_layers=False)
    dist.barrier()
    if engine == "ddp":
        if rank == 0:
            torch.save({"model": saved, "optimizer": optimizer.state_dict(), "step": 2}, output / "training.pt")
    else:
        assert runner.save_checkpoint(
            str(output / "deepspeed"), tag="step2", client_state={"step": 2}, save_latest=False
        )
    dist.barrier()
    release_runner(runner, engine)
    del runner, optimizer, hf_config
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # A fresh engine is required for a valid ZeRO-3 checkpoint resume.
    runner, optimizer, hf_config = make_runner(clean, config, engine, dtype, device, rank, offload)
    if engine == "ddp":
        checkpoint = torch.load(output / "training.pt", map_location=device, weights_only=True)
        runner.module.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        assert checkpoint["step"] == 2
    else:
        path, client = runner.load_checkpoint(str(output / "deepspeed"), tag="step2")
        assert path is not None and client["step"] == 2 and runner.global_steps == 2
    assert_offload_state(runner, offload, expected_step=2)
    restored_state = model_state(runner.module, stage)
    for name, tensor in saved.items():
        torch.testing.assert_close(restored_state[name], tensor, rtol=0, atol=0)
    runner.eval()
    with torch.no_grad():
        torch.testing.assert_close(forward(runner, tokens, kind).logits, expected_logits, rtol=0, atol=0)
    # Continuing the FP32 global-batch reference checks optimizer state too.
    training_step(runner, optimizer, reference, reference_optimizer, initial, engine, dtype, kind, 2, device, rank)
    assert_offload_state(runner, offload, expected_step=3)
    release_runner(runner, engine)
    del runner, optimizer, hf_config
    loaded = PeftModel.from_pretrained(
        deepcopy(clean), output / "adapter", adapter_name="renamed", autocast_adapter_dtype=False
    ).eval()
    attach_divergent_routing(loaded, rank, routing_mass=public_weighting == "routing")
    with torch.no_grad():
        torch.testing.assert_close(forward(loaded, tokens, kind).logits, expected_logits, rtol=0, atol=0)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["ddp", "zero0", "zero1", "zero2", "zero3"], required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--offload",
        choices=["none", "optimizer", "parameters"],
        default="none",
        help="ZeRO-3 CPU offload: optimizer only, or parameters plus optimizer; none preserves the default engine.",
    )
    parser.add_argument("--dtypes", nargs="+", choices=["float32", "bfloat16"], default=["float32", "bfloat16"])
    parser.add_argument("--inits", nargs="+", choices=["random", "sample"], default=["random", "sample"])
    parser.add_argument("--cases", nargs="+", choices=[case_name(*case) for case in DISTRIBUTED_CASES])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.offload != "none" and (args.engine != "zero3" or args.device != "cuda"):
        parser.error("--offload requires --engine zero3 --device cuda.")
    if "RANK" not in os.environ or int(os.environ.get("WORLD_SIZE", "1")) < 2:
        parser.error("Launch with torchrun and at least two workers.")
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if args.device == "cuda":
        if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
            parser.error("A CUDA device is required for every LOCAL_RANK.")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        if args.engine != "ddp":
            parser.error("CPU mode validates DDP test logic only; DeepSpeed cases require CUDA.")
        device = torch.device("cpu")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    dist.init_process_group("nccl" if device.type == "cuda" else "gloo", timeout=timedelta(seconds=300))
    bf16 = torch.tensor(
        int(device.type == "cpu" or torch.cuda.is_bf16_supported(including_emulation=False)), device=device
    )
    dist.all_reduce(bf16, op=dist.ReduceOp.MIN)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "engine": args.engine,
        "offload": args.offload,
        "rank": rank,
        "world_size": dist.get_world_size(),
        "device": str(device),
        "cases": [],
    }
    pending = None
    try:
        for kind, mode, public_weighting in DISTRIBUTED_CASES:
            if args.cases and case_name(kind, mode, public_weighting) not in args.cases:
                continue
            for init in args.inits:
                for dtype_name in args.dtypes:
                    pending = run_name(kind, mode, init, dtype_name, public_weighting)
                    if dtype_name == "bfloat16" and not bf16:
                        report["cases"].append(
                            {"case": pending, "status": "skipped", "reason": "Native BF16 unavailable"}
                        )
                        continue
                    if rank == 0:
                        print(json.dumps({"engine": args.engine, "case": pending, "status": "running"}), flush=True)
                    run_case(
                        args.engine,
                        kind,
                        mode,
                        init,
                        getattr(torch, dtype_name),
                        device,
                        rank,
                        args.output / pending,
                        public_weighting=public_weighting,
                        offload=args.offload,
                    )
                    report["cases"].append({"case": pending, "status": "passed"})
                    if rank == 0:
                        print(json.dumps(report["cases"][-1]), flush=True)
        report["status"] = "passed"
    except Exception:
        report["status"] = "failed"
        report["cases"].append({"case": pending, "status": "failed", "traceback": traceback.format_exc()})
        raise
    finally:
        (args.output / f"rank-{rank}.json").write_text(json.dumps(report, indent=2))
        unset_hf_deepspeed_config()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
