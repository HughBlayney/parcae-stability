# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Utility functions for training and inference."""

import inspect
import math
import shutil
import sys
import os

from dataclasses import asdict, is_dataclass, dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Mapping,
    Optional,
    TypeVar,
    Union,
    Literal,
    Callable,
    Type,
    TypedDict,
)
from functools import partial
from contextlib import nullcontext, contextmanager
from datetime import timedelta

import lightning as L
import torch
import torch.nn as nn
import torch.utils._device
import numpy as np
import random
import yaml
import time
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities.load import _lazy_load as lazy_load
from lightning.fabric.utilities.apply_func import convert_tensors_to_scalars, convert_to_tensors
from lightning.pytorch.loggers import WandbLogger
from typing_extensions import Self

if TYPE_CHECKING:
    import torch.distributed


def num_parameters(module: nn.Module, requires_grad: Optional[bool] = None) -> int:
    total: int = 0
    for p in module.parameters():
        if requires_grad is None or p.requires_grad == requires_grad:
            if hasattr(p, "quant_state"):
                # bitsandbytes 4bit layer support
                total += math.prod(p.quant_state.shape)  # type: ignore
            else:
                total += p.numel()
    return total

def num_recurrent_parameters(module: nn.Module, requires_grad: Optional[bool] = None) -> int:
    total: int = 0
    if hasattr(module, "transformer") and hasattr(module.transformer, "core_block") and module.transformer.core_block is not None:
        for p in module.transformer.core_block.parameters():
            if requires_grad is None or p.requires_grad == requires_grad:
                if hasattr(p, "quant_state"):
                    # bitsandbytes 4bit layer support
                    total += math.prod(p.quant_state.shape)  # type: ignore
                else:
                    total += p.numel()
        return total
    
    return 0


T = TypeVar("T")


def slice_logits_remap_labels(
    logits: torch.Tensor, targets: torch.Tensor, target_range: tuple, ignore_indices: list[int] = [-1]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slices logits to a specific index range, and remaps target labels to the new range, zero indexed."""
    assert min(target_range) >= 0, "smallest target_range value must be non-negative"
    assert max(target_range) < logits.size(-1), "largest target_range value must be less than the number of logits"
    assert sorted(target_range) == list(target_range), "target_range must be sorted already, sanity convention"

    label_id_to_tgt_id = {label_id: tgt_id for tgt_id, label_id in enumerate(target_range)}
    logits = logits[:, target_range]
    for label_id, tgt_id in label_id_to_tgt_id.items():
        targets[targets == label_id] = tgt_id

    return logits, targets


def chunked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 128,
    ignore_indices: list[int] = [-1],
    reduction: Optional[str] = None,
    training: bool = True,
    z_loss_eps=0.0,
    target_range: Optional[tuple[int]] = None,
    return_logits_targets: bool = False,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    ignore_index = ignore_indices[0] or -100
    for additional_ignore in ignore_indices[1:]:
        if additional_ignore is not None and additional_ignore != ignore_index:
            targets[targets == additional_ignore] = ignore_index

    cross_entropy_fn = partial(
        torch.nn.functional.cross_entropy,
        ignore_index=ignore_index,
    )

    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)

    if target_range is not None:
        # print("Using slice loss!")
        logits, targets = slice_logits_remap_labels(logits, targets, target_range, ignore_indices)
    else:
        # print("Using normal loss!")
        pass

    if chunk_size == 0:
        # no chunking at all
        if reduction is not None:
            loss = cross_entropy_fn(input=logits, target=targets, reduction=reduction)
        else:
            loss = cross_entropy_fn(input=logits, target=targets)
    else:
        # chunk cross entropy
        logit_chunks = logits.split(chunk_size)
        target_chunks = targets.split(chunk_size)
        losses = torch.zeros_like(targets, dtype=logits.dtype, device=logits.device)  # prealloc required for compile

        for idx, (logit_chunk, target_chunk) in enumerate(zip(logit_chunks, target_chunks)):
            loss_chunk = cross_entropy_fn(input=logit_chunk, target=target_chunk, reduction="none")
            losses[idx * chunk_size : (idx + 1) * chunk_size] = loss_chunk

        non_masked_elems = (targets != ignore_index).sum().clamp(min=1.0)
        loss = losses.sum() / non_masked_elems

    if z_loss_eps > 0.0 and training:
        loss += z_loss_eps * torch.logsumexp(logits, dim=-1).pow(2).mean()

    if not return_logits_targets:
        return loss
    else:
        non_masked_mask = targets != ignore_index
        return loss, logits[non_masked_mask, :], targets[non_masked_mask]


def map_old_state_dict_weights(state_dict: Dict, mapping: Mapping, prefix: str) -> Dict:
    for checkpoint_name, attribute_name in mapping.items():
        full_checkpoint_name = prefix + checkpoint_name
        if full_checkpoint_name in state_dict:
            full_attribute_name = prefix + attribute_name
            state_dict[full_attribute_name] = state_dict.pop(full_checkpoint_name)
    return state_dict


def get_default_supported_precision(training: bool) -> str:
    from lightning.fabric.accelerators import MPSAccelerator

    if MPSAccelerator.is_available() or (torch.cuda.is_available() and not torch.cuda.is_bf16_supported()):
        return "16-mixed" if training else "16-true"
    return "bf16-mixed" if training else "bf16-true"


def load_checkpoint(fabric: L.Fabric, model: nn.Module, checkpoint_path: str, strict: bool = True) -> None:
    if isinstance(fabric.strategy, FSDPStrategy):
        fabric.load_raw(checkpoint_path, model, strict=strict)
    elif fabric.strategy == "AxoNNFabric":
        state_dict = lazy_load(checkpoint_path)
        state_dict = state_dict.get("model", state_dict)
        model.module.load_state_dict(state_dict, strict=strict)
    else:
        state_dict = lazy_load(checkpoint_path)
        state_dict = state_dict.get("model", state_dict)
        model.load_state_dict(state_dict, strict=strict)


def flops_per_param(max_seq_length: int, n_layer: int, n_embd: int, n_params: int) -> int:
    flops_per_token = 2 * n_params  # each parameter is used for a MAC (2 FLOPS) per network operation
    flops_per_seq = flops_per_token * max_seq_length
    attn_flops_per_seq = n_layer * 2 * 2 * (n_embd * (max_seq_length**2))
    return flops_per_seq + attn_flops_per_seq


def estimate_flops(model, training: bool) -> int:
    """Measures estimated FLOPs for MFU.

    Refs:
        * https://ar5iv.labs.arxiv.org/html/2205.05198#A1
        * https://ar5iv.labs.arxiv.org/html/2204.02311#A2
    """
    n_trainable_params = num_parameters(model, requires_grad=True)
    trainable_flops = flops_per_param(
        model.max_seq_length, model.config.n_layer, model.config.n_embd, n_trainable_params
    )
    ops_per_step = 3 if training else 1
    n_frozen_params = num_parameters(model, requires_grad=False)
    frozen_flops = flops_per_param(model.max_seq_length, model.config.n_layer, model.config.n_embd, n_frozen_params)
    frozen_ops_per_step = 2 if training else 1
    return ops_per_step * trainable_flops + frozen_ops_per_step * frozen_flops


def simple_gptneox_tflops(metrics, fabric, cfg, batch_size=None):
    """Estimate the TFLOPs using the GPT-NeoX napkin math.

    https://github.com/EleutherAI/gpt-neox/blob/main/megatron/logging.py#L82
    Think about significant approximations and potential correctness issues under FSDP or AxoNN.
    """
    iter_time_s = metrics["seconds/step"]

    world_size = fabric.world_size
    vocab_size = cfg.model_config.vocab_size
    batch_size = cfg.world_batch_size if batch_size is None else batch_size
    seq_len = cfg.block_size
    hidden_size = cfg.model_config.n_embd
    num_layers = cfg.model_config.n_layer
    ckpt_activations_factor = 4 if cfg.gradient_checkpointing else 3
    flops_per_iteration = (
        24
        * ckpt_activations_factor
        * batch_size
        * seq_len
        * num_layers
        * (hidden_size**2)
        * (1.0 + (seq_len / (6.0 * hidden_size)) + (vocab_size / (16.0 * num_layers * hidden_size)))
    )
    return (flops_per_iteration / (iter_time_s * world_size)) / 1e12


def simple_axonn_tflops(metrics, fabric, cfg, batch_size):
    """This function copied in from megatron_logging.py.
    The GemmaMLP catch is probably wrong. Unsure about usage."""

    world_size = fabric.world_size
    iter_time_s = metrics["seconds/step"]
    config = cfg.model_config

    N = config.n_layer
    B = batch_size
    S = config.block_size
    V = config.padded_vocab_size
    H = config.n_embd
    IH = config.intermediate_size

    if config.mlp_class_name == "LLaMAMLP" or config.mlp_class_name == "GemmaMLP":
        linear_flops = N * (32 * B * S * H * H + 24 * B * S * H * IH)
    elif config.mlp_class_name == "GptNeoxMLP":
        linear_flops = N * (32 * B * S * H * H + 16 * B * S * H * IH)
    else:
        raise NotImplementedError
    attention_flops = N * (16 * B * S * S * H)
    head_flops = 6 * B * S * H * V
    if cfg.gradient_checkpointing:
        flops = linear_flops + attention_flops + head_flops
    else:
        flops = 3 / 4 * (linear_flops + attention_flops) + head_flops

    return flops / 1e12 / iter_time_s / world_size


class CycleIterator:
    """An iterator that cycles through an iterable indefinitely.

    Example:
        >>> iterator = CycleIterator([1, 2, 3])
        >>> [next(iterator) for _ in range(5)]
        [1, 2, 3, 1, 2]

    Note:
        Unlike ``itertools.cycle``, this iterator does not cache the values of the iterable.
    """

    def __init__(self, iterable: Iterable) -> None:
        self.iterable = iterable
        self.epoch = 0
        self._iterator = None

    def __next__(self) -> Any:
        if self._iterator is None:
            self._iterator = iter(self.iterable)
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.iterable)
            self.epoch += 1
            return next(self._iterator)

    def __iter__(self) -> Self:
        return self


def copy_config_files(source_dir: Path, out_dir: Path) -> None:
    """Copies the specified configuration and tokenizer files into the output directory."""

    config_files = ["generation_config.json", "model_config.yaml"]
    tokenizer_files = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"]

    for file_name in config_files + tokenizer_files:
        src_path = source_dir / file_name
        if src_path.exists():
            shutil.copy(src_path, out_dir)


def CLI(*args: Any, **kwargs: Any) -> Any:
    from jsonargparse import CLI, set_config_read_mode, set_docstring_parse_options

    set_docstring_parse_options(attribute_docstrings=True)
    set_config_read_mode(urls_enabled=True)

    kwargs.setdefault("as_positional", False)

    return CLI(*args, **kwargs)


def capture_hparams() -> Dict[str, Any]:
    """Captures the local variables ('hyperparameters') from where this function gets called."""
    caller_frame = inspect.currentframe().f_back  # type: ignore
    locals_of_caller = caller_frame.f_locals  # type: ignore
    hparams = {}
    for name, value in locals_of_caller.items():
        if value is None or isinstance(value, (int, float, str, bool, Path)):
            hparams[name] = value
        elif is_dataclass(value):
            hparams[name] = asdict(value)  # type: ignore
        else:
            hparams[name] = str(value)
    return hparams


def save_hyperparameters(function: Callable, checkpoint_dir: Path) -> None:
    """Captures the CLI parameters passed to `function` without running `function` and saves them to the checkpoint."""
    from jsonargparse import capture_parser

    # TODO: Make this more robust
    # This hack strips away the subcommands from the top-level CLI
    # to parse the file as if it was called as a script
    known_commands = [
        ("finetune", "full"),
        ("finetune", "lora"),
        ("finetune", "adapter"),
        ("finetune", "adapter_v2"),
        ("pretrain",),
    ]
    for known_command in known_commands:
        unwanted = slice(1, 1 + len(known_command))
        if tuple(sys.argv[unwanted]) == known_command:
            sys.argv[unwanted] = []

    parser = capture_parser(lambda: CLI(function))
    config = parser.parse_args()
    parser.save(config, checkpoint_dir / "hyperparameters.yaml", overwrite=True)


def save_config(config, checkpoint_dir: Path) -> None:
    config_dict = asdict(config)
    with open(checkpoint_dir / "model_config.yaml", "w", encoding="utf-8") as fp:
        yaml.dump(config_dict, fp)


def parse_devices(devices: Union[str, int]) -> int:
    if devices in (-1, "auto"):
        return torch.cuda.device_count() or 1
    if isinstance(devices, int) and devices > 0:
        return devices
    raise ValueError(f"Devices must be 'auto' or a positive integer, got: {devices!r}")


def choose_logger(
    logger_name: Literal["csv", "tensorboard", "wandb"],
    out_dir: Path,
    name: str,
    log_interval: int = 1,
    resume: Optional[bool] = None,
    **kwargs: Any,
):
    if logger_name == "csv":
        return CSVLogger(root_dir=(out_dir / "logs"), name="csv", flush_logs_every_n_steps=log_interval, **kwargs)
    if logger_name == "tensorboard":
        return TensorBoardLogger(root_dir=(out_dir / "logs"), name="tensorboard", **kwargs)
    if logger_name == "wandb":
        return WandbLogger(project=name, resume=resume, **kwargs)
    else:
        raise ValueError(
            f"`--logger_name={logger_name}` is not a valid option. Choose from 'csv', 'tensorboard', 'wandb'."
        )


def extend_checkpoint_dir(checkpoint_dir: Path) -> Path:
    new_checkpoint_dir = "checkpoints" / checkpoint_dir
    should_return_new_dir = (
        not checkpoint_dir.is_dir()
        and checkpoint_dir.parts[0] != "checkpoints"
        and not checkpoint_dir.is_absolute()
        and new_checkpoint_dir.exists()
    )
    return new_checkpoint_dir if should_return_new_dir else checkpoint_dir


def fsdp_auto_wrap_policy(set_of_transformer_layers: set[Type[torch.nn.Module]]):
    import functools
    from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy

    def lambda_policy_fn(module):
        if (
            len(list(module.named_children())) == 0
            and getattr(module, "weight", None) is not None
            and module.weight.requires_grad
        ):
            return True
        return False

    lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls=set_of_transformer_layers
    )

    auto_wrap_policy = functools.partial(_or_policy, policies=[lambda_policy, transformer_wrap_policy])

    return auto_wrap_policy


# wrapper for lightning fabrics
class LightningFabric:
    def __init__(self, devices, strategy, precision, loggers=[], num_nodes: int = 1):
        fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision, loggers=loggers, num_nodes=num_nodes)
        self._underlying_fabric = fabric
        self._precision = precision
        self.strategy_name = strategy.__class__.__name__

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self._underlying_fabric, attr)

    def get_prefix_for_checkpoint(self):
        return f"checkpoints-{self.strategy_name}"

    def log_dict(self, metrics, step=None):
        """Log metrics to wandb, converting tensors to scalars."""
        if self.global_rank == 0:
            metrics = convert_tensors_to_scalars(metrics)
            for logger in self.loggers:
                logger.log_metrics(metrics=metrics, step=step)

    def log_to_summary(self, dict_of_values):
        if self.global_rank == 0:
            experiment = self.loggers[0].experiment
            if hasattr(experiment, "summary"):  # wandb
                for key, value in dict_of_values.items():
                    experiment.summary[key] = value
            else:  # e.g. CSVLogger: record summary values as a metrics row without a step
                self.loggers[0].log_metrics({f"summary/{k}": v for k, v in dict_of_values.items()})

    def log_chart(self, dict_of_charts):
        if self.global_rank == 0:
            experiment = self.loggers[0].experiment
            if hasattr(experiment, "log"):
                experiment.log(dict_of_charts)

    def setup(self, model: torch.nn.Module, compile: bool = False, compile_ddp: bool = True):
        """Compiling DDP turns out not to improve speed on the A6000 ada cards"""

        if "DDP" in self.strategy_name and compile and compile_ddp:
            # the default dtype needs to be set before the lightning fabric handles it, otherwise DDP cannot be compiled
            # due to default dtypes changes in the compiled program
            if "bf16-true" in self._precision:
                torch.set_default_dtype(torch.bfloat16)
            elif "16-true" in self._precision:
                torch.set_default_dtype(torch.float16)

            model = self._underlying_fabric.setup(model)
            if compile:
                model = torch.compile(model, fullgraph=False, dynamic=False, mode="max-autotune-no-cudagraphs")  # type: ignore
        else:
            if compile:
                # error on dynamic shape
                model = torch.compile(model, fullgraph=False, dynamic=False, mode="max-autotune-no-cudagraphs")  # type: ignore
            model = self._underlying_fabric.setup(model)

        return model


@torch.no_grad()
def _allreduce_bucketed(model, unaveraged_grads, world_size=1):
    """
    Minimal memory implementation with correct gradient handling
    """
    bucket_size = 255 * 1024 * 1024 // 4  # 255MB in fp32 elements
    current_bucket = []
    current_numel = 0

    for grad in unaveraged_grads:
        grad_numel = grad.numel()

        # Check if adding this gradient would overflow bucket
        if current_numel + grad_numel > bucket_size and current_bucket:
            # Reduce current bucket first
            bucket_tensor = torch.cat(current_bucket)
            print(f"BUCKETGRAD: Reducing {bucket_tensor.numel()}")
            torch.distributed.all_reduce(bucket_tensor)
            print("Reduce passed")
            # Copy reduced values back to original grads
            torch.cuda.current_stream().synchronize()
            print("Reduce really passed")
            offset = 0
            for g in current_bucket:
                g.copy_(bucket_tensor[offset : offset + g.numel()])
                offset += g.numel()

            # Reset bucket
            current_bucket = []
            current_numel = 0

        # Add gradient to current or new bucket
        current_bucket.append(grad.view(-1))
        current_numel += grad_numel

    # Handle final bucket
    if current_bucket:
        bucket_tensor = torch.cat(current_bucket).div_(world_size)
        torch.distributed.all_reduce(bucket_tensor)

        offset = 0
        for grad in current_bucket:
            grad.copy_(bucket_tensor[offset : offset + grad.numel()])
            offset += grad.numel()

    # Update model gradients
    for param, grad in zip(model.parameters(), unaveraged_grads):
        param.grad = grad.view_as(param).data.div_(world_size)


@torch.no_grad()
def _allreduce_coalesced(model, unaveraged_grads, world_size=1):
    """This helps a tiny bit in multi-node settings and doesn't appear to hurt single-node performance."""
    concat_grad = torch.cat([g.reshape(-1) / world_size for g in unaveraged_grads])
    torch.distributed.all_reduce(concat_grad, async_op=False)

    pointer = 0
    for param in model.parameters():
        num_param = param.numel()
        param.grad = concat_grad[pointer : pointer + num_param].view_as(param).data
        pointer += num_param


# @torch.compile() # funny hang on A100s + very large model?
# maybe the graph is too long
@torch.no_grad()
def _allreduce_chunk_stream(model, world_size=1, device=torch.device("cpu"), safety_goggles_on=False):
    """Simple implementation with fixed MB chunks that can span gradients"""
    chunk_size = 1024 * 1024 * 64 // 4  # 64MB fp32 as in warmup

    chunk = torch.empty(chunk_size, dtype=torch.float32, device=device)
    chunk_index = 0
    param_refs = []

    for p in model.parameters():
        if p.grad is not None:
            grad_index = 0
            while grad_index < p.grad.numel():
                # Fold
                n = min(chunk_size - chunk_index, p.grad.numel() - grad_index)
                chunk[chunk_index : chunk_index + n] = p.grad.view(-1)[grad_index : grad_index + n]
                param_refs.append((p, grad_index, chunk_index, n))
                chunk_index += n
                grad_index += n

                if chunk_index == chunk_size:
                    # Average over ranks
                    torch.distributed.all_reduce(chunk)
                    chunk.div_(world_size)
                    if safety_goggles_on:
                        torch.cuda.current_stream().synchronize()
                    # Unfold
                    for param, start_p, start_c, numel in param_refs:
                        param.grad.view(-1)[start_p : start_p + numel] = chunk[start_c : start_c + numel]

                    # Reset
                    chunk = torch.empty(chunk_size, dtype=torch.float32, device=device)
                    chunk_index = 0
                    param_refs = []
    # Handle final chunk:
    if chunk_index > 0:
        torch.distributed.all_reduce(chunk)  # keep consistent MB size
        chunk.div_(world_size)
        for param, start_p, start_c, numel in param_refs:
            param.grad.view(-1)[start_p : start_p + numel].copy_(chunk[start_c : start_c + numel])


@torch.no_grad()
def _allreduce_chunk_stream_extra_copy(model, world_size=1, device=torch.device("cpu"), safety_goggles_on=False):
    """Simplified version at the cost of an extra gradient copy"""
    chunk_size = 1024 * 1024 * 64 // 4  # 64MB fp32 as in warmup

    chunk = torch.empty(chunk_size, dtype=torch.float32, device=device)

    flat_grads = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])

    grad_index = 0
    while grad_index < flat_grads.numel():
        chunk = flat_grads[grad_index : grad_index + chunk_size]
        torch.distributed.all_reduce(chunk)
        chunk.div_(world_size)
        if safety_goggles_on:
            torch.cuda.current_stream().synchronize()

        flat_grads[grad_index : grad_index + chunk_size] = chunk
        grad_index += chunk_size

    flat_grad_index = 0
    for p in model.parameters():
        p.grad.copy_(flat_grads[flat_grad_index : flat_grad_index + p.numel()].view_as(p))
        flat_grad_index += p.numel()


class SimpleFabric:
    """Simple ddp-based fabric without lightning cruft, can be a template for other fabrics"""

    verbose = False

    def __init__(self, precision="bf16-true", loggers=None, local_device_init=True, use_dumb_allreduce=True):
        self.precision = precision
        self.rank = int(os.getenv("SLURM_PROCID", os.getenv("RANK", "0")))
        self.local_device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")

        torch.distributed.init_process_group(
            backend="nccl",  # can disable to build both nccl and gloo backends, but gloo may be a gamble on frontier
            rank=self.rank,
            world_size=int(os.getenv("SLURM_NTASKS", os.getenv("WORLD_SIZE", -1))),
            device_id=self.local_device if local_device_init else None,  # this immediately forms the NCCL communicator
            timeout=timedelta(minutes=15),
        )
        loggers = loggers if loggers is not None else []
        self._loggers = loggers if isinstance(loggers, list) else [loggers]

        if precision == "bf16-true":
            self.dtype = torch.bfloat16
        elif precision in ["fp16-mixed", "bf16-mixed", "16-mixed"]:
            self.dtype = torch.float32
        else:
            raise ValueError(f"Invalid precision type {precision} provided.")
        # Please be careful with vague constructors with this turned on, float32 constructors should say so:
        torch.set_default_dtype(self.dtype)
        torch.cuda.set_device(self.local_device)
        self.global_rank_for_creating_dataloader = self.rank
        self.world_size = torch.distributed.get_world_size()
        self.local_devices = min(int(os.getenv("SLURM_NTASKS_PER_NODE", torch.cuda.device_count())), self.world_size)
        self.local_rank = int(os.getenv("LOCAL_RANK", self.rank % self.local_devices))
        assert self.local_rank == self.rank % self.local_devices
        self.strategy_name = "SimpleDDP"
        self.use_dumb_allreduce = use_dumb_allreduce
        self.no_sync = False

    def launch(self):
        pass

    def print(self, msg, ranks=[0]):
        if torch.distributed.get_rank() in ranks:
            print(msg, flush=True)

    def log_dict(self, metrics, step=None):
        if self.rank == 0:
            metrics = convert_tensors_to_scalars(metrics)
            for logger in self._loggers:
                logger.log_metrics(metrics=metrics, step=step)

    def log_to_summary(self, dict_of_values):
        if self.rank == 0:
            for key, value in dict_of_values.items():
                self.logger.experiment.summary[key] = value

    def log_chart(self, dict_of_charts):
        if self.rank == 0:
            self.logger.experiment.log(dict_of_charts)

    @property
    def device(self):
        return self.local_device

    @property
    def global_rank(self):
        return self.rank

    @property
    def logger(self):
        return self._loggers[0]

    @property
    def strategy(self):
        return "SimpleDDP"

    def setup_dataloaders(self, train_dataloader, val_dataloader):
        return train_dataloader, val_dataloader

    def setup(self, model, compile: bool = False):
        model = model.to(self.device)
        # wrap DDP and AMP
        if not self.use_dumb_allreduce:
            self._ddp_model_ref = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.rank])
        else:
            self._ddp_model_ref = model
        self.model_ref = _FabricModule(
            forward_module=self._ddp_model_ref,
            original_module=model,
            precision=self.precision,
        )
        # compile after DDP to compile DDP calls
        if compile:
            self.model_ref = torch.compile(
                self.model_ref, fullgraph=False, dynamic=False, mode="max-autotune-no-cudagraphs"
            )
        return self.model_ref

    def setup_optimizers(self, optimizers):
        return optimizers

    def all_reduce(self, data, reduce_op="mean") -> torch.Tensor:
        """
        All reduce over data parallel groups
        """
        op = torch.distributed.ReduceOp.AVG
        if reduce_op == "sum":
            op = torch.distributed.ReduceOp.SUM
        data = convert_to_tensors(data, device=self.device)

        torch.distributed.all_reduce(data, op=op)
        return data

    def barrier(self):
        torch.distributed.barrier(device_ids=[self.device.index])

    def broadcast(self, obj, src: int = 0):
        if not torch.distributed.is_initialized():
            return obj

        obj = [obj]
        torch.distributed.broadcast_object_list(obj, src)
        return obj[0]

    def backward(self, loss, model=None):
        loss.backward()
        if not self.no_sync and self.use_dumb_allreduce:
            _allreduce_chunk_stream(model, world_size=self.world_size, device=self.local_device)
            # _allreduce_chunk_stream_extra_copy(model, world_size=self.world_size, device=self.local_device)

    def clip_gradients(self, model, optimizer, max_norm, error_if_nonfinite=False):
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm, norm_type=2.0, error_if_nonfinite=error_if_nonfinite
        )
        return grad_norm

    def save(self, checkpoint_file_location, state, overwrite=True, strict=False):
        """New strategy: First 8 local ranks save!"""
        save_state_dict = {}
        for key, value in state.items():
            if key in ["optimizer", "model"] or "train_dataloader" in key:
                save_state_dict[key] = value.state_dict()  # form states on all devices to prevent deadlock!
            elif key == "packing_collator" and value is not None:
                save_state_dict[key] = value.state_dict()  # save packing collator buffer
            else:
                save_state_dict[key] = value

        if self.rank < self.local_devices:  # first 8 ranks!!
            checkpoint_folder = os.path.dirname(checkpoint_file_location)
            if not os.path.exists(checkpoint_folder):
                os.makedirs(checkpoint_folder)
            checkpoint_file = f"{checkpoint_file_location}_{self.local_rank}.pth"
            if os.path.exists(checkpoint_file) and not overwrite:
                raise ValueError(f"Checkpoint {checkpoint_file} already exists")
            elif os.path.exists(checkpoint_file):
                print(f"Checkpoint {checkpoint_file} overwritten with new version.")
            torch.save(save_state_dict, checkpoint_file)
            if self.verbose:
                print(f"Rank {self.rank} - local {self.local_rank} saved {checkpoint_file}.")

    def load(self, checkpoint_file_location, state, strict=False):
        checkpoint_file = f"{checkpoint_file_location}_{self.local_rank}.pth"
        if self.verbose:
            print(f"Rank {self.rank} - local {self.local_rank} loaded {checkpoint_file}.")
        checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"), weights_only=False)
        
        # Track what was loaded for debugging
        loaded_keys = []
        skipped_keys = []
        
        for key, value in checkpoint.items():
            if key in ["optimizer", "model"] or "train_dataloader" in key:
                try:
                    state[key].load_state_dict(value)
                    loaded_keys.append(f"{key} (state_dict)")
                except Exception as e:
                    print(f"WARNING: Failed to load {key}: {e}")
                    skipped_keys.append(key)
            elif key == "packing_collator" and state.get("packing_collator") is not None:
                state[key].load_state_dict(value)
                loaded_keys.append(f"{key} (state_dict)")
            elif key in ["tokenizer", "val_dataloader", "data_scheduler", "lr_scheduler", "flop_breakdown"]:
                # Skip objects that don't need to be restored or are recreated fresh
                skipped_keys.append(key)
            else:
                state[key] = value
                loaded_keys.append(key)
        
        if self.verbose or self.rank == 0:
            print(f"Checkpoint loaded. Keys restored: {len(loaded_keys)}, skipped: {len(skipped_keys)}")
            if skipped_keys:
                print(f"  Skipped keys: {skipped_keys}")

    @contextmanager
    def no_backward_sync(self, model, enabled=True):
        try:
            context = self._ddp_model_ref.no_sync() if enabled and not self.use_dumb_allreduce else nullcontext()
            with context:
                self.no_sync = bool(enabled)
                yield
        finally:
            self.no_sync = False

    def get_prefix_for_checkpoint(self):
        return "checkpoints-sane-ddp"

    @contextmanager
    def init_module(self, empty_init=False):
        yield None

    def seed_everything(self, seed: Optional[int] = None, workers: bool = False):
        max_seed_value = np.iinfo(np.uint32).max
        min_seed_value = np.iinfo(np.uint32).min
        if seed is None:
            env_seed = os.environ.get("PL_GLOBAL_SEED")
            if env_seed is None:
                seed = 0
                self.print(f"Warning: No seed found, seed set to {seed}")
            else:
                try:
                    seed = int(env_seed)
                except ValueError:
                    seed = 0
                    self.print(f"Warning: Invalid seed found: {repr(env_seed)}, seed set to {seed}")
        elif not isinstance(seed, int):
            seed = int(seed)

        if not (min_seed_value <= seed <= max_seed_value):
            self.print(f"Warning: {seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}")
            seed = 0

        # print(rank_prefixed_message(f"Seed set to {seed}", self.global_rank))
        os.environ["PL_GLOBAL_SEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"

        return seed


class _FabricModule(torch.nn.Module):
    def __init__(self, forward_module, precision, original_module=None) -> None:
        """The FabricModule is a thin wrapper around the :class:`torch.nn.Module` and handles precision / autocast
        automatically for the forward pass.

        The underlying wrapped module can be accessed via the property :attr:`module`.

        Args:
            forward_module: The module to wrap the ``forward`` method on.
            strategy: Reference to the strategy for handling precision etc.
            original_module: The original, unmodified module as passed into the
                :meth:`lightning.fabric.fabric.Fabric.setup` method. This is needed when attribute lookup
                on this wrapper should pass through to the original module.

        """
        super().__init__()
        self._forward_module = forward_module
        self._original_module = original_module or forward_module
        self.precision = precision

        class AmpSettings(TypedDict):
            device_type: str
            dtype: torch.dtype
            enabled: bool

        self.amp_settings: AmpSettings = {
            "device_type": "cuda",
            "enabled": "mixed" in precision,
            "dtype": torch.bfloat16 if "bf16" in precision else torch.float16,
        }
        self._fabric_module_initialized = True

    @property
    def module(self):
        return self._original_module or self._forward_module

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Casts all inputs to the right precision and handles autocast for operations in the module forward method."""
        with torch.autocast(**self.amp_settings):
            output = self._forward_module(*args, **kwargs)
        return output

    def __getattr__(self, item: Any) -> Any:
        try:
            # __getattr__ gets called as a last resort if the attribute does not exist
            # call nn.Module's implementation first
            return super().__getattr__(item)
        except AttributeError:
            # If the attribute is not available on the _FabricModule wrapper, redirect to the wrapped nn.Module
            original_module = super().__getattr__("_original_module")
            attr = getattr(original_module, item)
            return attr

    def __setattr__(self, name: str, value: Any) -> None:
        if not getattr(self, "_fabric_module_initialized", False):
            super().__setattr__(name, value)
            return

        original_has_attr = hasattr(self._original_module, name)
        fabric_has_attr = name in dir(self)

        if not (original_has_attr or fabric_has_attr):
            setattr(self._original_module, name, value)
            return

        if original_has_attr:
            setattr(self._original_module, name, value)
        if fabric_has_attr:
            super().__setattr__(name, value)


def param_count_estimator(
    width=None, depth=None, vocab_size=None, n_head=None, head_size=None, n_query_groups=None, intermediate_size=None
):
    # Embedding layer parameters
    embedding_params = vocab_size * width  # type: ignore

    # Attention parameters: attn + proj
    attn_shape = (n_head + 2 * n_query_groups) * head_size  # type: ignore
    attn_params = (width * attn_shape) + (head_size * n_head * width)  # type: ignore

    # MLP parameters: fc_1 + fc_2 + proj
    mlp_params = (width * intermediate_size) + (width * intermediate_size) + (intermediate_size * width)  # type: ignore

    # RMSNorm parameters: 2 per block + 1 final norm
    norm_params_per_block = 2 * width  # type: ignore

    # Total per block
    total_block_params = attn_params + mlp_params + norm_params_per_block

    # All layers (blocks)
    total_params = total_block_params * depth

    # Final LayerNorm and LM Head
    final_norm_params = width
    lm_head_params = width * vocab_size  # type: ignore

    # Total model parameters
    return total_params + embedding_params + final_norm_params + lm_head_params


T = TypeVar("T")

global hash_table
hash_table = None
table_size = 1_000_003


def _load_hash_table(device):
    global hash_table
    rng = torch.Generator(device=device)
    rng.manual_seed(2971215073)  # fib47 is prime
    hash_table = torch.rand(table_size, device=device, generator=rng)
    print(f"Goldfish hash table successfully constructed from seed {rng.initial_seed()}.")
    return hash_table


@dataclass
class GoldfishConfig:
    k_token_loss_dropout: int = 4
    start_position: int = 0
    context_width: int = 13
    strategy: Optional[str] = None  # off by default, set to "hash-table" or "hash-avalanche" to enable


def apply_tld(
    labels: torch.Tensor,
    settings: GoldfishConfig,
    ignore_index=0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a mask to a tensor to ignore every k-th token.
    `labels` is NOT updated in-place so apply_tld can be indepdently called for analysis/debugging/logging.

    Args:
        target: The target to apply the TLD mask to.
        strategy: The strategy to use for TLD.
            options implemented:
                - "static": Ignore every k-th token starting from `tld_start_position`.
                - "hash-legacy": Ignore tokens based on a hash of the context. For debugging purposes only.
                - "hash-table": Ignore tokens based on a hash of the context using a precomputed table.
                - "hash-avalanche": Ignore tokens based on a hash of the context using a hash function.
        k: The frequency with which tokens are ignored?
        tld_start_position: The position to start ignoring tokens from.
        context_width: Context width for hash-based approaches.

    Returns:
        The target with the mask applied and the indices of the dropped tokens.
    """
    strategy = settings.strategy
    k = settings.k_token_loss_dropout
    tld_start_position = settings.start_position
    tld_context_width = settings.context_width

    device = labels.device
    mbs, block_size = labels.shape
    masked_labels = labels.clone()

    if strategy == "static":
        dropped_token_indices = torch.arange(block_size, device=device)[tld_start_position::k].long()
        masked_labels[:, dropped_token_indices] = ignore_index
    elif strategy == "seeded_random":
        random_tensor = torch.randint(1, k + 1, size=labels.size())
        dropped_token_indices = (random_tensor == k).int()  # probability of dropping a token is 1/k
        masked_labels[dropped_token_indices] = ignore_index
    elif strategy == "pure_random":
        seed = int(time.time())
        generator = torch.Generator().manual_seed(seed)
        random_tensor = torch.randint(1, k + 1, size=labels.size(), generator=generator)
        dropped_token_indices = (random_tensor == k).int()  # probability of dropping a token is 1/k
        masked_labels[dropped_token_indices] = ignore_index
    elif strategy == "random_but_consisent_for_given_micro_batch":  # long ass name but better than unclear
        # TODO confirm that for given sample, its microbatch neighbours remains same through epochs
        micro_batch_seed = int(labels.sum().item())
        generator = torch.Generator().manual_seed(micro_batch_seed)
        random_tensor = torch.randint(1, k + 1, size=labels.size(), generator=generator)
        dropped_token_indices = (random_tensor == k).int()  # probability of dropping a token is 1/k
        masked_labels[dropped_token_indices] = ignore_index
    elif strategy == "hash-legacy":
        # Old hash for sanity checks, do not use
        dropped_token_indices = torch.zeros_like(labels)
        rng = torch.Generator(device=device)
        for b in range(mbs):
            for s in range(tld_context_width, block_size):
                prf_key = labels[b, s - tld_context_width : s].prod()
                rng.manual_seed(prf_key.item() % (2**64 - 1))
                dropped_token_indices[b, s] = torch.rand((1,), device=device) < 1 / k
        masked_labels[dropped_token_indices] = ignore_index
    elif strategy == "hash-table":
        global hash_table
        if hash_table is None:
            hash_table = _load_hash_table(device)
        hashed_keys = hash_table[labels.unfold(1, tld_context_width, 1).prod(dim=-1) % table_size]
        dropped_token_indices = hashed_keys < 1 / k
        masked_labels[:, tld_context_width - 1 :][dropped_token_indices] = ignore_index
        dropped_token_indices = dropped_token_indices.int()
    elif strategy == "hash-avalanche":
        keys = labels.unfold(1, tld_context_width, 1).prod(dim=-1).to(dtype=torch.uint64)
        hashed_keys = hashint(keys, width=32).long()
        dropped_token_indices = hashed_keys < ((1 << 32) - 1) / k
        masked_labels[:, tld_context_width - 1 :][dropped_token_indices] = ignore_index
    else:
        raise NotImplementedError(f"{strategy} TLD strategy is not implemented. Try 'static' instead.")

    return masked_labels, dropped_token_indices


@torch.compile  # required for uint64 support
def hashint(key: torch.Tensor, width: int = 32):
    """
    For any 1<k<=64, let mask=(1<<k)-1. hash_64() is a bijection on [0,1<<k), which means
    hash_64(x, mask)==hash_64(y, mask) if and only if x==y. hash_64i() is the inversion of
    hash_64(): hash_64i(hash_64(x, mask), mask) == hash_64(hash_64i(x, mask), mask) == x.
    """
    # thomas wang 64bit
    mask = (1 << width) - 1
    key = (~key + (key << 21)) & mask
    key = (key << 21) - key - 1
    key = key ^ key >> 24
    key = ((key + (key << 3)) + (key << 8)) & mask
    key = key * 265
    key = key ^ key >> 14
    key = ((key + (key << 2)) + (key << 4)) & mask
    key = key * 21
    key = key ^ key >> 28
    key = (key + (key << 31)) & mask
    return key


def get_abacus_param_groups(
    named_parameters, lr, no_weight_decay_for_bias_and_norm_params=True, increase_abacus_lr_multiplier=0.0
):
    param_groups = []
    abacus_param = None
    if no_weight_decay_for_bias_and_norm_params:
        wd_params = []
        no_wd_params = []

        for name, param in named_parameters:
            if ("abacus" in name.lower()) and (increase_abacus_lr_multiplier > 0.0):
                abacus_param = param
                continue

            no_wd = "norm" in name.lower() or "bias" in name.lower()
            if no_wd:
                no_wd_params.append(param)
            else:
                wd_params.append(param)

        if wd_params:
            param_groups.append({"params": wd_params, "lr": lr})
        if no_wd_params:
            param_groups.append({"params": no_wd_params, "lr": lr, "weight_decay": 0.0})
    else:
        params = []
        for n, p in named_parameters:
            if ("abacus" in n.lower()) and (increase_abacus_lr_multiplier > 0.0):
                abacus_param = p
                continue
            params.append(p)
        param_groups.append({"params": params, "lr": lr})

    if increase_abacus_lr_multiplier > 0.0:
        assert abacus_param is not None, "abacus param is not but you are trying to use it"
        param_groups.append(
            {
                "params": [abacus_param],
                "lr": lr * increase_abacus_lr_multiplier,
                "increase_abacus_lr_multiplier": increase_abacus_lr_multiplier,
            }
        )

    return param_groups


# Torch internal API compatibility: _reload_python_module_in_subproc was removed/renamed in newer torch versions.
try:
    from torch._inductor.codecache import _reload_python_module, _reload_python_module_in_subproc, ModuleType  # type: ignore[attr-defined]
except Exception:
    from torch._inductor.codecache import _reload_python_module  # type: ignore
    try:
        from torch._inductor.codecache import ModuleType  # type: ignore[attr-defined]
    except Exception:
        from types import ModuleType  # fallback to stdlib definition

    def _reload_python_module_in_subproc(key, path):
        # Fallback: reload in-process if subproc helper is unavailable in this torch version.
        return _reload_python_module(key, path)


def load_by_key_path_with_retry(
    cls,
    key: str,
    path: str,
    linemap: Optional[list[tuple[int, str]]] = None,
    attrs: Optional[Dict[str, Any]] = None,
) -> ModuleType:
    if linemap is None:
        linemap = []
    if key not in cls.cache:
        # Only retry the module load operation
        for attempt in range(10):
            try:
                mod = _reload_python_module(key, path)
                cls.cache.setdefault(key, mod)
                break
            except OSError:
                print("CACHE LOAD FAILURE")
                if attempt == 9:
                    raise
                time.sleep(min(0.1 * 2**attempt, 8.0))

        cls.linemaps[path] = list(zip(*linemap))
        if attrs is not None:
            for k, v in attrs.items():
                setattr(mod, k, v)
        if not (linemap or attrs):
            mod._reload_in_subproc = partial(_reload_python_module_in_subproc, key, path)  # type: ignore

    return cls.cache[key]


# original:
def load_by_key_path(
    cls,
    key: str,
    path: str,
    linemap: Optional[list[tuple[int, str]]] = None,
    attrs: Optional[Dict[str, Any]] = None,
) -> ModuleType:
    if linemap is None:
        linemap = []
    if key not in cls.cache:
        mod = _reload_python_module(key, path)

        # another thread might set this first
        cls.cache.setdefault(key, mod)
        # unzip into separate lines/nodes lists
        cls.linemaps[path] = list(zip(*linemap))

        if attrs is not None:
            for k, v in attrs.items():
                setattr(mod, k, v)

        if not (linemap or attrs):
            mod._reload_in_subproc = partial(_reload_python_module_in_subproc, key, path)  # type: ignore

    return cls.cache[key]


class ChunkedCE(torch.autograd.Function):
    """Horace He version"""

    @staticmethod
    def forward(ctx, _input, weight, target, compiled=True, ignore_index=-100, CHUNK_SIZE=1024):
        def compute_loss(input_chunk, weight, target):
            logits = torch.mm(input_chunk, weight.t())
            logits = logits.float()
            loss = torch.nn.functional.cross_entropy(logits, target, ignore_index=ignore_index)
            return loss

        grad_weight = torch.zeros_like(weight)
        grad_inputs = []
        loss_acc = torch.zeros((), device=_input.device)

        chunks = _input.shape[0] // CHUNK_SIZE

        def accumulate_chunk(input_chunk, target_chunk):
            (chunk_grad_input, chunk_grad_weight), chunk_loss = torch.func.grad_and_value(compute_loss, argnums=(0, 1))(
                input_chunk, weight, target_chunk
            )
            grad_weight.add_(chunk_grad_weight)
            loss_acc.add_(chunk_loss)
            return chunk_grad_input

        if compiled:
            accumulate_chunk = torch.compile(accumulate_chunk)
        input_chunks = torch.chunk(_input, chunks=chunks, dim=0)
        target_chunks = torch.chunk(target, chunks=chunks, dim=0)
        for input_chunk, target_chunk in zip(input_chunks, target_chunks):
            grad_inputs.append(accumulate_chunk(input_chunk, target_chunk))

        ctx.save_for_backward(
            torch.cat(grad_inputs, dim=0) / chunks,
            grad_weight / chunks,
        )
        return loss_acc / chunks

    @staticmethod
    def backward(ctx, grad_output):
        (grad_input, grad_weight) = ctx.saved_tensors
        return (grad_input, grad_weight, None, None, None, None)


class LinearCrossEntropyLoss(torch.nn.Linear):  # an instance of nn.Linear to be identified as such
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device=None,
        dtype=None,
        ignore_index: int = -100,
        chunk_size: int = 1024,
        init_method=None,
    ):
        self.init_method = None  # double-init because haven't done better integration
        super().__init__(in_features, out_features, bias=False, device=device, dtype=dtype)

        self.ignore_index = ignore_index
        self.chunk_size = chunk_size

        self.init_method = init_method
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.init_method is not None:
            self.init_method(self.weight)
        else:
            std = math.sqrt(1 / self.in_features)
            torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x, y=None):
        if y is None:
            return torch.matmul(x, self.weight.t())
        if x.is_meta:
            return torch.nn.functional.cross_entropy(torch.mm(x, self.weight.t()), y, ignore_index=self.ignore_index)
        else:
            return ChunkedCE.apply(
                x.view(-1, self.in_features), self.weight, y.view(-1), False, self.ignore_index, self.chunk_size
            )

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias=False"
