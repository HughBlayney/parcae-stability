"""Export a training checkpoint to the Hugging Face layout used by the released models.

Training checkpoints are single Lightning Fabric files holding the complete run
state: model weights under ``model``, plus optimizer, dataloader, collator, RNG,
and FLOP-accounting objects. Unpickling that file needs this repository's
training code importable, so analysis code should not read it directly. This
script extracts the model weights and the model configuration into the same
two-file layout as ``SandyResearch/parcae-140m``:

    <run_dir>/export/config.json         # model_config.json plus "_class_name"
    <run_dir>/export/pytorch_model.bin   # plain state dict, same keys as the release

That directory loads with ``receval.models.parcae.ModelingParcae.from_pretrained``
and can be uploaded to the Hub unchanged with ``huggingface_hub.upload_folder``.

Run from the repository root inside the training environment:

    python scripts/export_hf.py /datasets/$USER/parcae-outputs/parcae-small-140m-rec1

The newest ``step-*`` checkpoint under ``checkpoints-*/`` is used unless
``--checkpoint`` names one. ``export_info.json`` records which file was exported.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
# The checkpoint pickles ``cost.FLOPBreakdown`` and ``parcae_lm`` config objects.
for path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def newest_checkpoint(run_dir: Path) -> Path:
    candidates = [
        p for p in run_dir.glob("checkpoints-*/step-*") if re.match(r"step-(\d+)-", p.name)
    ]
    if not candidates:
        raise FileNotFoundError(f"No step-* checkpoints under {run_dir}/checkpoints-*/")
    return max(candidates, key=lambda p: int(re.match(r"step-(\d+)-", p.name).group(1)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="Training output directory containing model_config.json")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Specific checkpoint file (default: newest)")
    parser.add_argument("--out", type=Path, default=None, help="Export directory (default: <run_dir>/export)")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    checkpoint = (args.checkpoint or newest_checkpoint(run_dir)).resolve()
    out_dir = (args.out or run_dir / "export").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "model_config.json") as f:
        config = json.load(f)
    config["_class_name"] = config.get("model_class_name", "Parcae") + "Config"

    print(f"Reading {checkpoint}")
    # No mmap: memory-mapping a 1.5 GB file on the NFS scratch volume raises SIGBUS.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights = {key: tensor.detach().clone() for key, tensor in state["model"].items()}
    step = int(state.get("microbatch_step", re.match(r"step-(\d+)-", checkpoint.name).group(1)))

    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    torch.save(weights, out_dir / "pytorch_model.bin")
    with open(out_dir / "export_info.json", "w") as f:
        json.dump(
            {
                "source_checkpoint": str(checkpoint),
                "microbatch_step": step,
                "optimizer_step": int(state.get("optimizer_step", -1)),
                "num_tensors": len(weights),
            },
            f,
            indent=2,
        )
    size = sum(t.numel() for t in weights.values())
    print(f"Wrote {len(weights)} tensors ({size / 1e6:.1f}M elements) at step {step} to {out_dir}")


if __name__ == "__main__":
    main()
