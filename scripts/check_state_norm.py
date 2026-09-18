"""Held-out depth curve from a single recurrence trajectory: loss/PPL, ||h_T||, ||h_T - h_{T-1}|| for every T.

Runs the recurrence once to max(depths) and reads out loss at every requested depth along the way, which is
equivalent to running the model with num_steps = T for each T (same h0 draw) at a fraction of the cost.
Documents are evaluated like receval's val_loss task (one document per sequence, truncated to block_size+1,
right-padded within a batch; pads are masked out of the loss).

Usage (trained run):
    PYTHONPATH=. python scripts/check_state_norm.py --out_dir outputs/stab/poisson_rho1.25 \
        --val_data_dir /path/to/val --depths 1,2,4,8,16,32,64 --max_samples 1000 --output outputs/stab/poisson_rho1.25/eval/depth_curve.json
Usage (Step-0 probe on the released checkpoint with the learned decay replaced by a scalar):
    PYTHONPATH=. python scripts/check_state_norm.py --hf_repo SandyResearch/parcae-140m --override_fixed_decay 1.25 \
        --val_data_dir /path/to/val --max_samples 100 --output results/probe_rho1.25.json
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from receval.tasks.val_loss import load_val_texts_from_parquet  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--out_dir", type=str, help="training run directory (run_config.json + checkpoints-*/step-*)")
    src.add_argument("--hf_repo", type=str, help="HF repo or local dir with config.json + pytorch_model.bin")
    p.add_argument("--checkpoint_path", type=str, default=None, help="explicit checkpoint (default: latest in out_dir)")
    p.add_argument("--step", type=int, default=None, help="pick the checkpoint at this step from out_dir")
    p.add_argument("--tokenizer_path", type=str, default=None, help="override tokenizer dir (default: run_config / HF)")
    p.add_argument("--val_data_dir", type=str, required=True, help="directory with validation parquet shards")
    p.add_argument("--max_files", type=int, default=5)
    p.add_argument("--max_samples", type=int, default=1000)
    p.add_argument("--depths", type=str, default="1,2,3,4,6,8,10,12,16,20,24,32,48,64")
    p.add_argument("--override_fixed_decay", type=float, default=None,
                   help="replace the adapter's decay with this scalar at inference (monkeypatch, weights unchanged)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=None, help="default: model block_size")
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=0, help="seed for the random initial state h0")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=str, default=None, help="JSON output path (default: <out_dir>/eval/depth_curve.json)")
    return p.parse_args()


def load_model_and_tokenizer(args):
    from parcae_lm.tokenizer import Tokenizer
    from receval.models.parcae import ModelingParcae, load_state_dict_checked

    if args.hf_repo is not None:
        model = ModelingParcae.from_pretrained(args.hf_repo, device=args.device)
        tokenizer = Tokenizer(args.tokenizer_path) if args.tokenizer_path else Tokenizer.from_pretrained("SandyResearch/parcae-tokenizer")
        source = {"hf_repo": args.hf_repo}
    else:
        from receval.settings import CLISettings

        settings = CLISettings(out_dir=args.out_dir, checkpoint_path=args.checkpoint_path, step=args.step)
        assert settings.model_config is not None, f"no run_config.json / model config found in {args.out_dir}"
        assert settings.checkpoint_path is not None, f"no checkpoint found in {args.out_dir}"
        model = ModelingParcae(settings.model_config)
        state = torch.load(settings.checkpoint_path, map_location="cpu", weights_only=False)
        load_state_dict_checked(model, state["model"] if "model" in state else state)
        model.to(args.device).eval()
        tok_path = args.tokenizer_path or settings.tokenizer_path
        assert tok_path, "tokenizer_path not found in run_config.json; pass --tokenizer_path"
        tokenizer = Tokenizer(tok_path)
        source = {"out_dir": args.out_dir, "checkpoint_path": settings.checkpoint_path}

    adapter = model.transformer.adapter
    assert hasattr(adapter, "fixed_decay"), "check_state_norm.py expects a DiagonalInjection adapter"
    if args.override_fixed_decay is not None:
        adapter.fixed_decay = float(args.override_fixed_decay)  # _decay() now returns the scalar; A_log unused
    return model, tokenizer, source


@torch.no_grad()
def main():
    args = parse_args()
    depths = sorted({int(d) for d in args.depths.split(",") if d.strip()})
    t_max = depths[-1]
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    model, tokenizer, source = load_model_and_tokenizer(args)
    cfg = model.config
    seq_len = args.seq_len or cfg.block_size
    adapter = model.transformer.adapter
    decay_max, decay_mean = adapter.get_spectral_norm(), adapter.get_contraction_factor()
    print(f"model: {source} | adapter decay max={decay_max:.4f} mean={decay_mean:.4f} "
          f"| fixed_decay={adapter.fixed_decay} | state_init={cfg.state_init} | mean_recurrence={cfg.mean_recurrence}")

    texts = load_val_texts_from_parquet(args.val_data_dir, max_files=args.max_files)[: args.max_samples]
    print(f"{len(texts)} documents from {args.val_data_dir}")
    docs = []
    for text in texts:
        toks = tokenizer.encode(text, return_tensors=False)
        toks = toks.tolist() if not isinstance(toks, list) else toks
        if len(toks) >= 2:
            docs.append(toks[: seq_len + 1])
    docs.sort(key=len, reverse=True)  # length-sorted batches minimise padding

    autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.precision]
    autocast = torch.autocast(device_type=device.type, dtype=autocast_dtype) if autocast_dtype is not None and device.type == "cuda" else torch.autocast("cpu", enabled=False)
    pad_id = tokenizer.pad_id if getattr(tokenizer, "pad_id", None) is not None else 0

    nats = torch.zeros(len(depths), dtype=torch.float64)
    state_norm = torch.zeros(t_max + 1, dtype=torch.float64)      # index t = state after t recurrences (0 = h0)
    step_norm = torch.zeros(t_max + 1, dtype=torch.float64)       # ||h_t - h_{t-1}||
    cos_prev = torch.zeros(t_max + 1, dtype=torch.float64)        # cos(h_t, h_{t-1})
    n_tokens = 0
    nonfinite_depths = set()
    t0 = time.time()

    n_prelude = len(model.transformer.prelude)
    n_core = len(model.transformer.core_block)
    coda_ve_offset = n_prelude + n_core

    for b in range(0, len(docs), args.batch_size):
        batch = docs[b : b + args.batch_size]
        L = max(len(d) for d in batch)
        ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
        valid = torch.zeros((len(batch), L - 1), dtype=torch.bool)
        for i, d in enumerate(batch):
            ids[i, : len(d)] = torch.tensor(d)
            valid[i, : len(d) - 1] = True
        ids, valid = ids.to(device), valid.to(device)
        input_ids, targets = ids[:, :-1], ids[:, 1:]
        n_valid = int(valid.sum())
        n_tokens += n_valid
        freqs_cis = model.freqs_cis[:, : input_ids.shape[1]]
        model._current_input_ids = input_ids

        with autocast:
            x = model.transformer.wte(input_ids)
            if model.emb_scale != 1:
                x = x * model.emb_scale
            for i, block in enumerate(model.transformer.prelude):
                ve = model.value_embeds[str(i)](input_ids) if str(i) in model.value_embeds else None
                x = block(x, freqs_cis, None, ve=ve)
            if cfg.prelude_norm:
                x = model.transformer.ln_prelude(x)
            input_embeds = x
            h = model.initialize_state(input_embeds)
            total_steps = torch.tensor(t_max, device=device)

            def stats(h_new, h_old, t):
                hn = h_new.float().norm(dim=-1)
                state_norm[t] += float(hn[valid].sum())
                if h_old is not None:
                    diff = (h_new.float() - h_old.float()).norm(dim=-1)
                    step_norm[t] += float(diff[valid].sum())
                    cos = torch.nn.functional.cosine_similarity(h_new.float(), h_old.float(), dim=-1)
                    cos_prev[t] += float(cos[valid].sum())

            def readout_loss(h_t):
                y = model.transformer.C(h_t)
                for i, block in enumerate(model.transformer.coda):
                    key = str(coda_ve_offset + i)
                    ve = model.value_embeds[key](input_ids) if key in model.value_embeds else None
                    y = block(y, freqs_cis, None, ve=ve)
                y = model.transformer.ln_f(y)
                logits = model.lm_head(y).float() * cfg.init.logit_scale
                if getattr(cfg, "logit_softcap", None) is not None:
                    logits = cfg.logit_softcap * torch.tanh(logits / cfg.logit_softcap)
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
                return float(loss.view_as(targets)[valid].double().sum())

            stats(h, None, 0)
            for t in range(1, t_max + 1):
                h_prev = h
                h = model.core_block_forward(h, input_embeds, freqs_cis, None, step=torch.tensor(t - 1, device=device), total_steps=total_steps)
                stats(h, h_prev, t)
                if t in depths:
                    idx = depths.index(t)
                    nats_t = readout_loss(h)
                    if math.isfinite(nats_t):
                        nats[idx] += nats_t
                    else:
                        nonfinite_depths.add(t)
        done = min(b + args.batch_size, len(docs))
        if done % (args.batch_size * 10) < args.batch_size or done == len(docs):
            print(f"  {done}/{len(docs)} docs | {n_tokens:,} tokens | {time.time() - t0:.0f}s", flush=True)

    loss = {d: (float(nats[i]) / n_tokens if d not in nonfinite_depths else float("nan")) for i, d in enumerate(depths)}
    result = {
        "source": source,
        "override_fixed_decay": args.override_fixed_decay,
        "config": {k: getattr(cfg, k, None) for k in ["injection_fixed_decay", "max_recurrence", "sampling_scheme",
                                                       "mean_recurrence", "mean_backprop_depth", "state_init"]},
        "adapter_decay_max": decay_max,
        "adapter_decay_mean": decay_mean,
        "num_docs": len(docs),
        "num_tokens": n_tokens,
        "seed": args.seed,
        "precision": args.precision,
        "depths": depths,
        "loss": {str(d): loss[d] for d in depths},
        "ppl": {str(d): (math.exp(loss[d]) if math.isfinite(loss[d]) and loss[d] < 100 else float("inf")) for d in depths},
        "state_norm": {str(t): float(state_norm[t]) / n_tokens for t in range(t_max + 1)},
        "step_norm": {str(t): float(step_norm[t]) / n_tokens for t in range(1, t_max + 1)},
        "cos_prev": {str(t): float(cos_prev[t]) / n_tokens for t in range(1, t_max + 1)},
        "nonfinite_depths": sorted(nonfinite_depths),
    }

    print(f"\n{'T':>4} {'loss':>8} {'ppl':>9} {'|h_T|':>12} {'|h_T-h_T-1|':>12} {'cos(h_T,h_T-1)':>15}")
    for d in depths:
        print(f"{d:>4} {loss[d]:>8.4f} {result['ppl'][str(d)]:>9.3f} {result['state_norm'][str(d)]:>12.4g} "
              f"{result['step_norm'][str(d)]:>12.4g} {result['cos_prev'][str(d)]:>15.4f}")

    out = args.output or (str(Path(args.out_dir) / "eval" / "depth_curve.json") if args.out_dir else None)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
