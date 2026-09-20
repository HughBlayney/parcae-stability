"""Sanity-check the recurrence-ablated control configs against their recurrent siblings.

Run inside the container (needs torch):
    python scripts/check_rec1_controls.py
"""

import yaml
import torch

from parcae_lm.models.config import Config

SIZES = ["small-140m", "medium-370m", "large-770m", "xlarge-1_3b"]

ok = True
for size in SIZES:
    with open(f"launch_configs/parcae-{size}-rec1.yaml") as f:
        overwrite = yaml.safe_load(f)["model_overwrite"]

    rec_cfg = Config.from_name(f"parcae-{size}", vocab_size=32768, prelude_norm=True)
    ctl_cfg = Config.from_name(f"parcae-{size}", **overwrite)

    with torch.device("meta"):
        rec, ctl = rec_cfg.construct_model(), ctl_cfg.construct_model()

    rec_params = sum(p.numel() for p in rec.parameters())
    ctl_params = sum(p.numel() for p in ctl.parameters())

    ctl.train()
    n, k = ctl._sample_batch_depths(8)

    # ve_gate placement depends on layer_id parity, which shifts with mean_recurrence via the
    # coda offset o = n_prelude + n_core * mean_recurrence. Assert it did not move.
    rec_gates = sorted(nm for nm, _ in rec.named_parameters() if "ve_gate" in nm)
    ctl_gates = sorted(nm for nm, _ in ctl.named_parameters() if "ve_gate" in nm)

    checks = {
        "param count identical": rec_params == ctl_params,
        "exactly 0 no-grad steps": bool((n == 0).all()),
        "exactly 1 grad step": bool((k == 1).all()),
        "ve_gate placement identical": rec_gates == ctl_gates,
        "init std identical": rec_cfg.init.get_std("in_proj") == ctl_cfg.init.get_std("in_proj"),
    }
    ok &= all(checks.values())
    print(f"{size:12s} params={ctl_params:,}  " + "  ".join(f"{'PASS' if v else 'FAIL'}:{key}" for key, v in checks.items()))

print("\nAll checks passed." if ok else "\nFAILURES above.")
raise SystemExit(0 if ok else 1)
