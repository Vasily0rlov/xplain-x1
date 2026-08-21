"""CLI (S-#12): xplain-x1 run --dataset wine [--config path] [--override k=v ...]"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

from .data.registry import available, get_dataset
from .data.splits import make_splits
from .model.mlp import build_model
from .train.reference import reference_ceiling
from .train.settle import evaluate, null_statistics, settle
from .util.io import save_json
from .util.provenance import write_provenance

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def load_config(path: str | None, overrides: list[str] | None) -> dict:
    cfg = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text())
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, args.override)
    cfg = copy.deepcopy(cfg)
    cfg["data"]["dataset"] = args.dataset
    seed = int(args.seed)

    ds = get_dataset(args.dataset)
    splits = make_splits(ds, int(cfg["data"]["split_seed"]),
                         int(cfg["data"]["probe_size"]),
                         tuple(cfg["data"]["fractions"]))
    null_stats = null_statistics(ds, splits)

    ref = reference_ceiling(ds, splits, cfg)
    model = build_model(ds.d, [int(cfg["model"]["init_width"])] *
                        int(cfg["model"]["init_layers"]), ds.task, ds.n_classes,
                        seed=seed)
    res = settle(model, ds, splits, cfg, seed=seed, pressures=None)
    val = evaluate(model, ds, splits, splits.val, null_stats)
    test = evaluate(model, ds, splits, splits.test, null_stats)

    run_dir = ROOT / "runs" / args.dataset.replace(":", "_") / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(run_dir / "model.pt")
    save_json(run_dir / "result.json", {
        "dataset": args.dataset, "seed": seed, "epochs": res.epochs_run,
        "val": val, "test": test, "ref": ref, "history": res.history})
    write_provenance(run_dir, config=cfg, data_hash=ds.data_hash(),
                     seeds={"run_seed": seed,
                            "split_seed": int(cfg["data"]["split_seed"])})
    print(f"{args.dataset}: val fid {val['fidelity']:.3f} "
          f"(ref {ref['fid_ref_val']:.3f}, winner {ref['winner']}) -> {run_dir}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    for name in available():
        print(name)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="xplain-x1")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run the pipeline on one dataset")
    p_run.add_argument("--dataset", required=True)
    p_run.add_argument("--config", default=None)
    p_run.add_argument("--seed", default=0, type=int)
    p_run.add_argument("--override", action="append", default=[])
    p_run.set_defaults(fn=cmd_run)

    p_list = sub.add_parser("list", help="list registered datasets")
    p_list.set_defaults(fn=cmd_list)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
