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


def cmd_certify(args: argparse.Namespace) -> int:
    """Full pipeline with certification: restarts + CPSS + reality -> DAG + certificate."""
    from .certify.certify import certify
    from .data.registry import get_dataset as _get
    from .extract.certificate import build_certificate, render_markdown
    from .extract.dag import build_dag, to_dot
    from .util.box import wait_until_free

    cfg = load_config(args.config, args.override)
    cfg["data"]["dataset"] = args.dataset
    wait_until_free(float(cfg["compute"]["load_threshold"]))

    cert = certify(args.dataset, cfg)
    ds = _get(args.dataset)
    run_dir = ROOT / "runs" / args.dataset.replace(":", "_") / "certified"
    run_dir.mkdir(parents=True, exist_ok=True)

    cert_doc = build_certificate(cert, ds, cert["splits"], cfg)
    dag = build_dag(cert["model"], ds, cert["splits"], cert["concepts"], cfg)
    cert["model"].save(run_dir / "model.pt")
    save_json(run_dir / "concepts.json", cert["concepts"])
    save_json(run_dir / "dag.json", dag)
    (run_dir / "dag.dot").write_text(to_dot(dag))
    save_json(run_dir / "certificate.json", cert_doc)
    (run_dir / "certificate.md").write_text(render_markdown(cert_doc))
    write_provenance(run_dir, config=cfg, data_hash=ds.data_hash(),
                     seeds={"restarts": cert["R"], "cpss_runs": 2 * cert["B"],
                            "split_seed": int(cfg["data"]["split_seed"])})
    print(f"{args.dataset}: {cert['n_core']} CORE / "
          f"{len(cert['concepts']) - cert['n_core']} periphery, "
          f"fid {cert['main']['fidelity']:.3f} vs ref {cert['main']['fid_ref']:.3f}, "
          f"E[V]<={cert['ev_bound']:.3g} -> {run_dir}")
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

    p_cert = sub.add_parser("certify", help="full certification pipeline on one dataset")
    p_cert.add_argument("--dataset", required=True)
    p_cert.add_argument("--config", default=None)
    p_cert.add_argument("--override", action="append", default=[])
    p_cert.set_defaults(fn=cmd_certify)

    p_list = sub.add_parser("list", help="list registered datasets")
    p_list.set_defaults(fn=cmd_list)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
