"""Relational (ER) model of an XPLAIN-x1 certified model's STRUCTURE.

Renders the three-layer certified structure as the schema you would use to store
the model in a relational database: entities (Feature, Component, Unit, Route,
Collinearity-Group, the Function, the Decision) + the M:N junction tables an RDB
needs (Component-Support, Carrier, Unit-Input), with crow's-foot cardinalities.
Populated with the actual counts + key instances from a committed run.

Self-contained HTML, no external libs, theme-aware (matches the dashboards).

Run:  .venv/bin/python experiments/build_ermodels.py taiwan_credit bike
Out:  experiments/dashboards/<name>_ermodel.html
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xplain_x1.data.registry import get_dataset            # noqa: E402
from xplain_x1.extract.dashboard import (_CSS, _build_hier,  # noqa: E402
                                         _transitive_feature_supports)

OUT = ROOT / "experiments" / "dashboards"


# --------------------------------------------------------------- extract model
def extract(name: str) -> dict:
    d = json.loads((OUT / f"{name}.json").read_text())
    cert = d["cert"]; ident = cert["identification"]
    fdec = cert["function_decomposition"]; rts = cert["routes"]
    ds = get_dataset(name)
    comps = fdec["components"]
    core_c = [c for c in comps if c["label"] == "CORE"]
    outs = [n for n in d["dag"]["nodes"] if n["kind"] == "output"]
    is_mc = ident["task"] == "classification" and len(outs) > 2
    hier = _build_hier(cert, d["dag"], is_mc, len(outs), d.get("member_contrib"))
    units = [n for n in d["dag"]["nodes"] if n["kind"] == "unit"]
    trans = _transitive_feature_supports(d["dag"])
    n_carriers = sum(len(c["members"]) for c in hier["components"])
    n_unit_inputs = sum(len(trans.get(u["id"], [])) for u in units)
    n_comp_support = sum(len(c.get("support_names", [])) for c in comps)
    groups = rts.get("feature_groups", [])
    multi = [g for g in groups if len(g) > 1]
    core_r = [r for r in rts.get("rows", []) if r["label"] == "CORE"]
    # key instances (top by share) for the populated view
    core_c_sorted = sorted(core_c, key=lambda c: -(c.get("share_main") or 0))
    return {
        "dataset": name, "task": ident["task"], "n": ident["n"], "d": ident["d"],
        "n_classes": len(outs), "is_mc": is_mc,
        "counts": {
            "FEATURE": ds.d,
            "COLLINEARITY_GROUP": len(groups),
            "ROUTE": len(rts.get("rows", [])),
            "ROUTE_core": len(core_r),
            "COMPONENT": len(comps),
            "COMPONENT_core": len(core_c),
            "COMPONENT_SUPPORT": n_comp_support,
            "UNIT": len(units),
            "UNIT_INPUT": n_unit_inputs,
            "CARRIER": n_carriers,
        },
        "fnode": {"recon_r2": fdec.get("recon_r2"),
                  "coverage": d.get("core_share_sum")},
        "instances": {
            "components": [{"support": c["support_names"],
                            "share": round(c.get("share_main", 0), 3),
                            "Pi": c["Pi"], "pi": c["pi"], "label": c["label"]}
                           for c in core_c_sorted[:8]],
            "routes": [{"support": r.get("support_names"), "Pi": r["Pi"],
                        "pi": r["pi"], "label": r["label"]} for r in core_r],
            "units": [{"uid": u["id"], "layer": u["layer"],
                       "mu": round(u.get("mu") or 0, 2), "tag": u["tag"],
                       "reads": u.get("support_names", [])} for u in units],
            "groups": [{"members": g} for g in multi],
        },
    }


# --------------------------------------------------------------- ER schema (fixed)
# entity: (id, title, layer-tag, accent, x, y, w, [ (attr, key) ... ])
def _entities(m: dict) -> list:
    c = m["counts"]; fn = m["fnode"]
    task = m["task"]
    return [
        ("GRP", "COLLINEARITY_GROUP", "Layer S substrate", "peri", 30, 40, 210, [
            ("group_id", "PK"), ("member_features", ""), ("size", ""),
            ("method  |Spearman|>=0.8", "")], c["COLLINEARITY_GROUP"]),
        ("ROUTE", "ROUTE", "Layer S · certified reliance", "core", 30, 250, 210, [
            ("route_id", "PK"), ("group_id", "FK"), ("certified_support", ""),
            ("Pi  Pi_stability", ""), ("pi  cpss", ""), ("label CORE|PERIPH", "")],
            f'{c["ROUTE_core"]} CORE / {c["ROUTE"]}'),
        ("FEAT", "FEATURE", "input parameter", "accent", 300, 120, 200, [
            ("feature_id", "PK"), ("name", ""),
            ("type cont|ord|pred", ""), ("group_id", "FK")], c["FEATURE"]),
        ("CSUP", "COMPONENT_SUPPORT", "junction (M:N)", "peri", 560, 40, 190, [
            ("component_id", "FK"), ("feature_id", "FK")], c["COMPONENT_SUPPORT"]),
        ("UINP", "UNIT_INPUT", "junction (M:N)", "peri", 560, 470, 190, [
            ("unit_id", "FK"), ("feature_id", "FK")], c["UNIT_INPUT"]),
        ("COMP", "COMPONENT", "Layer F · certified term", "core", 820, 70, 220, [
            ("component_id", "PK"), ("support", ""), ("form", ""),
            ("share % target var", ""), ("Pi", ""), ("pi", ""),
            ("label CORE|PERIPH", "")], f'{c["COMPONENT_core"]} CORE / {c["COMPONENT"]}'),
        ("UNIT", "UNIT", "physical neuron", "peri", 820, 470, 220, [
            ("unit_id", "PK"), ("layer", ""), ("mu monosemanticity", ""),
            ("form", ""), ("tag CORE|PERIPH", "")], c["UNIT"]),
        ("CARR", "CARRIER", "junction (M:N, measured)", "peri", 820, 300, 220, [
            ("component_id", "FK"), ("unit_id", "FK"),
            ("contribution  d-share", ""), ("is_structural", "")], c["CARRIER"]),
        ("FUNC", "FUNCTION_F", "certified function  f = mu + Sigma terms", "accent",
            1110, 90, 210, [
            ("function_id", "PK"), ("intercept mu", ""),
            (f"recon_R2  {fn['recon_r2']:.2f}" if fn['recon_r2'] else "recon_R2", ""),
            (f"coverage  {fn['coverage']:.2f}" if fn['coverage'] is not None else "coverage", "")], 1),
        ("DEC", "DECISION", "output", "accent", 1110, 320, 210, [
            ("decision_id", "PK"),
            (f"task  {task}", ""),
            ("link  " + ("softmax" if task == "classification" else "identity"), ""),
            (f"n_classes  {m['n_classes']}", "")], 1),
    ]


# relationships: (a, side_a, b, side_b, card_a, card_b, label)
RELS = [
    ("FEAT", "l", "GRP", "r", "many", "one", "member of"),
    ("ROUTE", "t", "GRP", "b", "one", "one", "certifies reliance on"),
    ("COMP", "l", "CSUP", "r", "one", "many", "uses"),
    ("FEAT", "r", "CSUP", "l", "one", "many", "feeds"),
    ("UNIT", "l", "UINP", "r", "one", "many", "reads via"),
    ("FEAT", "r", "UINP", "l", "one", "many", "read by"),
    ("COMP", "b", "CARR", "t", "one", "many", "carried by"),
    ("UNIT", "t", "CARR", "b", "one", "many", "carries"),
    ("COMP", "r", "FUNC", "l", "many", "one", "sums into"),
    ("FUNC", "b", "DEC", "t", "one", "one", "link(f) ->"),
]

_ROWH = 19; _HEAD = 34


def _ent_h(attrs):
    return _HEAD + len(attrs) * _ROWH + 6


def _anchor(ent, side):
    _id, _t, _s, _a, x, y, w, attrs, _c = ent
    h = _ent_h(attrs)
    return {"l": (x, y + h / 2), "r": (x + w, y + h / 2),
            "t": (x + w / 2, y), "b": (x + w / 2, y + h)}[side]


def _crow(x, y, side, card, out=True):
    """crow's-foot glyph at (x,y) pointing away from the entity along `side`."""
    dx, dy = {"l": (-1, 0), "r": (1, 0), "t": (0, -1), "b": (0, 1)}[side]
    px, py = -dy, dx  # perpendicular
    g = []
    if card == "many":                       # three-prong fork
        for s in (-1, 0, 1):
            g.append(f'<line x1="{x}" y1="{y}" x2="{x+dx*11+px*7*s}" '
                     f'y2="{y+dy*11+py*7*s}" stroke="var(--ink2)" stroke-width="1.2"/>')
    else:                                     # one: a single bar across the line
        bx, by = x + dx * 7, y + dy * 7
        g.append(f'<line x1="{bx-px*6}" y1="{by-py*6}" x2="{bx+px*6}" '
                 f'y2="{by+py*6}" stroke="var(--ink2)" stroke-width="1.4"/>')
    return "".join(g)


def _render_svg(m: dict) -> str:
    ents = _entities(m)
    by = {e[0]: e for e in ents}
    W, H = 1360, 720
    parts = []
    # relationships first (behind boxes)
    for a, sa, b, sb, ca, cb, lab in RELS:
        pa = _anchor(by[a], sa); pb = _anchor(by[b], sb)
        mx, my = (pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2
        parts.append(f'<path d="M{pa[0]} {pa[1]} C {mx} {pa[1]}, {mx} {pb[1]}, '
                     f'{pb[0]} {pb[1]}" fill="none" stroke="var(--edge)" stroke-width="1.4"/>')
        parts.append(_crow(pa[0], pa[1], sa, ca))
        parts.append(_crow(pb[0], pb[1], sb, cb))
        parts.append(f'<rect x="{mx-len(lab)*3.1-4}" y="{my-8}" width="{len(lab)*6.2+8}" '
                     f'height="16" rx="3" fill="var(--surface)" stroke="var(--ring)"/>')
        parts.append(f'<text x="{mx}" y="{my+3.5}" text-anchor="middle" '
                     f'style="font:10px system-ui;fill:var(--ink2)">{lab}</text>')
    # entities
    for _id, title, sub, accent, x, y, w, attrs, cnt in ents:
        h = _ent_h(attrs)
        acol = {"core": "var(--core)", "accent": "var(--accent)",
                "peri": "var(--peri)"}[accent]
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" '
                     f'fill="var(--surface)" stroke="{acol}" stroke-width="1.6"/>')
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{_HEAD}" rx="7" '
                     f'fill="{acol}" fill-opacity="0.14"/>')
        parts.append(f'<text x="{x+10}" y="{y+15}" style="font:12px system-ui;'
                     f'font-weight:700;fill:var(--ink)">{title}</text>')
        parts.append(f'<text x="{x+10}" y="{y+27}" style="font:8.5px system-ui;'
                     f'fill:var(--muted)">{sub}</text>')
        # count badge
        badge = str(cnt)
        parts.append(f'<rect x="{x+w-len(badge)*6.5-14}" y="{y+6}" '
                     f'width="{len(badge)*6.5+10}" height="17" rx="8" fill="{acol}"/>')
        parts.append(f'<text x="{x+w-7}" y="{y+18}" text-anchor="end" '
                     f'style="font:10px system-ui;font-weight:700;fill:#fff">{badge}</text>')
        for i, (attr, key) in enumerate(attrs):
            ry = y + _HEAD + i * _ROWH + 13
            kcol = {"PK": "var(--accent)", "FK": "var(--warn)"}.get(key, "var(--muted)")
            if key:
                parts.append(f'<text x="{x+8}" y="{ry}" style="font:8px system-ui;'
                             f'font-weight:700;fill:{kcol}">{key}</text>')
            parts.append(f'<text x="{x+30}" y="{ry}" style="font:10px ui-monospace,'
                         f'monospace;fill:var(--ink2)">{attr}</text>')
            if i:
                parts.append(f'<line x1="{x}" y1="{y+_HEAD+i*_ROWH}" x2="{x+w}" '
                             f'y2="{y+_HEAD+i*_ROWH}" stroke="var(--grid)"/>')
    return f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto">{"".join(parts)}</svg>'


def _instances_html(m: dict) -> str:
    ins = m["instances"]
    def tbl(title, head, rows):
        h = "".join(f"<th>{c}</th>" for c in head)
        b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
        return (f'<div class="card"><h2>{title}</h2>'
                f'<table class="t"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>')
    comp_rows = [["·".join(c["support"]), f'{c["share"]:.3f}', f'{c["Pi"]:.2f}',
                  f'{c["pi"]:.2f}', c["label"]] for c in ins["components"]]
    route_rows = [["·".join(r["support"] or []), f'{r["Pi"]:.2f}', f'{r["pi"]:.2f}',
                   r["label"]] for r in ins["routes"]] or [["— none —", "", "", ""]]
    unit_rows = [[u["uid"], u["layer"], f'{u["mu"]:.2f}', u["tag"],
                  ", ".join(u["reads"])] for u in ins["units"]]
    grp_rows = [["{" + ", ".join(g["members"]) + "}"] for g in ins["groups"]] or [["— all singletons —"]]
    return (
        '<div class="wgrid">'
        + tbl("COMPONENT rows (Layer F CORE)", ["support", "share", "Π", "π", "label"], comp_rows)
        + tbl("ROUTE rows (Layer S CORE)", ["certified_support", "Π", "π", "label"], route_rows)
        + '</div><div class="wgrid">'
        + tbl("UNIT rows (physical neurons)", ["unit_id", "layer", "μ", "tag", "reads (features)"], unit_rows)
        + tbl("COLLINEARITY_GROUP rows (multi-feature)", ["members"], grp_rows)
        + '</div>')


def render(m: dict) -> str:
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>XPLAIN-x1 — {m['dataset']} relational model</title>
<style>{_CSS}
.erwrap {{ background:var(--page); border:1px solid var(--ring); border-radius:10px;
  padding:14px; overflow-x:auto; }}
.wgrid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
@media (max-width:900px){{ .wgrid {{ grid-template-columns:1fr; }} }}
.lg {{ display:flex; gap:16px; flex-wrap:wrap; font-size:11.5px; color:var(--ink2); margin:10px 0; }}
.lg b {{ color:var(--ink); }}
</style></head><body><div class="wrap">
  <div class="top"><div>
    <h1>{m['dataset']} — certified model as a relational schema</h1>
    <div class="sub">{m['task']} · n={m['n']:,} · d={m['d']} · the three-layer certified
      structure modelled as an ER schema (crow's-foot). Numbers on each entity are this
      model's row counts.</div></div></div>

  <div class="erwrap">{_render_svg(m)}</div>
  <div class="lg">
    <span><b>PK</b> primary key · <b>FK</b> foreign key</span>
    <span>crow's foot = “many”, bar = “one”</span>
    <span><b>Layer S</b> = routes/groups · <b>Layer F</b> = components (the certified claims)
      · UNIT = physical neurons · FUNCTION_F = f = μ + Σ components → DECISION</span>
  </div>

  <div class="card"><h2>How to read this schema</h2>
    <div class="note">Each <b>COMPONENT</b> (Layer F certified term) <b>uses</b> one or more
      <b>FEATURE</b>s (its support, via the COMPONENT_SUPPORT junction) and is physically
      <b>carried by</b> one or more <b>UNIT</b>s (measured, via CARRIER — a unit can carry a
      term through a <i>collinear</i> feature it reads, which is why CARRIER is distinct from
      COMPONENT_SUPPORT). Components <b>sum into</b> the <b>FUNCTION_F</b> (f = μ + Σ terms),
      which drives the <b>DECISION</b> via its link. FEATUREs are clustered into
      <b>COLLINEARITY_GROUP</b>s; a <b>ROUTE</b> (Layer S) certifies reliance on a group —
      the coarse, carving-invariant claim, distinct from any single unit or component.</div>
  </div>

  {_instances_html(m)}

  <div class="foot">Generated from the committed run (config + data + git hashes in the
    certificate). This is the schema of the model's certified STRUCTURE — not a deployment
    scoring schema. The M:N junction tables (COMPONENT_SUPPORT, UNIT_INPUT, CARRIER) are the
    associative tables a relational implementation would require.</div>
</div></body></html>"""


def main():
    targets = sys.argv[1:] or ["taiwan_credit", "bike"]
    for name in targets:
        m = extract(name)
        (OUT / f"{name}_ermodel.html").write_text(render(m), encoding="utf-8")
        c = m["counts"]
        print(f"{name}_ermodel.html: FEATURE {c['FEATURE']} · COMPONENT "
              f"{c['COMPONENT_core']}/{c['COMPONENT']} · UNIT {c['UNIT']} · ROUTE "
              f"{c['ROUTE_core']}/{c['ROUTE']} · CARRIER {c['CARRIER']}")


if __name__ == "__main__":
    main()
