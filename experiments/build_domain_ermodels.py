"""Domain-specific relational (ER) model for a dataset — the real-world entities
the flat ML table flattens, normalised, with the certified findings mapped onto
it.

A flat ML table can't be fully auto-normalised (it needs column *semantics*), but
the load-bearing structural insight is derivable: repeating column groups
(PAY_0..6, BILL_AMT1..6, …) are a repeating group ⇒ a child table.  The schemas
below apply that plus domain knowledge of each dataset; the payoff is that the
normalised domain model *explains* the certified collinearity groups (they are
adjacent rows of a repeating group, or two measures of one entity).

Self-contained styled HTML, crow's-foot; matches the dashboards.
Run:  .venv/bin/python experiments/build_domain_ermodels.py
Out:  experiments/dashboards/<name>_domain_ermodel.html
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xplain_x1.extract.dashboard import _CSS               # noqa: E402

OUT = ROOT / "experiments" / "dashboards"
_ROWH = 19; _HEAD = 34

# ---- entity = (id, title, subtitle, accent, x, y, w, [(attr,key,note)]) ------
# ---- rel = (a, side_a, b, side_b, card_a, card_b, label) ---------------------


def _ent_h(attrs):
    return _HEAD + len(attrs) * _ROWH + 6


def _anchor(e, side):
    _i, _t, _s, _a, x, y, w, attrs = e
    h = _ent_h(attrs)
    return {"l": (x, y + h / 2), "r": (x + w, y + h / 2),
            "t": (x + w / 2, y), "b": (x + w / 2, y + h)}[side]


def _crow(x, y, side, card):
    dx, dy = {"l": (-1, 0), "r": (1, 0), "t": (0, -1), "b": (0, 1)}[side]
    px, py = -dy, dx
    if card == "many":
        return "".join(
            f'<line x1="{x}" y1="{y}" x2="{x+dx*11+px*7*s}" y2="{y+dy*11+py*7*s}" '
            f'stroke="var(--ink2)" stroke-width="1.2"/>' for s in (-1, 0, 1))
    bx, by = x + dx * 7, y + dy * 7
    return (f'<line x1="{bx-px*6}" y1="{by-py*6}" x2="{bx+px*6}" y2="{by+py*6}" '
            f'stroke="var(--ink2)" stroke-width="1.4"/>')


def _svg(entities, rels, W, H):
    by = {e[0]: e for e in entities}
    p = []
    for a, sa, b, sb, ca, cb, lab in rels:
        pa = _anchor(by[a], sa); pb = _anchor(by[b], sb)
        mx, my = (pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2
        p.append(f'<path d="M{pa[0]} {pa[1]} C {mx} {pa[1]}, {mx} {pb[1]}, '
                 f'{pb[0]} {pb[1]}" fill="none" stroke="var(--edge)" stroke-width="1.4"/>')
        p.append(_crow(*pa, sa, ca)); p.append(_crow(*pb, sb, cb))
        p.append(f'<rect x="{mx-len(lab)*3.1-4}" y="{my-8}" width="{len(lab)*6.2+8}" '
                 f'height="16" rx="3" fill="var(--surface)" stroke="var(--ring)"/>')
        p.append(f'<text x="{mx}" y="{my+3.5}" text-anchor="middle" '
                 f'style="font:10px system-ui;fill:var(--ink2)">{lab}</text>')
    for _id, title, sub, accent, x, y, w, attrs in entities:
        h = _ent_h(attrs)
        acol = {"fact": "var(--accent)", "dim": "var(--core)",
                "child": "var(--warn)", "root": "var(--accent)"}.get(accent, "var(--peri)")
        p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" '
                 f'fill="var(--surface)" stroke="{acol}" stroke-width="1.6"/>')
        p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{_HEAD}" rx="7" '
                 f'fill="{acol}" fill-opacity="0.14"/>')
        p.append(f'<text x="{x+10}" y="{y+15}" style="font:12px system-ui;'
                 f'font-weight:700;fill:var(--ink)">{title}</text>')
        p.append(f'<text x="{x+10}" y="{y+27}" style="font:8.5px system-ui;'
                 f'fill:var(--muted)">{sub}</text>')
        for i, (attr, key, note) in enumerate(attrs):
            ry = y + _HEAD + i * _ROWH + 13
            kcol = {"PK": "var(--accent)", "FK": "var(--warn)",
                    "T": "var(--core)"}.get(key, "var(--muted)")
            if key:
                lab = "▶" if key == "T" else key
                p.append(f'<text x="{x+8}" y="{ry}" style="font:8px system-ui;'
                         f'font-weight:700;fill:{kcol}">{lab}</text>')
            p.append(f'<text x="{x+32}" y="{ry}" style="font:10px ui-monospace,'
                     f'monospace;fill:var(--ink2)">{attr}</text>')
            if note:
                p.append(f'<text x="{x+w-8}" y="{ry}" text-anchor="end" '
                         f'style="font:8.5px system-ui;fill:{kcol}">{note}</text>')
            if i:
                p.append(f'<line x1="{x}" y1="{y+_HEAD+i*_ROWH}" x2="{x+w}" '
                         f'y2="{y+_HEAD+i*_ROWH}" stroke="var(--grid)"/>')
    return f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto">{"".join(p)}</svg>'


# =============================================================== TAIWAN domain
TAIWAN = dict(
    name="taiwan_credit", title="Taiwan Credit — domain relational model",
    sub="consumer credit default · one flat ML row = one CLIENT with a 6-month "
        "repeating statement group (normalised here)",
    W=1180, H=520,
    entities=[
        ("CLIENT", "CLIENT", "the borrower · demographics", "root", 40, 40, 250, [
            ("client_id", "PK", ""), ("sex", "", "protected"),
            ("age", "", "protected"), ("education", "", ""),
            ("marital_status", "", "protected")]),
        ("ACCT", "CREDIT_ACCOUNT", "the credit line", "dim", 40, 300, 250, [
            ("account_id", "PK", ""), ("client_id", "FK", ""),
            ("credit_limit", "", "= LIMIT_BAL")]),
        ("STMT", "MONTHLY_STATEMENT", "repeating group — 6 months / account", "child",
            440, 150, 290, [
            ("statement_id", "PK", ""), ("account_id", "FK", ""),
            ("month_offset", "", "0=recent … 5"),
            ("repayment_status", "", "= PAY_0..6"),
            ("bill_amount", "", "= BILL_AMT*"),
            ("payment_amount", "", "= PAY_AMT*")]),
        ("DEF", "DEFAULT_OUTCOME", "the label", "fact", 850, 210, 250, [
            ("account_id", "FK", ""),
            ("defaulted_next_month", "T", "TARGET")]),
    ],
    rels=[
        ("CLIENT", "b", "ACCT", "t", "one", "many", "holds"),
        ("ACCT", "r", "STMT", "l", "one", "many", "has monthly (×6)"),
        ("ACCT", "r", "DEF", "l", "one", "one", "results in"),
    ],
    mapping=[
        ("PAY_0 — dominant certified driver (Π=1.0)",
         "MONTHLY_STATEMENT.repayment_status at month_offset=0 (the most recent month). "
         "The decision reliably flows through the latest repayment behaviour."),
        ("LIMIT_BAL — 2nd certified driver",
         "CREDIT_ACCOUNT.credit_limit — a per-account attribute."),
        ("Collinearity groups {PAY_3,PAY_4}, {BILL_AMT1..4}, {BILL_AMT5,6}",
         "adjacent rows of the SAME MONTHLY_STATEMENT time series — serial correlation is a "
         "DOMAIN consequence of the repeating group, which is exactly why the certified PAY_3 "
         "term is physically carried by units reading PAY_0/PAY_2/PAY_4."),
        ("Protected non-reliance (SEX, AGE, MARRIAGE)",
         "the CLIENT demographic attributes — none appears in a certified component (and none "
         "is a proxy): the decision structure lives on the ACCOUNT/STATEMENT, not the CLIENT."),
    ],
)

# =============================================================== BIKE domain
BIKE = dict(
    name="bike", title="Bike Sharing — domain relational model",
    sub="hourly rental demand · a star schema: one fact per hour, a calendar "
        "dimension, a weather-condition lookup",
    W=1150, H=470,
    entities=[
        ("CAL", "CALENDAR_DATE", "date dimension", "dim", 40, 60, 250, [
            ("date_id", "PK", ""), ("date", "", ""), ("year", "", "= year (trend)"),
            ("month", "", ""), ("season", "", ""), ("weekday", "", "= weekday"),
            ("is_holiday", "", ""), ("is_workingday", "", "")]),
        ("OBS", "RENTAL_OBSERVATION", "fact — one row per hour", "fact", 450, 40, 280, [
            ("observation_id", "PK", ""), ("date_id", "FK", ""),
            ("weather_id", "FK", ""), ("hour", "", "= hour (driver)"),
            ("temp", "", "= temp (driver)"), ("feel_temp", "", ""),
            ("humidity", "", ""), ("windspeed", "", ""),
            ("count", "T", "TARGET"), ("casual", "", ""), ("registered", "", "")]),
        ("WX", "WEATHER_CONDITION", "weather lookup", "dim", 850, 90, 230, [
            ("weather_id", "PK", ""),
            ("description", "", "clear/misty/rain")]),
    ],
    rels=[
        ("CAL", "r", "OBS", "l", "one", "many", "observed on (×24/day)"),
        ("WX", "l", "OBS", "r", "one", "many", "under condition"),
    ],
    mapping=[
        ("hour — dominant certified driver",
         "RENTAL_OBSERVATION.hour (a fact attribute)."),
        ("hour × weekday — the certified interaction",
         "RENTAL_OBSERVATION.hour × CALENDAR_DATE.weekday — a FACT × DIMENSION interaction "
         "(the commute demand curve reshapes by day-of-week / working-day)."),
        ("temp — 2nd driver; {temp, feel_temp} collinearity group",
         "two weather MEASURES of the same physical quantity on the fact row — same-entity "
         "redundancy, so the model uses either interchangeably."),
        ("year — certified driver",
         "CALENDAR_DATE.year — the system-growth trend (2011→2012), a calendar attribute, "
         "not a behavioural driver."),
    ],
)


def _mapping_html(mapping):
    rows = "".join(
        f'<tr><td style="white-space:nowrap"><b>{a}</b></td><td>{b}</td></tr>'
        for a, b in mapping)
    return (f'<div class="card"><h2>Certified findings → domain model</h2>'
            f'<div class="note">Where each certified result lives in the domain schema — '
            f'and why the collinearity groups exist.</div>'
            f'<table class="t"><thead><tr><th>certified finding</th>'
            f'<th>domain mapping</th></tr></thead><tbody>{rows}</tbody></table></div>')


def render(spec):
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{spec['name']} — domain relational model</title>
<style>{_CSS}
.erwrap {{ background:var(--page); border:1px solid var(--ring); border-radius:10px;
  padding:14px; overflow-x:auto; }}
.lg {{ display:flex; gap:16px; flex-wrap:wrap; font-size:11.5px; color:var(--ink2); margin:10px 0; }}
</style></head><body><div class="wrap">
  <div class="top"><div><h1>{spec['title']}</h1>
    <div class="sub">{spec['sub']}</div></div></div>
  <div class="erwrap">{_svg(spec['entities'], spec['rels'], spec['W'], spec['H'])}</div>
  <div class="lg">
    <span><b>PK</b> primary key · <b>FK</b> foreign key · <b style="color:var(--core)">▶</b> target</span>
    <span>crow's foot = “many”, bar = “one”</span>
    <span><b style="color:var(--accent)">blue</b> fact/target · <b style="color:var(--core)">green</b> dimension ·
      <b style="color:var(--warn)">amber</b> repeating-group child</span>
  </div>
  <div class="card"><h2>How this was derived</h2>
    <div class="note">The flat ML table has one row per observation. The load-bearing
      structure was derived from the column semantics: repeating indexed column groups
      (PAY_0..6, BILL_AMT1..6, PAY_AMT1..6 for Taiwan) are a <b>repeating group</b> ⇒ a child
      table (MONTHLY_STATEMENT). Bike's columns split cleanly into a calendar dimension, a
      weather lookup, and per-hour measures ⇒ a star schema. Domain knowledge supplies the
      entity names; the normalisation is what makes the certified collinearity findings
      <i>legible</i> — they fall out of the domain structure.</div></div>
  {_mapping_html(spec['mapping'])}
  <div class="foot">Domain-specific relational model (the real-world entities), distinct from
    the certified-structure meta-model in <code>{spec['name']}_ermodel.html</code>.</div>
</div></body></html>"""


def main():
    for spec in (TAIWAN, BIKE):
        (OUT / f"{spec['name']}_domain_ermodel.html").write_text(render(spec), encoding="utf-8")
        print(f"{spec['name']}_domain_ermodel.html: {len(spec['entities'])} entities, "
              f"{len(spec['rels'])} relationships, {len(spec['mapping'])} finding-mappings")


if __name__ == "__main__":
    main()
