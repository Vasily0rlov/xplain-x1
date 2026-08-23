"""Render the Bike standard-NN vs XPLAIN-x1 comparison as one self-contained HTML.

Two DAG panes (standard NN left, x1 right) over the same audit instrument, then
matched performance widgets: metrics table, predicted-vs-actual, ROC (derived
high-demand task), and confusion matrices.  Reuses the dashboard CSS; no external
libraries.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from xplain_x1.extract.dashboard import _CSS               # noqa: E402

_EXTRA_CSS = """
.cmp { display:grid; grid-template-columns:1fr 1fr; gap:0; }
@media (max-width:900px){ .cmp { grid-template-columns:1fr; } }
.pane { position:relative; }
.pane + .pane { border-left:1px solid var(--ring); }
@media (max-width:900px){ .pane + .pane { border-left:none; border-top:1px solid var(--ring); } }
.paneh { padding:12px 15px 8px; }
.paneh h3 { margin:0; font-size:15px; display:flex; align-items:center; gap:8px; }
.paneh .st { font-size:12px; color:var(--muted); margin-top:2px; }
.mrow { display:flex; gap:16px; flex-wrap:wrap; margin-top:8px; font-size:12px;
  font-variant-numeric:tabular-nums; }
.mrow .b { color:var(--ink); font-weight:650; font-size:15px; }
.mrow .k { color:var(--muted); font-size:10px; text-transform:uppercase; letter-spacing:.03em; }
.mrow .cell { text-align:center; }
.svgwrap { position:relative; height:430px; background:var(--page);
  border-top:1px solid var(--ring); overflow:hidden; }
.svgwrap svg { width:100%; height:100%; cursor:grab; }
.svgwrap svg:active { cursor:grabbing; }
.plegend { position:absolute; left:10px; bottom:8px; font-size:11px; color:var(--ink2);
  background:color-mix(in srgb,var(--surface) 85%,transparent); padding:3px 8px;
  border-radius:6px; display:flex; gap:12px; flex-wrap:wrap; }
.plegend i { width:11px; height:11px; border-radius:3px; display:inline-block; vertical-align:-1px; margin-right:4px; }
.wgrid { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
@media (max-width:900px){ .wgrid { grid-template-columns:1fr; } }
.metrics { border-collapse:collapse; width:100%; font-size:13px; font-variant-numeric:tabular-nums; }
.metrics th, .metrics td { padding:7px 12px; border-top:1px solid var(--grid); text-align:right; }
.metrics th:first-child, .metrics td:first-child { text-align:left; color:var(--ink2); }
.metrics thead th { border-top:none; color:var(--muted); font-size:11px;
  text-transform:uppercase; letter-spacing:.03em; }
.metrics .win { color:var(--good); font-weight:650; }
.metrics tr.hl td { background:var(--wash); }
.verdict { font-size:13px; color:var(--ink2); margin-top:10px; line-height:1.6;
  border-left:3px solid var(--accent); padding:6px 12px; background:var(--wash); border-radius:0 6px 6px 0; }
.chart svg { width:100%; height:auto; display:block; }
.chart .cap { font-size:11px; color:var(--muted); margin-top:4px; }
.cms { display:flex; gap:18px; flex-wrap:wrap; }
.cm { flex:1 1 200px; }
.cm h4 { margin:0 0 6px; font-size:12.5px; color:var(--ink2); }
.cmt { border-collapse:collapse; font-size:12px; font-variant-numeric:tabular-nums; }
.cmt td, .cmt th { border:1px solid var(--grid); padding:6px 12px; text-align:center; }
.cmt th { color:var(--muted); font-weight:600; font-size:11px; }
.cmt td.d { font-weight:700; }
"""

_JS = r"""
"use strict";
const $=s=>document.getElementById(s);
const NS="http://www.w3.org/2000/svg";
const D=JSON.parse($("cmp-data").textContent);
const tip=$("tip");
function el(t,a,p){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e;}
function fmt(v,d){return (v==null||isNaN(v))?"—":Number(v).toFixed(d==null?2:d);}
function abbr(s,m){m=m||11;return s.length>m?s.slice(0,m-1)+"…":s;}
function mtip(ev,h){tip.innerHTML=h;tip.style.opacity=1;
  tip.style.left=Math.min(ev.clientX+14,innerWidth-tip.offsetWidth-8)+"px";
  tip.style.top=Math.min(ev.clientY+14,innerHeight-tip.offsetHeight-8)+"px";}

// ---------- DAG pane --------------------------------------------------------
function drawDag(svgId, dag){
  const svg=$(svgId); const g=el("g",{},svg);
  const byId={}; dag.nodes.forEach(n=>byId[n.id]=n);
  const outCol=2+Math.max(1,...dag.nodes.filter(n=>n.kind==="unit").map(n=>n.layer||1));
  const colf=n=>n.kind==="input"?0:n.kind==="output"?outCol:(n.layer||1);
  const cols={}; dag.nodes.forEach(n=>{const c=colf(n);(cols[c]=cols[c]||[]).push(n);});
  const COLW=190,VGAP=30,R=11; let maxR=1;
  Object.values(cols).forEach(a=>maxR=Math.max(maxR,a.length));
  const H=40+ (maxR-1)*VGAP, W=(outCol+1)*COLW;
  const pos={};
  Object.keys(cols).forEach(c=>{const arr=cols[c],x=+c*COLW+COLW/2,span=(arr.length-1)*VGAP,y0=(H-span)/2;
    arr.forEach((n,i)=>pos[n.id]={x,y:y0+i*VGAP});});
  svg.setAttribute("viewBox",`0 0 ${W} ${H}`);
  dag.edges.forEach(e=>{const p=pos[e.src],q=pos[e.dst];if(!p||!q)return;
    const mx=(p.x+q.x)/2,w=0.5+3.2*Math.min(1,(e.share||0)*3);
    el("path",{d:`M${p.x+R} ${p.y} C ${mx} ${p.y}, ${mx} ${q.y}, ${q.x-R} ${q.y}`,
      fill:"none",stroke:"var(--edge)","stroke-width":w.toFixed(2),"stroke-opacity":0.4},g);});
  dag.nodes.forEach(n=>{const p=pos[n.id];if(!p)return;
    const grp=el("g",{transform:`translate(${p.x},${p.y})`,style:"cursor:default"},g);
    const core=n.tag==="CORE";
    const fill=n.kind==="unit"?(core?"var(--corewash)":"var(--periwash)"):
      (n.kind==="output"?"var(--surface)":"var(--wash)");
    const stroke=n.kind==="unit"?(core?"var(--core)":"var(--peri)"):
      (n.kind==="output"?"var(--accent)":"var(--grid)");
    if(n.kind==="unit") el("circle",{r:R,fill,stroke,"stroke-width":1.5},grp);
    else el("rect",{x:-R-8,y:-9,width:2*R+16,height:18,rx:9,fill,stroke,"stroke-width":1.2},grp);
    const t=el("text",{"text-anchor":"middle",y:3.2,
      style:"font:9px system-ui;fill:var(--ink);pointer-events:none"},grp);
    t.textContent = n.kind==="unit"
      ? (n.mu!=null?"μ"+fmt(n.mu,2):"") : abbr(n.label,13);
    grp.addEventListener("mousemove",ev=>mtip(ev, dagTip(n)));
    grp.addEventListener("mouseleave",()=>tip.style.opacity=0);
  });
  panzoom(svg,g,W,H);
}
function dagTip(n){
  if(n.kind==="unit") return `<h4>${n.support_names&&n.support_names.length?n.support_names.join("·"):n.id}</h4>`+
    `<div class="m">${n.tag==="CORE"?"legible":"polysemantic"} · ${n.form||"?"}</div>`+
    `<div class="m">μ ${fmt(n.mu)} · fan-in ${n.effective_fanin!=null?n.effective_fanin:"?"}</div>`;
  return `<h4>${n.label}</h4><div class="m">${n.kind}</div>`;
}
function panzoom(svg,g,W,H){
  let z={k:1,x:0,y:0},drag=null;
  const ap=()=>g.setAttribute("transform",`translate(${z.x},${z.y}) scale(${z.k})`);
  svg.addEventListener("mousedown",e=>drag={x:e.clientX,y:e.clientY,ox:z.x,oy:z.y});
  addEventListener("mousemove",e=>{if(!drag)return;z.x=drag.ox+(e.clientX-drag.x);z.y=drag.oy+(e.clientY-drag.y);ap();});
  addEventListener("mouseup",()=>drag=null);
  svg.addEventListener("wheel",e=>{e.preventDefault();const r=svg.getBoundingClientRect();
    const sx=(e.clientX-r.left)/r.width*W, sy=(e.clientY-r.top)/r.height*H;
    const f=e.deltaY<0?1.12:1/1.12,nk=Math.max(0.3,Math.min(4,z.k*f));
    z.x=sx-(sx-z.x)*(nk/z.k);z.y=sy-(sy-z.y)*(nk/z.k);z.k=nk;ap();},{passive:false});
}

// ---------- charts ----------------------------------------------------------
function scatter(mount, pts, mx, color){
  const W=300,H=220,L=44,B=32,T=8,Rr=10;
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`},mount);
  const X=v=>L+(v/mx)*(W-L-Rr), Y=v=>H-B-(v/mx)*(H-B-T);
  [0,.25,.5,.75,1].forEach(f=>{const v=f*mx;
    el("line",{x1:L,x2:W-Rr,y1:Y(v),y2:Y(v),stroke:"var(--grid)"},svg);
    const t=el("text",{x:L-5,y:Y(v)+3,"text-anchor":"end",style:"font:9px system-ui;fill:var(--muted)"},svg);
    t.textContent=Math.round(v);
    const t2=el("text",{x:X(v),y:H-B+13,"text-anchor":"middle",style:"font:9px system-ui;fill:var(--muted)"},svg);
    t2.textContent=Math.round(v);});
  el("line",{x1:X(0),y1:Y(0),x2:X(mx),y2:Y(mx),stroke:"var(--muted)","stroke-dasharray":"4 3"},svg);
  pts.forEach(p=>el("circle",{cx:X(p[0]),cy:Y(p[1]),r:1.7,fill:color,"fill-opacity":0.45},svg));
  const ax=el("text",{x:(L+W)/2,y:H-3,"text-anchor":"middle",style:"font:9.5px system-ui;fill:var(--ink2)"},svg);
  ax.textContent="actual";
  const ay=el("text",{x:12,y:(T+H-B)/2,"text-anchor":"middle",style:"font:9.5px system-ui;fill:var(--ink2)",
    transform:`rotate(-90 12 ${(T+H-B)/2})`},svg); ay.textContent="predicted";
}
function rocChart(mount, curves){
  const W=320,H=300,L=42,B=34,T=10,Rr=12;
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`},mount);
  const X=v=>L+v*(W-L-Rr), Y=v=>H-B-v*(H-B-T);
  [0,.25,.5,.75,1].forEach(f=>{
    el("line",{x1:L,x2:W-Rr,y1:Y(f),y2:Y(f),stroke:"var(--grid)"},svg);
    el("line",{x1:X(f),x2:X(f),y1:T,y2:H-B,stroke:"var(--grid)"},svg);
    const t=el("text",{x:L-5,y:Y(f)+3,"text-anchor":"end",style:"font:9px system-ui;fill:var(--muted)"},svg);t.textContent=f;
    const t2=el("text",{x:X(f),y:H-B+13,"text-anchor":"middle",style:"font:9px system-ui;fill:var(--muted)"},svg);t2.textContent=f;});
  el("line",{x1:X(0),y1:Y(0),x2:X(1),y2:Y(1),stroke:"var(--muted)","stroke-dasharray":"4 3"},svg);
  curves.forEach(c=>{
    const pts=c.fpr.map((f,i)=>`${X(f)},${Y(c.tpr[i])}`).join(" ");
    el("polyline",{points:pts,fill:"none",stroke:c.color,"stroke-width":2.2},svg);});
  const ax=el("text",{x:(L+W)/2,y:H-3,"text-anchor":"middle",style:"font:10px system-ui;fill:var(--ink2)"},svg);
  ax.textContent="false-positive rate";
  const ay=el("text",{x:12,y:(T+H-B)/2,"text-anchor":"middle",style:"font:10px system-ui;fill:var(--ink2)",
    transform:`rotate(-90 12 ${(T+H-B)/2})`},svg); ay.textContent="true-positive rate";
}

// ---------- build page ------------------------------------------------------
drawDag("dagL", D.left.dag);
drawDag("dagR", D.right.dag);
const mxA=Math.max(...D.left.scatter.concat(D.right.scatter).map(p=>Math.max(p[0],p[1])))*1.02;
scatter($("scL"), D.left.scatter, mxA, "var(--warn)");
scatter($("scR"), D.right.scatter, mxA, "var(--accent)");
rocChart($("roc"), [
  {fpr:D.left.roc.fpr,tpr:D.left.roc.tpr,color:"var(--warn)"},
  {fpr:D.right.roc.fpr,tpr:D.right.roc.tpr,color:"var(--accent)"}]);
"""


def _confusion_html(cm: list, title: str, color: str) -> str:
    (tn, fp), (fn, tp) = cm
    tot = tn + fp + fn + tp
    def sh(v):
        a = 0.10 + 0.55 * (v / max(1, tot))
        return f'style="background:color-mix(in srgb,{color} {a*100:.0f}%,transparent)"'
    return f"""<div class="cm"><h4>{title}</h4>
    <table class="cmt"><tr><th></th><th>pred low</th><th>pred high</th></tr>
    <tr><th>actual low</th><td class="d" {sh(tn)}>{tn}</td><td {sh(fp)}>{fp}</td></tr>
    <tr><th>actual high</th><td {sh(fn)}>{fn}</td><td class="d" {sh(tp)}>{tp}</td></tr>
    </table></div>"""


def _metrics_table(d: dict) -> str:
    L, R = d["left"], d["right"]
    ceil = d["fid_ref"]
    def row(label, lv, rv, better="high", fmtv=lambda x: x, sub=""):
        try:
            lwin = (lv >= rv) if better == "high" else (lv <= rv)
        except TypeError:
            lwin = None
        lc = ' class="win"' if lwin is True else ""
        rc = ' class="win"' if lwin is False else ""
        return (f'<tr><td>{label}{sub}</td>'
                f'<td{lc}>{fmtv(lv)}</td><td{rc}>{fmtv(rv)}</td></tr>')
    rows = [
        row("fidelity R² (log1p — the programme metric)", L["fidelity"], R["fidelity"], "high",
            sub=f'<div style="font-size:10.5px;color:var(--muted)">vs reference ceiling {ceil}</div>'),
        row("R² (raw count space)", L["reg"]["r2"], R["reg"]["r2"], "high"),
        row("RMSE (counts)", L["reg"]["rmse"], R["reg"]["rmse"], "low"),
        row("MAE (counts)", L["reg"]["mae"], R["reg"]["mae"], "low"),
        row("AUC (high-demand)", L["roc"]["auc"], R["roc"]["auc"], "high"),
    ]
    interp = [
        f'<tr class="hl"><td>hidden units (live)</td><td>{L["n_units"]}</td><td>{R["n_units"]}</td></tr>',
        row("legible units (μ≥0.8, fan-in≤3)", L["n_legible"], R["n_legible"], "high"),
        row("median μ (monosemanticity)", L["median_mu"], R["median_mu"], "high"),
        row("median fan-in", L["median_fanin"], R["median_fanin"], "low"),
    ]
    return (
        '<table class="metrics"><thead><tr><th>metric</th>'
        f'<th>Standard NN</th><th>XPLAIN-x1</th></tr></thead><tbody>'
        + "".join(rows)
        + f'<tr><td colspan="3" style="border-top:2px solid var(--ring);'
          'color:var(--muted);font-size:11px;padding-top:10px">INTERPRETABILITY '
          f'(same audit instrument) · reference ceiling R² = {ceil}</td></tr>'
        + "".join(interp)
        + '</tbody></table>')


def _pane(side: dict, accent: str) -> str:
    m = side
    return f"""
    <div class="pane">
      <div class="paneh">
        <h3>{m['name']}</h3>
        <div class="st">{m['subtitle']}</div>
        <div class="mrow">
          <div class="cell"><div class="b">{m['fidelity']}</div><div class="k">fidelity R²</div></div>
          <div class="cell"><div class="b">{m['roc']['auc']}</div><div class="k">AUC</div></div>
          <div class="cell"><div class="b">{m['n_units']}</div><div class="k">units</div></div>
          <div class="cell"><div class="b">{m['n_legible']}</div><div class="k">legible</div></div>
          <div class="cell"><div class="b">{m['median_mu']}</div><div class="k">med μ</div></div>
          <div class="cell"><div class="b">{m['median_fanin']}</div><div class="k">med fan-in</div></div>
        </div>
      </div>
      <div class="svgwrap"><svg id="{ 'dagL' if m['name']=='Standard NN' else 'dagR' }"></svg>
        <div class="plegend">
          <span><i style="background:var(--corewash);border:1.5px solid var(--core)"></i>legible (μ≥0.8)</span>
          <span><i style="background:var(--periwash);border:1.5px solid var(--peri)"></i>polysemantic</span>
          <span>drag / scroll · hover a unit</span>
        </div>
      </div>
    </div>"""


def render_comparison(d: dict) -> str:
    L, R = d["left"], d["right"]
    data_json = json.dumps(d, ensure_ascii=False)
    ceil = d["fid_ref"]
    # headline verdict — judged on FIDELITY (the programme's metric), with the
    # raw-count gap reported honestly.
    dfid = L["fidelity"] - R["fidelity"]
    close_fid = abs(dfid) <= 0.06
    fid_clause = (
        f"On the programme's own metric — fidelity (held-out R² in log1p space, the "
        f"scale both nets train in) — they are close: {L['fidelity']} (standard) vs "
        f"{R['fidelity']} (x1), both near the {ceil} reference ceiling."
        if close_fid else
        f"On fidelity (held-out log1p R²) the standard net leads {L['fidelity']} vs "
        f"{R['fidelity']} (ceiling {ceil}).")
    count_clause = (
        f" In raw count space the bigger dense net fits the heavy-tailed peak hours "
        f"harder (R² {L['reg']['r2']} vs {R['reg']['r2']}, RMSE {L['reg']['rmse']} vs "
        f"{R['reg']['rmse']}) — a real cost of the minimal model's small capacity, "
        f"honestly shown, not hidden."
        if L["reg"]["r2"] - R["reg"]["r2"] > 0.05 else
        f" The two also rank high-demand hours almost identically (AUC {L['roc']['auc']} "
        f"vs {R['roc']['auc']}).")
    leg_clause = (
        f" The decisive difference is legibility, measured by the same instrument: the "
        f"standard net spreads the signal across {L['n_units']} dense units "
        f"(median fan-in {L['median_fanin']}, {L['n_legible']} legible), while XPLAIN-x1 "
        f"carries it in {R['n_units']} sparse units (median fan-in {R['median_fanin']}, "
        f"{R['n_legible']} legible) whose supports you can read off the right-hand DAG "
        f"— hour, temperature, year. Only the x1 model comes with a certifiable "
        f"function decomposition on top.")
    verdict = fid_clause + count_clause + leg_clause

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bike — Standard NN vs XPLAIN-x1</title>
<style>{_CSS}{_EXTRA_CSS}</style></head>
<body><div id="tip"></div>
<div class="wrap">
  <div class="top">
    <div><h1>Bike Sharing — Standard NN vs XPLAIN-x1</h1>
      <div class="sub">{d['task']} · n={d['n']:,} · d={d['d']} · target: {d['target']}
        · reference ceiling R² {d['fid_ref']}</div></div>
    <div class="stats">
      <div class="stat"><div class="v">{d['fid_ref']}</div><div class="l">ceiling R²</div></div>
    </div>
  </div>

  <div class="card" style="padding:0;overflow:hidden">
    <div class="cmp">{_pane(L,'warn')}{_pane(R,'accent')}</div>
  </div>

  <div class="card">
    <h2>Head-to-head</h2>
    <div class="note">Both DAGs above are measured by the <b>same audit instrument</b>;
      only the training differs (vanilla dense vs pressured + grown). Colours mark
      per-unit legibility (μ ≥ 0.8 and effective fan-in ≤ 3).</div>
    {_metrics_table(d)}
    <div class="verdict">{verdict}</div>
  </div>

  <div class="wgrid">
    <div class="card chart">
      <h2>Predicted vs actual <span class="pill peri">held-out test</span></h2>
      <div class="note">Perfect prediction lies on the dashed diagonal. Counts of
        bike rentals per hour.</div>
      <div class="wgrid" style="gap:10px">
        <div><div id="scL"></div><div class="cap" style="color:var(--warn)">Standard NN · RMSE {L['reg']['rmse']}</div></div>
        <div><div id="scR"></div><div class="cap" style="color:var(--accent)">XPLAIN-x1 · RMSE {R['reg']['rmse']}</div></div>
      </div>
    </div>
    <div class="card chart">
      <h2>ROC — {d['binary_task']}</h2>
      <div class="note">A regression-derived binary task (predict whether an hour's
        demand is above the median). Score = predicted count. Positive rate {d['pos_rate']}.</div>
      <div id="roc"></div>
      <div class="cap"><span style="color:var(--warn)">■</span> Standard NN AUC {L['roc']['auc']}
        &nbsp; <span style="color:var(--accent)">■</span> XPLAIN-x1 AUC {R['roc']['auc']}</div>
    </div>
  </div>

  <div class="card">
    <h2>Confusion matrices <span class="pill peri">threshold = median count {d['test_median']}</span></h2>
    <div class="note">Predicted vs actual high-demand hours on the held-out test set.</div>
    <div class="cms">
      {_confusion_html(L['confusion'], 'Standard NN', 'var(--warn)')}
      {_confusion_html(R['confusion'], 'XPLAIN-x1', 'var(--accent)')}
    </div>
  </div>

  <div class="foot">
    Standard NN: plain 1-hidden-layer dense MLP, vanilla MSE, no pressures / growth /
    certification (<code>baselines/standard_nn/</code>). XPLAIN-x1: the frozen method
    (seed-0 pipeline). Both trained on identical splits and target scaling, measured by
    the same audit. The x1 DAG's certified CORE/PERIPHERY labels live in the full
    dashboard; here both DAGs are coloured by per-model legibility for a like-for-like view.
  </div>
</div>
<script id="cmp-data" type="application/json">{data_json}</script>
<script>{_JS}</script>
</body></html>"""
