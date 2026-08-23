"""Self-contained HTML dashboards for owner review (Build Plan M6).

One `render_dashboard(data)` call -> a single standalone HTML file: no external
libraries, theme-aware, with an interactive drill-down concept DAG plus the
three certified layers (F: function components, S: routes, R: reliance) and the
certificate header.  Principles borrowed from the xplain-v4 dashboards (inline
CSS/JS, hand-rolled SVG DAG with pan/zoom + hover tooltips, embedded JSON) but
rebuilt around x1's as-built artefacts.

`data` is the JSON-serialisable dict assembled by `experiments/build_dashboards.py`
(dag, certificate document, tau sweep).  The renderer is pure: data in, HTML out.
"""
from __future__ import annotations

import json

# --------------------------------------------------------------------------- CSS
_CSS = """
:root {
  --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#8b8983;
  --grid:#e3e2db; --ring:rgba(11,11,11,.10); --wash:rgba(42,120,214,.08);
  --core:#1baf7a; --coreink:#0d7a54; --corewash:rgba(27,175,122,.12);
  --peri:#9a988f; --periwash:rgba(154,152,143,.12);
  --accent:#2a78d6; --warn:#eb6834; --good:#0ca30c;
  --edge:#b9b8ae;
}
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) {
  --page:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c8c7bd; --muted:#8b8983;
  --grid:#2c2c2a; --ring:rgba(255,255,255,.11); --wash:rgba(57,135,229,.14);
  --core:#199e70; --coreink:#5fd3a6; --corewash:rgba(25,158,112,.16);
  --peri:#6f6e67; --periwash:rgba(120,118,110,.16);
  --accent:#3987e5; --warn:#d95926; --good:#3fb23f; --edge:#55554e; } }
:root[data-theme="dark"] {
  --page:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c8c7bd; --muted:#8b8983;
  --grid:#2c2c2a; --ring:rgba(255,255,255,.11); --wash:rgba(57,135,229,.14);
  --core:#199e70; --coreink:#5fd3a6; --corewash:rgba(25,158,112,.16);
  --peri:#6f6e67; --periwash:rgba(120,118,110,.16);
  --accent:#3987e5; --warn:#d95926; --good:#3fb23f; --edge:#55554e; }
* { box-sizing:border-box; }
body { background:var(--page); color:var(--ink); margin:0;
  font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }
a { color:var(--accent); }
.wrap { max-width:1180px; margin:0 auto; padding:20px 20px 64px; }
.top { display:flex; align-items:flex-start; gap:16px; flex-wrap:wrap;
  border-bottom:1px solid var(--ring); padding-bottom:14px; margin-bottom:18px; }
.top h1 { font-size:20px; margin:0 0 3px; }
.top .sub { font-size:12.5px; color:var(--muted); }
.back { font-size:12.5px; }
.stats { display:flex; gap:22px; margin-left:auto; flex-wrap:wrap; }
.stat { text-align:right; }
.stat .v { font-size:19px; font-weight:650; font-variant-numeric:tabular-nums; }
.stat .l { font-size:10.5px; color:var(--muted); text-transform:uppercase;
  letter-spacing:.04em; }
.pill { display:inline-block; font-size:11px; font-weight:700; padding:2px 9px;
  border-radius:11px; letter-spacing:.03em; }
.pill.core { background:var(--corewash); color:var(--coreink); }
.pill.peri { background:var(--periwash); color:var(--ink2); }
.pill.warn { background:color-mix(in srgb,var(--warn) 15%,transparent); color:var(--warn); }
.pill.ok   { background:color-mix(in srgb,var(--good) 14%,transparent); color:var(--good); }
.card { background:var(--surface); border:1px solid var(--ring); border-radius:10px;
  padding:15px 17px; margin:16px 0; }
.card h2 { font-size:14px; margin:0 0 3px; display:flex; gap:9px; align-items:center;
  flex-wrap:wrap; }
.card .note { font-size:12px; color:var(--muted); margin:2px 0 12px; line-height:1.5; }
.layerbar { display:flex; gap:0; height:5px; border-radius:3px; overflow:hidden;
  margin:4px 0 2px; background:var(--grid); }
.dagcard { padding:0; overflow:hidden; }
.daghdr { display:flex; align-items:center; gap:10px; flex-wrap:wrap;
  padding:12px 15px 10px; border-bottom:1px solid var(--ring); }
.daghdr h2 { margin:0; }
.seg { display:inline-flex; border:1px solid var(--grid); border-radius:7px; overflow:hidden; }
.seg button { border:none; border-radius:0; padding:4px 11px; background:var(--surface);
  color:var(--ink2); font-size:12px; cursor:pointer; }
.seg button.on { background:var(--accent); color:#fff; }
.ctrls { display:flex; gap:7px; align-items:center; flex-wrap:wrap; margin-left:auto; }
.ctrls button, .ctrls label { font-size:12px; background:var(--surface); color:var(--ink);
  border:1px solid var(--grid); border-radius:6px; padding:3px 9px; cursor:pointer; }
.ctrls label { display:inline-flex; align-items:center; gap:5px; }
.dagbody { display:grid; grid-template-columns:1fr 300px; }
@media (max-width:820px){ .dagbody { grid-template-columns:1fr; } }
.dagwrap { position:relative; height:540px; background:var(--page);
  border-right:1px solid var(--ring); overflow:hidden; }
.dagwrap svg { width:100%; height:100%; display:block; cursor:grab; }
.dagwrap svg:active { cursor:grabbing; }
.legend { position:absolute; left:12px; bottom:10px; display:flex; gap:13px;
  flex-wrap:wrap; font-size:11px; color:var(--ink2);
  background:color-mix(in srgb,var(--surface) 84%,transparent);
  padding:4px 9px; border-radius:6px; }
.legend i { width:11px; height:11px; border-radius:3px; display:inline-block;
  vertical-align:-1px; margin-right:4px; }
.detail { padding:14px 15px; min-height:0; overflow-y:auto; max-height:540px; }
.detail .empty { color:var(--muted); font-size:12.5px; line-height:1.6; }
.detail h3 { margin:0 0 2px; font-size:14px; word-break:break-word; }
.detail .did { font-size:11px; color:var(--muted); font-family:ui-monospace,monospace; }
.detail dl { display:grid; grid-template-columns:92px 1fr; gap:3px 10px;
  margin:11px 0 4px; font-size:12.5px; font-variant-numeric:tabular-nums; }
.detail dt { color:var(--muted); } .detail dd { margin:0; }
.detail .chips { margin-top:9px; }
.chip { display:inline-block; font-size:10.5px; border:1px solid var(--grid);
  border-radius:10px; padding:1px 8px; margin:2px 3px 0 0; color:var(--ink2); }
.chip.r { border-color:color-mix(in srgb,var(--warn) 45%,var(--grid));
  color:var(--warn); }
table.t { border-collapse:collapse; width:100%; font-size:12.5px;
  font-variant-numeric:tabular-nums; }
table.t th, table.t td { text-align:left; padding:5px 12px 5px 0;
  border-top:1px solid var(--grid); vertical-align:top; }
table.t th { color:var(--muted); font-weight:600; border-top:none;
  font-size:11px; text-transform:uppercase; letter-spacing:.03em; }
table.t td.num, table.t th.num { text-align:right; }
.sharebar { display:inline-block; height:9px; border-radius:2px;
  background:var(--accent); vertical-align:0; min-width:1px; }
.grp { font-size:12px; color:var(--ink2); margin:2px 0; }
.grp b { color:var(--ink); font-weight:600; }
.bars { display:grid; gap:5px; margin-top:4px; }
.bars .row { display:flex; gap:9px; align-items:baseline; font-size:12.5px; }
.bars .row .u { min-width:150px; color:var(--ink2); }
.bars .row.pass .u::before { content:"\\2713 "; color:var(--good); }
.bars .row.fail .u::before { content:"\\2717 "; color:var(--warn); }
#tip { position:fixed; pointer-events:none; background:var(--surface); color:var(--ink);
  border:1px solid var(--ring); border-radius:8px; padding:8px 11px; font-size:12px;
  box-shadow:0 4px 18px rgba(0,0,0,.18); opacity:0; z-index:30; max-width:300px;
  transition:opacity .08s; }
#tip h4 { margin:0 0 3px; font-size:12.5px; }
#tip .m { font-size:11px; color:var(--muted); font-variant-numeric:tabular-nums; }
.foot { font-size:11.5px; color:var(--muted); margin-top:26px; line-height:1.6;
  border-top:1px solid var(--ring); padding-top:12px; }
"""

# --------------------------------------------------------------------------- JS
_JS = r"""
"use strict";
const $ = s => document.getElementById(s);
const NS = "http://www.w3.org/2000/svg";
const D = JSON.parse($("dash-data").textContent);
const tip = $("tip");
function el(t,a,p){ const e=document.createElementNS(NS,t);
  for(const k in a) e.setAttribute(k,a[k]); if(p) p.appendChild(e); return e; }
function fmt(v,d){ return (v==null||isNaN(v))?"—":Number(v).toFixed(d==null?2:d); }
function abbr(s,m){ m=m||13; return s.length>m ? s.slice(0,m-1)+"…" : s; }

// Two models over the same run: HIER (drill-down concept hierarchy) and UNITS
// (the raw model graph). MODE switches between them.
const HIER = D.hier, UNITS = D.units;
let MODE = "hier", SEL=null, EXP={}, SHOW_PERI=true, SHOW_IN=true;
let VZ={k:1,x:0,y:0}, drag=null, W=900, H=560;
const svg=$("dag"), g=$("dagg");

function M(){ return MODE==="hier" ? HIER : UNITS; }
function byId(){ const m={}; M().nodes.forEach(n=>m[n.id]=n); return m; }

// ---------- visibility: expansion-set + reachability ------------------------
// A node is visible iff reachable from a root through EXPANDED ancestors.
// Member units are SHARED between components, so visibility is recomputed by
// BFS rather than per-node flags (collapsing one parent keeps a shared child
// visible while another expanded parent still declares it).
let VIS = new Set();
function recomputeVis(){
  VIS = new Set(); const B=byId();
  const queue = M().nodes.filter(n=>n.start).map(n=>n.id);
  queue.forEach(id=>VIS.add(id));
  while(queue.length){
    const id = queue.shift(), n = B[id];
    if(!n || !EXP[id]) continue;
    (n.drill||[]).forEach(cid=>{
      if(!VIS.has(cid)){ VIS.add(cid); queue.push(cid); } });
  }
}
function visible(n){
  if(MODE==="units"){
    if(n.kind==="input" && !SHOW_IN) return false;
    if(n.kind==="unit" && n.tag!=="CORE" && !SHOW_PERI) return false;
    return true;
  }
  return VIS.has(n.id);
}
function hasChildren(n){ return n.drill && n.drill.length; }
function isExpanded(n){ return !!EXP[n.id]; }
function toggleDrill(n){
  if(!hasChildren(n)) return;
  EXP[n.id] = !EXP[n.id];
  layout(); draw();
}

// ---------- layout: columns by level, spread visible nodes vertically -------
let POS={};
function level(n){
  if(MODE==="hier") return n.lvl;
  return n.kind==="input"?0 : n.kind==="output"?UNITS.outCol : (n.layer||1);
}
function layout(){
  if(MODE==="hier") recomputeVis();
  const vis=M().nodes.filter(visible);
  const cols={}; vis.forEach(n=>{ const c=level(n); (cols[c]=cols[c]||[]).push(n); });
  const keys=Object.keys(cols).map(Number).sort((a,b)=>a-b);
  const COLW=232, VGAP=60, TOP=46;
  let maxRows=1; keys.forEach(k=>maxRows=Math.max(maxRows,cols[k].length));
  H=TOP*2 + (maxRows-1)*VGAP + 30;
  W=(keys.length? (Math.max(...keys)+1):1)*COLW;
  POS={};
  keys.forEach(k=>{ const arr=cols[k], x=k*COLW+COLW/2;
    const span=(arr.length-1)*VGAP, y0=(H-span)/2;
    arr.forEach((n,i)=>POS[n.id]={x,y:y0+i*VGAP}); });
  svg.setAttribute("viewBox",`0 0 ${W} ${H}`);
}

const R=17;
function nodeFill(n){
  if(n.kind==="feature"||n.kind==="input") return "var(--wash)";
  if(n.kind==="output") return "var(--surface)";
  return (n.tag==="CORE") ? "var(--corewash)" : "var(--periwash)";
}
function nodeStroke(n){
  if(n.kind==="component"||n.kind==="unit") return n.tag==="CORE"?"var(--core)":"var(--peri)";
  if(n.kind==="output") return "var(--accent)";
  return "var(--grid)";
}
function nlabel(n){
  if(n.kind==="component") return n.support_names.map(s=>abbr(s,10)).join("·");
  if(n.kind==="unit") return (n.support_names&&n.support_names.length)
      ? n.support_names.map(s=>abbr(s,9)).join("·") : n.id;
  return abbr(n.label,16);
}

function draw(){
  g.textContent="";
  const B=byId();
  M().edges.forEach(e=>{
    const a=B[e.src], b=B[e.dst]; if(!a||!b||!visible(a)||!visible(b)) return;
    const p=POS[e.src], q=POS[e.dst]; if(!p||!q) return;
    const mx=(p.x+q.x)/2, w=0.8+5*Math.min(1,(e.share||0)*3);
    el("path",{d:`M${p.x+R} ${p.y} C ${mx} ${p.y}, ${mx} ${q.y}, ${q.x-R} ${q.y}`,
      fill:"none",stroke:"var(--edge)","stroke-width":w.toFixed(2),
      "stroke-opacity":(SEL&&SEL!==e.src&&SEL!==e.dst)?0.14:0.5},g);
  });
  M().nodes.forEach(n=>{
    if(!visible(n)) return; const p=POS[n.id]; if(!p) return;
    const dim=SEL&&SEL!==n.id&&!adjacent(n.id,SEL,B);
    const grp=el("g",{transform:`translate(${p.x},${p.y})`,
      style:`cursor:pointer;opacity:${dim?0.3:1}`},g);
    if(n.kind==="output"){
      el("rect",{x:-R-18,y:-14,width:2*R+36,height:28,rx:7,fill:nodeFill(n),
        stroke:nodeStroke(n),"stroke-width":1.5},grp);
    } else if(n.kind==="feature"||n.kind==="input"){
      el("rect",{x:-R-12,y:-12,width:2*R+24,height:24,rx:12,fill:nodeFill(n),
        stroke:nodeStroke(n),"stroke-width":1.2},grp);
    } else {
      el("circle",{r:R,fill:nodeFill(n),stroke:nodeStroke(n),
        "stroke-width":(n.id===SEL)?3:1.8},grp);
    }
    const t=el("text",{"text-anchor":"middle",y:4,
      style:"font:11px system-ui;fill:var(--ink);pointer-events:none"},grp);
    t.textContent=nlabel(n);
    // sub-label: μ for units, share for components
    let sub=null;
    if(n.kind==="unit"&&n.mu!=null) sub="μ "+fmt(n.mu,2);
    else if(n.kind==="component"&&n.share!=null) sub=(100*n.share).toFixed(0)+"%";
    if(sub){ const st=el("text",{"text-anchor":"middle",y:R+12,
      style:"font:9.5px system-ui;fill:var(--muted);pointer-events:none"},grp);
      st.textContent=sub; }
    // drill affordance: a small +/- badge when the node has children
    if(MODE==="hier"&&hasChildren(n)){
      const open=isExpanded(n);
      el("circle",{cx:R-2,cy:-R+2,r:6.5,fill:"var(--surface)",
        stroke:nodeStroke(n),"stroke-width":1.2},grp);
      const bt=el("text",{x:R-2,y:-R+5.4,"text-anchor":"middle",
        style:"font:11px system-ui;font-weight:700;fill:var(--ink2);pointer-events:none"},grp);
      bt.textContent=open?"–":"+";
    }
    grp.addEventListener("mouseenter",ev=>hover(n,ev));
    grp.addEventListener("mousemove",moveTip);
    grp.addEventListener("mouseleave",()=>tip.style.opacity=0);
    grp.addEventListener("click",ev=>{ev.stopPropagation(); select(n.id);});
    grp.addEventListener("dblclick",ev=>{ev.stopPropagation();
      if(MODE==="hier") toggleDrill(n);});
  });
  applyZoom();
}
function adjacent(a,b,B){ return M().edges.some(e=>(e.src===a&&e.dst===b)||(e.src===b&&e.dst===a)); }

// ---------- hover + detail --------------------------------------------------
function hover(n,ev){
  let h=`<h4>${nlabel(n)}</h4>`;
  if(n.kind==="unit")
    h+=`<div class="m">${n.tag} · layer ${n.layer} · ${n.form||"?"}</div>`+
       `<div class="m">μ ${fmt(n.mu)} · Π ${fmt(n.Pi)} · π ${fmt(n.pi)} · Δ ${fmt(n.delta)}</div>`;
  else if(n.kind==="component")
    h+=`<div class="m">${n.tag} function component · share ${fmt(100*n.share,1)}%</div>`+
       `<div class="m">Π ${fmt(n.Pi)} · π ${fmt(n.pi)} · ${n.n_members||0} member unit${n.n_members===1?"":"s"}`+
       `${hasChildren(n)?" · double-click to drill":""}</div>`;
  else if(n.kind==="output")
    h+=`<div class="m">${n.sub||"output"}${hasChildren(n)?" · double-click to drill into its components":""}</div>`;
  else h+=`<div class="m">input feature</div>`;
  tip.innerHTML=h; tip.style.opacity=1; moveTip(ev);
}
function moveTip(ev){
  tip.style.left=Math.min(ev.clientX+14, innerWidth-tip.offsetWidth-8)+"px";
  tip.style.top=Math.min(ev.clientY+14, innerHeight-tip.offsetHeight-8)+"px";
}
function select(id){ SEL=(SEL===id)?null:id; draw(); renderDetail(); }

function renderDetail(){
  const box=$("detail"), B=byId();
  if(!SEL || !B[SEL]){ box.innerHTML=emptyHelp(); return; }
  const n=B[SEL];
  if(n.kind==="unit"){
    let h=`<h3>${nlabel(n)}</h3><div class="did">${n.id}</div>`+
      `<div style="margin-top:7px"><span class="pill ${n.tag==="CORE"?"core":"peri"}">${n.tag}</span></div>`+
      `<dl><dt>support</dt><dd>${(n.support_names||[]).join(", ")||"—"}</dd>`+
      `<dt>form</dt><dd>${n.form||"—"}</dd><dt>layer</dt><dd>${n.layer}</dd>`+
      `<dt>μ mono</dt><dd>${fmt(n.mu)}</dd><dt>Π stab</dt><dd>${fmt(n.Pi)}</dd>`+
      `<dt>π cpss</dt><dd>${fmt(n.pi)}</dd><dt>Δ real</dt><dd>${fmt(n.delta,3)}</dd></dl>`;
    h+=(n.reasons&&n.reasons.length)
      ? `<div class="chips">`+n.reasons.map(r=>`<span class="chip r">${r}</span>`).join("")+`</div>`
      : `<div class="chips"><span class="chip">certified: stable, frequent, real</span></div>`;
    box.innerHTML=h; return;
  }
  if(n.kind==="component"){
    box.innerHTML=`<h3>${nlabel(n)}</h3><div class="did">function component</div>`+
      `<div style="margin-top:7px"><span class="pill ${n.tag==="CORE"?"core":"peri"}">${n.tag}</span></div>`+
      `<dl><dt>support</dt><dd>${n.support_names.join(", ")}</dd>`+
      `<dt>share</dt><dd>${fmt(100*n.share,1)}% of function variance</dd>`+
      `<dt>Π stab</dt><dd>${fmt(n.Pi)}</dd><dt>π cpss</dt><dd>${fmt(n.pi)}</dd>`+
      `<dt>members</dt><dd>${n.n_members||0} unit(s) carry this claim</dd></dl>`+
      `<div class="chips"><span class="chip">${hasChildren(n)
        ? (n.n_members ? "double-click: expand member units" : "double-click: features (no localised carrier)")
        : "leaf"}</span></div>`;
    return;
  }
  box.innerHTML=`<h3>${n.label}</h3><div class="did">${n.kind}</div>`+
    `<div class="empty" style="margin-top:10px">${n.kind==="output"
      ? "Model output. Double-click to drill into the certified components that drive it."
      : "Input feature (one monosemantic column)."}</div>`;
}
function emptyHelp(){
  if(MODE==="hier") return `<div class="empty"><b>Layer F DAG.</b> The certified function
    components (the “laws”) are shown by default. <b>Double-click</b> a component to expand
    its <b>member units</b> — the model units its mass physically flows through — then
    double-click a unit to expand its parents down to input features: the end-to-end
    parameter flow, with edge width = real contribution share. A <b>+</b> badge marks an
    expandable node; <b>–</b> collapses it.<br><br><b>Green</b> = CORE (certified);
    <b>grey</b> = periphery. Single-click any node for its numbers. Drag to pan, scroll to zoom.</div>`;
  return `<div class="empty"><b>Model units.</b> Circles are the network's hidden units,
    labelled by their support and μ (monosemanticity). <b>Green</b> = CORE, <b>grey</b> =
    periphery (labelled with why it failed to certify). This is the raw graph for judging
    node nameability. Click a unit for detail; toggle inputs/periphery above.</div>`;
}

// ---------- pan / zoom ------------------------------------------------------
function applyZoom(){ g.setAttribute("transform",`translate(${VZ.x},${VZ.y}) scale(${VZ.k})`); }
svg.addEventListener("mousedown",e=>{drag={x:e.clientX,y:e.clientY,ox:VZ.x,oy:VZ.y};});
addEventListener("mousemove",e=>{ if(!drag)return;
  VZ.x=drag.ox+(e.clientX-drag.x); VZ.y=drag.oy+(e.clientY-drag.y); applyZoom(); });
addEventListener("mouseup",()=>drag=null);
svg.addEventListener("click",()=>{ if(SEL){SEL=null; draw(); renderDetail();} });
svg.addEventListener("wheel",e=>{ e.preventDefault();
  const r=svg.getBoundingClientRect(), sx=(e.clientX-r.left)/r.width*W, sy=(e.clientY-r.top)/r.height*H;
  const f=e.deltaY<0?1.12:1/1.12, nk=Math.max(0.3,Math.min(3.2,VZ.k*f));
  VZ.x=sx-(sx-VZ.x)*(nk/VZ.k); VZ.y=sy-(sy-VZ.y)*(nk/VZ.k); VZ.k=nk; applyZoom();
},{passive:false});
function fit(){ VZ={k:1,x:0,y:0}; applyZoom(); }

// ---------- mode + controls -------------------------------------------------
function setMode(m){
  MODE=m; SEL=null;
  $("m_hier").classList.toggle("on",m==="hier");
  $("m_units").classList.toggle("on",m==="units");
  $("unitctrls").style.display = m==="units" ? "" : "none";
  $("hierctrls").style.display = m==="hier" ? "" : "none";
  layout(); draw(); renderDetail();
}
$("m_hier").addEventListener("click",()=>setMode("hier"));
$("m_units").addEventListener("click",()=>setMode("units"));
$("b_fit").addEventListener("click",()=>{ // reset: back to the Layer F view
  EXP={}; HIER.nodes.filter(n=>n.start).forEach(n=>EXP[n.id]=true);
  layout(); draw(); fit(); });
$("b_expand").addEventListener("click",()=>{ // expand every component's members
  HIER.nodes.forEach(n=>{ if(n.start||n.kind==="component") EXP[n.id]=true; });
  layout(); draw(); });
$("t_peri").addEventListener("change",e=>{SHOW_PERI=e.target.checked; layout(); draw();});
$("t_in").addEventListener("change",e=>{SHOW_IN=e.target.checked; layout(); draw();});
// default view: the Layer F DAG (root expanded -> components visible)
HIER.nodes.filter(n=>n.start).forEach(n=>EXP[n.id]=true);
setMode("hier"); fit();
"""


def _esc(s: object) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _comp_share(c: dict) -> float:
    return c.get("share", c.get("share_main", 0)) or 0


def _comp_names(c: dict) -> str:
    return ", ".join(c.get("support_names", c.get("names", [])))


def _sharebar_table(components: list[dict]) -> str:
    if not components:
        return '<div class="note">No function components above threshold.</div>'
    comps = sorted(components, key=lambda c: -_comp_share(c))
    mx = max(_comp_share(c) for c in comps) or 1.0
    rows = []
    for c in comps:
        share = _comp_share(c)
        w = max(1, round(150 * share / mx))
        lab = c.get("label", "")
        pill = "core" if lab == "CORE" else "peri"
        rows.append(
            f'<tr><td>{_esc(_comp_names(c))}</td>'
            f'<td class="num">{share:.3f}</td>'
            f'<td><span class="sharebar" style="width:{w}px"></span></td>'
            f'<td class="num">{(c.get("Pi", 0) or 0):.2f}</td>'
            f'<td class="num">{(c.get("pi", 0) or 0):.2f}</td>'
            f'<td><span class="pill {pill}">{_esc(lab)}</span></td></tr>')
    return (
        '<table class="t"><thead><tr><th>component (support)</th>'
        '<th class="num">share</th><th></th><th class="num">Π</th>'
        '<th class="num">π</th><th>label</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>')


def _routes_block(routes: dict) -> str:
    groups = routes.get("feature_groups", [])
    rows = routes.get("rows", [])
    gtxt = ""
    multi = [g for g in groups if len(g) > 1]
    if multi:
        gtxt = '<div class="note">Collinearity groups (|Spearman| ≥ 0.8): ' + \
            "; ".join("<b>{" + ", ".join(_esc(x) for x in g) + "}</b>" for g in multi) + \
            ". Singletons omitted.</div>"
    else:
        gtxt = '<div class="note">All feature groups are singletons ' \
            '(no strong collinearity) — routes coincide with unit supports.</div>'
    if not rows:
        return gtxt + '<div class="note">No routes above threshold.</div>'
    body = []
    for r in rows:
        sup = ", ".join(r.get("support_names", r.get("support", []) if isinstance(
            r.get("support"), list) else []))
        lab = r.get("label", "")
        pill = "core" if lab == "CORE" else "peri"
        nvar = len(r.get("variants", []) or [])
        core = r.get("common_core_names") or r.get("common_core") or []
        core_txt = ", ".join(map(_esc, core)) if core else "—"
        body.append(
            f'<tr><td>{_esc(sup)}</td>'
            f'<td class="num">{r.get("Pi", 0):.2f}</td>'
            f'<td class="num">{r.get("pi", 0):.2f}</td>'
            f'<td class="num">{nvar}</td>'
            f'<td>{core_txt}</td>'
            f'<td>{"yes" if r.get("multiplicitous") else "no"}</td>'
            f'<td><span class="pill {pill}">{_esc(lab)}</span></td></tr>')
    return gtxt + (
        '<table class="t"><thead><tr><th>route (modal support)</th>'
        '<th class="num">Π</th><th class="num">π</th>'
        '<th class="num">variants</th><th>common core</th>'
        '<th>multiplic.</th><th>label</th></tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table>')


def _reliance_block(reliance: list[dict]) -> str:
    if not reliance:
        return '<div class="note">No reliance rows.</div>'
    mx = max((r.get("max_reliance", 0) or 0) for r in reliance) or 1.0
    rows = []
    for r in sorted(reliance, key=lambda x: -(x.get("min_reliance", 0) or 0)):
        lo, hi = r.get("min_reliance", 0), r.get("max_reliance", 0)
        x0, x1 = round(150 * lo / mx), round(150 * hi / mx)
        rows.append(
            f'<tr><td>{_esc(r.get("group", ""))}</td>'
            f'<td class="num">{lo:.3f}</td><td class="num">{hi:.3f}</td>'
            f'<td><svg width="156" height="12" style="vertical-align:-2px">'
            f'<line x1="{x0}" x2="{x1}" y1="6" y2="6" stroke="var(--accent)" '
            f'stroke-width="6" stroke-linecap="round"/></svg></td></tr>')
    return (
        '<div class="note">Every retraining relies on each group by at least the '
        'min (a portfolio guarantee robust to which carving a run picked).</div>'
        '<table class="t"><thead><tr><th>feature group</th>'
        '<th class="num">min</th><th class="num">max</th><th>range</th>'
        '</tr></thead><tbody>' + "".join(rows) + '</tbody></table>')


def _bars_block(data: dict) -> str:
    b = data.get("bars", [])
    if not b:
        return ""
    rows = []
    for row in b:
        cls = "pass" if row.get("pass") else "fail"
        rows.append(f'<div class="row {cls}"><span class="u">{_esc(row["id"])}</span>'
                    f'<span>{_esc(row["detail"])}</span></div>')
    return '<div class="bars">' + "".join(rows) + '</div>'


def _build_units(dag: dict) -> dict:
    """The raw model graph (inputs -> units -> outputs) for nameability review."""
    units = [n for n in dag["nodes"] if n["kind"] == "unit"]
    out_col = 2 + max([1] + [n.get("layer", 1) for n in units])
    return {"nodes": dag["nodes"], "edges": dag["edges"], "outCol": out_col}


def _transitive_feature_supports(dag: dict) -> dict:
    """Resolve each unit's support_names to INPUT features, recursively.

    Layer-1 units name features directly; deeper units name parent unit ids —
    resolvable inside the DAG itself, so saved artefacts re-render with no new
    producer data."""
    unit_nodes = {n["id"]: n for n in dag["nodes"] if n["kind"] == "unit"}
    memo: dict = {}

    def resolve(uid, seen=frozenset()):
        if uid in memo:
            return memo[uid]
        out = set()
        for name in (unit_nodes[uid].get("support_names") or []):
            if name in unit_nodes and name not in seen:
                out |= resolve(name, seen | {uid})
            elif name not in unit_nodes:
                out.add(name)
        memo[uid] = out
        return out

    return {uid: resolve(uid) for uid in unit_nodes}


def _build_hier(cert: dict, dag: dict, is_multiclass: bool,
                n_classes: int) -> dict:
    """The Layer F DAG with drill-down into the physical model.

    Levels: output (0) -> certified components (1) -> member units (2..) ->
    input features.  A unit is a MEMBER of a component iff the component's
    feature support is contained in the unit's transitive input support — the
    units through which that certified claim's mass physically flows.  A unit
    drills into its physical parents (deeper units, then features) with edge
    widths taken from the real masked-weight contribution shares, so the fully
    expanded view is the end-to-end parameter flow.
    """
    fdec = cert["function_decomposition"]
    ident = cert["identification"]
    comps = sorted(fdec.get("components", []), key=lambda c: -_comp_share(c))
    core = [c for c in comps if c.get("label") == "CORE"]
    peri = [c for c in comps if c.get("label") != "CORE"][:6]
    chosen = core + peri

    unit_nodes = {n["id"]: n for n in dag["nodes"] if n["kind"] == "unit"}
    trans = _transitive_feature_supports(dag)
    n_layers = max([1] + [n.get("layer", 1) for n in unit_nodes.values()])
    feat_lvl = 2 + n_layers
    edge_share = {(e["src"], e["dst"]): e.get("share", 0)
                  for e in dag.get("edges", [])}

    nodes, edges = [], []
    feats: dict = {}
    used_units: dict = {}

    def feat_node(name: str) -> str:
        fid = "F::" + name
        if fid not in feats:
            feats[fid] = {"id": fid, "kind": "feature", "label": name,
                          "lvl": feat_lvl}
        return fid

    def unit_node(uid: str) -> str:
        if uid in used_units:
            return uid
        u = unit_nodes[uid]
        lvl = 2 + (n_layers - u.get("layer", 1))
        used_units[uid] = {**u, "lvl": lvl, "drill": []}   # placeholder first
        drill = []
        for pname in (u.get("support_names") or []):
            if pname in unit_nodes:
                drill.append(unit_node(pname))
                edges.append({"src": uid, "dst": pname,
                              "share": edge_share.get((pname, uid), 0.05)})
            else:
                drill.append(feat_node(pname))
                edges.append({"src": uid, "dst": "F::" + pname,
                              "share": edge_share.get((pname, uid), 0.05)})
        used_units[uid]["drill"] = drill
        return uid

    comp_ids: list = []
    for i, c in enumerate(chosen):
        cid = f"C{i}"
        comp_ids.append(cid)
        supp = c.get("support_names", c.get("names", []))
        sset = set(supp)
        members = [uid for uid, tf in trans.items() if sset and sset <= tf]
        drill = [unit_node(uid) for uid in members]
        for uid in members:
            edges.append({"src": cid, "dst": uid,
                          "share": unit_nodes[uid].get("coverage") or 0.02})
        if not members:   # unlocalised claim: drill straight to its features
            drill = [feat_node(f) for f in supp]
            for f in supp:
                edges.append({"src": cid, "dst": "F::" + f, "share": 0.03})
        nodes.append({
            "id": cid, "kind": "component", "lvl": 1,
            "support_names": supp, "share": _comp_share(c),
            "Pi": c.get("Pi", 0), "pi": c.get("pi", 0),
            "n_members": len(members),
            "tag": c.get("label", "PERIPHERY"), "drill": drill})

    root_label = (f"{n_classes}-class output" if is_multiclass
                  else ("class decision" if ident["task"] == "classification"
                        else "output"))
    root = {"id": "OUT", "kind": "output", "lvl": 0, "start": True,
            "label": root_label, "sub": f"{ident['task']} · {ident['dataset']}",
            "drill": comp_ids}
    for cid, c in zip(comp_ids, chosen):
        edges.append({"src": "OUT", "dst": cid, "share": _comp_share(c)})
    nodes = [root] + nodes + list(used_units.values()) + list(feats.values())
    dedup: dict = {}
    for e in edges:
        k = (e["src"], e["dst"])
        if k not in dedup or e.get("share", 0) > dedup[k].get("share", 0):
            dedup[k] = e
    return {"nodes": nodes, "edges": list(dedup.values())}


def render_dashboard(data: dict) -> str:
    """data -> a single self-contained HTML string."""
    cert = data["cert"]
    ident = cert["identification"]
    perf = cert["performance_and_limits"]
    struct = cert["interpretability_structure"]
    stat = cert["statistical_certification"]
    fdec = cert["function_decomposition"]
    routes = cert["routes"]
    reliance = cert["portfolio_reliance"]

    ratio = perf.get("fidelity_ratio")
    depth = struct.get("honest_depth", {})
    regime = stat.get("regime", "standard")
    regime_short = "small-n regime" if regime != "standard" else "standard regime"
    ev = fdec.get("ev_bound")

    # outputs: derive multiclass + class count from the DAG output nodes
    out_nodes = [n for n in data["dag"]["nodes"] if n["kind"] == "output"]
    is_clf = ident["task"] == "classification"
    n_classes = len(out_nodes) if is_clf else 1
    is_multiclass = is_clf and n_classes > 2

    # coverage is a clean [0,1] fraction only for a single output (regression /
    # binary logit); for multiclass the per-logit shares can sum past 1, so we
    # report the core-component count instead of a misleading "coverage".
    raw_cov = data.get("core_share_sum")
    cov_is_fraction = (not is_multiclass) and raw_cov is not None and raw_cov <= 1.05
    if cov_is_fraction:
        cov_stat = (f'<div class="stat"><div class="v">{raw_cov:.2f}</div>'
                    f'<div class="l">core coverage</div></div>')
    else:
        cov_stat = (f'<div class="stat"><div class="v">{fdec.get("recon_r2") or 0:.2f}</div>'
                    f'<div class="l">recon R²</div></div>')

    stats_html = (
        f'<div class="stat"><div class="v">{perf.get("fidelity_val", 0):.3f}</div>'
        f'<div class="l">fidelity</div></div>'
        f'<div class="stat"><div class="v">{ratio if ratio else "—"}'
        f'</div><div class="l">vs ceiling</div></div>'
        f'<div class="stat"><div class="v">{fdec.get("n_core", 0)}</div>'
        f'<div class="l">core components</div></div>'
        f'{cov_stat}')

    depth_txt = ""
    if depth:
        earned = depth.get("earned_depth", depth.get("depth"))
        depth_txt = (f'honest depth <b>{earned}</b> '
                     f'(widths {"×".join(map(str, struct.get("widths", [])))})')

    payload = {
        "units": _build_units(data["dag"]),
        "hier": _build_hier(cert, data["dag"], is_multiclass, n_classes),
    }
    data_json = json.dumps(payload, ensure_ascii=False, default=str)

    title = f"XPLAIN-x1 — {ident['dataset']}"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<div id="tip"></div>
<div class="wrap">
  <div class="top">
    <div>
      <div class="back"><a href="index.html">← all dashboards</a></div>
      <h1>{_esc(ident['dataset'])}</h1>
      <div class="sub">{_esc(ident['task'])} · n={ident['n']:,} · d={ident['d']}
        · <span class="pill {'warn' if regime!='standard' else 'ok'}">{_esc(regime_short)}</span></div>
    </div>
    <div class="stats">{stats_html}</div>
  </div>

  <div class="card dagcard">
    <div class="daghdr">
      <h2>Concept DAG</h2>
      <div class="seg">
        <button id="m_hier" class="on">Layer F DAG</button>
        <button id="m_units">Model units</button>
      </div>
      <div class="ctrls">
        <span id="hierctrls" class="ctrls" style="border:none;padding:0">
          <button id="b_expand">expand all members</button>
        </span>
        <span id="unitctrls" class="ctrls" style="border:none;padding:0;display:none">
          <label><input type="checkbox" id="t_peri" checked> periphery</label>
          <label><input type="checkbox" id="t_in" checked> inputs</label>
        </span>
        <button id="b_fit">reset view</button>
      </div>
    </div>
    <div class="dagbody">
      <div class="dagwrap">
        <svg id="dag" preserveAspectRatio="xMidYMid meet"><g id="dagg"></g></svg>
        <div class="legend">
          <span><i style="background:var(--corewash);border:1.5px solid var(--core)"></i>CORE</span>
          <span><i style="background:var(--periwash);border:1.5px solid var(--peri)"></i>periphery</span>
          <span><i style="background:var(--surface);border:1.5px solid var(--accent)"></i>output</span>
          <span><i style="background:var(--wash);border:1px solid var(--grid)"></i>feature</span>
          <span>double-click: component → member units → features · edge = real share</span>
        </div>
      </div>
      <div class="detail" id="detail"></div>
    </div>
  </div>

  <div class="card">
    <h2>Layer F — certified function components <span class="pill core">the certified claims</span></h2>
    <div class="note">{_esc(fdec['note'])} Reconstruction R² {fmt_or(fdec.get('recon_r2'))};
      E[V] ≤ {fmt_or(ev, 3)} over {stat.get('p_universe','?')}-support universe.</div>
    {_sharebar_table(fdec.get('components', []))}
  </div>

  <div class="card">
    <h2>Layer S — routes <span class="pill peri">inspectable equivalence classes</span></h2>
    {_routes_block(routes)}
  </div>

  <div class="card">
    <h2>Layer R — portfolio reliance</h2>
    {_reliance_block(reliance)}
  </div>

  <div class="card">
    <h2>Certificate summary</h2>
    <div class="note">{depth_txt} · core coverage {struct.get('core_coverage_share',0):.2f}
      · periphery reasons: {_esc(', '.join(struct.get('periphery_reasons', [])) or 'none')}.
      Regime: {_esc(regime)}.</div>
    {_bars_block(data)}
    <div class="note" style="margin-top:10px"><b>Non-claims.</b>
      {_esc(' '.join(cert.get('non_claims', [])))}</div>
  </div>

  <div class="foot">
    XPLAIN-x1 audit dashboard · git {_esc(ident.get('git_commit','')[:10])}
    · config {_esc(ident.get('config_hash','')[:10])}
    · data {_esc(ident.get('data_hash','')[:10])}
    · R={ident['seeds']['restarts']} restarts, {ident['seeds']['cpss_runs']} CPSS runs.
    Certified claims are the Layer F components; the DAG is inspectable, the periphery labelled.
  </div>
</div>
<script id="dash-data" type="application/json">{data_json}</script>
<script>{_JS}</script>
</body>
</html>"""


def fmt_or(v, d: int = 2) -> str:
    return "—" if v is None else f"{v:.{d}f}"


def render_index(metas: list[dict]) -> str:
    """metas: [{dataset, task, n, d, fidelity, ratio, n_core, coverage, regime, file}]"""
    cards = []
    for m in sorted(metas, key=lambda x: (x.get("regime") == "standard" and 0 or 1)):
        rp = "ok" if (m.get("ratio") and m["ratio"] >= 0.98) else "warn"
        reg = "small-n" if m.get("regime") != "standard" else "standard"
        cards.append(f"""
      <a class="ix" href="{_esc(m['file'])}">
        <div class="ixh"><b>{_esc(m['dataset'])}</b>
          <span class="pill {'warn' if reg!='standard' else 'ok'}">{reg}</span></div>
        <div class="ixs">{_esc(m['task'])} · n={m['n']:,} · d={m['d']}</div>
        <div class="ixstats">
          <span><b>{m.get('fidelity',0):.3f}</b> fid</span>
          <span><b>{m.get('ratio','—')}</b> vs ceiling</span>
          <span><b>{m.get('n_core',0)}</b> core comp</span>
          <span><b>{(m.get('coverage') or 0):.2f}</b> coverage</span>
        </div>
      </a>""")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>XPLAIN-x1 — dashboards</title>
<style>{_CSS}
.ixgrid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr));
  gap:14px; margin-top:18px; }}
a.ix {{ display:block; text-decoration:none; color:inherit; background:var(--surface);
  border:1px solid var(--ring); border-radius:10px; padding:15px 16px; }}
a.ix:hover {{ border-color:var(--accent); }}
.ixh {{ display:flex; gap:9px; align-items:center; font-size:15px; }}
.ixs {{ font-size:12px; color:var(--muted); margin:3px 0 10px; }}
.ixstats {{ display:flex; gap:14px; flex-wrap:wrap; font-size:12px; color:var(--ink2);
  font-variant-numeric:tabular-nums; }}
.ixstats b {{ color:var(--ink); font-size:14px; }}
</style></head>
<body><div class="wrap">
  <div class="top"><div><h1>XPLAIN-x1 — audit dashboards</h1>
    <div class="sub">Owner review (M6): inspect each model's concept DAG for node
    nameability, and the three certified layers.</div></div></div>
  <div class="ixgrid">{"".join(cards)}</div>
  <div class="foot">One self-contained HTML per processed dataset. Certified claims
    are the Layer F function components; the unit DAG is inspectable and the
    periphery is labelled, never certified.</div>
</div></body></html>"""
