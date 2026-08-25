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
function abbr(s,m){ m=m||15; return s.length>m ? s.slice(0,m-1)+"…" : s; }

const HIER = D.hier, UNITS = D.units;
let MODE="hier", SEL=null, EXP={}, SHOW_PERI=true, SHOW_IN=true;
let VZ={k:1,x:0,y:0}, drag=null, W=980, H=560, POS={};
const svg=$("dag"), g=$("dagg");

// each component starts collapsed; the LR skeleton (features -> boxes -> output)
// is the default Layer F view.
function moveTip(ev){ tip.style.left=Math.min(ev.clientX+14,innerWidth-tip.offsetWidth-8)+"px";
  tip.style.top=Math.min(ev.clientY+14,innerHeight-tip.offsetHeight-8)+"px"; }
function showTip(html,ev){ tip.innerHTML=html; tip.style.opacity=1; moveTip(ev); }
function hideTip(){ tip.style.opacity=0; }

// segmented horizontal contribution bar. segs = [{share, label}] (shares need
// not be normalised); drawn left->right into parent group at (x,y,width).
const SEG=["#2a78d6","#1baf7a","#eb6834","#eda100","#8a63d2","#e0567f","#159e8e",
           "#6b7bd6","#c06a2b","#3f9a3f","#a05fd0","#5a9fd6"];
function drawBar(parent,x,y,w,h,segs){
  const tot=segs.reduce((s,z)=>s+(z.share||0),0)||1;
  el("rect",{x,y,width:w,height:h,rx:2,fill:"var(--grid)","fill-opacity":0.5},parent);
  let cx=x;
  segs.forEach((z,i)=>{ const sw=w*(z.share||0)/tot; if(sw<=0)return;
    const r=el("rect",{x:cx,y,width:Math.max(0.5,sw),height:h,
      fill:z.color||SEG[i%SEG.length]},parent);
    r.style.cursor="default";
    r.addEventListener("mousemove",ev=>{ev.stopPropagation();
      showTip(`<h4>${z.label}</h4><div class="m">${(100*(z.share||0)/tot).toFixed(1)}% of this bar</div>`,ev);});
    r.addEventListener("mouseleave",hideTip);
    cx+=sw; });
}

// =================================================================== LAYER F
const FEATCX=96, COMPX=360, COMPW=232, ROWH=24, HEADH=42, BARZONE=17, BOXGAP=20, PAD=10;
const OUTCX=COMPX+COMPW+150;
function rowCount(c){ const base=c.members.length||c.feats_direct.length;
  return base + ((c.extra_members||0)>0?1:0); }
function barH(c){ return c.measured ? BARZONE : 0; }   // contribution bar zone
function compHeight(c){ return HEADH + barH(c) +
  (EXP[c.id] ? Math.max(1,rowCount(c))*ROWH + PAD : 0); }
function layoutHier(){
  const comps=HIER.components;
  let y=40; const cpos={};
  comps.forEach(c=>{ const h=compHeight(c); cpos[c.id]={y, h, top:y}; y+=h+BOXGAP; });
  const compsBottom=y;
  const feats=HIER.features;
  const fh=Math.max(compsBottom, 40+feats.length*30);
  H=Math.max(fh, 200)+30; W=OUTCX+150;
  POS={cpos, feat:{}, };
  const fy0=(H-(feats.length-1)*30)/2;
  feats.forEach((f,i)=>POS.feat[f.id]={x:FEATCX,y:fy0+i*30});
  POS.out={x:OUTCX,y:H/2};
  svg.setAttribute("viewBox",`0 0 ${W} ${H}`);
}
function memberRowY(c,i){ return POS.cpos[c.id].top+HEADH+barH(c)+i*ROWH+ROWH/2; }
function tagStroke(t){ return t==="CORE"?"var(--core)":"var(--peri)"; }
function tagFill(t){ return t==="CORE"?"var(--corewash)":"var(--periwash)"; }

function edgePath(x1,y1,x2,y2){ const mx=(x1+x2)/2;
  return `M${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`; }
function drawEdge(x1,y1,x2,y2,share,dim){
  const w=0.7+5*Math.min(1,(share||0)*3);
  el("path",{d:edgePath(x1,y1,x2,y2),fill:"none",stroke:"var(--edge)",
    "stroke-width":w.toFixed(2),"stroke-opacity":dim?0.12:0.5},g); }

function drawHier(){
  g.textContent="";
  const focus = SEL;
  const featSel = new Set();  // features touched by the selected node (for focus dim)
  if(focus){
    const c=HIER.components.find(c=>c.id===focus);
    if(c){ (c.members.length?c.members.flatMap(m=>m.feats):c.support_names).forEach(f=>featSel.add(f)); }
  }
  // ---- edges (drawn first, behind) ----
  HIER.components.forEach(c=>{
    const cx1=COMPX, cy=POS.cpos[c.id].top+HEADH/2;
    const dimC = focus && focus!==c.id;
    // component -> output
    drawEdge(COMPX+COMPW, POS.cpos[c.id].top+compHeight(c)/2, OUTCX-46, POS.out.y, c.share, dimC);
    if(!EXP[c.id]){
      // collapsed: feature -> component (the support), summary width
      c.support_names.forEach(f=>{ const fp=POS.feat["F::"+f]; if(!fp)return;
        drawEdge(fp.x+56, fp.y, cx1, cy, c.share, dimC); });
    } else {
      // expanded: feature -> each member row (real flow)
      const rows = c.members.length?c.members:c.feats_direct.map(f=>({feats:[f],direct:true}));
      rows.forEach((m,i)=>{ const ry=memberRowY(c,i);
        (m.feats||[]).forEach(f=>{ const fp=POS.feat["F::"+f]; if(!fp)return;
          const dim = dimC || (focus && !featSel.has(f) && focus!==c.id);
          drawEdge(fp.x+56, fp.y, cx1, ry, (m.coverage||0.04), dim); }); });
    }
  });
  // ---- feature nodes (left lane) ----
  HIER.features.forEach(f=>{ const p=POS.feat[f.id]; if(!p)return;
    const dim=focus && !featSel.has(f.label) && !HIER.components.some(c=>c.id===focus && c.support_names.includes(f.label));
    const grp=el("g",{transform:`translate(${p.x},${p.y})`,style:`cursor:default;opacity:${dim?0.35:1}`},g);
    const wpx=Math.max(64, 8+7*abbr(f.label,18).length);
    el("rect",{x:-wpx/2,y:-11,width:wpx,height:22,rx:11,fill:"var(--wash)",
      stroke:"var(--grid)","stroke-width":1},grp);
    const t=el("text",{x:0,y:4,"text-anchor":"middle",
      style:"font:11px system-ui;fill:var(--ink);pointer-events:none"},grp);
    t.textContent=abbr(f.label,18);
    grp.addEventListener("mousemove",ev=>showTip(`<h4>${f.label}</h4><div class="m">input feature</div>`,ev));
    grp.addEventListener("mouseleave",hideTip);
  });
  // ---- component boxes (middle lane) ----
  HIER.components.forEach(c=>{
    const P=POS.cpos[c.id]; const h=compHeight(c);
    const dimC=focus && focus!==c.id && !(HIER.components.find(x=>x.id===focus));
    const grp=el("g",{transform:`translate(${COMPX},${P.top})`,
      style:`cursor:pointer;opacity:${(focus&&focus!==c.id)?0.5:1}`},g);
    el("rect",{x:0,y:0,width:COMPW,height:h,rx:9,fill:tagFill(c.tag),
      stroke:tagStroke(c.tag),"stroke-width":c.id===SEL?2.4:1.5},grp);
    // header
    const title=el("text",{x:12,y:19,style:"font:12.5px system-ui;font-weight:650;fill:var(--ink);pointer-events:none"},grp);
    title.textContent=abbr(c.support_names.join(" × "),24);
    const sub=el("text",{x:12,y:33,style:"font:10px system-ui;fill:var(--muted);pointer-events:none"},grp);
    sub.textContent=`${(100*c.share).toFixed(1)}% · Π ${fmt(c.Pi)} · ${c.n_members} carrier${c.n_members===1?"":"s"}`;
    // expand badge
    if(c.n_members||c.feats_direct.length){
      const bx=COMPW-16;
      el("circle",{cx:bx,cy:14,r:8,fill:"var(--surface)",stroke:tagStroke(c.tag),"stroke-width":1.2},grp);
      const bt=el("text",{x:bx,y:18,"text-anchor":"middle",
        style:"font:13px system-ui;font-weight:700;fill:var(--ink2);pointer-events:none"},grp);
      bt.textContent=EXP[c.id]?"–":"+";
    }
    // measured-contribution bar (how the component's mass distributes across
    // its member units, from per-unit ablation) — always visible on the box.
    if(c.measured){
      const segs=c.members.filter(m=>m.contrib>0)
        .map(m=>({share:m.contrib,label:m.uid+" — Δshare "+fmt(m.contrib,3)}));
      drawBar(grp,12,HEADH-3,COMPW-24,10,segs.length?segs:[{share:1,label:"no single dominant carrier"}]);
    }
    // member rows inside the box
    if(EXP[c.id]){
      const y0=HEADH+barH(c);
      const rows=c.members.length?c.members:c.feats_direct.map(f=>({uid:f,direct:true,feats:[f]}));
      rows.forEach((m,i)=>{
        const ry=y0+i*ROWH+ROWH/2;
        const rg=el("g",{transform:`translate(0,${ry})`,style:"cursor:pointer"},grp);
        el("rect",{x:8,y:-ROWH/2+2,width:COMPW-16,height:ROWH-4,rx:5,
          fill:m.direct?"var(--wash)":tagFill(m.tag),stroke:m.direct?"var(--grid)":tagStroke(m.tag),
          "stroke-width":1,"stroke-opacity":0.7},rg);
        const mt=el("text",{x:14,y:3.5,style:"font:10.5px system-ui;fill:var(--ink);pointer-events:none"},rg);
        // label with RESOLVED features (L2 units otherwise read as parent ids)
        mt.textContent = m.direct ? ("feature · "+abbr(m.uid,16))
          : `${m.uid} · ${(m.feats||[]).map(s=>abbr(s,7)).join("·")}`;
        if(!m.direct){
          // "reads directly" tag when the unit also literally contains the
          // component's support (vs carrying it through a collinear sibling)
          const pctX = m.structural ? COMPW-34 : COMPW-16;
          if(m.structural){
            const tg=el("text",{x:COMPW-16,y:3.5,"text-anchor":"end",
              style:"font:8.5px system-ui;fill:var(--core);pointer-events:none"},rg);
            tg.textContent="⌾ direct";
          }
          const mm=el("text",{x:pctX,y:3.5,"text-anchor":"end",
            style:"font:9.5px system-ui;fill:var(--muted);pointer-events:none"},rg);
          mm.textContent = (c.measured && c.contrib_sum>0)
            ? (fmt(100*m.contrib/c.contrib_sum,0)+"%") : ("μ "+fmt(m.mu,2));
          rg.addEventListener("mousemove",ev=>showTip(
            `<h4>${m.uid} · layer ${m.layer}</h4>`+
            `<div class="m">${m.tag} · ${m.form||"?"} · μ ${fmt(m.mu)}</div>`+
            `<div class="m">reads: ${m.feats.join(", ")}`+
            (m.structural?" (contains this term's features directly)":" (carries it via a collinear feature)")+`</div>`+
            (c.measured?`<div class="m">measured Δshare when ablated: ${fmt(m.contrib,3)}</div>`:""),ev));
          rg.addEventListener("mouseleave",hideTip);
          rg.addEventListener("click",ev=>{ev.stopPropagation(); selectMember(c,m);});
        }
      });
      if((c.extra_members||0)>0){
        const ry=y0+rows.length*ROWH+ROWH/2;
        const et=el("text",{x:COMPW/2,y:ry+3.5,"text-anchor":"middle",
          style:"font:9.5px system-ui;font-style:italic;fill:var(--muted);pointer-events:none"},grp);
        et.textContent=`+${c.extra_members} more carrier unit(s)`;
      }
    }
    grp.addEventListener("mousemove",ev=>{ if(ev.target.tagName!=="rect"||ev.offsetY==null){} showTip(
      `<h4>${c.support_names.join(" × ")}</h4>`+
      `<div class="m">${c.tag} function component · ${(100*c.share).toFixed(1)}% of variance</div>`+
      `<div class="m">Π ${fmt(c.Pi)} · π ${fmt(c.pi)} · ${c.n_members} member unit(s)`+
      `${(c.n_members||c.feats_direct.length)?" · double-click to open":""}</div>`,ev); });
    grp.addEventListener("mouseleave",hideTip);
    grp.addEventListener("click",ev=>{ev.stopPropagation(); select(c.id);});
    grp.addEventListener("dblclick",ev=>{ev.stopPropagation();
      if(c.n_members||c.feats_direct.length){ EXP[c.id]=!EXP[c.id]; layoutHier(); drawHier(); }});
  });
  // ---- output node (right) ----
  const o=HIER.output, op=POS.out;
  const og=el("g",{transform:`translate(${op.x},${op.y})`},g);
  el("rect",{x:-46,y:-16,width:92,height:32,rx:8,fill:"var(--surface)",
    stroke:"var(--accent)","stroke-width":1.6},og);
  const ot=el("text",{x:0,y:4,"text-anchor":"middle",
    style:"font:11px system-ui;fill:var(--ink);pointer-events:none"},og);
  ot.textContent=abbr(o.label,12);
  og.addEventListener("mousemove",ev=>showTip(`<h4>${o.label}</h4><div class="m">${o.sub}</div>`,ev));
  og.addEventListener("mouseleave",hideTip);
  applyZoom();
}

// =================================================================== UNITS
const UW=158, UH=44, UVGAP=20;   // unit box size + vertical gap
function drawUnits(){
  g.textContent="";
  const N=UNITS.nodes, E=UNITS.edges;
  const outCol=UNITS.outCol;
  const colf=n=>n.kind==="input"?0:n.kind==="output"?outCol:(n.layer||1);
  const vis=N.filter(n=>{ if(n.kind==="input"&&!SHOW_IN)return false;
    if(n.kind==="unit"&&n.tag!=="CORE"&&!SHOW_PERI)return false; return true; });
  const cols={}; vis.forEach(n=>{const c=colf(n);(cols[c]=cols[c]||[]).push(n);});
  const keys=Object.keys(cols).map(Number).sort((a,b)=>a-b);
  const COLW=232; let maxR=1; keys.forEach(k=>maxR=Math.max(maxR,cols[k].length));
  H=60+maxR*(UH+UVGAP); W=(Math.max(...keys,0)+1)*COLW;
  const pos={};
  keys.forEach(k=>{const arr=cols[k],x=k*COLW+COLW/2,span=(arr.length-1)*(UH+UVGAP),y0=(H-span)/2;
    arr.forEach((n,i)=>pos[n.id]={x,y:y0+i*(UH+UVGAP)});});
  svg.setAttribute("viewBox",`0 0 ${W} ${H}`);
  const visIds=new Set(vis.map(n=>n.id));
  const halfW=n=>n.kind==="unit"?UW/2:(n.kind==="output"?46:56);
  E.forEach(e=>{ if(!visIds.has(e.src)||!visIds.has(e.dst))return;
    const p=pos[e.src],q=pos[e.dst]; if(!p||!q)return;
    drawEdge(p.x+halfW(nById(e.src)),p.y,q.x-halfW(nById(e.dst)),q.y,e.share,
      SEL&&SEL!==e.src&&SEL!==e.dst); });
  function nById(id){ return N.find(n=>n.id===id); }
  vis.forEach(n=>{ const p=pos[n.id]; if(!p)return;
    const grp=el("g",{transform:`translate(${p.x},${p.y})`,style:"cursor:pointer"},g);
    if(n.kind==="unit"){
      const fill=n.tag==="CORE"?"var(--corewash)":"var(--periwash)";
      const stroke=n.tag==="CORE"?"var(--core)":"var(--peri)";
      el("rect",{x:-UW/2,y:-UH/2,width:UW,height:UH,rx:7,fill,stroke,
        "stroke-width":n.id===SEL?2.6:1.4},grp);
      // resolved-feature label (L2 units read as features, not parent ids)
      const feats=(n.feats&&n.feats.length)?n.feats:(n.support_names||[]);
      const lab=el("text",{x:0,y:-UH/2+15,"text-anchor":"middle",
        style:"font:10.5px system-ui;fill:var(--ink);pointer-events:none"},grp);
      lab.textContent=abbr(feats.map(s=>abbr(s,10)).join(", "),24);
      const sub=el("text",{x:-UW/2+8,y:-UH/2+28,
        style:"font:9px system-ui;fill:var(--muted);pointer-events:none"},grp);
      sub.textContent=`${n.id} · L${n.layer} · μ ${fmt(n.mu,2)}`;
      // per-input contribution bar (masked-weight contribution of each parent)
      if(n.in_contrib&&n.in_contrib.length)
        drawBar(grp,-UW/2+8,UH/2-13,UW-16,8,
          n.in_contrib.map(z=>({share:z.share,label:z.p+" — "+fmt(100*z.share,0)+"%"})));
    } else {
      const fill=n.kind==="output"?"var(--surface)":"var(--wash)";
      const stroke=n.kind==="output"?"var(--accent)":"var(--grid)";
      el("rect",{x:-halfW(n),y:-14,width:2*halfW(n),height:28,rx:8,fill,stroke,"stroke-width":1.3},grp);
      const t=el("text",{"text-anchor":"middle",y:4,style:"font:11px system-ui;fill:var(--ink);pointer-events:none"},grp);
      t.textContent=abbr(n.label,16);
    }
    grp.addEventListener("mousemove",ev=>showTip(unitTip(n),ev));
    grp.addEventListener("mouseleave",hideTip);
    grp.addEventListener("click",ev=>{ev.stopPropagation(); select(n.id);});
  });
  applyZoom();
}
function unitTip(n){ if(n.kind==="unit") return `<h4>${(n.feats||n.support_names||[]).join(", ")||n.id}</h4>`+
  `<div class="m">${n.id} · ${n.tag} · layer ${n.layer} · ${n.form||"?"}</div>`+
  `<div class="m">μ ${fmt(n.mu)} · Π ${fmt(n.Pi)} · π ${fmt(n.pi)}</div>`;
  return `<h4>${n.label}</h4><div class="m">${n.kind}</div>`; }

function draw(){ if(MODE==="hier"){layoutHier(); drawHier();} else drawUnits(); }

// =================================================================== detail
function select(id){ SEL=(SEL===id)?null:id; draw(); renderDetail(); }
function selectMember(c,m){ SEL=m.uid; draw();
  const box=$("detail");
  box.innerHTML=`<h3>${m.uid}</h3><div class="did">carrier of “${c.support_names.join(" × ")}”</div>`+
    `<div style="margin-top:7px"><span class="pill ${m.tag==="CORE"?"core":"peri"}">${m.tag}</span>`+
    (m.structural?` <span class="pill core">reads it directly</span>`:` <span class="pill peri">via collinear feature</span>`)+`</div>`+
    `<dl><dt>form</dt><dd>${m.form||"—"}</dd><dt>layer</dt><dd>${m.layer}</dd>`+
    `<dt>μ mono</dt><dd>${fmt(m.mu)}</dd><dt>reads</dt><dd>${m.feats.join(", ")}</dd>`+
    (c.measured?`<dt>Δshare</dt><dd>${fmt(m.contrib,3)} when ablated (${c.contrib_sum>0?fmt(100*m.contrib/c.contrib_sum,0):0}% of measured mass)</dd>`:"")+
    `</dl>`+
    `<div class="chips"><span class="chip">${m.structural
      ? "carries this component and literally reads its features"
      : "carries this component through a collinear feature ("+m.feats.join(", ")+")"}</span></div>`;
}
function renderDetail(){
  const box=$("detail");
  if(MODE==="units"){ const n=UNITS.nodes.find(n=>n.id===SEL);
    if(!n){ box.innerHTML=emptyHelp(); return; }
    if(n.kind==="unit"){ box.innerHTML=`<h3>${(n.support_names||[]).join("·")||n.id}</h3>`+
      `<div class="did">${n.id}</div><div style="margin-top:7px"><span class="pill ${n.tag==="CORE"?"core":"peri"}">${n.tag}</span></div>`+
      `<dl><dt>support</dt><dd>${(n.support_names||[]).join(", ")||"—"}</dd><dt>form</dt><dd>${n.form||"—"}</dd>`+
      `<dt>μ</dt><dd>${fmt(n.mu)}</dd><dt>Π</dt><dd>${fmt(n.Pi)}</dd><dt>π</dt><dd>${fmt(n.pi)}</dd><dt>Δ</dt><dd>${fmt(n.delta,3)}</dd></dl>`+
      ((n.reasons&&n.reasons.length)?`<div class="chips">`+n.reasons.map(r=>`<span class="chip r">${r}</span>`).join("")+`</div>`
        :`<div class="chips"><span class="chip">certified: stable, frequent, real</span></div>`);
      return; }
    box.innerHTML=`<h3>${n.label}</h3><div class="did">${n.kind}</div>`; return;
  }
  const c=HIER.components.find(c=>c.id===SEL);
  if(!c){ box.innerHTML=emptyHelp(); return; }
  box.innerHTML=`<h3>${c.support_names.join(" × ")}</h3><div class="did">certified function component</div>`+
    `<div style="margin-top:7px"><span class="pill ${c.tag==="CORE"?"core":"peri"}">${c.tag}</span></div>`+
    `<dl><dt>share</dt><dd>${(100*c.share).toFixed(1)}% of function variance</dd>`+
    `<dt>Π stab</dt><dd>${fmt(c.Pi)}</dd><dt>π cpss</dt><dd>${fmt(c.pi)}</dd>`+
    `<dt>carriers</dt><dd>${c.n_members} unit(s) carry it (measured)</dd></dl>`+
    `<div class="chips"><span class="chip">${(c.n_members||c.feats_direct.length)?"double-click the box to open its carrier units":"leaf"}</span></div>`;
}
function emptyHelp(){
  if(MODE==="hier") return `<div class="empty"><b>Layer F flow (left → right).</b>
    Input features on the left feed the certified <b>function components</b> (the middle
    boxes — additive terms of the model's function, NOT classes), which sum into the
    output on the right. The <b>coloured bar under each box header</b> is the
    <b>measured</b> split of that term's mass across its carrier units (from ablating each
    unit). <b>Double-click a box</b> to open it: its <b>carrier units</b> appear inside —
    identified by measurement, so a unit that carries the term through a collinear
    feature is included (a <span style="color:var(--core)">⌾ direct</span> tag marks the
    ones that also literally read the term's features). This is why the carrier here can
    match the CORE unit in the Model-units view. <b>Green</b> = CORE; <b>grey</b> =
    periphery. Drag to pan, scroll to zoom.</div>`;
  return `<div class="empty"><b>Model units.</b> The raw network graph: features → hidden
    units → output. Each unit is a <b>box labelled by its resolved input features</b>
    (layer-2 units too), and the <b>coloured bar at its base</b> shows each input's
    measured contribution (masked-weight share). <b>Green</b> = CORE, <b>grey</b> =
    periphery. Hover a bar segment for the feature and %. Toggle inputs/periphery above.</div>`;
}

// =================================================================== pan/zoom
function applyZoom(){ g.setAttribute("transform",`translate(${VZ.x},${VZ.y}) scale(${VZ.k})`); }
svg.addEventListener("mousedown",e=>{drag={x:e.clientX,y:e.clientY,ox:VZ.x,oy:VZ.y};});
addEventListener("mousemove",e=>{ if(!drag)return;
  VZ.x=drag.ox+(e.clientX-drag.x); VZ.y=drag.oy+(e.clientY-drag.y); applyZoom(); });
addEventListener("mouseup",()=>drag=null);
svg.addEventListener("click",()=>{ if(SEL){SEL=null; draw(); renderDetail();} });
svg.addEventListener("wheel",e=>{ e.preventDefault();
  const r=svg.getBoundingClientRect(), sx=(e.clientX-r.left)/r.width*W, sy=(e.clientY-r.top)/r.height*H;
  const f=e.deltaY<0?1.12:1/1.12, nk=Math.max(0.25,Math.min(3.2,VZ.k*f));
  VZ.x=sx-(sx-VZ.x)*(nk/VZ.k); VZ.y=sy-(sy-VZ.y)*(nk/VZ.k); VZ.k=nk; applyZoom(); },{passive:false});
function fit(){ VZ={k:1,x:0,y:0};
  const r=svg.getBoundingClientRect();   // fit whole graph into view
  const s=Math.min(1, (r.width||900)/W*0.98, (r.height||520)/H*0.98);
  VZ.k=Math.max(0.25,s); applyZoom(); }

// =================================================================== controls
function setMode(m){ MODE=m; SEL=null;
  $("m_hier").classList.toggle("on",m==="hier"); $("m_units").classList.toggle("on",m==="units");
  $("unitctrls").style.display=m==="units"?"":"none"; $("hierctrls").style.display=m==="hier"?"":"none";
  draw(); renderDetail(); fit(); }
$("m_hier").addEventListener("click",()=>setMode("hier"));
$("m_units").addEventListener("click",()=>setMode("units"));
$("b_fit").addEventListener("click",()=>{ EXP={}; draw(); fit(); });
$("b_expand").addEventListener("click",()=>{ HIER.components.forEach(c=>EXP[c.id]=!!(c.n_members||c.feats_direct.length)); draw(); });
$("t_peri").addEventListener("change",e=>{SHOW_PERI=e.target.checked; draw();});
$("t_in").addEventListener("change",e=>{SHOW_IN=e.target.checked; draw();});
setMode("hier");
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


def _protected_block(par: dict) -> str:
    if not par.get("declared"):
        return ""
    relies = par.get("relies_on_protected")
    pill = "warn" if relies else "ok"
    verdict_word = "RELIES — review" if relies else "NON-RELIANCE"
    rows = []
    for r in par.get("rows", []):
        stat = r["status"]
        cls = "core" if r["certified_component"] else ("peri" if r["appears_in_decomposition"] else "ok")
        rows.append(
            f'<tr><td>{_esc(r["attribute"])}</td>'
            f'<td>{"yes" if r["appears_in_decomposition"] else "no"}</td>'
            f'<td>{"YES" if r["certified_component"] else "no"}</td>'
            f'<td class="num">{r["max_share"]:.4f}</td>'
            f'<td><span class="pill {cls}">{_esc(stat)}</span></td></tr>')
    px = par.get("proxy", {})
    proxy_html = ""
    if px.get("drivers"):
        ppill = "ok" if px.get("flag") == "none" else "warn"
        prows = []
        for r in sorted(px["drivers"], key=lambda r: -r["max_rho"]):
            fcls = {"strong": "warn", "notable": "warn"}.get(r["flag"], "ok")
            prows.append(
                f'<tr><td>{_esc(r["driver"])}</td>'
                f'<td>{_esc(r["nearest_protected"])}</td>'
                f'<td class="num">{r["max_rho"]:.3f}</td>'
                f'<td><span class="pill {fcls}">{_esc(r["flag"])}</span></td></tr>')
        proxy_html = f"""
    <h2 style="margin-top:16px;font-size:13px">Proxy screen (indirect reliance)
      <span class="pill {ppill}">max |ρ| {px.get('max_rho',0):.3f} · {_esc(px.get('flag',''))}</span></h2>
    <div class="note">{_esc(px.get('verdict',''))}</div>
    <table class="t"><thead><tr><th>certified driver</th><th>nearest protected</th>
      <th class="num">max |ρ|</th><th>flag</th></tr></thead>
      <tbody>{''.join(prows)}</tbody></table>
    <div class="note" style="margin-top:6px">{_esc(px.get('method',''))}</div>"""
    return f"""
  <div class="card">
    <h2>Protected-attribute non-reliance <span class="pill {pill}">{verdict_word}</span>
      <span class="pill peri">fair lending · SR 11-7 / EU AI Act Art 10</span></h2>
    <div class="note">{_esc(par.get('verdict',''))}</div>
    <table class="t"><thead><tr><th>protected attribute</th><th>in decomposition?</th>
      <th>certified?</th><th class="num">max share</th><th>status</th></tr></thead>
      <tbody>{''.join(rows)}</tbody></table>
    <div class="note" style="margin-top:8px">{_esc(par.get('basis',''))}</div>
    {proxy_html}
  </div>"""


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
    """The raw model graph (inputs -> units -> outputs) for nameability review.

    Each unit node is enriched with its resolved input features (`feats`) and
    the per-parent contribution shares (`in_contrib`, the incoming masked-weight
    contributions normalised to sum 1) so the viewer can draw feature-named
    boxes with a bottom contribution bar."""
    unit_nodes = [n for n in dag["nodes"] if n["kind"] == "unit"]
    out_col = 2 + max([1] + [n.get("layer", 1) for n in unit_nodes])
    trans = _transitive_feature_supports(dag)
    # incoming edges per unit (parent -> unit), with contribution shares
    incoming: dict = {}
    for e in dag["edges"]:
        incoming.setdefault(e["dst"], []).append((e["src"], e.get("share", 0.0)))
    id2name = {n["id"]: n.get("label", n["id"]) for n in dag["nodes"]}
    for n in dag["nodes"]:
        if n["kind"] != "unit":
            continue
        n["feats"] = sorted(trans.get(n["id"], []))
        ins = incoming.get(n["id"], [])
        tot = sum(s for _, s in ins) or 1.0
        n["in_contrib"] = [
            {"p": id2name.get(src, src), "share": round(s / tot, 4)}
            for src, s in sorted(ins, key=lambda x: -x[1])]
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
                n_classes: int, member_contrib: dict | None = None) -> dict:
    """The Layer F DAG as a left-to-right flow with expandable component boxes.

    Three lanes: INPUT FEATURES (left) -> CERTIFIED COMPONENT boxes (middle) ->
    OUTPUT (right).  A component is a purified additive term of the learned
    function (NOT a class).  Its *member units* are the model units that
    physically carry that term's mass, identified by MEASUREMENT: ablate a unit
    and the component's purified share drops.  This sees through feature
    collinearity (a unit reading `workingday` can carry the `hour × weekday`
    term), unlike a structural containment test.  Units that ALSO literally read
    the component's features are tagged `structural`.  When no ablation
    measurement is available the membership falls back to structural containment.
    """
    fdec = cert["function_decomposition"]
    ident = cert["identification"]
    comps = sorted(fdec.get("components", []), key=lambda c: -_comp_share(c))
    core = [c for c in comps if c.get("label") == "CORE"]
    peri = [c for c in comps if c.get("label") != "CORE"][:6]
    chosen = core + peri

    unit_nodes = {n["id"]: n for n in dag["nodes"] if n["kind"] == "unit"}
    trans = _transitive_feature_supports(dag)

    feat_names: list = []       # left lane, in first-seen order
    def feat_id(name: str) -> str:
        if name not in feat_names:
            feat_names.append(name)
        return "F::" + name

    mc = member_contrib or {}
    components = []
    for i, c in enumerate(chosen):
        supp = c.get("support_names", c.get("names", []))
        sset = set(supp)
        for f in supp:
            feat_id(f)
        # structural carriers: units whose input features literally contain the
        # component's support (a guess from wiring, blind to feature collinearity)
        structural = {uid for uid, tf in trans.items() if sset and sset <= tf}
        # measured carriers: ablating the unit drops this component's purified
        # share (the ground truth — sees through workingday≈weekday collinearity).
        ckey = ",".join(map(str, sorted(c.get("support", []))))
        cdrop = mc.get(ckey, {})
        measured = bool(cdrop)
        # membership = MEASURED carriers when we have them (so the physical carrier
        # of a certified term shows up even if it reads a collinear sibling
        # feature); fall back to structural containment otherwise.
        if measured:
            member_uids = sorted(cdrop, key=lambda uid: -cdrop[uid])
        else:
            member_uids = sorted(structural,
                                 key=lambda uid: -(unit_nodes[uid].get("coverage") or 0))
        MAX_ROWS = 12
        shown, extra = member_uids[:MAX_ROWS], len(member_uids) - MAX_ROWS
        members = []
        for uid in shown:
            u = unit_nodes[uid]
            tfeats = sorted(trans[uid])
            for f in tfeats:
                feat_id(f)
            members.append({
                "uid": uid, "mu": u.get("mu"), "tag": u.get("tag", "PERIPHERY"),
                "form": u.get("form"), "layer": u.get("layer", 1),
                "immediate": u.get("support_names") or [],
                "feats": tfeats,                 # transitive input features
                "coverage": u.get("coverage"),
                "contrib": round(float(cdrop.get(uid, 0.0)), 4),
                # tag units that ALSO literally read the component's features
                "structural": uid in structural})
        contrib_sum = round(sum(m["contrib"] for m in members), 4)
        components.append({
            "id": f"C{i}", "kind": "component",
            "support_names": supp, "share": _comp_share(c),
            "Pi": c.get("Pi", 0), "pi": c.get("pi", 0),
            "tag": c.get("label", "PERIPHERY"),
            "n_members": len(member_uids), "members": members,
            "extra_members": max(0, extra), "measured": measured,
            "contrib_sum": contrib_sum,
            # a component with no localised carrier wires its features directly
            "feats_direct": supp if not member_uids else []})

    root_label = (f"{n_classes}-class output" if is_multiclass
                  else ("class decision" if ident["task"] == "classification"
                        else "output"))
    features = [{"id": "F::" + n, "kind": "feature", "label": n}
                for n in feat_names]
    return {
        "features": features,
        "components": components,
        "output": {"id": "OUT", "kind": "output", "label": root_label,
                   "sub": f"{ident['task']} · {ident['dataset']}"},
    }


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
        "hier": _build_hier(cert, data["dag"], is_multiclass, n_classes,
                            data.get("member_contrib")),
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
        <button id="m_hier" class="on">Layer F flow</button>
        <button id="m_units">Model units</button>
      </div>
      <div class="ctrls">
        <span id="hierctrls" class="ctrls" style="border:none;padding:0">
          <button id="b_expand">open all boxes</button>
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
          <span><i style="background:var(--surface);border:1.5px solid var(--accent)"></i>output (right)</span>
          <span><i style="background:var(--wash);border:1px solid var(--grid)"></i>input feature (left)</span>
          <span>features → components → output · bar = measured contribution split · double-click a box to open it</span>
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

  {_protected_block(cert.get('protected_attribute_reliance', {}))}

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
