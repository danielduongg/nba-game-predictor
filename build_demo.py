import json
M=json.load(open("web_model.json"))
HTML=r'''<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>NBA Win Probability — live demo</title>
<style>
 :root{--bg:#0b0f17;--panel:#121826;--panel2:#0e1420;--line:#1f2a3c;--ink:#e8eef9;--mut:#8aa0bf;
   --home:#fb923c;--away:#38bdf8;--accent:#fb923c;--warn:#ffb454;}
 *{box-sizing:border-box}
 body{margin:0;background:radial-gradient(1100px 650px at 75% -10%,#33210f 0%,var(--bg) 55%);color:var(--ink);
   font:16px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Inter,sans-serif;-webkit-font-smoothing:antialiased}
 .wrap{max-width:900px;margin:0 auto;padding:36px 20px 80px}
 .eyebrow{letter-spacing:.16em;text-transform:uppercase;font-size:12px;color:var(--accent);font-weight:700}
 h1{font-size:29px;margin:.25em 0 .15em;line-height:1.15}
 .sub{color:var(--mut);max-width:66ch}
 .presets{display:flex;gap:8px;flex-wrap:wrap;margin-top:18px}
 .pz{font-size:12.5px;border:1px solid var(--line);background:#0c1422;color:var(--mut);padding:6px 11px;border-radius:999px;cursor:pointer}
 .pz:hover{border-color:var(--accent);color:var(--ink)}
 .result{background:linear-gradient(180deg,var(--panel),var(--panel2));border:1px solid var(--line);border-radius:16px;padding:22px;margin-top:16px;box-shadow:0 20px 50px -30px #000}
 .split{display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:10px}
 .side{font-weight:800}
 .side .nm{font-size:13px;color:var(--mut);font-weight:600;letter-spacing:.03em;text-transform:uppercase}
 .side .p{font-size:40px;font-variant-numeric:tabular-nums;line-height:1}
 .pbar{height:18px;border-radius:999px;overflow:hidden;display:flex;border:1px solid var(--line)}
 .pbar .h{background:var(--home)}.pbar .a{background:var(--away)}
 .teams{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}
 @media(max-width:720px){.teams{grid-template-columns:1fr}}
 .card{background:linear-gradient(180deg,var(--panel),var(--panel2));border:1px solid var(--line);border-radius:16px;padding:18px 20px}
 .card h3{margin:0 0 10px;font-size:15px}
 .hh{color:var(--home)}.aa{color:var(--away)}
 .row{margin:11px 0}
 .row .lab{display:flex;justify-content:space-between;font-size:13px;color:var(--mut);margin-bottom:4px}
 .row .lab b{color:var(--ink);font-variant-numeric:tabular-nums;font-size:14.5px}
 input[type=range]{width:100%;height:5px}
 .adv{margin-top:14px;border-top:1px dashed var(--line);padding-top:8px}
 .adv summary{cursor:pointer;color:var(--mut);font-size:13px}
 h2{font-size:14px;margin:18px 0 8px;color:var(--mut);font-weight:600;letter-spacing:.02em;text-transform:uppercase}
 .imp{display:flex;flex-direction:column;gap:7px}
 .ib{display:grid;grid-template-columns:120px 1fr;align-items:center;gap:9px;font-size:13px;color:var(--mut)}
 .ib .t{height:14px;background:#0a0f1a;border:1px solid var(--line);border-radius:5px;overflow:hidden}
 .ib .f{height:100%;background:linear-gradient(90deg,#fb923c,#f97316)}
 .note{margin-top:16px;border-left:3px solid var(--warn);background:#1a160c;padding:11px 14px;border-radius:0 10px 10px 0;color:#ecd9b0;font-size:14px}
 .foot{margin-top:22px;color:var(--mut);font-size:13px;text-align:center}
 a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
</style></head>
<body><div class="wrap">
 <div class="eyebrow">nba-game-predictor · live demo</div>
 <h1>Who wins before tip-off?</h1>
 <p class="sub">The repo's <b>calibrated, leakage-free logistic model</b> (test accuracy 67%, ROC-AUC 0.72), in your browser. Every input is known before the game starts — no peeking at the final score. Set each team's Elo and rest.</p>
 <div class="presets" id="presets"></div>

 <div class="result">
   <div class="split">
     <div class="side hh"><div class="nm">Home win</div><div class="p" id="ph">—</div></div>
     <div class="side aa" style="text-align:right"><div class="nm">Away win</div><div class="p" id="pa">—</div></div>
   </div>
   <div class="pbar"><div class="h" id="bh"></div><div class="a" id="ba"></div></div>
 </div>

 <div class="teams">
  <div class="card"><h3 class="hh">🏠 Home team</h3>
    <div class="row"><div class="lab"><span>Elo rating</span><b id="vhe"></b></div><input type="range" id="he" min="1380" max="1680" step="1" value="1520"></div>
    <div class="row"><div class="lab"><span>Rest days</span><b id="vhr"></b></div><input type="range" id="hr" min="0" max="5" step="1" value="2"></div>
  </div>
  <div class="card"><h3 class="aa">✈️ Away team</h3>
    <div class="row"><div class="lab"><span>Elo rating</span><b id="vae"></b></div><input type="range" id="ae" min="1380" max="1680" step="1" value="1500"></div>
    <div class="row"><div class="lab"><span>Rest days</span><b id="var"></b></div><input type="range" id="ar" min="0" max="5" step="1" value="2"></div>
  </div>
 </div>

 <details class="adv"><summary>Advanced features (recent form &amp; net rating)</summary>
   <div class="teams" style="margin-top:12px">
     <div class="card">
       <div class="row"><div class="lab"><span>Form diff (home−away win% L10)</span><b id="vform"></b></div><input type="range" id="form" min="-0.5" max="0.5" step="0.05" value="0"></div>
     </div>
     <div class="card">
       <div class="row"><div class="lab"><span>Net-rating diff (home−away)</span><b id="vnet"></b></div><input type="range" id="net" min="-12" max="12" step="0.5" value="0"></div>
     </div>
   </div>
 </details>

 <h2>What the model weights</h2>
 <div class="imp" id="imp"></div>
 <div class="note">Team strength + home court (bundled into <b>Elo</b>) dominates; rest is a small edge; recent form and net rating add little once Elo is known. The model is <b>calibrated</b> (a 60% really means 60%) and strictly <b>leakage-free</b>.</div>
 <div class="foot">Calibrated logistic regression · Elo + rolling features, no post-game leakage · <a href="https://github.com/danielduongg/nba-game-predictor" target="_blank">source, backtest &amp; calibration curve →</a></div>
</div>
<script>
const M=__MODEL__; const F=M.features; const HC=M.home_court_elo;
function feats(){
  const he=+he_.value, ae=+ae_.value, hr=+hr_.value, ar=+ar_.value, form=+form_.value, net=+net_.value;
  return {elo_diff:(he+HC)-ae, form_diff:form, netrtg_diff:net, rest_diff:hr-ar,
          home_b2b:hr===0?1:0, away_b2b:ar===0?1:0};
}
function prob(){
  const fv=feats(); const x=F.map(k=>fv[k]); let out=0;
  for(const m of M.members){let s=m.intercept;for(let i=0;i<F.length;i++)s+=((x[i]-m.mean[i])/m.scale[i])*m.coef[i];out+=1/(1+Math.exp(m.a*s+m.b));}
  return out/M.members.length;
}
const he_=document.getElementById('he'),ae_=document.getElementById('ae'),hr_=document.getElementById('hr'),
      ar_=document.getElementById('ar'),form_=document.getElementById('form'),net_=document.getElementById('net');
function render(){
  document.getElementById('vhe').textContent=he_.value;
  document.getElementById('vae').textContent=ae_.value;
  document.getElementById('vhr').textContent=hr_.value+(hr_.value==='0'?' (back-to-back)':'');
  document.getElementById('var').textContent=ar_.value+(ar_.value==='0'?' (back-to-back)':'');
  document.getElementById('vform').textContent=(+form_.value).toFixed(2);
  document.getElementById('vnet').textContent=(+net_.value).toFixed(1);
  const p=prob();
  document.getElementById('ph').textContent=(p*100).toFixed(0)+'%';
  document.getElementById('pa').textContent=((1-p)*100).toFixed(0)+'%';
  document.getElementById('bh').style.width=(p*100).toFixed(1)+'%';
  document.getElementById('ba').style.width=((1-p)*100).toFixed(1)+'%';
}
[he_,ae_,hr_,ar_,form_,net_].forEach(el=>el.addEventListener('input',render));
const PRESETS={
 "Even matchup":[1500,1500,2,2,0,0],
 "Contender at home":[1620,1470,2,2,0.2,6],
 "Road favorite":[1470,1620,2,2,-0.2,-6],
 "Home on a back-to-back":[1500,1500,0,3,0,0],
};
function setP(a){he_.value=a[0];ae_.value=a[1];hr_.value=a[2];ar_.value=a[3];form_.value=a[4];net_.value=a[5];render();}
const pc=document.getElementById('presets');
Object.keys(PRESETS).forEach(k=>{const c=document.createElement('span');c.className='pz';c.textContent=k;c.onclick=()=>setP(PRESETS[k]);pc.appendChild(c);});
const NAMES={elo_diff:"Elo + home court",rest_diff:"Rest advantage",form_diff:"Recent form",netrtg_diff:"Net rating",home_b2b:"Home back-to-back",away_b2b:"Away back-to-back"};
const ent=Object.entries(M.importance).sort((a,b)=>b[1]-a[1]);const mx=ent[0][1];
document.getElementById('imp').innerHTML=ent.map(([k,v])=>`<div class="ib"><div>${NAMES[k]||k}</div><div class="t"><div class="f" style="width:${Math.max(v/mx*100,1.5).toFixed(0)}%"></div></div></div>`).join('');
render();
</script>
</body></html>'''
HTML=HTML.replace("__MODEL__",json.dumps(M))
open("index.html","w").write(HTML); print("wrote index.html",round(len(HTML)/1024,1),"KB")
