"""Local tool for hand-labelling specimen sheets, to get truth independent of GBIF.

Every other label set here comes from GBIF, which also supplies the strongest
scientificName confidence signal. These are read off the image instead.

Pipeline candidates are offered to make labelling fast, but shuffled with their
source hidden, and nothing is preselected -- so accepting one cannot
systematically favour any pipeline.

    python nbs/label_tool.py --images transcription/data/hand-50
    # then open http://127.0.0.1:842

Writes transcription/data/hand-50-truth/{taxons,dates,collectors,
catalognumbers,localities}.txt -- the same format the evals read, so:

    HERBARIA_GT_DIR=transcription/data/hand-50-truth python nbs/model_compare.py

Free: reads local caches, no API calls.
"""
import argparse
import json
import os
import random
import sys

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

# field -> ground-truth filename, matching nbs/gbif_fetch.py's layout
FIELD_FILE = {
    "scientificName": "taxons.txt",
    "eventDate": "dates.txt",
    "recordedBy": "collectors.txt",
    "barcode": "catalognumbers.txt",
    "location": "localities.txt",
}
FIELDS = list(FIELD_FILE)

# AVOID names the mistake each field attracts -- mostly other labels on the same
# sheet: institution stamps, exchange labels, determination slips.
HINT = {
    "scientificName": "Genus + species, currently accepted name. Authority not needed.",
    "eventDate": "Four-digit year always -- '5/21/19' cannot be parsed. Only the year is scored.",
    "recordedBy": "The collector, usually after 'Collected by' or 'Leg.'",
    "barcode": "The number under the printed barcode. Digits only; prefixes are ignored.",
    "location": "Where the plant was collected: state, county, locality.",
}
AVOID = {
    "scientificName": "If the sheet shows an old synonym, write today's accepted name -- "
                      "the pipelines emit that, so a verbatim synonym scores them wrong.",
    "eventDate": "Not the determination date, which is often later and in a different hand.",
    "recordedBy": "Not 'det.' / 'ex herb.' / donors / later annotators.",
    "barcode": "Not the handwritten 'No. xxx' by the collector -- that is the field "
               "number, a different DWC field we do not score.",
    "location": "Not the herbarium's own address or stamp, which is usually the "
                "largest text on the sheet.",
}

CACHES = [
    ("transcription/results/ocr_cache/azure", "fields"),
    ("transcription/results/ocr_cache/google", "fields"),
    ("transcription/results/model_cache/claude-sonnet-5", None),
]

app = FastAPI()
STATE = {}


def read_gt(d):
    out = {f: {} for f in FIELDS}
    for f, fn in FIELD_FILE.items():
        p = os.path.join(d, fn)
        if os.path.exists(p):
            for line in open(p, encoding="utf-8", errors="replace"):
                if ":" in line:
                    k, v = line.split(":", 1)
                    out[f][k.strip()] = v.strip()
    return out


def write_gt(d, labels):
    os.makedirs(d, exist_ok=True)
    for f, fn in FIELD_FILE.items():
        with open(os.path.join(d, fn), "w", encoding="utf-8") as fh:
            for occid in sorted(labels[f]):
                fh.write(f"{occid}: {labels[f][occid]}\n")


def candidates(occid):
    """Shuffled, source-hidden candidate values per field."""
    out = {f: [] for f in FIELDS}
    for root, key in CACHES:
        p = os.path.join(root, occid + ".json")
        if not os.path.exists(p):
            continue
        try:
            rec = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(rec, dict):
            continue
        src = rec.get(key) if key else rec
        if not isinstance(src, dict):
            continue
        for f in FIELDS:
            v = src.get(f)
            if isinstance(v, dict):          # Azure's structured location
                v = ", ".join(str(x) for x in v.values()
                              if x and str(x).upper() != "UNKNOWN")
            v = (str(v).strip() if v is not None else "")
            if v and v.upper() != "UNKNOWN" and v not in out[f]:
                out[f].append(v)
    for f in FIELDS:
        random.shuffle(out[f])
    return out


class Save(BaseModel):
    occid: str
    values: dict


@app.get("/api/next")
def next_specimen():
    done = {o for o in STATE["labels"]["scientificName"]}
    todo = [o for o in STATE["occids"] if o not in done]
    if not todo:
        return JSONResponse({"done": True, "total": len(STATE["occids"])})
    occid = todo[0]
    return JSONResponse({
        "done": False,
        "occid": occid,
        "index": len(STATE["occids"]) - len(todo) + 1,
        "total": len(STATE["occids"]),
        "fields": FIELDS,
        "hints": HINT,
        "avoid": AVOID,
        "candidates": candidates(occid),
    })


@app.get("/img/{occid}")
def image(occid: str):
    for ext in (".jpeg", ".jpg", ".png"):
        p = os.path.join(STATE["images"], occid + ext)
        if os.path.exists(p):
            return FileResponse(p)
    return JSONResponse({"error": "not found"}, status_code=404)


@app.post("/api/save")
def save(s: Save):
    for f in FIELDS:
        STATE["labels"][f][s.occid] = (s.values.get(f) or "UNKNOWN").strip() or "UNKNOWN"
    write_gt(STATE["out"], STATE["labels"])
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


PAGE = """
<!doctype html><meta charset=utf-8><title>Specimen labelling</title>
<style>
 :root{--bg:#121614;--panel:#1b211e;--line:#2c3733;--ink:#e6ebe6;--soft:#93a09a;--accent:#57bc9d}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,sans-serif;
      display:grid;grid-template-columns:1fr 430px;height:100vh;overflow:hidden}
 #imgwrap{position:relative;overflow:hidden;background:#0a0d0c;cursor:grab}
 #imgwrap.drag{cursor:grabbing}
 img{position:absolute;transform-origin:0 0;user-select:none;-webkit-user-drag:none}
 #hint{position:absolute;left:12px;bottom:12px;background:#000a;padding:6px 10px;
       border-radius:6px;font-size:12px;color:#cfd8d3}
 #side{border-left:1px solid var(--line);background:var(--panel);overflow-y:auto;padding:18px}
 h2{margin:0 0 2px;font-size:15px}
 .prog{color:var(--soft);font-size:12px;margin-bottom:16px}
 .f{margin-bottom:18px;border-bottom:1px solid var(--line);padding-bottom:14px}
 .fname{font-weight:600;font-size:13px}
 .fhint{color:var(--soft);font-size:11.5px}
 .favoid{color:#d6a54a;font-size:11.5px;margin:3px 0 7px;padding-left:8px;
         border-left:2px solid #6a5320}
 .cand{display:block;width:100%;text-align:left;margin:3px 0;padding:7px 9px;border-radius:6px;
       border:1px solid var(--line);background:#151a18;color:var(--ink);font:13px system-ui;cursor:pointer}
 .cand:hover{border-color:var(--accent)}
 .cand.sel{border-color:var(--accent);background:#14302a}
 input{width:100%;margin-top:5px;padding:7px 9px;border-radius:6px;border:1px solid var(--line);
       background:#0f1412;color:var(--ink);font:13px system-ui}
 input:focus{outline:2px solid var(--accent);outline-offset:-1px}
 button.go{width:100%;padding:11px;border:0;border-radius:7px;background:var(--accent);
           color:#07110d;font-weight:700;font-size:14px;cursor:pointer;margin-top:4px}
 button.go:disabled{opacity:.4;cursor:not-allowed}
 .unk{background:none;border:0;color:var(--soft);font-size:11.5px;cursor:pointer;padding:3px 0;text-decoration:underline}
 .kbd{color:var(--soft);font-size:11px;text-align:center;margin-top:10px}
</style>
<div id=imgwrap><img id=sheet><div id=hint>scroll = zoom &middot; drag = pan &middot; double-click = fit</div></div>
<div id=side>
  <h2 id=occid>loading</h2><div class=prog id=prog></div>
  <div id=fields></div>
  <button class=go id=save disabled>Save &amp; next</button>
  <div class=kbd>Ctrl+Enter saves</div>
</div>
<script>
let cur=null, chosen={};
const img=document.getElementById('sheet'), wrap=document.getElementById('imgwrap');
let z=1,ox=0,oy=0;
function apply(){img.style.transform=`translate(${ox}px,${oy}px) scale(${z})`}
function fit(){const r=wrap.getBoundingClientRect();
  z=Math.min(r.width/img.naturalWidth,r.height/img.naturalHeight)||1;
  ox=(r.width-img.naturalWidth*z)/2; oy=(r.height-img.naturalHeight*z)/2; apply()}
img.onload=fit; wrap.ondblclick=fit;
wrap.onwheel=e=>{e.preventDefault();const r=wrap.getBoundingClientRect();
  const mx=e.clientX-r.left,my=e.clientY-r.top,f=e.deltaY<0?1.15:1/1.15;
  ox=mx-(mx-ox)*f; oy=my-(my-oy)*f; z*=f; apply()};
let dr=null;
wrap.onmousedown=e=>{dr=[e.clientX-ox,e.clientY-oy];wrap.classList.add('drag')};
addEventListener('mouseup',()=>{dr=null;wrap.classList.remove('drag')});
addEventListener('mousemove',e=>{if(dr){ox=e.clientX-dr[0];oy=e.clientY-dr[1];apply()}});

function ready(){return cur && cur.fields.every(f=>(chosen[f]||'').trim())}
function refresh(){document.getElementById('save').disabled=!ready()}

async function load(){
  const d=await (await fetch('/api/next')).json();
  if(d.done){document.body.innerHTML=
    `<div style="padding:60px;font:16px system-ui;color:#e6ebe6">
     <h2>All ${d.total} specimens labelled.</h2>
     <p style="color:#93a09a">Written to the truth directory. Score with:<br>
     <code>HERBARIA_GT_DIR=transcription/data/hand-50-truth python nbs/model_compare.py</code></p></div>`;
    document.body.style.display='block';return}
  cur=d; chosen={};
  document.getElementById('occid').textContent=d.occid;
  document.getElementById('prog').textContent=`specimen ${d.index} of ${d.total}`;
  img.src='/img/'+d.occid;
  const box=document.getElementById('fields'); box.innerHTML='';
  d.fields.forEach(f=>{
    const el=document.createElement('div'); el.className='f';
    el.innerHTML=`<div class=fname>${f}</div><div class=fhint>${d.hints[f]||''}</div>`
                 +(d.avoid[f]?`<div class=favoid>${d.avoid[f]}</div>`:'');
    (d.candidates[f]||[]).forEach(v=>{
      const b=document.createElement('button'); b.className='cand'; b.textContent=v;
      b.onclick=()=>{chosen[f]=v;
        el.querySelectorAll('.cand').forEach(x=>x.classList.remove('sel'));
        b.classList.add('sel'); el.querySelector('input').value=v; refresh()};
      el.appendChild(b)});
    const i=document.createElement('input'); i.placeholder='or type what you read';
    i.oninput=()=>{chosen[f]=i.value;
      el.querySelectorAll('.cand').forEach(x=>x.classList.remove('sel')); refresh()};
    el.appendChild(i);
    const u=document.createElement('button'); u.className='unk'; u.textContent="can't read this";
    u.onclick=()=>{chosen[f]='UNKNOWN'; i.value='UNKNOWN';
      el.querySelectorAll('.cand').forEach(x=>x.classList.remove('sel')); refresh()};
    el.appendChild(u);
    box.appendChild(el)});
  refresh()}

document.getElementById('save').onclick=async()=>{
  if(!ready())return;
  await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({occid:cur.occid,values:chosen})});
  load()};
addEventListener('keydown',e=>{if(e.ctrlKey&&e.key==='Enter')document.getElementById('save').click()});
load();
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="transcription/data/hand-50")
    ap.add_argument("--out", default=None, help="default: <images>-truth")
    ap.add_argument("--port", type=int, default=842)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--redo", action="store_true",
                    help="present every specimen again, overwriting existing labels. "
                         "Use after a protocol change, so early batches are not scored "
                         "under different rules from later ones.")
    args = ap.parse_args()

    out = args.out or args.images.rstrip("/\\") + "-truth"
    if os.path.abspath(out) == os.path.abspath(args.images):
        raise SystemExit("--out must differ from --images, or GBIF labels get overwritten")

    occids = sorted(os.path.splitext(f)[0] for f in os.listdir(args.images)
                    if f.lower().endswith((".jpeg", ".jpg", ".png")))
    if not occids:
        raise SystemExit(f"no images in {args.images}")

    random.seed(args.seed)
    existing = read_gt(out)
    prior = len(existing["scientificName"])
    if args.redo:
        existing = {f: {} for f in FIELDS}
    STATE.update(images=args.images, out=out, occids=occids, labels=existing)
    print(f"{len(occids)} specimens in {args.images}")
    if args.redo:
        print(f"writing labels to {out}  (--redo: re-labelling all {len(occids)}, "
              f"{prior} previous will be overwritten)")
    else:
        print(f"writing labels to {out}  ({prior} already done)")
    print(f"\n  open http://127.0.0.1:{args.port}\n")

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
