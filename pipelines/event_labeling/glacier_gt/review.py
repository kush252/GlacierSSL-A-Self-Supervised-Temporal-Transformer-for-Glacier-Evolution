"""Human verification (TimeSync-style) + final ground-truth tiers.

1. `build_review_sheet` writes review.html next to the evidence figures. Reviewers see the
   chips/plots and pick a class + confidence. BLIND by default: the automatic label is hidden
   so the reviewer is not anchored to it. Decisions export as CSV (one file per reviewer).
2. `merge_reviews` combines labels.jsonl with any number of reviewer CSVs:
     - Cohen's kappa for every reviewer pair, and auto-vs-consensus agreement + confusion matrix
     - final gt_tier:
         Gold    >= 2 reviewers agree with each other AND with the automatic label (auto tier A/B),
                 no reviewer at low confidence
         Silver  1 reviewer agrees with auto; or auto tier A unreviewed;
                 or >= 2 reviewers agree but overrule auto (human label kept, flagged)
         Bronze  unreviewed tier B/C, or reviewers disagree
         Reject  consensus says Insufficient / can't tell
"""
from __future__ import annotations

import csv
import html
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

from . import CLASSES

REVIEW_CLASSES = CLASSES + ["Ambiguous", "Insufficient"]


def build_review_sheet(records, out_dir: Path, blind: bool = True) -> Path:
    cards = []
    for r in records:
        img = Path(r["figure"]).relative_to(out_dir).as_posix()
        auto = "" if blind else (f"<p class='auto'>auto: <b>{r['primary_label']}</b> · tier {r['auto_tier']}"
                                 f" · agree {', '.join(r['agree']) or '–'}</p>")
        opts = "".join(f"<label><input type='radio' name='{r['window_id']}' value='{c}'> {c}</label>"
                       for c in REVIEW_CLASSES)
        cards.append(f"""
<section class="card" data-id="{r['window_id']}">
  <h2>{html.escape(r['region'])} · {r['glims_id']} · {r['start_year']}–{r['end_year']}
      <span>{r['n_valid']}/{r['n_epochs']} valid epochs</span></h2>
  <a href="{img}" target="_blank"><img src="{img}" loading="lazy" alt="evidence panel"></a>
  {auto}
  <div class="opts">{opts}</div>
  <div class="row">
    <label>confidence <select class="conf"><option>high</option><option selected>medium</option><option>low</option></select></label>
    <label class="also"><input type="checkbox" class="also_retreat"> also retreating</label>
    <input class="notes" placeholder="notes (what you saw, e.g. 'split at 2021 between tongues')">
  </div>
</section>""")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Glacier Event Review</title>
<style>
:root{{--bg:#f9f9f7;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3de;--acc:#2a78d6}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0d0d0d;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#383835;--acc:#3987e5}}}}
body{{background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif;margin:0;padding:16px}}
header{{position:sticky;top:0;background:var(--bg);padding:8px 0 12px;border-bottom:1px solid var(--line);z-index:2}}
h1{{font-size:18px;margin:0 0 6px}} .card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin:14px 0}}
h2{{font-size:14px;margin:0 0 8px}} h2 span{{color:var(--ink2);font-weight:400}} img{{max-width:100%;border-radius:4px}}
.opts{{display:flex;flex-wrap:wrap;gap:6px 14px;margin:8px 0}} .row{{display:flex;flex-wrap:wrap;gap:10px;align-items:center}}
.notes{{flex:1;min-width:220px;padding:6px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink)}}
button,input#rev{{padding:6px 12px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--ink)}}
button.primary{{background:var(--acc);color:#fff;border-color:var(--acc)}} .auto{{color:var(--ink2)}}
.done{{border-color:var(--acc)}} #count{{color:var(--ink2)}}
</style></head><body>
<header><h1>Glacier event review ({'blind' if blind else 'auto label shown'})</h1>
<p>Judge the event from the imagery and measurements only. Classes: Stable, Retreat, Advance, Fragmentation
(ice body splits into separate parts that stay separate), Disappearance (ice gone for good), Ambiguous, Insufficient.</p>
<input id="rev" placeholder="your reviewer id"> <button class="primary" onclick="exportCsv()">Export CSV</button>
<span id="count"></span></header>
{''.join(cards)}
<script>
const KEY='glacier_review_'+location.pathname;
function save(){{const d={{rev:document.getElementById('rev').value,items:{{}}}};
 document.querySelectorAll('.card').forEach(c=>{{const id=c.dataset.id,s=c.querySelector('input[type=radio]:checked');
  d.items[id]={{label:s?s.value:'',conf:c.querySelector('.conf').value,also:c.querySelector('.also_retreat').checked,notes:c.querySelector('.notes').value}};
  c.classList.toggle('done',!!s);}});
 const n=Object.values(d.items).filter(x=>x.label).length;
 document.getElementById('count').textContent=n+' / '+Object.keys(d.items).length+' reviewed';
 try{{localStorage.setItem(KEY,JSON.stringify(d));}}catch(e){{}} return d;}}
function load(){{let d=null;try{{d=JSON.parse(localStorage.getItem(KEY));}}catch(e){{}}
 if(d){{document.getElementById('rev').value=d.rev||'';
  document.querySelectorAll('.card').forEach(c=>{{const x=(d.items||{{}})[c.dataset.id];if(!x)return;
   if(x.label){{const r=c.querySelector(`input[value="${{x.label}}"]`);if(r)r.checked=true;}}
   c.querySelector('.conf').value=x.conf||'medium';c.querySelector('.also_retreat').checked=!!x.also;c.querySelector('.notes').value=x.notes||'';}});}}
 save();}}
function exportCsv(){{const d=save();if(!d.rev){{alert('Enter a reviewer id first');return;}}
 const q=s=>'"'+String(s).replaceAll('"','""')+'"';
 const rows=[['window_id','reviewer','label','confidence','also_retreat','notes']];
 for(const[id,x]of Object.entries(d.items))if(x.label)rows.push([id,d.rev,x.label,x.conf,x.also,x.notes]);
 const blob=new Blob([rows.map(r=>r.map(q).join(',')).join('\\n')],{{type:'text/csv'}});
 const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='review_'+d.rev+'.csv';a.click();}}
document.addEventListener('change',save);document.addEventListener('input',save);load();
</script></body></html>"""
    path = out_dir / ("review_blind.html" if blind else "review_with_auto.html")
    path.write_text(page, encoding="utf-8")
    return path


# ----------------------------------------------------------------------------- agreement
def cohen_kappa(a: list, b: list) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / n**2
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def merge_reviews(labels_path: Path, review_csvs: list[Path], out_path: Path) -> dict:
    recs = [json.loads(l) for l in labels_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    reviews: dict[str, dict[str, dict]] = {}
    for p in review_csvs:
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                reviews.setdefault(row["window_id"], {})[row["reviewer"]] = row
    reviewers = sorted({r for v in reviews.values() for r in v})

    report = {"reviewers": reviewers, "pairwise_kappa": {}, "auto_vs_consensus": {}}
    for r1, r2 in combinations(reviewers, 2):
        both = [w for w, v in reviews.items() if r1 in v and r2 in v]
        report["pairwise_kappa"][f"{r1}~{r2}"] = {
            "n": len(both),
            "kappa": round(cohen_kappa([reviews[w][r1]["label"] for w in both],
                                       [reviews[w][r2]["label"] for w in both]), 3),
        }

    auto_l, cons_l = [], []
    for r in recs:
        v = reviews.get(r["window_id"], {})
        labels = [x["label"] for x in v.values()]
        lowconf = any(x["confidence"] == "low" for x in v.values())
        consensus = labels[0] if labels and len(set(labels)) == 1 else None
        auto = r["primary_label"]
        if not labels:
            tier = "Silver" if r["auto_tier"] == "A" else "Bronze"
            final = auto
        elif consensus is None:
            tier, final = "Bronze", auto
        elif consensus in ("Insufficient", "Ambiguous"):
            tier, final = "Reject", consensus
        elif len(labels) >= 2 and consensus == auto and r["auto_tier"] in ("A", "B") and not lowconf:
            tier, final = "Gold", auto
        elif consensus == auto:
            tier, final = "Silver", auto
        elif len(labels) >= 2:
            tier, final = "Silver", consensus
            r.setdefault("flags", []).append("auto_overruled_by_reviewers")
        else:
            tier, final = "Bronze", auto
        r.update(gt_label=final, gt_tier=tier, reviews=v)
        if consensus:
            auto_l.append(auto)
            cons_l.append(consensus)

    if auto_l:
        cm = Counter(zip(auto_l, cons_l))
        report["auto_vs_consensus"] = {
            "n": len(auto_l),
            "accuracy": round(sum(a == c for a, c in zip(auto_l, cons_l)) / len(auto_l), 3),
            "kappa": round(cohen_kappa(auto_l, cons_l), 3),
            "confusion(auto,human)": {f"{a}|{c}": n for (a, c), n in sorted(cm.items())},
        }
    report["gt_tier_counts"] = dict(Counter(r["gt_tier"] for r in recs))
    report["gt_label_by_tier"] = {t: dict(Counter(r["gt_label"] for r in recs if r["gt_tier"] == t))
                                  for t in ("Gold", "Silver", "Bronze", "Reject")}
    with open(out_path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    return report
