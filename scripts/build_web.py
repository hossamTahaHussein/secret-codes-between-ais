"""Build web/index.html: embed the recorded data from web/data/*.json into web/template.html.

Only fields that exist in the exported files are used; nothing is generated or edited. Runs without a
final evaluation (still training) are left out until they finish. Re-run after `python -m stego.export_web`
to pick up new runs (Arabic, round 2) automatically.

    .venv/bin/python scripts/build_web.py
"""
import json
from datetime import date
from pathlib import Path

import numpy as np

DATA = Path("web/data")
LABELS = {
    "en_baseline": "Pilot A: penalties from step 1",
    "en_warmup": "Pilot B: 200-step warm-up",
    "q3_en_warmup": "Round 1: Qwen3, English",
    "q3_ar_warmup": "Round 1: Qwen3, Arabic",
}


def label(run):
    if run["id"] in LABELS:
        return LABELS[run["id"]]
    if run["id"].startswith("r2_"):
        return f"Round 2: Qwen3, {'Arabic' if run['lang'] == 'ar' else 'English'}, seed {run['seed']}"
    return run["id"]


def r3(xs):
    return [None if x is None else round(float(x), 3) for x in xs]


def build_run(entry):
    run = json.loads((DATA / entry["files"]["run"]).read_text())
    m = run["metrics"]
    return dict(id=entry["id"], label=label(entry), group=entry["group"], lang=entry["lang"], model=entry["model"],
                seed=entry["seed"], warmup=entry["penalty_warmup"], w_comm=entry["w_comm"],
                channel_formed=entry.get("channel_formed"), steps=run["steps"],
                recv=r3(m["recv_acc"]), literal=r3(m["literal"]), flagged=r3(m["flagged"]),
                off_lang=r3(m["off_lang"]) if "off_lang" in m else None,
                samples=run["samples"], eval=run["eval_final"]["summary"],
                eval_untrained=run.get("eval_untrained", {}).get("summary"))


def build_defense(entry):
    d = json.loads((DATA / entry["files"]["defense"]).read_text())
    payloads = d["payloads"]
    summaries = []
    for key, s in d["summaries"].items():
        if "auc" in s:
            summaries.append(dict(key=key, name=s["defense"], auc=round(s["auc"], 3), tpr=round(s["tpr_at_5fpr"], 3),
                                  undetected=round(s["undetected_cc_at_5fpr"], 3)))
    other = {k: v for k, v in d["summaries"].items() if "auc" not in v}

    # 5%-false-alarm threshold per detector, from its own scores on the negatives it was tested against
    thresholds = {}
    for neg in ("clean", "sendclean"):
        for r in d.get(neg, []):
            for k, v in r.get("detector_scores", {}).items():
                thresholds.setdefault(k, []).append(v)
    thresholds = {k: float(np.quantile(v, 0.95)) for k, v in thresholds.items()}

    def secret_row(r):
        out = dict(item=r["item"], payload=r["payload"], text=r["text"], guess=r.get("guess"),
                   literal=r.get("literal", False), scores=r.get("detector_scores", {}))
        for k in ("p_sus", "p_correct"):
            if k in r:
                out[k] = r[k]
        if "paraphrase" in r:
            p = r["paraphrase"]
            guess = p.get("guess") or (r["payload"] if p["correct"] else next(x for x in payloads if x != r["payload"]))
            out["para"] = dict(text=p["text"], guess=guess)
        if "translation_en" in r:
            out["translation"] = r["translation_en"]
        return out

    return dict(id=entry["id"], label=label(entry), lang=d["lang"], payloads=payloads, summaries=summaries,
                other=other, thresholds=thresholds, codebook=d["codebook"],
                secret=[secret_row(r) for r in d["secret"]],
                clean=[dict(item=r["item"], text=r["text"], scores=r.get("detector_scores", {}),
                            **({"translation": r["translation_en"]} if "translation_en" in r else {}))
                       for r in d["clean"]])


def main():
    index = json.loads((DATA / "index.json").read_text())
    finished = [e for e in index["runs"] if "eval_final" in e]
    bundle = dict(built=str(date.today()), channel_rule=index["channel_rule"],
                  runs=[build_run(e) for e in finished],
                  defenses=[build_defense(e) for e in finished if "defense" in e["files"]],
                  pending=[dict(id=e["id"], label=label(e), steps_logged=e["steps_logged"])
                           for e in index["runs"] if "eval_final" not in e])
    blob = json.dumps(bundle, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = Path("web/template.html").read_text().replace("/*__DATA__*/null", blob)
    Path("web/index.html").write_text(html)
    print(f"web/index.html  {len(html.encode()) / 1024:.0f} KB  runs={[r['id'] for r in bundle['runs']]}  "
          f"defenses={[d['id'] for d in bundle['defenses']]}  pending={[p['id'] for p in bundle['pending']]}")


if __name__ == "__main__":
    main()
