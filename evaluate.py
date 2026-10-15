"""
evaluate.py — วัดผลเทียบระบบเดิม (ข้อ 1) กับระบบปรับปรุง (ข้อ 2) เพื่อใช้ตอบข้อ 3

รัน:  python evaluate.py             วัดการค้นคืนอย่างเดียว (ไม่เสียค่า LLM)
      python evaluate.py --answers   วัดคุณภาพคำตอบของ 5 คำถามหลักด้วย (เรียก LLM ไม่เกิน ~10 ครั้ง)

ตัวชี้วัด (คิดจากรหัส section ของต้นฉบับ ชิ้นย่อยหลายชิ้นจาก section เดียวกันนับเป็น 1)
  Hit@5       มี section ที่ถูกต้องอย่างน้อย 1 ชิ้นในชิ้นที่ส่งให้ LLM หรือไม่
  Recall@5    สัดส่วน section ที่ถูกต้องที่ค้นเจอในชิ้นที่ส่งให้ LLM
  MRR@5       1 / อันดับของ section ที่ถูกต้องชิ้นแรก (ยิ่งใกล้ 1 ยิ่งเจอเร็ว)
  Precision   สัดส่วนชิ้นที่ส่งให้ LLM ที่มาจาก section ที่ถูกต้อง (ยิ่งสูง ยิ่งมีของไม่เกี่ยวปนน้อย)
  ชิ้นที่ส่ง   จำนวนชิ้นเฉลี่ยที่ส่งให้ LLM (การกรองทำให้น้อยกว่า 5)
  BM25 เจอ    จำนวนผลที่การค้นแบบจับคำส่งกลับมา
  มีบริบท     สัดส่วนของชิ้นที่ส่งให้ LLM แล้วมีบริบทติดไปด้วย

ผลลัพธ์: results/evaluation.json และ results/evaluation.md
"""

import argparse
import json
import statistics
import time

import rag_improved as R


MULTI_HOP_IDS = {"Q7", "Q19", "Q20", "Q21"}


def section_order(results: list[dict]) -> list[str]:
    seen, order = set(), []
    for r in results:
        s = r["section_id"]
        if s not in seen:
            seen.add(s)
            order.append(s)
    return order


def metrics(results: list[dict], gold: list[str]) -> dict:
    order = section_order(results)
    gold_set = set(gold)
    first = next((i for i, s in enumerate(order, 1) if s in gold_set), None)
    m = {
        "hit": 1.0 if first else 0.0,
        "recall": len(gold_set & set(order)) / len(gold_set),
        "mrr": 1.0 / first if first else 0.0,
    }
    if results:
        m["precision"] = sum(1 for r in results if r["section_id"] in gold_set) / len(results)
    return m


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 3) if xs else 0.0


def apply_threshold(results: list[dict], t: float) -> list[dict]:
    return [r for r in results if (r["rerank_score"] or 0) >= t] if t > 0 else list(results)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", action="store_true", help="สร้างคำตอบด้วย LLM เทียบกันด้วย")
    args = ap.parse_args()

    qs = json.loads((R.ROOT / "eval_questions.json").read_text(encoding="utf-8"))["questions"]
    client = R.opensearch_client()
    models = R.Models(load_reranker=True)
    llm = R.LLM() if args.answers else None
    base = R.BaselineRAG(models, client, llm)
    imp = R.ImprovedRAG(models, client, llm)
    if not base.available():
        raise SystemExit("ไม่พบ anthropic-* index ของระบบเดิม ให้รัน python original/authenticRAG.py ก่อน")
    if not imp.available():
        raise SystemExit("ไม่พบ improved-* index ให้รัน python improved_ingest.py ก่อน")

    t_default = R.RERANK_THRESHOLD
    labels = {
        "baseline": "ระบบเดิม (ข้อ 1)",
        "improved_no_rerank": "ปรับปรุง A+B+C (ยังไม่ rerank)",
        "improved_rerank": "ปรับปรุง A+B+C+D (rerank ไม่กรอง)",
        "improved": f"ปรับปรุง A+B+C+D+D2 (rerank + กรอง ≥ {t_default})",
        "improved_graph": f"ปรับปรุงครบ + GraphRAG (ระบบสุดท้าย)",
    }

    rows = []
    for q in qs:
        row = {"id": q["id"], "question": q["question"], "gold": q["gold"], "systems": {}}
        b = base.search(q["question"])
        nr = imp.search(q["question"], use_reranker=False, use_graph=False)
        rr = imp.search(q["question"], use_reranker=True, threshold=0, use_graph=False)
        rg = imp.search(q["question"], use_reranker=True, threshold=0, use_graph=True)
        variants = {
            "baseline": (b, b["results"]),
            "improved_no_rerank": (nr, nr["results"]),
            "improved_rerank": (rr, rr["results"]),
            "improved": (rr, apply_threshold(rr["results"], t_default)),
            "improved_graph": (rg, apply_threshold(rg["results"], t_default)),
        }
        for key, (found, res) in variants.items():
            entry = {
                "top": [r["id"] for r in res],
                "top_sections": section_order(res),
                "kept": len(res),
                "sparse_count": found["sparse_count"],
                "graph_count": found.get("graph_count", 0),
                "from_graph": sum(1 for r in res if r.get("graph_rank")),
                "context_coverage": (sum(1 for r in res if r["context"].strip()) / len(res)) if res else None,
                "seconds": found["seconds"],
            }
            if q["gold"]:
                entry.update(metrics(res, q["gold"]))
            row["systems"][key] = entry
        # เก็บคะแนน rerank ของ 5 อันดับแรกของระบบสุดท้ายไว้ทดลอง threshold
        row["rerank_top5"] = [{"id": r["id"], "section_id": r["section_id"], "score": r["rerank_score"]}
                              for r in rg["results"]]
        row["graph_paths"] = {r["id"]: r["graph_path"] for r in rg["results"] if r.get("graph_path")}
        rows.append(row)
        R.log(f"{q['id']:<4} " + " | ".join(
            f"{k}: hit={v.get('hit', '-')} ส่ง={v['kept']} bm25={v['sparse_count']}"
            for k, v in row["systems"].items()))

    summary = {}
    for key, label in labels.items():
        scored = [r["systems"][key] for r in rows if r["gold"]]
        allr = [r["systems"][key] for r in rows]
        summary[key] = {
            "label": label,
            "hit@5": mean([s["hit"] for s in scored]),
            "recall@5": mean([s["recall"] for s in scored]),
            "mrr@5": mean([s["mrr"] for s in scored]),
            "precision": mean([s.get("precision") for s in scored]),
            "avg_kept": mean([s["kept"] for s in allr]),
            "avg_bm25_results": mean([s["sparse_count"] for s in allr]),
            "zero_bm25_questions": sum(1 for s in allr if s["sparse_count"] == 0),
            "context_coverage": mean([s["context_coverage"] for s in allr]),
            "avg_from_graph": mean([s["from_graph"] for s in allr]),
            "avg_search_seconds": mean([s["seconds"] for s in allr]),
        }

    # ---- ทดลองหลาย threshold (เทียบกับตารางในบทความ jev-reranker) ----
    sweep = []
    for t in R.THRESHOLD_SWEEP:
        ans_rows, kept_all, no_ans_rejected, ans_rejected = [], [], 0, 0
        for r in rows:
            kept = [x for x in r["rerank_top5"] if (x["score"] or 0) >= t] if t > 0 else r["rerank_top5"]
            kept_all.append(len(kept))
            if r["gold"]:
                ans_rows.append(metrics(kept, r["gold"]))
                ans_rejected += 0 if kept else 1
            else:
                no_ans_rejected += 0 if kept else 1
        sweep.append({
            "threshold": t, "avg_kept": mean(kept_all),
            "removed": round(1 - sum(kept_all) / max(1, sum(len(r["rerank_top5"]) for r in rows)), 3),
            "hit@5": mean([m["hit"] for m in ans_rows]),
            "recall@5": mean([m["recall"] for m in ans_rows]),
            "precision": mean([m.get("precision") for m in ans_rows]),
            "answerable_rejected": ans_rejected,
            "unanswerable_rejected": no_ans_rejected,
            "unanswerable_total": sum(1 for r in rows if not r["gold"]),
        })

    # ---- ตัวอย่างการตัดคำ standard vs thai (สไลด์หน้า 23) ----
    samples = ["อาการของหัดเยอรมัน", "วัคซีน MMR ควรฉีดตอนอายุเท่าไร"]
    analyzer_demo = {s: {a: R.analyze_tokens(client, s, a) for a in ("standard", "thai")} for s in samples}

    # ---- คำตอบ (ถ้าเลือก --answers) ----
    answers = []
    if args.answers:
        for q in qs[:5]:
            pair = {"id": q["id"], "question": q["question"]}
            for key, rag in (("baseline", base), ("improved", imp)):
                found = rag.search(q["question"])
                t0 = time.time()
                ans = rag.answer(q["question"], found["results"])
                pair[key] = {"answer": ans["text"], **R.answer_flags(ans["text"], ans["finish_reason"]),
                             "seconds": round(time.time() - t0, 1)}
            answers.append(pair)
            R.log(f"{q['id']} ตอบแล้ว: เดิม {pair['baseline']['chars']} ตัวอักษร | "
                  f"ใหม่ {pair['improved']['chars']} ตัวอักษร")

    info_path = R.ROOT / "results" / "ingest_info.json"
    ingest_info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    if ingest_info.get("provisional_contexts"):
        R.log(f"\n! ระบบปรับปรุงยังใช้บริบทชั่วคราว {len(ingest_info['provisional_contexts'])} section "
              "ผลนี้ยังไม่ใช่ผลสุดท้าย")

    out = {"summary": summary, "threshold_sweep": sweep, "threshold_default": t_default,
           "ingest_info": ingest_info,
           "per_question": rows, "analyzer_demo": analyzer_demo, "answers": answers}
    (R.ROOT / "results" / "evaluation.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    (R.ROOT / "results" / "evaluation.md").write_text(render_md(out, qs), encoding="utf-8")
    R.log("\nสรุป:")
    for v in summary.values():
        R.log(f"  {v['label']}: Hit@5={v['hit@5']} Recall@5={v['recall@5']} MRR@5={v['mrr@5']} "
              f"Precision={v['precision']} ส่งเฉลี่ย={v['avg_kept']} BM25 เฉลี่ย={v['avg_bm25_results']} "
              f"มีบริบท={v['context_coverage']} จากกราฟ={v['avg_from_graph']}")
    R.log("\nทดลอง threshold (ระบบสุดท้าย):")
    for s in sweep:
        R.log(f"  t={s['threshold']:<5} ส่งเฉลี่ย={s['avg_kept']} ตัดทิ้ง={s['removed']:.0%} "
              f"Hit@5={s['hit@5']} Precision={s['precision']} "
              f"ปฏิเสธคำถามที่ตอบได้={s['answerable_rejected']} "
              f"ปฏิเสธคำถามที่ตอบไม่ได้={s['unanswerable_rejected']}/{s['unanswerable_total']}")
    R.log("บันทึกที่ results/evaluation.md และ results/evaluation.json")


def render_md(out: dict, qs: list[dict]) -> str:
    s = out["summary"]
    L = ["# ผลการวัดผล: ระบบเดิม vs ระบบปรับปรุง", "",
         f"คำถาม {len(qs)} ข้อ (คิดคะแนนการค้นคืน {sum(1 for q in qs if q['gold'])} ข้อที่เอกสารมีคำตอบ)", ""]
    prov = out.get("ingest_info", {}).get("provisional_contexts") or []
    if prov:
        L += [f"> **ผลชั่วคราว:** ระบบปรับปรุงยังใช้บริบทชั่วคราว (ชื่อเอกสาร + หัวข้อ) {len(prov)} section "
              "แทนบริบทที่ LLM เขียน ต้องรันใหม่หลังสร้างบริบทครบ", ""]
    L += [
         "## สรุป", "",
         "| ระบบ | Hit@5 | Recall@5 | MRR@5 | Precision | ชิ้นที่ส่งเฉลี่ย | ชิ้นที่กราฟช่วยหา | BM25 เจอเฉลี่ย | คำถามที่ BM25 เจอ 0 | มีบริบท | เวลาค้น (วินาที) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for v in s.values():
        L.append(f"| {v['label']} | {v['hit@5']} | {v['recall@5']} | {v['mrr@5']} | {v['precision']} "
                 f"| {v['avg_kept']} | {v['avg_from_graph']} | {v['avg_bm25_results']} | {v['zero_bm25_questions']} "
                 f"| {v['context_coverage']:.0%} | {v['avg_search_seconds']} |")
    multi = [r for r in out["per_question"] if r["id"] in MULTI_HOP_IDS and r["gold"]]
    if multi:
        L += ["", "## คำถามที่ต้องต่อข้อมูลข้ามเอกสาร (multi-hop)", "",
              "| ข้อ | คำถาม | Recall@5 เดิม | Recall@5 ไม่มีกราฟ | Recall@5 มีกราฟ |", "|---|---|---|---|---|"]
        for r in multi:
            sy = r["systems"]
            L.append(f"| {r['id']} | {r['question']} | {sy['baseline']['recall']:.2f} "
                     f"| {sy['improved']['recall']:.2f} | {sy['improved_graph']['recall']:.2f} |")
        L += ["", "เส้นทางในกราฟที่ระบบสุดท้ายใช้หาหลักฐาน:", ""]
        for r in multi:
            for cid, p in r.get("graph_paths", {}).items():
                L.append(f"- {r['id']} `{cid}`: {p}")
    L += ["", "## ทดลอง threshold ของ relevance filtering (D2)", "",
          "ใช้คะแนน rerank ชุดเดียวกัน เปลี่ยนแค่เกณฑ์ตัดทิ้ง เทียบแนวคิดกับตาราง relevance_rerank ในบทความ jev-reranker", "",
          "| threshold | ชิ้นที่ส่งเฉลี่ย | ตัดทิ้ง | Hit@5 | Recall@5 | Precision | คำถามที่ตอบได้แต่ถูกปฏิเสธ | คำถามที่ตอบไม่ได้และถูกปฏิเสธ |",
          "|---|---|---|---|---|---|---|---|"]
    for w in out["threshold_sweep"]:
        mark = " (ค่าที่ใช้)" if w["threshold"] == out["threshold_default"] else ""
        L.append(f"| {w['threshold']}{mark} | {w['avg_kept']} | {w['removed']:.0%} | {w['hit@5']} | {w['recall@5']} "
                 f"| {w['precision']} | {w['answerable_rejected']} | {w['unanswerable_rejected']}/{w['unanswerable_total']} |")
    L += ["", "## รายคำถาม", "",
          "| ข้อ | คำถาม | section ที่ถูก | เดิม: ชิ้นที่ส่ง (MRR) | ปรับปรุงครบ: ชิ้นที่ส่ง (MRR) | คะแนน rerank 5 อันดับแรก |",
          "|---|---|---|---|---|---|"]
    for r in out["per_question"]:
        b, i = r["systems"]["baseline"], r["systems"]["improved"]
        fmt = lambda e: (", ".join(e["top_sections"]) or "(ไม่ส่ง)") + (f" ({e['mrr']:.2f})" if r["gold"] else "")
        scores = ", ".join(f"{x['score']:.2f}" for x in r["rerank_top5"] if x["score"] is not None)
        L.append(f"| {r['id']} | {r['question']} | {', '.join(r['gold']) or '(เอกสารไม่มี)'} | {fmt(b)} | {fmt(i)} | {scores} |")
    L += ["", "## การตัดคำของ analyzer (เกี่ยวกับสไลด์หน้า 23)", ""]
    for text, d in out["analyzer_demo"].items():
        L.append(f"- `{text}`")
        for a, toks in d.items():
            L.append(f"  - {a}: {' | '.join(toks)}")
    if out["answers"]:
        L += ["", "## คุณภาพคำตอบ (5 คำถามหลัก)", "",
              "| ข้อ | ระบบ | ความยาว | ถูกตัดกลางคัน | มีอักษรจีน | จำนวนอ้างอิง | ปฏิเสธเพราะไม่มีหลักฐาน |",
              "|---|---|---|---|---|---|---|"]
        for a in out["answers"]:
            for key, label in (("baseline", "เดิม"), ("improved", "ปรับปรุง")):
                e = a[key]
                L.append(f"| {a['id']} | {label} | {e['chars']} | {'ใช่' if e['truncated'] else '-'} "
                         f"| {'ใช่' if e['has_cjk'] else '-'} | {e['citations']} | {'ใช่' if e['no_evidence'] else '-'} |")
        L += ["", "### คำตอบเต็ม", ""]
        for a in out["answers"]:
            L += [f"#### {a['id']}: {a['question']}", "", "**ระบบเดิม**", "", a["baseline"]["answer"], "",
                  "**ระบบปรับปรุง**", "", a["improved"]["answer"], ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
