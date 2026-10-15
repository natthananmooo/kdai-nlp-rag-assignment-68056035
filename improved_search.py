"""
improved_search.py — ถามตอบด้วยระบบปรับปรุง (คู่กับ original/onlysearchAuthenticRAG.py)

รัน:  python improved_search.py              ใช้ reranker และกรองด้วย threshold (ค่าเริ่มต้น)
      RERANK_THRESHOLD=0 python improved_search.py   rerank อย่างเดียว ไม่กรอง
      USE_RERANKER=0 python improved_search.py  ปิด reranker ถ้าเครื่องหน่วยความจำไม่พอ

ใช้คำถามชุดเดียวกับต้นฉบับ 5 ข้อ และบันทึกผลที่ results/improved_search_results.json
"""

import json

import rag_improved as R

QUESTIONS = [
    "โรคหัดและโรคหัดเยอรมันแตกต่างกันอย่างไร?",
    "อธิบายสาเหตุของโรคหัดเยอรมันและการป้องกัน",
    "ทำไมโรคหัดเยอรมันจึงมีอันตรายกับหญิงตั้งครรภ์?",
    "ถ้าคนที่ฉีดวัคซีนป้องกันโรคหัดเยอรมันแล้ว จะมีโอกาสติดเชื้อหรือไม่?",
    "โรคหัดเยอรมันมีผลกระทบอย่างไรต่อระบบสาธารณสุขและเศรษฐกิจของประเทศ?",
]


def main() -> None:
    models = R.Models()
    rag = R.ImprovedRAG(models, llm=R.LLM())
    if not rag.available():
        raise SystemExit("ยังไม่มี improved-* index ให้รัน python improved_ingest.py ก่อน")

    output = []
    for i, q in enumerate(QUESTIONS, 1):
        R.log(f"\n===== คำถามที่ {i}/{len(QUESTIONS)} =====\nคำถาม: {q}")
        found = rag.search(q)
        R.log(f"Sparse (Thai BM25) เจอ {found['sparse_count']} | Dense เจอ {found['dense_count']} | "
              f"กราฟเจอ {found['graph_count']} | "
              f"rerank {'เปิด' if found['reranked'] else 'ปิด'} | "
              f"กรองทิ้ง {found['filtered_out']} ชิ้น (threshold {found['threshold']}) | "
              f"ค้น {found['seconds']} วินาที")
        for r in found["results"]:
            score = f"rerank {r['rerank_score']:.3f}" if r["rerank_score"] is not None else f"RRF {r['rrf']:.4f}"
            R.log(f"  {r['rank']}. [{r['id']}] {score} | {r['title']}")
        ans = rag.answer(q, found["results"])
        flags = R.answer_flags(ans["text"], ans["finish_reason"])
        R.log(f"\nคำตอบ: {ans['text']}\n({flags})")
        output.append({"question": q, "answer": ans["text"], "answer_flags": flags,
                       "search": {k: v for k, v in found.items() if k not in ("results", "pool")},
                       "results": found["results"]})

    path = R.ROOT / "results" / "improved_search_results.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    R.log(f"\nบันทึกผลลัพธ์ไปยัง {path.relative_to(R.ROOT)} สำเร็จ")


if __name__ == "__main__":
    main()
