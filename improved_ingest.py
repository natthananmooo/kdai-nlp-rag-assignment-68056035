"""
improved_ingest.py — นำเข้าเอกสารเข้าระบบปรับปรุง (คู่กับ original/authenticRAG.py)

รัน:  python improved_ingest.py
      python improved_ingest.py --dry-run   ดูผลการแบ่ง chunk โดยไม่เรียก LLM/OpenSearch

สิ่งที่ทำ
  1. แบ่ง section ด้วย MarkdownNodeParser แบบต้นฉบับ (ได้ doc_0..doc_52 ตรงกัน)
  2. [A1, A2] ให้ Qwen เขียนบริบทภาษาไทยของแต่ละ section จาก "เอกสารเต็ม"
     ผลเก็บใน results/contexts_cache.json รันซ้ำไม่ต้องเสียค่า LLM อีก
  3. [A3] แบ่ง section ยาวเป็นชิ้นย่อยไม่เกิน ~1,000 ตัวอักษร
  4. [B, C] ลบและสร้าง improved-* index ใหม่ (ไม่แตะ anthropic-* ของระบบเดิม)
     BM25 ใช้ Thai analyzer และ vector index เก็บบริบทไว้ด้วย
"""

import argparse
import sys

import rag_improved as R


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="แสดงการแบ่ง chunk เท่านั้น")
    ap.add_argument("--no-llm", action="store_true",
                    help="ไม่เรียก LLM: section ที่ยังไม่มีบริบทใน cache ใช้ชื่อเอกสาร+หัวข้อเป็นบริบทชั่วคราว")
    args = ap.parse_args()

    sections, full_docs = R.load_sections()
    R.log(f"แบ่ง section แบบต้นฉบับได้ {len(sections)} section")
    if len(sections) != 53:
        R.log("! จำนวน section ไม่เท่ากับ 53 ของระบบเดิม การเทียบผลด้วยรหัส doc_x อาจคลาดเคลื่อน")
    chunks = R.build_chunks(sections, full_docs)
    longest = max(len(s["text"]) for s in sections)
    R.log(f"แบ่ง section ที่ยาวเกิน {R.MAX_CHUNK_CHARS} ตัวอักษรแล้วได้ {len(chunks)} chunk "
          f"(section ยาวสุดเดิม {longest} ตัวอักษร, chunk ยาวสุดตอนนี้ "
          f"{max(len(c.content) for c in chunks)} ตัวอักษร)")

    if args.dry_run:
        for c in chunks:
            R.log(f"{c.chunk_id:<12} {len(c.content):>5}  {c.title}")
        return

    llm = None if args.no_llm else R.LLM()
    client = R.opensearch_client()
    client.info()  # ตรวจว่าเชื่อมต่อ OpenSearch ได้ก่อนเสียค่า LLM

    # ---- บริบทต่อ section (เรียก LLM 1 ครั้งต่อ section เท่าต้นฉบับ) ----
    cache = R.load_context_cache()
    section_context: dict[str, str] = {}
    provisional = []
    for i, s in enumerate(sections, 1):
        key = R._cache_key(s["source"], s["text"])
        if key in cache:
            section_context[s["section_id"]] = cache[key]
            continue
        if args.no_llm:
            # บริบทชั่วคราว ไม่บันทึกลง cache รันแบบปกติภายหลังเพื่อให้ LLM เขียนแทน
            doc_title = R._first_heading(full_docs[s["source"]])
            section_context[s["section_id"]] = (
                f"ส่วน '{R._first_heading(s['text'])}' จากบทความเรื่อง {doc_title}")
            provisional.append(s["section_id"])
            continue
        try:
            ctx = R.generate_section_context(llm, full_docs[s["source"]], s["text"])
        except R.OutOfCredits as e:
            done = sum(1 for x in sections if R._cache_key(x["source"], x["text"]) in cache)
            R.log(f"\n! หยุดที่ section {i}/{len(sections)}: {e}\n  ทำเสร็จแล้ว {done} section")
            sys.exit(2)
        cache[key] = ctx
        R.save_context_cache(cache)
        section_context[s["section_id"]] = ctx
        R.log(f"[{i}/{len(sections)}] {s['section_id']}: {ctx[:80]}")
    R.log(f"บริบทครบ {len(section_context)} section (cache: {R.CONTEXT_CACHE.name})")
    if provisional:
        R.log(f"! โหมด --no-llm: {len(provisional)} section ใช้บริบทชั่วคราว (ชื่อเอกสาร + หัวข้อ) "
              f"ผลวัดยังไม่ใช่ผลสุดท้าย เมื่อเติมเครดิตแล้วให้รัน python improved_ingest.py อีกครั้ง")
    (R.ROOT / "results" / "ingest_info.json").write_text(
        R.json.dumps({"sections": len(sections), "chunks": len(chunks),
                      "llm_contexts": len(sections) - len(provisional),
                      "provisional_contexts": provisional}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    for c in chunks:
        c.context = section_context[c.section_id]

    # ---- embedding ----
    models = R.Models(load_reranker=False)
    R.log(f"กำลังสร้าง embedding {len(chunks)} chunk ...")
    vectors = models.embed([c.text_for_embedding() for c in chunks])

    # ---- สร้าง index ใหม่ ----
    bm25_body, vector_body = R.improved_index_bodies(models.dim)
    for name, body in ((R.IMPROVED_BM25_INDEX, bm25_body), (R.IMPROVED_VECTOR_INDEX, vector_body)):
        if client.indices.exists(index=name):
            client.indices.delete(index=name)
        client.indices.create(index=name, body=body)
        R.log(f"สร้าง index {name} ใหม่แล้ว")

    bm25_ops, vec_ops = [], []
    for c, v in zip(chunks, vectors):
        common = {"chunk_id": c.chunk_id, "section_id": c.section_id, "source": c.source,
                  "title": c.title, "content": c.content, "contextualized_content": c.context}
        bm25_ops += [{"index": {"_index": R.IMPROVED_BM25_INDEX, "_id": c.chunk_id}}, common]
        vec_ops += [{"index": {"_index": R.IMPROVED_VECTOR_INDEX, "_id": c.chunk_id}},
                    {**common, "embedding": v}]

    for name, ops in ((R.IMPROVED_BM25_INDEX, bm25_ops), (R.IMPROVED_VECTOR_INDEX, vec_ops)):
        resp = client.bulk(body=ops)
        if resp.get("errors"):
            bad = [it for it in resp["items"] if it["index"].get("error")]
            R.log(f"! นำเข้า {name} ผิดพลาด {len(bad)} รายการ ตัวอย่าง: {bad[0]['index']['error']}")
            sys.exit(1)
        client.indices.refresh(index=name)
        count = client.count(index=name)["count"]
        R.log(f"นำเข้า {name} สำเร็จ {count} chunk")

    # ---- โชว์ผลของ Thai analyzer เทียบ standard (ประกอบสไลด์หน้า 23) ----
    sample = "อาการของหัดเยอรมัน"
    for analyzer in ("standard", "thai"):
        tokens = R.analyze_tokens(client, sample, analyzer)
        R.log(f"{analyzer:>8} analyzer ตัดคำ '{sample}' ได้: {tokens}")
    R.log("เสร็จแล้ว ต่อด้วย: python improved_search.py")


if __name__ == "__main__":
    main()
