"""
webui.py — หน้าเว็บถามตอบ เทียบระบบเดิม (ข้อ 1) กับระบบปรับปรุง (ข้อ 2) ข้างกัน

รัน:  pip install flask
      python webui.py
แล้วเปิด http://localhost:7860

ต้องมี OpenSearch ทำงานอยู่ และนำเข้าเอกสารครบทั้งสองระบบแล้ว
ถ้าไม่ได้ตั้ง OPENROUTER_API_KEY หน้าเว็บยังค้นเอกสารได้ แต่จะสร้างคำตอบไม่ได้
"""

import json  # อ่านไฟล์คำถาม eval_questions.json
import os  # อ่านพอร์ตจาก environment variable
import threading  # ใช้ Lock ให้ประมวลผลทีละคำขอ
import time  # จับเวลาการสร้างคำตอบ

# Flask = web framework ขนาดเล็กของ Python
# render_template ส่งหน้า HTML, jsonify ส่งข้อมูล JSON, request อ่านข้อมูลที่หน้าเว็บส่งมา
from flask import Flask, jsonify, render_template, request

import rag_improved as R

app = Flask(__name__)  # สร้างเว็บแอป
_lock = threading.Lock()   # เครื่อง RAM 8 GB: ให้ประมวลผลทีละคำขอ
_state: dict = {}  # เก็บโมเดลและการเชื่อมต่อที่โหลดแล้ว (โหลดครั้งเดียว)

# โหลดคำถามตัวอย่าง และ gold (section ที่ถูกต้อง) ไว้แสดงว่าระบบค้นเจอถูกหรือไม่
QUESTIONS = json.loads((R.ROOT / "eval_questions.json").read_text(encoding="utf-8"))["questions"]
GOLD = {q["question"].strip(): q["gold"] for q in QUESTIONS}


# โหลดทรัพยากรครั้งแรกที่ถูกเรียก แล้วเก็บใน _state ใช้ซ้ำ (lazy loading)
# โมเดลชุดเดียวใช้ร่วมกันทั้งระบบเดิมและระบบปรับปรุง ประหยัด RAM
def resources():
    if not _state:
        client = R.opensearch_client()
        models = R.Models(load_reranker=R.USE_RERANKER)
        # ถ้าไม่มี API key ยังเปิดเว็บได้ (ค้นได้ แต่สร้างคำตอบไม่ได้)
        try:
            llm = R.LLM()
        except RuntimeError:
            llm = None
        _state.update(client=client, models=models, llm=llm,
                      baseline=R.BaselineRAG(models, client, llm),
                      improved=R.ImprovedRAG(models, client, llm))
    return _state


# นับจำนวนชิ้นใน index (ไม่มี index หรือ error คืน None) ใช้แสดงสถานะ
def index_count(client, name):
    try:
        return client.count(index=name)["count"] if client.indices.exists(index=name) else None
    except Exception:  # noqa: BLE001
        return None


# สถิติกราฟสำหรับแสดงในหน้าเว็บ (สร้างไม่ได้คืน None หน้าเว็บจะไม่พัง)
def _graph_stats(s):
    try:
        import graph_rag
        st = graph_rag.graph_stats(s["improved"].graph.G)
        return {"nodes": st["nodes"], "edges": st["edges"], "cross": st["cross_document_edges"]}
    except Exception:  # noqa: BLE001
        return None


# หน้าแรก: ส่งไฟล์ templates/index.html
@app.get("/")
def home():
    return render_template("index.html")


# API สถานะ: จำนวนชิ้นในแต่ละ index, การตั้งค่า reranker/กราฟ/LLM และคำถามตัวอย่าง
# หน้าเว็บเรียกตอนเปิดเพื่อแสดงสถานะระบบ
@app.get("/api/status")
def status():
    s = resources()
    c = s["client"]
    return jsonify({
        "baseline": {"bm25": index_count(c, R.BASELINE_BM25_INDEX),
                     "vector": index_count(c, R.BASELINE_VECTOR_INDEX)},
        "improved": {"bm25": index_count(c, R.IMPROVED_BM25_INDEX),
                     "vector": index_count(c, R.IMPROVED_VECTOR_INDEX)},
        "reranker": R.USE_RERANKER, "reranker_model": R.RERANK_MODEL,
        "threshold": R.RERANK_THRESHOLD,
        "graph": R.USE_GRAPH,
        "graph_stats": _graph_stats(s),
        "llm": s["llm"] is not None, "llm_model": R.LLM_MODEL,
        "samples": [q["question"] for q in QUESTIONS],
    })


# API ถามตอบ: รับคำถามจากหน้าเว็บ ค้นด้วยทั้ง 2 ระบบ แล้วส่งผลกลับเป็น JSON ให้แสดงข้างกัน
@app.post("/api/ask")
def ask():
    # อ่าน JSON ที่หน้าเว็บส่งมา
    body = request.get_json(force=True)
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify({"error": "พิมพ์คำถามก่อนกดค้น"}), 400
    # ตัวเลือกจากหน้าเว็บ: สร้างคำตอบไหม, เปิด rerank / กรอง / กราฟไหม (ใช้ทดลองเปิดปิดแต่ละเทคนิค)
    want_answer = bool(body.get("answer"))
    use_rerank = bool(body.get("rerank", True)) and R.USE_RERANKER
    threshold = R.RERANK_THRESHOLD if body.get("filter", True) else 0.0
    use_graph = bool(body.get("graph", True))

    s = resources()
    if want_answer and s["llm"] is None:
        return jsonify({"error": "ยังไม่ได้ตั้ง OPENROUTER_API_KEY จึงสร้างคำตอบไม่ได้ "
                                 "ปิดตัวเลือกสร้างคำตอบ หรือตั้ง key แล้วเปิดเว็บใหม่"}), 400

    # gold = section ที่ถูกต้อง (ถ้าเป็นคำถามในชุดวัดผล)
    out = {"question": question, "gold": GOLD.get(question, []), "systems": {}}
    # ล็อกให้ทำทีละคำขอ (เครื่อง RAM 8 GB รันโมเดลหลายงานพร้อมกันไม่ไหว)
    with _lock:
        for key in ("baseline", "improved"):
            rag = s[key]
            if not rag.available():
                out["systems"][key] = {"error": "ยังไม่ได้นำเข้าเอกสารของระบบนี้"}
                continue
            try:
                # ระบบปรับปรุงส่งตัวเลือกเปิด/ปิดไปด้วย ส่วนระบบเดิมค้นแบบต้นฉบับ
                found = rag.search(question, use_reranker=use_rerank, threshold=threshold,
                                   use_graph=use_graph) if key == "improved" else rag.search(question)
                entry = {k: v for k, v in found.items() if k != "pool"}  # ตัด pool (ผู้สมัคร 20 ชิ้น) ออก ลดขนาดข้อมูล
                if want_answer:
                    t0 = time.time()
                    ans = rag.answer(question, found["results"])
                    entry["answer"] = ans["text"]
                    entry["answer_flags"] = R.answer_flags(ans["text"], ans["finish_reason"])
                    entry["answer_seconds"] = round(time.time() - t0, 1)
                out["systems"][key] = entry
            # error ของระบบหนึ่งไม่ทำให้อีกระบบพัง แสดงข้อความ error แทน
            except Exception as e:  # noqa: BLE001
                out["systems"][key] = {"error": f"{type(e).__name__}: {e}"}
    return jsonify(out)


# เริ่มเว็บเซิร์ฟเวอร์: โหลดโมเดลก่อน แล้วเปิดที่ 127.0.0.1 (เข้าได้จากเครื่องตัวเองเท่านั้น)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    print(f"กำลังโหลดโมเดล ... เมื่อขึ้น Running on แล้วเปิด http://localhost:{port}")
    resources()
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
