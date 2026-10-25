"""
graph_rag.py — GraphRAG แบบ rule-based (ตามแนวบทความของอาจารย์ Aekanun:
"Why Vector RAG Answers Only Half of a Legal Question: GraphRAG + MCP in 5 Steps")

แนวคิดจากบทความที่นำมาใช้
  - Knowledge graph เก็บใน NetworkX ในหน่วยความจำ ไม่ต้องใช้ graph database
  - สร้างกราฟด้วยกฎ (rule-based) ไม่ใช้ LLM: เร็ว ไม่มีค่า API และได้กราฟเดิมทุกครั้งที่รัน
  - Semantic nodes จากรายการคำที่ผู้พัฒนาคัดเลือกเอง
  - Vector หาจุดเริ่มต้น (seed) แล้วกราฟเดินต่อด้วย BFS ไม่เกิน 2 hop ทั้งขาเข้าและขาออก
  - จำกัดจำนวนผล และติดป้าย [hopN][ความสัมพันธ์] ให้ LLM รู้ว่าแต่ละชิ้นมาถึงด้วยความสัมพันธ์อะไร
  - Keyword สองระดับ (แนวคิดจาก LightRAG): low-level = ชื่อโรคและคำเฉพาะ, high-level = ประเภทหัวข้อ
    (บทความใช้ LLM แยก keyword; งานนี้ใช้การจับคำจากรายการเดียวกับที่ใช้สร้างกราฟ จึงไม่เสียค่า API)

ปัญหาที่แก้ (context fragmentation) ในเอกสารชุดนี้ เช่น
  ส่วนสาเหตุของต้อกระจก (doc_29) บอกว่าต้อกระจกแต่กำเนิดเกิดจากแม่เป็นหัดเยอรมันตอนตั้งครรภ์
  ส่วนภาวะแทรกซ้อนในเอกสารหัดเยอรมัน (doc_6) บอกว่าทารกอาจเป็นต้อกระจกและต้อหิน
  ข้อความเหล่านี้อยู่คนละเอกสาร เชื่อมกันด้วยชื่อโรค ไม่ใช่ด้วยความคล้ายของข้อความ

รันเดี่ยว:  python graph_rag.py   สร้างกราฟ ตรวจความถูกต้อง และบันทึก results/knowledge_graph.json
"""

# =====================================================================
# ส่วนที่ 0: import ไลบรารี
# =====================================================================
from __future__ import annotations

import json  # บันทึกกราฟเป็นไฟล์ JSON
import re  # regular expression ใช้จับชื่อโรค/คำเชื่อมในข้อความ
from collections import deque  # คิวสองทาง ใช้ทำ BFS (Breadth-First Search)
from pathlib import Path

# NetworkX: ไลบรารีกราฟของ Python เก็บกราฟในหน่วยความจำ (ตามบทความ ไม่ต้องใช้ graph database)
import networkx as nx

ROOT = Path(__file__).resolve().parent  # โฟลเดอร์โปรเจกต์
GRAPH_PATH = ROOT / "results" / "knowledge_graph.json"  # ไฟล์บันทึกกราฟ (เปิดดูโครงสร้างได้)

MAX_HOPS = 2          # เท่าบทความ
MAX_SEEDS = 10        # seed จากคำในคำถาม + seed จาก vector 3 ชิ้น
MAX_SECTIONS = 10     # จำนวน section สูงสุดที่ส่งต่อเข้า RRF

# ---------------------------------------------------------------------------
# Schema: โรค 4 โรค (1 เอกสาร = 1 โรค)
# ---------------------------------------------------------------------------

# 1 โรค = 1 เอกสาร: key -> ชื่อไทย, ไฟล์ต้นทาง, และ regex ของชื่อเรียกต่าง ๆ (ไทย/อังกฤษ/ชื่อพ้อง)
# ใช้ทั้งตอนสร้างกราฟ (section เอ่ยถึงโรคไหน) และตอนค้น (คำถามถามถึงโรคไหน)
DISEASES = {
    "rubella": {"name": "หัดเยอรมัน", "source": "./corpus_input/1.md",
                "patterns": [r"หัดเยอรมัน", r"เหือด", r"รูเบลลา", r"rubella", r"german measles", r"หัดสามวัน"]},
    "cholera": {"name": "อหิวาตกโรค", "source": "./corpus_input/2.md",
                "patterns": [r"อหิวาต", r"cholera", r"วิบริโอ"]},
    "cataract": {"name": "ต้อกระจก", "source": "./corpus_input/44.md",
                 "patterns": [r"ต้อกระจก", r"cataract"]},
    "gerd": {"name": "กรดไหลย้อน", "source": "./corpus_input/5555.md",
             "patterns": [r"กรดไหลย้อน", r"gerd", r"น้ำย่อยไหลกลับ"]},
}

# ประเภทหัวข้อ: (รหัส, ชื่อไทย, คำในหัวข้อ section, คำในคำถาม) ตรวจตามลำดับ ตัวแรกที่ตรงชนะ
# คำในหัวข้อ section ใช้ตอนสร้างกราฟ, คำในคำถามใช้ตอนค้น (high-level keyword แบบ LightRAG)
TOPICS = [
    ("reference", "แหล่งอ้างอิง", ["เอกสารอ้างอิง", "เมดไทย"], []),
    ("related", "เรื่องที่เกี่ยวข้อง", ["เรื่องที่เกี่ยวข้อง"], ["เกี่ยวข้อง", "เกิดร่วม"]),
    ("differential", "การแยกโรค", ["การแยกโรค"], ["แตกต่าง", "ต่างกัน", "แยกโรค", "แบบไหน"]),
    ("diagnosis", "การวินิจฉัย", ["วินิจฉัย", "สิ่งที่ตรวจพบ"], ["วินิจฉัย", "ตรวจ", "รู้ได้อย่างไร"]),
    ("complication", "ภาวะแทรกซ้อน", ["ภาวะแทรกซ้อน"], ["ภาวะแทรกซ้อน", "อันตราย", "ผลกระทบ", "เสี่ยงเป็น"]),
    ("risk", "ปัจจัยเสี่ยง", ["ปัจจัยเสี่ยง"], ["ปัจจัยเสี่ยง", "กำเริบ"]),
    ("cause", "สาเหตุ", ["สาเหตุ"], ["สาเหตุ", "เกิดจาก", "ติดต่อ", "เชื้อ"]),
    ("symptom", "อาการ", ["อาการ"], ["อาการ"]),
    ("prevention", "การป้องกัน", ["ป้องกัน"], ["ป้องกัน", "วัคซีน"]),
    ("treatment", "การรักษา", ["รักษา", "ผ่าตัด", "สมุนไพร"], ["รักษา", "ผ่าตัด"]),
    ("advice", "คำแนะนำ", ["คำแนะนำ", "พบแพทย์", "ข้อควรรู้"], ["ควรทำ", "พบแพทย์", "ปฏิบัติตัว"]),
    ("overview", "ภาพรวม", [], ["คืออะไร"]),
]
TOPIC_NAME = {t[0]: t[1] for t in TOPICS}  # แปลงรหัสหัวข้อ -> ชื่อไทย

# Semantic nodes ที่ผู้จัดทำคัดเลือกเอง (แนวเดียวกับบทความ) — คำที่เชื่อมเอกสารหลายเรื่องเข้าหากัน
CONCEPTS = {
    "pregnancy": ("การตั้งครรภ์", [r"ตั้งครรภ์", r"ทารกในครรภ์"]),
    "congenital": ("ความผิดปกติแต่กำเนิด", [r"แต่กำเนิด", r"congenital"]),
    "deafness": ("หูหนวก", [r"หูหนวก"]),
    "encephalitis": ("สมองอักเสบ", [r"สมองอักเสบ"]),
    "arthritis": ("ข้ออักเสบ", [r"ข้ออักเสบ"]),
    "measles": ("โรคหัด", [r"หัด(?!เยอรมัน|สามวัน)", r"(?<!german )(?<!three-day )measles"]),  # (?!...) = negative lookahead: 'หัด' ที่ไม่ใช่ 'หัดเยอรมัน'/'หัดสามวัน'
    "dengue": ("ไข้เลือดออก", [r"ไข้เลือดออก"]),
    "vaccine": ("วัคซีน", [r"วัคซีน", r"\bmmr\b", r"เอ็มเอ็มอาร์"]),
    "diabetes": ("เบาหวาน", [r"เบาหวาน"]),
    "hypertension": ("ความดันโลหิตสูง", [r"ความดันโลหิตสูง"]),
    "glaucoma": ("ต้อหิน", [r"ต้อหิน", r"glaucoma"]),
    "steroid": ("ยาสเตียรอยด์", [r"สเตียรอยด์"]),
    "smoking": ("บุหรี่", [r"บุหรี่"]),
    "alcohol": ("แอลกอฮอล์", [r"แอลกอฮอล์", r"เหล้า"]),
    "obesity": ("โรคอ้วน", [r"อ้วน", r"น้ำหนักเกิน"]),
    "dehydration": ("ภาวะขาดน้ำ", [r"ขาดน้ำ"]),
    "antibiotic": ("ยาปฏิชีวนะ", [r"ยาปฏิชีวนะ"]),
    "handwashing": ("การล้างมือ", [r"ล้างมือ"]),
    "esophagus": ("หลอดอาหาร", [r"หลอดอาหาร"]),
    "lens_implant": ("เลนส์แก้วตาเทียม", [r"แก้วตาเทียม"]),
    "surgery": ("การผ่าตัด", [r"ผ่าตัด"]),
    "antacid": ("ยาลดกรด", [r"ยาลดกรด"]),
    "cancer": ("มะเร็ง", [r"มะเร็ง"]),
}

# น้ำหนักของเส้นแต่ละประเภทตอนให้คะแนน (เส้นที่เจาะจงกว่าได้น้ำหนักมากกว่า)
EDGE_WEIGHT = {"ABOUT_TOPIC": 1.0, "MENTIONS_DISEASE": 0.8, "MENTIONS": 0.7, "HAS_TOPIC": 0.6}
HOP_DECAY = {1: 1.0, 2: 0.5}  # ยิ่งห่างจาก seed คะแนนยิ่งลด: hop ที่ 2 ได้ครึ่งเดียว


# รวม regex หลายตัวเป็นตัวเดียวด้วย | (หรือ) ไม่สนตัวพิมพ์เล็ก/ใหญ่ คอมไพล์ครั้งเดียวแล้วใช้ซ้ำ
def _compile(patterns):
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)


# คอมไพล์ regex ของทุกโรคและทุกคำเชื่อมไว้ล่วงหน้า
DISEASE_RE = {k: _compile(v["patterns"]) for k, v in DISEASES.items()}
CONCEPT_RE = {k: _compile(v[1]) for k, v in CONCEPTS.items()}


# หัวข้อ = บรรทัดแรกที่ไม่ว่าง ตัดเครื่องหมาย # ออก
def _heading(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip().lstrip("#").strip()
    return ""


# จัดประเภทหัวข้อของ section ด้วยกฎ (rule-based)
# section แรกของไฟล์ = ภาพรวม, นอกนั้นดูว่าหัวข้อมีคำใน TOPICS ไหม (ตัวแรกที่ตรงชนะ)
# ไม่ตรงเลย = ภาพรวม
def classify_topic(heading: str, is_doc_title: bool) -> str:
    if is_doc_title:
        return "overview"
    for code, _name, heading_words, _q in TOPICS:
        if any(w in heading for w in heading_words):
            return code
    return "overview"


# ---------------------------------------------------------------------------
# สร้างกราฟ
# ---------------------------------------------------------------------------

def build_graph(sections: list[dict]) -> nx.DiGraph:
    """sections = ผลจาก rag_improved.load_sections() (doc_0..doc_52 ตรงกับระบบเดิม)"""
    # DiGraph = กราฟมีทิศทาง (เส้นมีหัวลูกศร)
    # โครงสร้างกราฟ (schema):
    #   Disease --HAS_TOPIC--> Topic <--ABOUT_TOPIC-- Section --MENTIONS_DISEASE--> Disease (โรคอื่น)
    #                                                Section --MENTIONS--> Concept
    G = nx.DiGraph()
    source_to_disease = {v["source"]: k for k, v in DISEASES.items()}  # ไฟล์ -> โรค
    # 1) สร้างโหนดโรค และโหนดคำเชื่อม (concept)
    for key, d in DISEASES.items():
        G.add_node(f"disease:{key}", label="Disease", name=d["name"])
    for key, (name, _p) in CONCEPTS.items():
        G.add_node(f"concept:{key}", label="Concept", name=name)

    # หา section แรกของแต่ละไฟล์ (= หัวเรื่อง/ภาพรวม)
    first_of_source = {}
    for s in sections:
        first_of_source.setdefault(s["source"], s["section_id"])

    # 2) สร้างโหนด section ทีละชิ้น แล้วเชื่อมเส้นตามกฎ
    for s in sections:
        sid, text = s["section_id"], s["text"]
        disease = source_to_disease.get(s["source"])
        if disease is None:
            continue
        heading = _heading(text)
        topic = classify_topic(heading, first_of_source[s["source"]] == sid)
        # โหนดหัวข้อแยกตามโรค เช่น topic:rubella:symptom (อาการของหัดเยอรมัน)
        # ถ้ายังไม่มี ให้สร้างและเชื่อมจากโหนดโรค
        topic_node = f"topic:{disease}:{topic}"
        if topic_node not in G:
            G.add_node(topic_node, label="Topic", name=f"{TOPIC_NAME[topic]} ({DISEASES[disease]['name']})",
                       disease=disease, topic=topic)
            G.add_edge(f"disease:{disease}", topic_node, relationship="HAS_TOPIC")
        # โหนด section เชื่อมไปหาหัวข้อของตัวเอง
        sec_node = f"section:{sid}"
        G.add_node(sec_node, label="Section", name=heading, section_id=sid, disease=disease, topic=topic)
        G.add_edge(sec_node, topic_node, relationship="ABOUT_TOPIC")

        # เส้นข้ามเอกสาร: section ที่เอ่ยถึงโรคอื่น
        for other, rx in DISEASE_RE.items():
            n = len(rx.findall(text)) if other != disease else 0  # ไม่นับโรคของเอกสารตัวเอง
            if n:
                G.add_edge(sec_node, f"disease:{other}", relationship="MENTIONS_DISEASE", count=n)
        # section ที่เอ่ยถึงคำเชื่อม เช่น 'ตั้งครรภ์' -> เชื่อมกับ concept นั้น (count = จำนวนครั้งที่เอ่ย)
        for key, rx in CONCEPT_RE.items():
            n = len(rx.findall(text))
            if n:
                G.add_edge(sec_node, f"concept:{key}", relationship="MENTIONS", count=n)

    # ตัด concept ที่ไม่มี section ใดเอ่ยถึง
    G.remove_nodes_from([n for n in list(G) if n.startswith("concept:") and G.degree(n) == 0])
    return G


# สถิติของกราฟ ใช้ตรวจความถูกต้องและแสดงในรายงาน/หน้าเว็บ
def graph_stats(G: nx.DiGraph) -> dict:
    labels, rels = {}, {}
    for _n, d in G.nodes(data=True):
        labels[d["label"]] = labels.get(d["label"], 0) + 1
    for _a, _b, d in G.edges(data=True):
        rels[d["relationship"]] = rels.get(d["relationship"], 0) + 1
    # เส้นข้ามเอกสาร = section ที่เอ่ยถึงโรคของเอกสารอื่น
    cross = [(a, b) for a, b, d in G.edges(data=True) if d["relationship"] == "MENTIONS_DISEASE"]
    # 5 concept ที่เชื่อมกับ section มากที่สุด
    hubs = sorted(((n, G.degree(n)) for n, d in G.nodes(data=True) if d["label"] == "Concept"),
                  key=lambda x: -x[1])[:5]
    return {
        "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
        "nodes_by_label": labels, "edges_by_relationship": rels,
        "weakly_connected": nx.is_weakly_connected(G),  # ทุกโหนดเชื่อมถึงกันหมด (ไม่สนทิศทาง) หรือไม่
        "average_degree": round(2 * G.number_of_edges() / max(G.number_of_nodes(), 1), 2),  # จำนวนเส้นเฉลี่ยต่อโหนด
        "cross_document_edges": len(cross),
        "concept_hubs": [{"node": n, "name": G.nodes[n]["name"], "degree": deg} for n, deg in hubs],
    }


# บันทึกกราฟเป็น JSON (nodes + edges) ให้เปิดดูหรือนำไปวาดภาพได้
def save_graph(G: nx.DiGraph, path: Path = GRAPH_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"stats": graph_stats(G),
            "nodes": [{"id": n, **d} for n, d in G.nodes(data=True)],
            "edges": [{"source": a, "target": b, **d} for a, b, d in G.edges(data=True)]}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------------------
# ค้นด้วยกราฟ
# ---------------------------------------------------------------------------

# ตัวค้นด้วยกราฟ: หา seed จากคำถาม -> เดิน BFS ไม่เกิน 2 hop -> ให้คะแนน section ที่เดินถึง
class GraphRetriever:
    def __init__(self, G: nx.DiGraph):
        self.G = G
        self.U = G.to_undirected(as_view=True)   # เดินได้ทั้งขาเข้าและขาออก เหมือนบทความ

    # สร้าง retriever จาก sections ได้ในบรรทัดเดียว
    @classmethod
    def from_sections(cls, sections) -> "GraphRetriever":
        return cls(build_graph(sections))

    def find_seeds(self, query: str, dense_section_ids: list[str] | None = None) -> list[tuple[str, float, str]]:
        """คืนรายการ (node, น้ำหนัก, เหตุผล)"""
        # หา keyword สองระดับในคำถาม (แนวคิด LightRAG)
        #   low-level  = ชื่อโรค และคำเชื่อม (concept)
        #   high-level = ประเภทหัวข้อ เช่น อาการ สาเหตุ การรักษา
        q = query.lower()
        diseases = [k for k, rx in DISEASE_RE.items() if rx.search(q)]
        topics = [code for code, _n, _h, qwords in TOPICS if any(w in q for w in qwords)]
        concepts = [k for k, rx in CONCEPT_RE.items() if rx.search(q) and f"concept:{k}" in self.G]

        # สร้างรายการ seed พร้อมน้ำหนัก (ยิ่งเจาะจง น้ำหนักยิ่งมาก)
        seeds: list[tuple[str, float, str]] = []
        for d in diseases:           # high-level × low-level: หัวข้อของโรคที่ถาม เจาะจงที่สุด
            for t in topics:
                node = f"topic:{d}:{t}"
                if node in self.G:
                    seeds.append((node, 1.5, f"คำถามถามเรื่อง{TOPIC_NAME[t]}ของ{DISEASES[d]['name']}"))
        for d in diseases:  # seed: โหนดโรคที่ถาม (น้ำหนัก 1.0)
            seeds.append((f"disease:{d}", 1.0, f"คำถามเอ่ยถึง{DISEASES[d]['name']}"))
        for c in concepts:  # seed: คำเชื่อมที่ถาม (น้ำหนัก 1.0)
            seeds.append((f"concept:{c}", 1.0, f"คำถามเอ่ยถึง{CONCEPTS[c][0]}"))
        for sid in (dense_section_ids or [])[:3]:   # vector หาจุดเริ่มต้น (แนวคิดหลักของบทความ)
            node = f"section:{sid}"
            if node in self.G:
                seeds.append((node, 0.8, "vector search เจอเป็นอันดับต้น"))

        # ตัด seed ซ้ำ (เก็บตัวแรก ซึ่งน้ำหนักสูงสุด) และจำกัดไม่เกิน MAX_SEEDS
        seen, unique = set(), []
        for s in seeds:
            if s[0] not in seen:
                seen.add(s[0])
                unique.append(s)
        return unique[:MAX_SEEDS]

    def search(self, query: str, dense_section_ids: list[str] | None = None,
               k: int = MAX_SECTIONS) -> list[dict]:
        seeds = self.find_seeds(query, dense_section_ids)
        # scores = คะแนนรวมของแต่ละ section, paths = เส้นทางที่ดีที่สุดที่เดินมาถึง section นั้น
        scores: dict[str, float] = {}
        paths: dict[str, tuple[float, str]] = {}
        for seed, w, why in seeds:
            # ถ้า seed เป็น section เอง (มาจาก vector) ให้คะแนนตรง ๆ
            if self.G.nodes[seed]["label"] == "Section":
                sid = self.G.nodes[seed]["section_id"]
                scores[sid] = scores.get(sid, 0) + w
                paths.setdefault(sid, (w, f"[seed] {why}"))
            # BFS ไม่เกิน MAX_HOPS hop เก็บ path ที่ให้คะแนนสูงสุดของแต่ละโหนด
            best = {seed: (w, [])}  # best[โหนด] = (คะแนนดีสุด, เส้นทาง)
            queue = deque([(seed, 0)])  # คิวเก็บ (โหนด, จำนวน hop)
            while queue:
                cur, hop = queue.popleft()
                if hop >= MAX_HOPS:  # เดินครบ 2 hop แล้ว หยุด
                    continue
                # ดูเพื่อนบ้านทุกตัว (กราฟไม่มีทิศ = เดินได้ทั้งขาเข้าและขาออก)
                for nb in self.U.neighbors(cur):
                    edge = self.G.edges.get((cur, nb)) or self.G.edges.get((nb, cur))  # หาเส้นจริงในกราฟมีทิศ (อาจเป็น cur->nb หรือ nb->cur)
                    rel = edge["relationship"]
                    # section ที่เอ่ยถึงคำนั้นบ่อยกว่าได้น้ำหนักมากขึ้นเล็กน้อย (สูงสุด +25%)
                    freq = 1 + 0.05 * min(edge.get("count", 1) - 1, 5)
                    # คะแนนของเพื่อนบ้าน = คะแนนโหนดปัจจุบัน × น้ำหนักเส้น × ความถี่ × การลดตาม hop
                    val = best[cur][0] * EDGE_WEIGHT.get(rel, 0.5) * freq * HOP_DECAY[hop + 1]
                    if nb in best and best[nb][0] >= val:  # เคยมาถึงด้วยคะแนนดีกว่าแล้ว ไม่ต้องเดินซ้ำ
                        continue
                    best[nb] = (val, best[cur][1] + [(rel, nb)])
                    queue.append((nb, hop + 1))
            # สะสมคะแนนให้ section ที่เดินถึง (seed หลายตัวเดินถึง section เดียวกัน คะแนนบวกกัน)
            for node, (val, path) in best.items():
                if node == seed or self.G.nodes[node]["label"] != "Section":
                    continue
                sid = self.G.nodes[node]["section_id"]
                scores[sid] = scores.get(sid, 0) + val
                if val > paths.get(sid, (0, ""))[0]:
                    paths[sid] = (val, self._describe(seed, path, why))

        # เรียงคะแนนมากไปน้อย เอา k อันดับแรก
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:k]
        return [{"section_id": sid, "score": round(sc, 3), "path": paths[sid][1]} for sid, sc in ranked]

    def _describe(self, seed: str, path: list[tuple[str, str]], why: str) -> str:
        """สร้างป้ายแบบบทความ เช่น [hop1][MENTIONS_DISEASE] ต้อกระจก"""
        hop = len(path)
        via = " → ".join(f"{rel}" for rel, _n in path)
        return f"[hop{hop}][{via}] จาก {self.G.nodes[seed]['name']} ({why})"


# ---------------------------------------------------------------------------
# รันเดี่ยว: สร้าง ตรวจ และบันทึกกราฟ
# ---------------------------------------------------------------------------

def main() -> None:
    import rag_improved as R
    sections, _full = R.load_sections()
    # สร้างกราฟ คำนวณสถิติ และบันทึกไฟล์
    G = build_graph(sections)
    stats = graph_stats(G)
    save_graph(G)

    print(f"กราฟ: {stats['nodes']} โหนด {stats['edges']} เส้น (average degree {stats['average_degree']})")
    print("โหนดตามประเภท:", stats["nodes_by_label"])
    print("เส้นตามประเภท:", stats["edges_by_relationship"])
    print(f"เส้นข้ามเอกสาร (MENTIONS_DISEASE): {stats['cross_document_edges']}")

    # ตรวจความถูกต้องแบบบทความ: นับ section ให้ครบ และกราฟต้องเชื่อมกันเป็นก้อนเดียว
    n_sections = stats["nodes_by_label"].get("Section", 0)
    print(f"ตรวจ: section ครบ {n_sections}/{len(sections)} {'ผ่าน' if n_sections == len(sections) else 'ไม่ผ่าน'}")
    print(f"ตรวจ: เชื่อมเป็นก้อนเดียว (weakly connected) {'ผ่าน' if stats['weakly_connected'] else 'ไม่ผ่าน'}")
    print("concept ที่เชื่อมมากที่สุด:", ", ".join(f"{h['name']} ({h['degree']})" for h in stats["concept_hubs"]))

    print("\nตัวอย่างเส้นข้ามเอกสาร:")
    for a, b, d in G.edges(data=True):
        if d["relationship"] == "MENTIONS_DISEASE":
            print(f"  {a} ({G.nodes[a]['name'][:40]}) → {G.nodes[b]['name']}")

    # ทดลองค้นด้วยคำถาม multi-hop ที่ต้องเชื่อม 2 เอกสาร (หัดเยอรมัน + ต้อกระจก)
    q = "ถ้าแม่ติดเชื้อหัดเยอรมันระหว่างตั้งครรภ์ ลูกเสี่ยงเป็นต้อกระจกไหม และต้อกระจกในทารกรักษาอย่างไร"
    print(f"\nทดลองค้น: {q}")
    for r in GraphRetriever(G).search(q)[:6]:
        print(f"  {r['section_id']:<7} {r['score']:<6} {r['path']}")
    print(f"\nบันทึกกราฟที่ {GRAPH_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
