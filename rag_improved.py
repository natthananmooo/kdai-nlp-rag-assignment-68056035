"""
rag_improved.py — แกนหลักของระบบ RAG ฉบับปรับปรุง (งานข้อ 2)

ไฟล์นี้รวมทุกอย่างที่ระบบปรับปรุงใช้ และมีระบบเดิม (Baseline) จำลองไว้ด้วย
เพื่อให้สคริปต์วัดผลและหน้าเว็บเรียกใช้ทั้งสองระบบได้โดยโหลดโมเดลชุดเดียว

สิ่งที่ปรับปรุงจากโค้ดต้นฉบับ (original/authenticRAG.py, original/onlysearchAuthenticRAG.py)
  A  Chunking + Contextual Retrieval
     A1 สร้างบริบทของแต่ละ chunk จาก "เอกสารเต็ม" ตามวิธีของ Anthropic
        (ต้นฉบับส่ง chunk ตัวเองแทนเอกสารเต็ม: full_doc_content = doc.page_content)
     A2 ให้ LLM เขียนบริบทเป็นภาษาไทยและระบุชื่อโรค คำถามภาษาไทยจึงค้นเจอด้วย BM25
     A3 แบ่ง section ที่ยาวเกิน (สูงสุด 8,404 ตัวอักษร) เป็นชิ้นย่อยไม่เกิน ~1,000 ตัวอักษร
        แต่ละชิ้นติดหัวข้อเรื่องไว้ข้างหน้า และใช้บริบทของ section ร่วมกัน
        (จำนวนครั้งที่เรียก LLM จึงเท่าต้นฉบับ คือ 1 ครั้งต่อ section)
  B  BM25 ใช้ Thai analyzer ของ OpenSearch แทน standard analyzer
  C  เก็บบริบทใน vector index ด้วย บริบทจึงไม่หายเมื่อผลลัพธ์มาจากฝั่ง vector
  D  ดึงผู้สมัคร 20 ชิ้นด้วย Hybrid + RRF แล้ว rerank ด้วย cross-encoder
     BAAI/bge-reranker-v2-m3 คัดเหลือ 5 ชิ้น
  D2 Relevance filtering: ตัดชิ้นที่คะแนน rerank ต่ำกว่า threshold ทิ้ง
     ถ้าไม่เหลือเลย ตอบว่าเอกสารไม่มีข้อมูลโดยไม่เรียก LLM
     (แนวคิดจาก jev-reranker ในบทความที่อาจารย์แนะนำ ซึ่งเรียกโมเดลผ่าน API และต้องใช้ API key
      งานนี้ใช้ bge-reranker-v2-m3 ที่รันบนเครื่องแทน)
  GR GraphRAG (graph_rag.py): knowledge graph แบบ rule-based เดินด้วย BFS 2 hop
     แล้วใช้ผลเป็นตัวค้นที่ 3 ใน RRF คู่กับ BM25 และ vector (ตามบทความของอาจารย์)
  G  Prompt ตอบคำถาม: ให้ตอบจากเอกสารเท่านั้น เป็นภาษาไทย อ้างอิงแหล่ง
     และเพิ่ม max_tokens กันคำตอบถูกตัด (LLM ยังเป็นตัวเดิม ตามเงื่อนไขโจทย์)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# ค่าตั้งต้น (เปลี่ยนได้ผ่าน environment variable)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
CORPUS_PATHS = [
    ROOT / "corpus_input" / "1.md",
    ROOT / "corpus_input" / "2.md",
    ROOT / "corpus_input" / "44.md",
    ROOT / "corpus_input" / "5555.md",
]

# LLM ตัวเดียวกับต้นฉบับ (โจทย์ข้อ 2 ห้ามเปลี่ยน LLM ที่ใช้สร้างคำตอบ)
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen/qwen-2.5-72b-instruct")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
USE_RERANKER = os.environ.get("USE_RERANKER", "1") != "0"
# GR GraphRAG: ใช้ knowledge graph เป็นตัวค้นที่ 3 ใน RRF (ดู graph_rag.py)
USE_GRAPH = os.environ.get("USE_GRAPH", "1") != "0"
GRAPH_CANDIDATES = 20
# D2 relevance filtering: เก็บเฉพาะชิ้นที่คะแนน rerank >= ค่านี้ (0 = ไม่กรอง)
# แนวคิดจาก jev-reranker (relevance_rerank) แต่ใช้กับ bge-reranker ที่รันบนเครื่องฟรี
RERANK_THRESHOLD = float(os.environ.get("RERANK_THRESHOLD", "0.1"))
THRESHOLD_SWEEP = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
NO_EVIDENCE_ANSWER = ("ไม่พบข้อมูลที่ตอบคำถามนี้ในเอกสารที่มี "
                      "(ไม่มีชิ้นใดผ่านเกณฑ์ความเกี่ยวข้องของ reranker จึงไม่ได้ส่งให้ LLM ตอบ)")

OPENSEARCH_HOST = os.environ.get("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.environ.get("OPENSEARCH_PORT", "9200"))

# index ของระบบเดิม (สร้างโดย original/authenticRAG.py) — อ่านอย่างเดียว ไม่แก้ไข
BASELINE_VECTOR_INDEX = "anthropic-vector-index"
BASELINE_BM25_INDEX = "anthropic-bm25-index"
# index ของระบบปรับปรุง — แยกชื่อ ไม่ทับของเดิม
IMPROVED_VECTOR_INDEX = "improved-vector-index"
IMPROVED_BM25_INDEX = "improved-bm25-index"

RRF_K = 60                 # ค่าเดียวกับต้นฉบับ
FINAL_K = 5                # ส่งให้ LLM 5 ชิ้นเท่าต้นฉบับ (เทียบกันได้ยุติธรรม)
CANDIDATES_PER_RETRIEVER = 20
RERANK_POOL = 20
MAX_CHUNK_CHARS = 1000
MAX_DOC_CHARS = 30000      # กันเอกสารยาวเกิน context ของ LLM
CONTEXT_CACHE = ROOT / "results" / "contexts_cache.json"

_CJK = re.compile(r"[一-鿿぀-ヿ]")  # อักษรจีน/ญี่ปุ่น


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# A: Chunking
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: str          # รหัสชิ้นในระบบปรับปรุง เช่น doc_19_p2
    section_id: str        # รหัส section ตามต้นฉบับ เช่น doc_19 (ใช้เทียบผล)
    source: str
    doc_title: str
    section_title: str
    part: int
    parts: int
    content: str
    context: str = ""

    @property
    def title(self) -> str:
        suffix = f" (ส่วนที่ {self.part}/{self.parts})" if self.parts > 1 else ""
        if self.section_title == self.doc_title:
            return f"{self.doc_title}{suffix}"
        return f"{self.doc_title} › {self.section_title}{suffix}"

    def text_for_embedding(self) -> str:
        return f"{self.title}\n{self.context}\n\n{self.content}"

    def text_for_rerank(self) -> str:
        return f"{self.title}\n{self.context}\n{self.content}"


def _first_heading(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line.lstrip("#").strip()[:120]
    return "(ไม่มีหัวข้อ)"


def load_sections(paths: list[Path] | None = None):
    """อ่านไฟล์และแบ่ง section ด้วย MarkdownNodeParser แบบเดียวกับต้นฉบับ
    ผลจึงตรงกับ doc_0..doc_52 ของระบบเดิมทุกชิ้น ใช้เทียบผลข้ามระบบได้"""
    from llama_index.core import Document
    from llama_index.core.node_parser import MarkdownNodeParser

    paths = paths or CORPUS_PATHS
    full_docs: dict[str, str] = {}
    docs = []
    for p in paths:
        text = Path(p).read_text(encoding="utf-8")
        source = f"./corpus_input/{Path(p).name}"
        full_docs[source] = text
        docs.append(Document(text=text, metadata={"source": source}))
    nodes = MarkdownNodeParser(chunk_size=256).get_nodes_from_documents(docs)
    sections = []
    for i, n in enumerate(nodes):
        sections.append({
            "section_id": f"doc_{i}",
            "source": n.metadata.get("source", ""),
            "text": n.text,
        })
    return sections, full_docs


def split_section(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """แบ่งตามย่อหน้า ไม่ตัดกลางประโยค ย่อหน้าที่ยาวเกินเองจะแบ่งตามบรรทัด"""
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n-{3,}\n", text) if p.strip()]
    units: list[str] = []
    for p in paras:
        if len(p) <= max_chars:
            units.append(p)
            continue
        buf = ""
        for line in p.splitlines():
            if buf and len(buf) + len(line) + 1 > max_chars:
                units.append(buf)
                buf = ""
            buf = f"{buf}\n{line}" if buf else line
        if buf:
            units.append(buf)

    pieces: list[str] = []
    buf = ""
    for u in units:
        if buf and len(buf) + len(u) + 2 > max_chars:
            pieces.append(buf)
            buf = ""
        buf = f"{buf}\n\n{u}" if buf else u
    if buf:
        pieces.append(buf)
    return pieces or [text]


def build_chunks(sections, full_docs) -> list[Chunk]:
    chunks: list[Chunk] = []
    for s in sections:
        doc_title = _first_heading(full_docs[s["source"]])
        section_title = _first_heading(s["text"])
        pieces = split_section(s["text"])
        for j, piece in enumerate(pieces, 1):
            cid = s["section_id"] if len(pieces) == 1 else f"{s['section_id']}_p{j}"
            chunks.append(Chunk(cid, s["section_id"], s["source"], doc_title,
                                section_title, j, len(pieces), piece))
    return chunks


# ---------------------------------------------------------------------------
# LLM (ผ่าน OpenRouter เหมือนต้นฉบับ)
# ---------------------------------------------------------------------------

class OutOfCredits(RuntimeError):
    """เครดิต OpenRouter ไม่พอ (HTTP 402)"""


class LLM:
    def __init__(self, model: str = LLM_MODEL):
        from openai import OpenAI
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("ยังไม่ได้ตั้ง OPENROUTER_API_KEY "
                               "ให้รัน: export OPENROUTER_API_KEY='sk-or-v1-...'")
        self.model = model
        self.client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")

    def chat(self, prompt: str, max_tokens: int, temperature: float,
             system: str = "You are a helpful assistant.", retries: int = 4) -> dict:
        last = None
        for attempt in range(1, retries + 1):
            try:
                r = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                choice = r.choices[0]
                return {"text": (choice.message.content or "").strip(),
                        "finish_reason": choice.finish_reason}
            except Exception as e:  # noqa: BLE001
                last = e
                status = getattr(e, "status_code", None)
                if status == 402:
                    # เครดิตไม่พอ: ลองซ้ำไม่ช่วย หยุดเลยพร้อมบอกวิธีแก้
                    raise OutOfCredits(
                        "เครดิต OpenRouter ไม่พอ เติมได้ที่ https://openrouter.ai/settings/credits "
                        "แล้วรันคำสั่งเดิมอีกครั้ง (บริบทที่สร้างเสร็จแล้วเก็บใน cache ไม่เสียเงินซ้ำ)") from e
                if status in (400, 401, 403, 404):
                    # คำขอผิดหรือ key ผิด: ลองซ้ำก็ได้ผลเดิม
                    raise RuntimeError(f"OpenRouter ปฏิเสธคำขอ (HTTP {status}): {str(e)[:300]}") from e
                wait = self._retry_after(e) or (5, 15, 30, 60)[min(attempt, 4) - 1]
                log(f"  ! เรียก LLM ไม่สำเร็จ (ครั้งที่ {attempt}/{retries}, HTTP {status}): "
                    f"{str(e)[:160]} ... รอ {wait} วินาทีแล้วลองใหม่")
                if attempt < retries:
                    time.sleep(wait)
        raise RuntimeError(f"เรียก LLM ไม่สำเร็จหลังลอง {retries} ครั้ง: {str(last)[:300]}")

    @staticmethod
    def _retry_after(e) -> int | None:
        """อ่านเวลาที่ OpenRouter บอกให้รอ (header Retry-After) จำกัดไม่เกิน 120 วินาที"""
        try:
            value = e.response.headers.get("retry-after")
            return min(int(float(value)), 120) if value else None
        except Exception:  # noqa: BLE001
            return None


CONTEXT_PROMPT = """<document>
{document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Write the context in Thai (1-2 sentences), name the disease the document is about, and mention the main topic of the chunk.
Answer only with the succinct context and nothing else."""


def _document_window(doc: str, chunk: str, limit: int = MAX_DOC_CHARS) -> str:
    """ใช้เอกสารเต็มถ้าไม่ยาวเกิน ถ้ายาวเกินให้ตัดช่วงรอบ ๆ chunk"""
    if len(doc) <= limit:
        return doc
    pos = max(doc.find(chunk[:200]), 0)
    start = max(0, min(pos - limit // 2, len(doc) - limit))
    return doc[start:start + limit]


def generate_section_context(llm: LLM, full_doc: str, section_text: str) -> str:
    prompt = CONTEXT_PROMPT.format(document=_document_window(full_doc, section_text),
                                   chunk=section_text)
    for _ in range(2):
        out = llm.chat(prompt, max_tokens=200, temperature=0.1)["text"]
        if out and not _CJK.search(out):
            return out
        log("  ! บริบทมีอักษรจีนหรือว่างเปล่า ลองใหม่")
    return _CJK.sub("", out).strip()


def _cache_key(source: str, text: str) -> str:
    return hashlib.sha1(f"{LLM_MODEL}|{source}|{text}".encode("utf-8")).hexdigest()


def load_context_cache() -> dict:
    if CONTEXT_CACHE.exists():
        return json.loads(CONTEXT_CACHE.read_text(encoding="utf-8"))
    return {}


def save_context_cache(cache: dict) -> None:
    CONTEXT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    CONTEXT_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------------------
# โมเดล embedding และ reranker (โหลดครั้งเดียว ใช้ร่วมกันทั้งสองระบบ)
# ---------------------------------------------------------------------------

class Models:
    def __init__(self, load_reranker: bool = USE_RERANKER):
        from sentence_transformers import SentenceTransformer
        log(f"กำลังโหลด embedding model {EMBED_MODEL} ...")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.dim = self.embedder.get_sentence_embedding_dimension()
        self._reranker = None
        self.want_reranker = load_reranker

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self.embedder.encode(texts, batch_size=8, normalize_embeddings=True,
                                    show_progress_bar=len(texts) > 16)
        return [v.tolist() for v in vecs]

    def embed_raw(self, text: str) -> list[float]:
        """แบบเดียวกับต้นฉบับทุกประการ ใช้กับระบบเดิม"""
        return self.embedder.encode(text).tolist()

    @property
    def reranker(self):
        if self._reranker is None and self.want_reranker:
            from sentence_transformers import CrossEncoder
            log(f"กำลังโหลด reranker {RERANK_MODEL} ...")
            self._reranker = CrossEncoder(RERANK_MODEL, max_length=512)
        return self._reranker


def opensearch_client():
    from opensearchpy import OpenSearch
    return OpenSearch(hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
                      use_ssl=False, timeout=60)


# ---------------------------------------------------------------------------
# RRF (สูตรเดียวกับต้นฉบับ: 1/(k + rank) โดย rank เริ่มที่ 0)
# ---------------------------------------------------------------------------

def rrf_fusion(rankings: list[list[tuple[str, float, dict]]], k: int = RRF_K,
               merge_sources: bool = False):
    """merge_sources=False จำลองต้นฉบับ: เก็บ _source จากผลชุดแรกที่เจอเท่านั้น
    merge_sources=True (ระบบปรับปรุง): รวม field จากทั้งสอง index"""
    scores: dict[str, float] = {}
    sources: dict[str, dict] = {}
    ranks: dict[str, dict[int, int]] = {}
    for rid, results in enumerate(rankings):
        for rank, (doc_id, _score, src) in enumerate(results):
            if doc_id not in scores:
                scores[doc_id] = 0.0
                sources[doc_id] = dict(src)
            elif merge_sources:
                for kk, vv in src.items():
                    if vv and not sources[doc_id].get(kk):
                        sources[doc_id][kk] = vv
            scores[doc_id] += 1.0 / (k + rank)
            ranks.setdefault(doc_id, {})[rid] = rank + 1
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(d, s, sources[d], ranks[d]) for d, s in ordered]


# ---------------------------------------------------------------------------
# ระบบเดิม (Baseline) — จำลองจาก original/onlysearchAuthenticRAG.py ทุกขั้น
# ---------------------------------------------------------------------------

class BaselineRAG:
    name = "ระบบเดิม (ข้อ 1)"

    def __init__(self, models: Models, client=None, llm: LLM | None = None):
        self.models, self.client, self.llm = models, client or opensearch_client(), llm

    def available(self) -> bool:
        return self.client.indices.exists(index=BASELINE_VECTOR_INDEX) and \
            self.client.indices.exists(index=BASELINE_BM25_INDEX)

    def search(self, query: str, k: int = FINAL_K) -> dict:
        t0 = time.time()
        sparse = self.client.search(index=BASELINE_BM25_INDEX, body={
            "query": {"multi_match": {"query": query,
                                      "fields": ["content", "contextualized_content"],
                                      "type": "best_fields"}},
            "size": k * 2})["hits"]["hits"]
        qv = self.models.embed_raw(query)
        dense = self.client.search(index=BASELINE_VECTOR_INDEX, body={
            "query": {"knn": {"embedding": {"vector": qv, "k": k * 2}}},
            "size": k * 2})["hits"]["hits"]
        fused = rrf_fusion([[(h["_id"], h["_score"], h["_source"]) for h in sparse],
                            [(h["_id"], h["_score"], h["_source"]) for h in dense]])
        results = []
        for rank, (doc_id, score, src, rr) in enumerate(fused[:k], 1):
            results.append({
                "rank": rank, "id": doc_id, "section_id": doc_id,
                "title": _first_heading(src.get("content", "")),
                "content": src.get("content", ""),
                "context": src.get("contextualized_content", ""),
                "rrf": score, "sparse_rank": rr.get(0), "dense_rank": rr.get(1),
                "rerank_score": None,
            })
        return {"results": results, "sparse_count": len(sparse), "dense_count": len(dense),
                "candidates": len(fused), "seconds": round(time.time() - t0, 2)}

    def answer(self, question: str, results: list[dict]) -> dict:
        """prompt และพารามิเตอร์เดียวกับต้นฉบับ (temperature 0.6, max_tokens 1000)"""
        context = "".join(f"Content: {r['content']}\nContext: {r['context']}\n\n" for r in results)
        prompt = ("You are a question answering assistant. Answer the question as truthful "
                  f"and helpful as possible.\n\nContext information:\n{context}\n\nQuestion: {question}")
        return self.llm.chat(prompt, max_tokens=1000, temperature=0.6)


# ---------------------------------------------------------------------------
# ระบบปรับปรุง
# ---------------------------------------------------------------------------

THAI_TEXT = {"type": "text", "analyzer": "thai"}


def improved_index_bodies(dim: int) -> tuple[dict, dict]:
    bm25 = {
        "settings": {"index": {"number_of_replicas": 0},
                     "similarity": {"default": {"type": "BM25"}}},
        "mappings": {"properties": {
            "chunk_id": {"type": "keyword"}, "section_id": {"type": "keyword"},
            "source": {"type": "keyword"},
            "title": THAI_TEXT, "content": THAI_TEXT, "contextualized_content": THAI_TEXT,
        }},
    }
    # ตั้งค่า kNN แบบเดียวกับต้นฉบับ (รันได้แน่นอนบน OpenSearch 2.19.1 ของเครื่องนี้)
    vector = {
        "settings": {"index.knn": True, "index.knn.space_type": "cosinesimil",
                     "index.number_of_replicas": 0},
        "mappings": {"properties": {
            "embedding": {"type": "knn_vector", "dimension": dim},
            "chunk_id": {"type": "keyword"}, "section_id": {"type": "keyword"},
            "source": {"type": "keyword"}, "title": {"type": "text"},
            "content": {"type": "text"}, "contextualized_content": {"type": "text"},
        }},
    }
    return bm25, vector


class ImprovedRAG:
    name = "ระบบปรับปรุง (ข้อ 2)"

    def __init__(self, models: Models, client=None, llm: LLM | None = None,
                 use_reranker: bool = USE_RERANKER, use_graph: bool = USE_GRAPH):
        self.models, self.client, self.llm = models, client or opensearch_client(), llm
        self.use_reranker = use_reranker
        self.use_graph = use_graph
        self._graph = None

    @property
    def graph(self):
        """สร้างกราฟจากไฟล์เอกสารครั้งแรกที่ใช้ (ใช้เวลาไม่ถึงวินาที ไม่เรียก LLM)"""
        if self._graph is None:
            import graph_rag
            sections, _full = load_sections()
            self._graph = graph_rag.GraphRetriever.from_sections(sections)
        return self._graph

    def _graph_ranking(self, query: str, dense) -> list[tuple[str, float, dict]]:
        """GR: กราฟจัดอันดับ section แล้วแปลงเป็นอันดับของ chunk เพื่อเข้า RRF"""
        dense_sections = []
        for _id, _s, src in dense:
            sid = src.get("section_id")
            if sid and sid not in dense_sections:
                dense_sections.append(sid)
        ranked = self.graph.search(query, dense_section_ids=dense_sections)
        if not ranked:
            return []
        order = {r["section_id"]: i for i, r in enumerate(ranked)}
        path = {r["section_id"]: r["path"] for r in ranked}
        body = {"query": {"terms": {"section_id": list(order)}}, "size": 200}
        hits = self.client.search(index=IMPROVED_BM25_INDEX, body=body)["hits"]["hits"]

        def part(cid: str) -> int:
            m = re.search(r"_p(\d+)$", cid)
            return int(m.group(1)) if m else 0

        hits.sort(key=lambda h: (order[h["_source"]["section_id"]], part(h["_id"])))
        out = []
        for h in hits[:GRAPH_CANDIDATES]:
            src = dict(h["_source"])
            src["graph_path"] = path[src["section_id"]]
            out.append((h["_id"], 0.0, src))
        return out

    def available(self) -> bool:
        return self.client.indices.exists(index=IMPROVED_VECTOR_INDEX) and \
            self.client.indices.exists(index=IMPROVED_BM25_INDEX)

    # ---- ค้นหา ----
    def _sparse(self, query: str, size: int):
        body = {"query": {"multi_match": {
            "query": query, "type": "best_fields", "tie_breaker": 0.3,
            "fields": ["title^2", "content", "contextualized_content"]}}, "size": size}
        hits = self.client.search(index=IMPROVED_BM25_INDEX, body=body)["hits"]["hits"]
        return [(h["_id"], h["_score"], h["_source"]) for h in hits]

    def _dense(self, query: str, size: int):
        qv = self.models.embed([query])[0]
        body = {"query": {"knn": {"embedding": {"vector": qv, "k": size}}}, "size": size,
                "_source": {"excludes": ["embedding"]}}
        hits = self.client.search(index=IMPROVED_VECTOR_INDEX, body=body)["hits"]["hits"]
        return [(h["_id"], h["_score"], h["_source"]) for h in hits]

    def search(self, query: str, k: int = FINAL_K, use_reranker: bool | None = None,
               threshold: float | None = None, use_graph: bool | None = None) -> dict:
        use_reranker = self.use_reranker if use_reranker is None else use_reranker
        use_graph = self.use_graph if use_graph is None else use_graph
        threshold = RERANK_THRESHOLD if threshold is None else threshold
        t0 = time.time()
        sparse = self._sparse(query, CANDIDATES_PER_RETRIEVER)
        dense = self._dense(query, CANDIDATES_PER_RETRIEVER)
        graph = self._graph_ranking(query, dense) if use_graph else []
        rankings = [sparse, dense] + ([graph] if use_graph else [])
        fused = rrf_fusion(rankings, merge_sources=True)[:RERANK_POOL]

        cands = []
        for doc_id, score, src, rr in fused:
            cands.append({
                "id": doc_id, "section_id": src.get("section_id", doc_id.split("_p")[0]),
                "title": src.get("title", ""), "content": src.get("content", ""),
                "context": src.get("contextualized_content", ""), "rrf": score,
                "sparse_rank": rr.get(0), "dense_rank": rr.get(1), "graph_rank": rr.get(2),
                "graph_path": src.get("graph_path", ""), "rerank_score": None,
            })

        reranker = self.models.reranker if use_reranker else None
        if reranker is not None and cands:
            pairs = [(query, f"{c['title']}\n{c['context']}\n{c['content']}") for c in cands]
            scores = reranker.predict(pairs, batch_size=4, show_progress_bar=False)
            for c, s in zip(cands, scores):
                c["rerank_score"] = float(s)
            cands.sort(key=lambda c: c["rerank_score"], reverse=True)

        filtered_out = 0
        if reranker is not None and threshold > 0:
            kept = [c for c in cands[:k] if c["rerank_score"] >= threshold]
            filtered_out = min(k, len(cands)) - len(kept)
            results = kept
        else:
            results = cands[:k]
        for i, r in enumerate(results, 1):
            r["rank"] = i
        return {"results": results, "pool": cands, "sparse_count": len(sparse),
                "dense_count": len(dense), "graph_count": len(graph), "graph_used": use_graph,
                "candidates": len(fused),
                "reranked": reranker is not None,
                "threshold": threshold if reranker is not None else None,
                "filtered_out": filtered_out, "seconds": round(time.time() - t0, 2)}

    # ---- ตอบคำถาม ----
    def answer(self, question: str, results: list[dict]) -> dict:
        if not results:   # D2: ไม่มีหลักฐานผ่านเกณฑ์ ไม่ต้องให้ LLM เดา
            return {"text": NO_EVIDENCE_ANSWER, "finish_reason": "no_evidence"}
        blocks = []
        for r in results:
            link = f"\nความสัมพันธ์ในกราฟ: {r['graph_path']}" if r.get("graph_path") else ""
            blocks.append(f"[{r['id']}] {r['title']}{link}\nบริบท: {r['context']}\nเนื้อหา:\n{r['content']}")
        prompt = (
            "ตอบคำถามโดยใช้ข้อมูลจากเอกสารด้านล่างเท่านั้น\n"
            "- ตอบเป็นภาษาไทยทั้งหมด ห้ามใช้ภาษาจีน\n"
            "- ใส่รหัสเอกสารที่ใช้ เช่น [doc_6] ท้ายประโยคที่อ้างอิง\n"
            "- ห้ามเพิ่มตัวเลขหรือข้อเท็จจริงที่ไม่มีในเอกสาร\n"
            "- ห้ามอนุมาน คาดเดา หรือสรุปเกินกว่าที่เอกสารเขียนไว้ตรง ๆ\n"
            "- ใส่รหัสเอกสารเฉพาะกับประโยคที่เอกสารนั้นเขียนไว้จริงเท่านั้น\n"
            "- ถ้าเอกสารไม่มีข้อมูลที่ตอบคำถามโดยตรง ให้ตอบสั้น ๆ ว่าเอกสารไม่ได้ระบุ แล้วหยุด "
            "ถ้ามีบางส่วน ให้ตอบเฉพาะส่วนที่เอกสารเขียนไว้ และบอกว่าส่วนใดเอกสารไม่ได้ระบุ\n"
            "- ส่วนที่เอกสารมีคำตอบ ให้ตอบครบทุกประเด็นที่เกี่ยวกับคำถาม อย่าตัดรายละเอียดที่มีในเอกสารทิ้ง\n"
            "- ถ้ามีหลายประเด็น ให้จัดเป็นหัวข้อหรือข้อย่อยให้อ่านง่าย\n"
            "- จบประโยคสมบูรณ์\n\n"
            "เอกสาร:\n" + "\n\n---\n\n".join(blocks) + f"\n\nคำถาม: {question}\nคำตอบ:"
        )
        out = self.llm.chat(prompt, max_tokens=2000, temperature=0.3,
                            system="You are a careful Thai medical information assistant. "
                                   "You only answer from the provided documents.")
        if _CJK.search(out["text"]):
            retry = self.llm.chat(prompt + "\n(ย้ำ: ตอบเป็นภาษาไทยเท่านั้น)", max_tokens=2000,
                                  temperature=0.2)
            if not _CJK.search(retry["text"]):
                out = retry
        return out


# ---------------------------------------------------------------------------
# ฟังก์ชันช่วยวัดผลข้อความคำตอบ
# ---------------------------------------------------------------------------

def answer_flags(text: str, finish_reason: str | None) -> dict:
    return {
        "chars": len(text),
        "truncated": finish_reason == "length",
        "no_evidence": finish_reason == "no_evidence",
        "has_cjk": bool(_CJK.search(text)),
        "citations": len(set(re.findall(r"\[(doc_\d+(?:_p\d+)?)\]", text))),
    }


def analyze_tokens(client, text: str, analyzer: str) -> list[str]:
    r = client.indices.analyze(body={"analyzer": analyzer, "text": text})
    return [t["token"] for t in r.get("tokens", [])]
