import json
import os
import re
import uuid
from typing import Any, Dict, List, Optional

import fitz
import numpy as np
import requests
import weaviate
from PIL import Image
from langchain_core.documents import Document
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter, MetadataQuery

from multimodal_models import MultiModalEmbedder


class VectorDB:
    """
    Weaviate-backed multimodal multi-vector index.

    - Ingestion: extracts text / tables / images from PDFs.
    - Embedding: produces multiple vectors per unit (text windows or image patches).
    - Retrieval: ANN candidate generation + exact late interaction (MaxSim) rerank.
    """

    def __init__(self, db_file: str = "db.json"):
        self.db_file = db_file
        self.collection_name = os.getenv("WEAVIATE_COLLECTION", "PaperMultiVector")
        self.embedder = MultiModalEmbedder(os.getenv("MM_EMBED_MODEL", "sentence-transformers/clip-ViT-B-32"))

        self.client = weaviate.connect_to_local(
            host=os.getenv("WEAVIATE_HOST", "localhost"),
            port=int(os.getenv("WEAVIATE_PORT", "8080")),
            grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
        )
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        collections = self.client.collections
        if collections.exists(self.collection_name):
            return

        collections.create(
            name=self.collection_name,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="paper_id", data_type=DataType.TEXT),
                Property(name="source_name", data_type=DataType.TEXT),
                Property(name="doc_id", data_type=DataType.TEXT),
                Property(name="subvector_id", data_type=DataType.TEXT),
                Property(name="modality", data_type=DataType.TEXT),
                Property(name="page", data_type=DataType.INT),
                Property(name="content", data_type=DataType.TEXT),
                Property(name="reference", data_type=DataType.TEXT),
                Property(name="image_path", data_type=DataType.TEXT),
                # Stored to run exact MaxSim rerank after ANN retrieval.
                Property(name="embedding_json", data_type=DataType.TEXT),
            ],
        )

    def load_db(self) -> Dict[str, Any]:
        if os.path.exists(self.db_file):
            with open(self.db_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save_db(self, data: Dict[str, Any]) -> None:
        with open(self.db_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def add_document_db(self, paper_id: str, object_ids: List[str]) -> None:
        db = self.load_db()
        db[paper_id] = {"ids": object_ids}
        self.save_db(db)

    def _safe_id(self, raw: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_.-]", "_", raw)

    def _split_text(self, text: str, chunk_size: int = 1400, overlap: int = 250) -> List[str]:
        text = re.sub(r"\s+", " ", (text or "")).strip()
        if not text:
            return []
        chunks = []
        i = 0
        while i < len(text):
            j = min(len(text), i + chunk_size)
            chunks.append(text[i:j])
            if j >= len(text):
                break
            i = max(0, j - overlap)
        return chunks

    def _extract_pdf_multimodal(self, pdf_path: str, image_dir: str = "uploads/images") -> List[Dict[str, Any]]:
        os.makedirs(image_dir, exist_ok=True)
        doc = fitz.open(pdf_path)
        base = self._safe_id(os.path.splitext(os.path.basename(pdf_path))[0])

        out: List[Dict[str, Any]] = []
        try:
            for pidx in range(len(doc)):
                page = doc[pidx]
                page_num = pidx + 1

                # text
                page_text = page.get_text("text")
                for chunk in self._split_text(page_text):
                    out.append({"modality": "text", "page": page_num, "content": chunk, "image_path": ""})

                # tables as text rows
                try:
                    tables = page.find_tables()
                    for tidx, table in enumerate(tables.tables):
                        rows = table.extract() or []
                        rows_txt = [" | ".join([(c or "") for c in row]) for row in rows]
                        table_text = "\n".join(rows_txt).strip()
                        if table_text:
                            out.append(
                                {
                                    "modality": "table",
                                    "page": page_num,
                                    "content": f"Table {tidx + 1} on page {page_num}:\n{table_text}",
                                    "image_path": "",
                                }
                            )
                except Exception:
                    pass

                # images
                images = page.get_images(full=True)
                for iidx, img in enumerate(images):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n - pix.alpha > 3:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    image_path = os.path.join(image_dir, f"{base}_p{page_num}_img{iidx+1}.png")
                    pix.save(image_path)
                    out.append(
                        {
                            "modality": "image",
                            "page": page_num,
                            "content": f"Figure {iidx + 1} on page {page_num}",
                            "image_path": image_path,
                        }
                    )
        finally:
            doc.close()

        return out

    def _vectors_for_item(self, item: Dict[str, Any]) -> List[List[float]]:
        if item["modality"] == "image" and item.get("image_path"):
            image = Image.open(item["image_path"]).convert("RGB")
            return self.embedder.embed_image_multi(image, grid=2)
        return self.embedder.embed_text_multi(item["content"])

    def store_pdf(self, pdf_path: str, paper_id: Optional[str] = None) -> Dict[str, Any]:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        paper_id = paper_id or os.path.splitext(os.path.basename(pdf_path))[0]
        items = self._extract_pdf_multimodal(pdf_path)
        collection = self.client.collections.get(self.collection_name)

        object_ids: List[str] = []
        doc_units = 0
        subvectors = 0

        with collection.batch.dynamic() as batch:
            for item in items:
                doc_units += 1
                doc_id = str(uuid.uuid4())
                reference = f"{paper_id} p.{item['page']} [{item['modality']}]"

                for vidx, vec in enumerate(self._vectors_for_item(item)):
                    oid = str(uuid.uuid4())
                    batch.add_object(
                        uuid=oid,
                        properties={
                            "paper_id": paper_id,
                            "source_name": os.path.basename(pdf_path),
                            "doc_id": doc_id,
                            "subvector_id": f"{doc_id}:{vidx}",
                            "modality": item["modality"],
                            "page": item["page"],
                            "content": item["content"],
                            "reference": reference,
                            "image_path": item.get("image_path", ""),
                            "embedding_json": json.dumps(vec),
                        },
                        vector=vec,
                    )
                    object_ids.append(oid)
                    subvectors += 1

        self.add_document_db(paper_id, object_ids)
        return {"paper_id": paper_id, "doc_units": doc_units, "subvectors": subvectors}

    def _dot(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def _maxsim_score(self, q_vectors: List[np.ndarray], d_vectors: List[np.ndarray]) -> float:
        if not q_vectors or not d_vectors:
            return 0.0
        token_max = []
        for qv in q_vectors:
            best = max(self._dot(qv, dv) for dv in d_vectors)
            token_max.append(best)
        return float(np.mean(token_max))

    def search(self, question: str, paper_id: Optional[str] = None, top_k: int = 8) -> List[Document]:
        collection = self.client.collections.get(self.collection_name)
        q_items = self.embedder.embed_query_multi(question)
        q_vectors = [np.array(q.vector, dtype=np.float32) for q in q_items]
        if not q_vectors:
            return []

        where_filter = Filter.by_property("paper_id").equal(paper_id) if paper_id else None

        # 1) candidate generation from ANN for each query token vector
        candidate_doc_ids: set[str] = set()
        for q in q_items:
            ann = collection.query.near_vector(
                near_vector=q.vector,
                limit=max(12, top_k * 4),
                filters=where_filter,
                return_metadata=MetadataQuery(distance=True),
            )
            for obj in ann.objects:
                candidate_doc_ids.add(obj.properties["doc_id"])

        if not candidate_doc_ids:
            return []

        # 2) exact MaxSim rerank using stored subvectors per candidate doc_id
        ranked = []
        for doc_id in candidate_doc_ids:
            objs = collection.query.fetch_objects(
                filters=Filter.by_property("doc_id").equal(doc_id),
                limit=256,
            )
            d_vectors: List[np.ndarray] = []
            representative = None
            for obj in objs.objects:
                props = obj.properties
                try:
                    d_vectors.append(np.array(json.loads(props["embedding_json"]), dtype=np.float32))
                except Exception:
                    continue
                if representative is None:
                    representative = props

            if not d_vectors or representative is None:
                continue

            score = self._maxsim_score(q_vectors, d_vectors)
            ranked.append((score, representative))

        ranked.sort(key=lambda x: x[0], reverse=True)
        top = ranked[:top_k]

        return [
            Document(
                page_content=item["content"],
                metadata={
                    "paper_id": item["paper_id"],
                    "source_name": item["source_name"],
                    "modality": item["modality"],
                    "page": item["page"],
                    "reference": item["reference"],
                    "image_path": item.get("image_path", ""),
                    "late_interaction_score": score,
                },
            )
            for score, item in top
        ]

    def delete_document_vectordb(self, paper_id: str) -> bool:
        db = self.load_db()
        if paper_id not in db:
            return False

        ids = db[paper_id]["ids"]
        collection = self.client.collections.get(self.collection_name)
        for oid in ids:
            try:
                collection.data.delete_by_id(oid)
            except Exception:
                pass

        del db[paper_id]
        self.save_db(db)
        return True


def download_arxiv_pdf(arxiv_id_or_url: str, target_dir: str = "uploads") -> str:
    os.makedirs(target_dir, exist_ok=True)

    raw = arxiv_id_or_url.strip()
    if "arxiv.org" in raw:
        raw = raw.rstrip("/").split("/")[-1]
    arxiv_id = raw.replace(".pdf", "")

    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    response = requests.get(pdf_url, timeout=40)
    response.raise_for_status()

    output_path = os.path.join(target_dir, f"{arxiv_id}.pdf")
    with open(output_path, "wb") as f:
        f.write(response.content)
    return output_path
