import os
import re
import json
import uuid
import requests
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image
import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter, MetadataQuery
from langchain_core.documents import Document

from multimodal_models import MultiModalEmbedder


class VectorDB:
    """Weaviate multi-vector store for multimodal arXiv paper retrieval with late interaction."""

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
        if not collections.exists(self.collection_name):
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
        chunks: List[str] = []
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
        out: List[Dict[str, Any]] = []
        base = self._safe_id(os.path.splitext(os.path.basename(pdf_path))[0])

        try:
            for pidx in range(len(doc)):
                page = doc[pidx]
                page_num = pidx + 1

                # text
                page_text = page.get_text("text")
                for chunk in self._split_text(page_text):
                    out.append(
                        {
                            "modality": "text",
                            "page": page_num,
                            "content": chunk,
                            "image_path": "",
                        }
                    )

                # tables (as textual rows)
                try:
                    tables = page.find_tables()
                    for tidx, table in enumerate(tables.tables):
                        rows = table.extract() or []
                        rows_txt = [" | ".join([(c or "") for c in r]) for r in rows]
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

                # images (extract raw image bytes and store path)
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

        # text/table multimodal vectors
        return self.embedder.embed_text_multi(item["content"])

    def store_pdf(self, pdf_path: str, paper_id: Optional[str] = None) -> Dict[str, Any]:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        paper_id = paper_id or os.path.splitext(os.path.basename(pdf_path))[0]
        items = self._extract_pdf_multimodal(pdf_path)
        collection = self.client.collections.get(self.collection_name)

        object_ids: List[str] = []
        doc_count = 0
        subvector_count = 0

        with collection.batch.dynamic() as batch:
            for item in items:
                doc_count += 1
                doc_id = str(uuid.uuid4())
                reference = f"{paper_id} p.{item['page']} [{item['modality']}]"
                vectors = self._vectors_for_item(item)

                for vidx, vec in enumerate(vectors):
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
                        },
                        vector=vec,
                    )
                    object_ids.append(oid)
                    subvector_count += 1

        self.add_document_db(paper_id, object_ids)
        return {
            "paper_id": paper_id,
            "doc_units": doc_count,
            "subvectors": subvector_count,
        }

    def search(self, question: str, paper_id: Optional[str] = None, top_k: int = 8) -> List[Document]:
        """
        Late-interaction style retrieval approximation:
        1) Produce multiple query token vectors.
        2) For each query vector, retrieve nearest subvectors from Weaviate.
        3) Aggregate by doc_id using MaxSim over query token hits.
        """
        collection = self.client.collections.get(self.collection_name)
        qvectors = self.embedder.embed_query_multi(question)
        if not qvectors:
            return []

        where_filter = Filter.by_property("paper_id").equal(paper_id) if paper_id else None

        # doc_id -> token -> best similarity, plus representative metadata
        scores: Dict[str, Dict[str, Any]] = {}

        for qv in qvectors:
            resp = collection.query.near_vector(
                near_vector=qv.vector,
                limit=max(12, top_k * 2),
                filters=where_filter,
                return_metadata=MetadataQuery(distance=True),
            )
            for obj in resp.objects:
                props = obj.properties
                doc_id = props["doc_id"]
                distance = obj.metadata.distance if obj.metadata and obj.metadata.distance is not None else 1.0
                similarity = max(0.0, 1.0 - float(distance))

                if doc_id not in scores:
                    scores[doc_id] = {
                        "token_sims": {},
                        "content": props["content"],
                        "metadata": {
                            "paper_id": props["paper_id"],
                            "source_name": props["source_name"],
                            "modality": props["modality"],
                            "page": props["page"],
                            "reference": props["reference"],
                            "image_path": props.get("image_path", ""),
                        },
                    }

                prev = scores[doc_id]["token_sims"].get(qv.token, 0.0)
                if similarity > prev:
                    scores[doc_id]["token_sims"][qv.token] = similarity

        ranked = []
        for doc_id, record in scores.items():
            token_values = list(record["token_sims"].values())
            # MaxSim aggregate (sum of best similarities per query token)
            score = sum(token_values) / max(1, len(qvectors))
            ranked.append((score, record))

        ranked.sort(key=lambda x: x[0], reverse=True)
        top = ranked[:top_k]

        return [
            Document(page_content=r["content"], metadata={**r["metadata"], "late_interaction_score": s})
            for s, r in top
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
