import os
import re
import json
import uuid
import requests
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter, MetadataQuery

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


class VectorDB:
    """Weaviate-backed multi-vector storage for multimodal arXiv papers."""

    def __init__(self, db_file: str = "db.json"):
        self.db_file = db_file
        self.collection_name = os.getenv("WEAVIATE_COLLECTION", "PaperChunk")
        self.embedding_model = OpenAIEmbeddings(model=os.getenv("EMBED_MODEL", "text-embedding-3-large"))

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
                    Property(name="modality", data_type=DataType.TEXT),
                    Property(name="vector_type", data_type=DataType.TEXT),
                    Property(name="page", data_type=DataType.INT),
                    Property(name="chunk_id", data_type=DataType.TEXT),
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="reference", data_type=DataType.TEXT),
                ],
            )

    def close(self) -> None:
        self.client.close()

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

    def _safe_paper_id(self, source_name: str) -> str:
        base = os.path.basename(source_name)
        paper_id = os.path.splitext(base)[0]
        return re.sub(r"[^a-zA-Z0-9_.-]", "_", paper_id)

    def _keyword_projection(self, text: str, max_tokens: int = 32) -> str:
        words = re.findall(r"[A-Za-z0-9\-]+", text.lower())
        seen = set()
        deduped = []
        for w in words:
            if w not in seen and len(w) > 2:
                seen.add(w)
                deduped.append(w)
            if len(deduped) >= max_tokens:
                break
        return " ".join(deduped)

    def _split_text(self, text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []

        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunks.append(text[start:end])
            if end == len(text):
                break
            start = max(0, end - overlap)
        return chunks

    def _extract_pdf_multimodal(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text, table text (when detectable), and image placeholders from PDF.
        - text modality: paragraph chunks
        - table modality: extracted table rows as text
        - image modality: image metadata placeholders
        """
        extracted: List[Dict[str, Any]] = []
        doc = fitz.open(pdf_path)

        try:
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                page_number = page_idx + 1

                # 1) text chunks
                page_text = page.get_text("text") or ""
                for chunk in self._split_text(page_text):
                    extracted.append(
                        {
                            "modality": "text",
                            "page": page_number,
                            "content": chunk,
                        }
                    )

                # 2) table chunks (PyMuPDF table detection, if available)
                try:
                    tables = page.find_tables()
                    for t_idx, table in enumerate(tables.tables):
                        rows = table.extract() or []
                        table_text_rows = [" | ".join([c or "" for c in row]) for row in rows]
                        table_text = "\n".join(table_text_rows).strip()
                        if table_text:
                            extracted.append(
                                {
                                    "modality": "table",
                                    "page": page_number,
                                    "content": f"Table {t_idx + 1} (page {page_number}):\n{table_text}",
                                }
                            )
                except Exception:
                    # keep robust if table detector isn't available in installed pymupdf build
                    pass

                # 3) image placeholders (no OCR/captioning in this backend-only iteration)
                images = page.get_images(full=True)
                for i_idx, _ in enumerate(images):
                    extracted.append(
                        {
                            "modality": "image",
                            "page": page_number,
                            "content": f"Image {i_idx + 1} on page {page_number}. Visual content placeholder.",
                        }
                    )
        finally:
            doc.close()

        return extracted

    def _build_vectors(self, content: str) -> List[Tuple[str, str, List[float]]]:
        """
        Multi-vector representation for one chunk:
        - semantic: original chunk text
        - keyword: compact keyword projection for lexical/term-centric matching
        """
        semantic = content
        keyword = self._keyword_projection(content)

        semantic_vec = self.embedding_model.embed_query(semantic)
        keyword_vec = self.embedding_model.embed_query(keyword if keyword else semantic[:300])

        return [
            ("semantic", semantic, semantic_vec),
            ("keyword", keyword if keyword else semantic, keyword_vec),
        ]

    def store_pdf(self, pdf_path: str, paper_id: Optional[str] = None) -> Dict[str, Any]:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        paper_id = paper_id or self._safe_paper_id(pdf_path)
        items = self._extract_pdf_multimodal(pdf_path)
        if not items:
            return {"paper_id": paper_id, "chunks": 0, "vectors": 0}

        collection = self.client.collections.get(self.collection_name)
        inserted_ids: List[str] = []
        vector_count = 0

        with collection.batch.dynamic() as batch:
            for item in items:
                chunk_id = str(uuid.uuid4())
                reference = f"{paper_id} p.{item['page']} [{item['modality']}]"

                for vector_type, projection_text, vector in self._build_vectors(item["content"]):
                    object_id = str(uuid.uuid4())
                    batch.add_object(
                        uuid=object_id,
                        properties={
                            "paper_id": paper_id,
                            "source_name": os.path.basename(pdf_path),
                            "modality": item["modality"],
                            "vector_type": vector_type,
                            "page": item["page"],
                            "chunk_id": chunk_id,
                            "content": projection_text,
                            "reference": reference,
                        },
                        vector=vector,
                    )
                    inserted_ids.append(object_id)
                    vector_count += 1

        self.add_document_db(paper_id, inserted_ids)
        return {"paper_id": paper_id, "chunks": len(items), "vectors": vector_count}

    def search(self, question: str, paper_id: Optional[str] = None, top_k: int = 6) -> List[Document]:
        collection = self.client.collections.get(self.collection_name)

        semantic_vec = self.embedding_model.embed_query(question)
        keyword_vec = self.embedding_model.embed_query(self._keyword_projection(question) or question)

        where_filter = None
        if paper_id:
            where_filter = Filter.by_property("paper_id").equal(paper_id)

        semantic_hits = collection.query.near_vector(
            near_vector=semantic_vec,
            limit=top_k,
            filters=where_filter,
            return_metadata=MetadataQuery(distance=True),
        )
        keyword_hits = collection.query.near_vector(
            near_vector=keyword_vec,
            limit=top_k,
            filters=where_filter,
            return_metadata=MetadataQuery(distance=True),
        )

        # Fuse by chunk_id with best (lowest) distance
        fused: Dict[str, Dict[str, Any]] = {}
        for resp in [semantic_hits, keyword_hits]:
            for obj in resp.objects:
                props = obj.properties
                chunk_key = props["chunk_id"]
                dist = obj.metadata.distance if obj.metadata and obj.metadata.distance is not None else 1.0
                if chunk_key not in fused or dist < fused[chunk_key]["distance"]:
                    fused[chunk_key] = {
                        "distance": dist,
                        "content": props["content"],
                        "metadata": {
                            "paper_id": props["paper_id"],
                            "source_name": props["source_name"],
                            "modality": props["modality"],
                            "vector_type": props["vector_type"],
                            "page": props["page"],
                            "chunk_id": props["chunk_id"],
                            "reference": props["reference"],
                        },
                    }

        ranked = sorted(fused.values(), key=lambda x: x["distance"])[:top_k]
        return [Document(page_content=r["content"], metadata=r["metadata"]) for r in ranked]

    def delete_document_vectordb(self, paper_id: str) -> bool:
        db = self.load_db()
        if paper_id not in db:
            return False

        ids = db[paper_id]["ids"]
        collection = self.client.collections.get(self.collection_name)
        for object_id in ids:
            try:
                collection.data.delete_by_id(object_id)
            except Exception:
                pass

        del db[paper_id]
        self.save_db(db)
        return True


def download_arxiv_pdf(arxiv_id_or_url: str, target_dir: str = "uploads") -> str:
    os.makedirs(target_dir, exist_ok=True)

    arxiv_id = arxiv_id_or_url.strip()
    if "arxiv.org" in arxiv_id:
        arxiv_id = arxiv_id.rstrip("/").split("/")[-1]
    arxiv_id = arxiv_id.replace(".pdf", "")

    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    response = requests.get(pdf_url, timeout=30)
    response.raise_for_status()

    output_path = os.path.join(target_dir, f"{arxiv_id}.pdf")
    with open(output_path, "wb") as f:
        f.write(response.content)

    return output_path
