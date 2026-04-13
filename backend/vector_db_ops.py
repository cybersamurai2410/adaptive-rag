import json
import os
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
    ColPali retrieval:
    - Each PDF page -> page image
    - Each page image -> multiple patch vectors
    - Query -> multiple vectors
    - ANN retrieves patch-level hits
    - Group by page_id
    - Late interaction MaxSim ranks pages
    """

    def __init__(self, db_file: str = "db.json"):
        self.db_file = db_file
        self.collection_name = os.getenv("WEAVIATE_COLLECTION", "PaperPatchVector")
        self.embedder = MultiModalEmbedder(
            model_name=os.getenv("COLPALI_MODEL", "vidore/colpali-v1.2"),
        )

        self.client = weaviate.connect_to_local(
            host=os.getenv("WEAVIATE_HOST", "localhost"),
            port=int(os.getenv("WEAVIATE_PORT", "8080")),
            grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
        )
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        if self.client.collections.exists(self.collection_name):
            return

        self.client.collections.create(
            name=self.collection_name,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="paper_id", data_type=DataType.TEXT),
                Property(name="source_name", data_type=DataType.TEXT),
                Property(name="page_id", data_type=DataType.TEXT),
                Property(name="page", data_type=DataType.INT),
                Property(name="patch_id", data_type=DataType.TEXT),
                Property(name="image_path", data_type=DataType.TEXT),
                Property(name="page_text", data_type=DataType.TEXT),
                Property(name="reference", data_type=DataType.TEXT),
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

    def _extract_pdf_pages(self, pdf_path: str, image_dir: str = "uploads/pages") -> List[Dict[str, Any]]:
        os.makedirs(image_dir, exist_ok=True)
        doc = fitz.open(pdf_path)
        base = os.path.splitext(os.path.basename(pdf_path))[0]

        pages: List[Dict[str, Any]] = []
        try:
            for pidx in range(len(doc)):
                page_num = pidx + 1
                page = doc[pidx]

                # Render page as image (page-level visual representation)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                image_path = os.path.join(image_dir, f"{base}_p{page_num}.png")
                pix.save(image_path)

                # keep page text as answer context (retrieval is still image-patch based)
                page_text = page.get_text("text") or ""

                pages.append(
                    {
                        "page": page_num,
                        "image_path": image_path,
                        "page_text": page_text,
                    }
                )
        finally:
            doc.close()

        return pages

    def store_pdf(self, pdf_path: str, paper_id: Optional[str] = None) -> Dict[str, Any]:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        paper_id = paper_id or os.path.splitext(os.path.basename(pdf_path))[0]
        pages = self._extract_pdf_pages(pdf_path)

        collection = self.client.collections.get(self.collection_name)
        object_ids: List[str] = []
        total_patch_vectors = 0

        with collection.batch.dynamic() as batch:
            for page in pages:
                page_id = f"{paper_id}:p{page['page']}"
                reference = f"{paper_id} p.{page['page']} [page]"

                image = Image.open(page["image_path"]).convert("RGB")
                patch_vectors = self.embedder.embed_image_multi(image)

                for pidx, vec in enumerate(patch_vectors):
                    oid = str(uuid.uuid4())
                    batch.add_object(
                        uuid=oid,
                        properties={
                            "paper_id": paper_id,
                            "source_name": os.path.basename(pdf_path),
                            "page_id": page_id,
                            "page": page["page"],
                            "patch_id": f"{page_id}:patch{pidx}",
                            "image_path": page["image_path"],
                            "page_text": page["page_text"],
                            "reference": reference,
                            "embedding_json": json.dumps(vec),
                        },
                        vector=vec,
                    )
                    object_ids.append(oid)
                    total_patch_vectors += 1

        self.add_document_db(paper_id, object_ids)
        return {
            "paper_id": paper_id,
            "pages": len(pages),
            "patch_vectors": total_patch_vectors,
        }

    def _dot(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def _maxsim_score(self, q_vectors: List[np.ndarray], page_vectors: List[np.ndarray]) -> float:
        if not q_vectors or not page_vectors:
            return 0.0
        per_q = []
        for qv in q_vectors:
            per_q.append(max(self._dot(qv, pv) for pv in page_vectors))
        return float(np.mean(per_q))

    def search(self, question: str, paper_id: Optional[str] = None, top_k: int = 8) -> List[Document]:
        collection = self.client.collections.get(self.collection_name)

        # Query -> multiple vectors
        q_items = self.embedder.embed_query_multi(question)
        q_vectors = [np.array(q.vector, dtype=np.float32) for q in q_items]
        if not q_vectors:
            return []

        where_filter = Filter.by_property("paper_id").equal(paper_id) if paper_id else None

        # 1) ANN patch-level retrieval per query vector
        candidate_pages: set[str] = set()
        for q in q_items:
            hits = collection.query.near_vector(
                near_vector=q.vector,
                limit=max(20, top_k * 8),
                filters=where_filter,
                return_metadata=MetadataQuery(distance=True),
            )
            for h in hits.objects:
                candidate_pages.add(h.properties["page_id"])

        if not candidate_pages:
            return []

        # 2) Group by page_id + exact MaxSim late interaction
        ranked_pages = []
        for page_id in candidate_pages:
            objs = collection.query.fetch_objects(
                filters=Filter.by_property("page_id").equal(page_id),
                limit=2048,
            )

            page_vectors: List[np.ndarray] = []
            rep = None
            for obj in objs.objects:
                props = obj.properties
                try:
                    page_vectors.append(np.array(json.loads(props["embedding_json"]), dtype=np.float32))
                except Exception:
                    continue
                if rep is None:
                    rep = props

            if not page_vectors or rep is None:
                continue

            score = self._maxsim_score(q_vectors, page_vectors)
            ranked_pages.append((score, rep))

        ranked_pages.sort(key=lambda x: x[0], reverse=True)
        top_pages = ranked_pages[:top_k]

        # Output is page-level documents (not patch-level)
        return [
            Document(
                page_content=page["page_text"],
                metadata={
                    "paper_id": page["paper_id"],
                    "source_name": page["source_name"],
                    "page": page["page"],
                    "page_id": page["page_id"],
                    "reference": page["reference"],
                    "image_path": page["image_path"],
                    "late_interaction_score": score,
                },
            )
            for score, page in top_pages
        ]

    def inspect_multivector(self, paper_id: str, max_pages: int = 5) -> Dict[str, Any]:
        collection = self.client.collections.get(self.collection_name)
        objs = collection.query.fetch_objects(
            filters=Filter.by_property("paper_id").equal(paper_id),
            limit=20000,
        )

        by_page: Dict[str, Dict[str, Any]] = {}
        for obj in objs.objects:
            p = obj.properties
            pid = p["page_id"]
            entry = by_page.setdefault(
                pid,
                {
                    "page_id": pid,
                    "reference": p.get("reference"),
                    "page": p.get("page"),
                    "patch_vectors": 0,
                    "vector_dim": 0,
                },
            )
            entry["patch_vectors"] += 1
            if entry["vector_dim"] == 0:
                try:
                    entry["vector_dim"] = len(json.loads(p["embedding_json"]))
                except Exception:
                    entry["vector_dim"] = 0

        samples = list(by_page.values())[:max_pages]
        total_patch_vectors = sum(v["patch_vectors"] for v in by_page.values())
        return {
            "paper_id": paper_id,
            "pages": len(by_page),
            "total_patch_vectors": total_patch_vectors,
            "samples": samples,
        }

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
