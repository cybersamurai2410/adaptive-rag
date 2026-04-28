from typing import Any, List, Optional
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from utilities import (
    answer_grader,
    format_docs_for_prompt,
    hallucination_grader,
    question_rewriter,
    question_router,
    rag_chain,
    retrieval_grader,
    web_search_tool,
)
from vector_db_ops import VectorDB


class GraphState(TypedDict):
    question: str
    paper_id: Optional[str]
    generation: str
    documents: List[Document]
    citations: List[Any]


vector_db = VectorDB()


def route_question(state: GraphState):
    print("---ROUTE QUESTION---")
    source = question_router.invoke({"question": state["question"]})
    if source.datasource == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    print("---ROUTE QUESTION TO PAPER RAG---")
    return "paper_rag"


def retrieve(state: GraphState):
    print("---RETRIEVE FROM WEAVIATE---")
    docs = vector_db.search(
        question=state["question"],
        paper_id=state.get("paper_id"),
        top_k=8,
    )
    return {"documents": docs, "question": state["question"], "paper_id": state.get("paper_id")}


def grade_documents(state: GraphState):
    print("---GRADE RETRIEVED DOCUMENTS---")
    filtered = []
    for d in state["documents"]:
        score = retrieval_grader.invoke(
            {"question": state["question"], "document": d.page_content}
        )
        if score.binary_score == "yes":
            filtered.append(d)
    return {"documents": filtered, "question": state["question"], "paper_id": state.get("paper_id")}


def decide_to_generate(state: GraphState):
    if not state["documents"]:
        print("---NO RELEVANT DOCS -> TRANSFORM QUERY---")
        return "transform_query"
    return "generate"


def transform_query(state: GraphState):
    print("---TRANSFORM QUERY---")
    better_question = question_rewriter.invoke({"question": state["question"]})
    return {
        "question": better_question,
        "documents": state.get("documents", []),
        "paper_id": state.get("paper_id"),
    }


def web_search(state: GraphState):
    print("---WEB SEARCH---")
    results = web_search_tool.invoke({"query": state["question"]})
    page_content = "\n".join([r.get("content", "") for r in results])
    d = Document(
        page_content=page_content,
        metadata={"reference": "web_search", "source_name": "web", "modality": "text", "page": 0},
    )
    return {"documents": [d], "question": state["question"], "paper_id": state.get("paper_id")}


def generate(state: GraphState):
    print("---GENERATE---")
    docs = state["documents"]
    context = format_docs_for_prompt(docs)
    generation = rag_chain.invoke({"question": state["question"], "context": context})

    citations = []
    for d in docs:
        ref = d.metadata.get("reference")
        if not ref:
            continue

        page_number = d.metadata.get("page")
        source_name = d.metadata.get("source_name", "")
        paper_id = state.get("paper_id") or source_name or "unknown"
        snippet = " ".join((d.page_content or "").split())
        snippet = snippet[:420] + ("..." if len(snippet) > 420 else "")

        citation_obj = {
            "id": ref,
            "raw": ref,
            "paperId": paper_id,
            "page": page_number,
            "label": d.metadata.get("modality", "page"),
            "display": ref,
            "snippet": snippet,
            "bbox": d.metadata.get("bbox"),
        }

        if not any(c.get("raw") == ref for c in citations):
            citations.append(citation_obj)

    return {
        "question": state["question"],
        "paper_id": state.get("paper_id"),
        "documents": docs,
        "generation": generation,
        "citations": citations,
    }


def grade_generation_v_documents_and_question(state: GraphState):
    print("---CHECK HALLUCINATIONS---")
    docs_text = "\n\n".join([d.page_content for d in state["documents"]])
    score = hallucination_grader.invoke(
        {"documents": docs_text, "generation": state["generation"]}
    )
    if score.binary_score == "yes":
        answer_score = answer_grader.invoke(
            {"question": state["question"], "generation": state["generation"]}
        )
        if answer_score.binary_score == "yes":
            return "useful"
        return "not_useful"
    return "not_supported"


def compile_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("web_search", web_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)

    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "web_search": "web_search",
            "paper_rag": "retrieve",
        },
    )

    workflow.add_edge("web_search", "generate")
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )

    workflow.add_edge("transform_query", "retrieve")

    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not_supported": "generate",
            "not_useful": "transform_query",
            "useful": END,
        },
    )

    return workflow.compile()
