from typing import List, Literal

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class RouteQuery(BaseModel):
    """Route question to paper RAG or web search."""

    datasource: Literal["paper_rag", "web_search"] = Field(
        ..., description="Route to paper_rag when question is about uploaded paper(s), otherwise web_search."
    )


question_router = (
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a routing agent. If the user asks about uploaded arXiv papers, methods, tables, figures,"
                " or citations from those papers, route to paper_rag. For current events or outside knowledge, route to web_search.",
            ),
            ("human", "{question}"),
        ]
    )
    | llm.with_structured_output(RouteQuery)
)


class GradeDocuments(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="Whether document is relevant to question")


retrieval_grader = (
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Grade whether the retrieved evidence is relevant to the question. "
                "Answer only with yes or no.",
            ),
            ("human", "Question:\n{question}\n\nEvidence:\n{document}"),
        ]
    )
    | llm.with_structured_output(GradeDocuments)
)


class GradeHallucinations(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="Whether answer is grounded in evidence")


hallucination_grader = (
    ChatPromptTemplate.from_messages(
        [
            ("system", "Decide if answer is grounded in provided evidence. Return yes or no."),
            ("human", "Evidence:\n{documents}\n\nAnswer:\n{generation}"),
        ]
    )
    | llm.with_structured_output(GradeHallucinations)
)


class GradeAnswer(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="Whether answer addresses the question")


answer_grader = (
    ChatPromptTemplate.from_messages(
        [
            ("system", "Decide if answer addresses the user question. Return yes or no."),
            ("human", "Question:\n{question}\n\nAnswer:\n{generation}"),
        ]
    )
    | llm.with_structured_output(GradeAnswer)
)


question_rewriter = (
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Rewrite the question to improve scholarly retrieval over paper text, tables, and figures.",
            ),
            ("human", "Original question:\n{question}"),
        ]
    )
    | llm
    | StrOutputParser()
)


rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an academic assistant. Use only provided context. "
            "If evidence is insufficient, say so clearly. "
            "Include concise inline references like [source p.X modality].",
        ),
        (
            "human",
            "Question:\n{question}\n\nContext:\n{context}\n\n"
            "Return a concise answer followed by a short 'References' list.",
        ),
    ]
)

rag_chain = rag_prompt | llm | StrOutputParser()

web_search_tool = TavilySearchResults(k=3)


def format_docs_for_prompt(docs: List) -> str:
    rows = []
    for i, d in enumerate(docs, start=1):
        ref = d.metadata.get("reference", "unknown")
        rows.append(f"[{i}] {ref}\n{d.page_content}")
    return "\n\n".join(rows)
