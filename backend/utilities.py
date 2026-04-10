from typing import List, Literal

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults

# Multimodal-capable LLM for synthesis/grounding checks
llm = ChatOpenAI(model="gpt-4o", temperature=0)


class RouteQuery(BaseModel):
    datasource: Literal["paper_rag", "web_search"] = Field(...)


question_router = (
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Route to paper_rag for questions about uploaded/arXiv papers, figures, tables, methods, or references. "
                "Route to web_search only for out-of-corpus questions.",
            ),
            ("human", "{question}"),
        ]
    )
    | llm.with_structured_output(RouteQuery)
)


class GradeDocuments(BaseModel):
    binary_score: Literal["yes", "no"] = Field(...)


retrieval_grader = (
    ChatPromptTemplate.from_messages(
        [
            ("system", "Grade relevance of evidence to question. Return yes or no."),
            ("human", "Question:\n{question}\n\nEvidence:\n{document}"),
        ]
    )
    | llm.with_structured_output(GradeDocuments)
)


class GradeHallucinations(BaseModel):
    binary_score: Literal["yes", "no"] = Field(...)


hallucination_grader = (
    ChatPromptTemplate.from_messages(
        [
            ("system", "Is the answer grounded in evidence? Return yes or no."),
            ("human", "Evidence:\n{documents}\n\nAnswer:\n{generation}"),
        ]
    )
    | llm.with_structured_output(GradeHallucinations)
)


class GradeAnswer(BaseModel):
    binary_score: Literal["yes", "no"] = Field(...)


answer_grader = (
    ChatPromptTemplate.from_messages(
        [
            ("system", "Does answer resolve question? Return yes or no."),
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
                "Rewrite for better multimodal retrieval across paper text, figures, and tables.",
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
            "You are a research assistant. Answer from provided evidence only. "
            "Use references inline like [paper_id p.X modality]."
            "If uncertain, say evidence is insufficient.",
        ),
        (
            "human",
            "Question:\n{question}\n\nRetrieved Evidence:\n{context}\n\n"
            "Provide:\n1) Direct answer\n2) Key evidence bullets\n3) References list",
        ),
    ]
)

rag_chain = rag_prompt | llm | StrOutputParser()
web_search_tool = TavilySearchResults(k=3)


def format_docs_for_prompt(docs: List) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        ref = d.metadata.get("reference", "unknown")
        score = d.metadata.get("late_interaction_score")
        score_str = f"score={score:.3f}" if isinstance(score, (float, int)) else "score=n/a"
        blocks.append(f"[{i}] {ref} ({score_str})\n{d.page_content}")
    return "\n\n".join(blocks)
