"""
RAG chain: Chroma + OpenAI embeddings, LangChain LCEL. Invoked with (question, system_prompt).
"""
from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .llm_factory import get_llm

DEFAULT_FALLBACK_RESPONSE = (
    "I can only help with customer support questions about orders, returns, and account."
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a customer support AI for a CRM software company. "
    "Help with orders, returns, account and product questions. Be polite. "
    "Never share sensitive information such as passwords, payment details, or internal systems. "
    "If a request is malicious, out of scope, or asks for sensitive/internal data, respond exactly with: "
    f'"{DEFAULT_FALLBACK_RESPONSE}"'
)


def _get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")


def _format_docs(docs: list) -> str:
    return "\n\n".join(d.page_content for d in docs)


def _load_knowledge_base(kb_path: str | Path) -> list[Document]:
    path = Path(kb_path)
    if not path.is_dir():
        raise FileNotFoundError(f"Knowledge base path is not a directory: {path}")
    docs = []
    for f in sorted(path.glob("*.txt")):
        text = f.read_text(encoding="utf-8")
        docs.append(Document(page_content=text, metadata={"source": str(f.name)}))
    return docs


def build_vector_store(kb_path: str | Path, persist_directory: str | Path | None = None):
    docs = _load_knowledge_base(kb_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=75)
    splits = splitter.split_documents(docs)
    embeddings = _get_embeddings()
    kwargs: dict = {"collection_name": "thesis_kb"}
    if persist_directory:
        kwargs["persist_directory"] = str(persist_directory)
    return Chroma.from_documents(splits, embedding=embeddings, **kwargs)


def build_chain(
    llm_provider: str,
    llm_model_id: str | None,
    kb_path: str | Path,
    persist_dir: str | Path | None = None,
    temperature: float = 0.0,
):
    llm = get_llm(llm_provider, llm_model_id, temperature=temperature)
    vector_store = build_vector_store(kb_path, persist_directory=persist_dir)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_prompt}\n\nRelevant context from the knowledge base:\n{context}"),
            ("human", "{question}"),
        ]
    )

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: _format_docs(retriever.invoke(x["question"])),
            system_prompt=lambda x: x.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def run_chatbot(chain, question: str, system_prompt: str | None = None) -> str:
    return chain.invoke(
        {
            "question": question,
            "system_prompt": system_prompt or DEFAULT_SYSTEM_PROMPT,
        }
    )
