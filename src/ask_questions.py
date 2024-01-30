from __future__ import annotations

import json
import os

import backoff
import openai
from haystack import Pipeline
from haystack.document_stores import FAISSDocumentStore, InMemoryDocumentStore
from haystack.nodes import AnswerParser, PromptNode, PromptTemplate, JoinDocuments, SentenceTransformersRanker
from haystack.nodes import EmbeddingRetriever, BM25Retriever
from haystack.pipelines import FAQPipeline
from tqdm import tqdm

from index_database import EMBEDDING_MODEL, load_db, EMBEDDING_MODEL_ASYM, DOCUMENT_STORE_INDEX_ASYM_EMB
from question_generator import MODEL_CHATGPT_16k

openai_api_key = os.getenv("OPENAI_API_KEY")


@backoff.on_exception(backoff.expo,
                      openai.RateLimitError,
                      max_time=10)
def perform_rag_request(query: str, pipe: [Pipeline | FAQPipeline], _top_k: int = 10, hybrid: bool = False) -> dict:
    if isinstance(pipe, Pipeline):
        if hybrid:
            result = pipe.run(query=query, params={
                "SparseRetriever": {"top_k": 10},
                "DenseRetriever": {"top_k": 10},
                # "JoinDocuments": {"top_k_join": 15},  # comment for debug
                "JoinDocuments": {"top_k_join": 15, "debug": True}, #uncomment for debug
                "ReRanker": {"top_k": _top_k},
            })
        else:
            result = pipe.run(query=query, params={"EmbeddingRetriever": {"top_k": _top_k}, "debug": True})
    else:
        result = pipe.run(query=query, params={"Retriever": {"top_k": _top_k}, "debug": True})

    print(result["answers"][0].answer)
    return result


def get_retrieved_doc_content(result):
    for i, doc in enumerate(result["documents"]):
        print(f"{i + 1}: {doc.content}")


def get_topics_and_labels(result):
    for entry in [[document.score, document.meta["document_title"], document.meta["label"]] for document in
                  result["documents"]]:
        print(entry)


def ask_simple_question():
    p = generate_prompt_pipeline()
    res = perform_rag_request("What can you do at your Camp in the game?", p)
    get_retrieved_doc_content(res)
    get_topics_and_labels(res)


def generate_prompt_node(model: str = MODEL_CHATGPT_16k) -> PromptNode:
    # question_answering_check = PromptTemplate("deepset/question-answering-check", output_parser=AnswerParser())
    # question_answering_per_doc = PromptTemplate("deepset/question-answering-per-document", output_parser=AnswerParser())
    question_answering_with_references = PromptTemplate("deepset/question-answering-with-references",
                                                        output_parser=AnswerParser(
                                                            reference_pattern=r"Document\[(\d+)\]"))

    return PromptNode(model_name_or_path=model,
                      api_key=openai_api_key,
                      default_prompt_template=question_answering_with_references,
                      model_kwargs={"temperature": 0.5}
                      )


def generate_hybrid_pipeline(dataset: str = "data/Baldur_documents_128.json", llm_model: str = MODEL_CHATGPT_16k,
                             embedding_model: str = EMBEDDING_MODEL) -> Pipeline:

    document_store = InMemoryDocumentStore(use_bm25=True, embedding_dim=768)
    sparse_retriever = BM25Retriever(document_store=document_store)
    dense_retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model=embedding_model,
        use_gpu=True,
        scale_score=False,
    )
    __dataset = json.loads(open(dataset).read())
    document_store.write_documents(__dataset)
    document_store.update_embeddings(retriever=dense_retriever)

    if embedding_model != EMBEDDING_MODEL:
        dataset = json.loads(open(dataset).read())
        document_store.delete_documents()
        document_store.write_documents(dataset)
        document_store.update_embeddings(retriever=dense_retriever)

    join_documents = JoinDocuments(join_mode="concatenate")
    rerank = SentenceTransformersRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2")
    prompt_node = generate_prompt_node(llm_model)

    pipeline = Pipeline()
    pipeline.add_node(component=sparse_retriever, name="SparseRetriever", inputs=["Query"])
    pipeline.add_node(component=dense_retriever, name="DenseRetriever", inputs=["Query"])
    pipeline.add_node(component=join_documents, name="JoinDocuments", inputs=["SparseRetriever", "DenseRetriever"])
    pipeline.add_node(component=rerank, name="ReRanker", inputs=["JoinDocuments"])
    pipeline.add_node(component=prompt_node, name="QAPromptNode", inputs=["ReRanker"])

    return pipeline


def generate_prompt_pipeline(document_store: FAISSDocumentStore, llm_model: str = MODEL_CHATGPT_16k,
                             embedding_model: str = EMBEDDING_MODEL) -> Pipeline:
    retriever = EmbeddingRetriever(document_store=document_store,
                                   embedding_model=embedding_model)
    prompt_node = generate_prompt_node(llm_model)

    p_prompt = Pipeline()
    p_prompt.add_node(component=retriever, name="EmbeddingRetriever", inputs=["Query"])
    p_prompt.add_node(component=prompt_node, name="QAPromptNode", inputs=["EmbeddingRetriever"])

    return p_prompt


def generate_retrieval_pipeline() -> FAQPipeline:
    document_store = load_db()
    retriever = EmbeddingRetriever(document_store=document_store,
                                   embedding_model=EMBEDDING_MODEL)
    return FAQPipeline(retriever)


def parse_doc_results(result: dict) -> list:
    docs = []
    for i, doc in enumerate(result["documents"]):
        docs.append({
            "rank": i + 1,
            "content": doc.content,
            "meta": doc.meta,
            "id": doc.id
        })
    return docs


def parse_rag_results(_q: dict, _res: dict) -> dict:
    contexts = parse_doc_results(_res)
    return {"question": _q["question"],
            "answer": _res["answers"][0].answer,
            "contexts": contexts,
            "ground_truth": _q["ground_truth"],
            "applied_contexts": parse_applied_contexts(contexts, _res["answers"][0].answer),
            "meta": {
                "id_source_doc": _q["meta"]["id_source_doc"],
                "id_query": _q["meta"]["id_query"],
                "source_context": _q["context"]
            }}


def parse_applied_contexts(_contexts: list[dict], answer: str):
    applied_contexts = []
    for context in _contexts:
        if f"[document {context['rank']}]" in answer.lower() or f"[{context['rank']}]" in answer.lower():
            applied_contexts.append(context)
    return applied_contexts


if __name__ == "__main__":
    with open("data/qa_dataset_bg3_lp.json", "r", encoding="utf-8") as f:
        _dataset = json.loads(f.read())
    db = load_db(DOCUMENT_STORE_INDEX_ASYM_EMB)
    # db = load_db()
    # p = generate_hybrid_pipeline(llm_model=MODEL_CHATGPT_16k)
    p = generate_prompt_pipeline(db, llm_model=MODEL_CHATGPT_16k, embedding_model=EMBEDDING_MODEL_ASYM)
    answers = []
    top_k = 5
    for q in tqdm(_dataset):
        res = perform_rag_request(q["question"], p, _top_k=top_k, hybrid=False)
        answers.append(parse_rag_results(q, res))

    with open(f"evaluation/rag_cos_128_topk{top_k}_asym_gpt35.json", "w", encoding="utf-8") as fp:
        json.dump(answers, fp)
