import argparse
import json
import os

from tqdm import tqdm

from ask_questions import generate_prompt_pipeline, perform_rag_request, parse_rag_results
from index_database import DOCUMENT_STORE_DB, DOCUMENT_STORE_INDEX, EMBEDDING_MODEL, load_db, \
    import_dataset_from_dict
from question_generator import MODEL_CHATGPT_16k
from src.evaluate_pipeline_results import evaluate_rag_complete_with_ragas, evaluate_rag_answers_with_ragas, \
    evaluate_rag_retrieval_with_ragas


def main(**kwargs):
    # Get the OpenAI API key from the environment variable
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Check if the API key is provided
    if openai_api_key is None:
        raise ValueError("OpenAI API key is not provided in the environment variable OPENAI_API_KEY")

    if os.path.isfile(kwargs["document_store_db"]):
        db = load_db(document_store_index=kwargs["document_store_index"])
    else:
        with open(kwargs["path_to_document_store_file"], "r", encoding="utf-8") as f:
            _documents = json.loads(f.read())
        db = import_dataset_from_dict(_documents, document_store_db=kwargs["document_store_db"],
                                      document_store_index=kwargs["document_store_index"], delete_store=False)

    p = generate_prompt_pipeline(document_store=db, llm_model=kwargs["synthesis_model"],
                                 embedding_model=kwargs["embedding_model"])

    with open(kwargs["path_to_dataset_file"], "r", encoding="utf-8") as f:
        _dataset = json.loads(f.read())

    answers = []
    for q in tqdm(_dataset):
        res = perform_rag_request(q["question"], p, _top_k=kwargs["top_k"])
        answers.append(parse_rag_results(q, res))

    with open(kwargs["output_path_rag"], "w", encoding="utf-8") as fp:
        json.dump(answers, fp)

    if kwargs["perform_ragas_evaluation_answer"] and kwargs["perform_ragas_evaluation_context"]:
        evaluate_rag_complete_with_ragas(kwargs["output_path_rag"])
    elif kwargs["perform_ragas_evaluation_answer"] and not kwargs["perform_ragas_evaluation_context"]:
        evaluate_rag_answers_with_ragas(kwargs["output_path_rag"])
    elif kwargs["perform_ragas_evaluation_context"] and not kwargs["perform_ragas_evaluation_answer"]:
        evaluate_rag_retrieval_with_ragas(kwargs["output_path_rag"])
    else:
        print("No evaluation with ragas performed on RAG output")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate your QA-Dataset with RAG and ragas")
    parser.add_argument('--path_to_document_store_file', type=str, default="data/Baldur_documents_256.json", help="Path to Document store in JSON format")
    parser.add_argument('--document_store_db', type=str, default=DOCUMENT_STORE_DB, help="Path and name for the RAG SQL database")
    parser.add_argument('--document_store_index', type=str, default=DOCUMENT_STORE_INDEX, help="Path and name for RAG SQL index")
    parser.add_argument('--path_to_dataset_file', type=str, default="data/qa_dataset_bg3_lp.json", help="Path to your QA-Dataset file based on the document store in JSON format")
    parser.add_argument('--output_path_rag', type=str, default="evaluation/rag_result.json", help="Output path for RAG results (without ragas) in JSON format")
    parser.add_argument('--output_path_ragas_evaluation', type=bool, default="evaluation/ragas_evaluation_result.json", help="Output path for ragas evaluation results of the RAG pipeline in JSON format")
    parser.add_argument('--perform_ragas_evaluation_answer', type=bool, default=False, help="Evaluate RAG pipeline answers with ragas")
    parser.add_argument('--perform_ragas_evaluation_context', type=bool, default=False, help="Output path for ragas evaluation results")
    parser.add_argument('--embedding_model', type=str, default=EMBEDDING_MODEL, help="Huggingface Embedding model for RAG")
    parser.add_argument('--synthesis_model', type=str, default=MODEL_CHATGPT_16k, help="LLM to answer the questions in the dataset after retrieval")
    parser.add_argument('--top_k', type=int, default="rag_result", help="TopK value for RAG retrieval")

    args = parser.parse_args()
    main(path_to_dataset=args.path_to_dataset, output_path=args.output_path)