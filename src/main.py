import argparse
import json
import os

from datasets import load_dataset
from tqdm import tqdm

from ask_questions import generate_prompt_pipeline, perform_rag_request, parse_rag_results
from index_database import DOCUMENT_STORE_DB, DOCUMENT_STORE_INDEX, EMBEDDING_MODEL, load_db, \
    import_dataset_from_dict, import_dataset_from_generator
from question_generator import MODEL_CHATGPT_16k
from evaluate_pipeline_results import evaluate_rag_complete_with_ragas, evaluate_rag_answers_with_ragas, \
    evaluate_rag_retrieval_with_ragas, evaluate_rag_with_traditional, calculate_mean_and_std

original_dataset = []


def generate_ms_marco():
    dataset = load_dataset('ms_marco', 'v1.1', split="test")
    for entry in dataset:
        for i, passage in enumerate(entry['passages']["passage_text"]):
            entry["content"] = entry['passages']['passage_text'][i]
            entry["label"] = entry['passages']['is_selected'][i]
            entry["url"] = entry['passages']["url"][i]
            yield entry


def generate_wiki_qa():
    dataset = load_dataset('wiki_qa', split="train")
    dataset = dataset.rename_column("answer", "content")
    for entry in dataset:
        yield entry


def YOUR_GENERATOR():
    raise RuntimeError(
        "Implement your custom dataset generator. Look to the functions above for inspiration. Provided are MSMarco and Wiki QA.")


def main(**kwargs):
    # Get the OpenAI API key from the environment variable
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Check if the API key is provided
    if openai_api_key is None:
        raise ValueError("OpenAI API key is not provided in the environment variable OPENAI_API_KEY")

    # Load FAISS document store or create if it does not exist
    if os.path.isfile(kwargs["document_store_index"]):
        db = load_db(document_store_index=kwargs["document_store_index"])
    else:
        if kwargs["apply_hf_dataset"]:
            db = import_dataset_from_generator(YOUR_GENERATOR, document_store_db=kwargs["document_store_db"],
                                               document_store_index=kwargs["document_store_index"],
                                               embedding_model=kwargs["embedding_model"], delete_store=False)
        else:
            with open(kwargs["path_to_document_store_file"], "r", encoding="utf-8") as f:
                _documents = json.loads(f.read())
            db = import_dataset_from_dict(_documents, document_store_db=kwargs["document_store_db"],
                                          document_store_index=kwargs["document_store_index"],
                                          embedding_model=kwargs["embedding_model"], delete_store=False)

    # Create Haystack RAG pipeline
    p = generate_prompt_pipeline(document_store=db, llm_model=kwargs["synthesis_model"],
                                 embedding_model=kwargs["embedding_model"])

    # Load QA dataset
    with open(kwargs["path_to_dataset_file"], "r", encoding="utf-8") as f:
        _dataset = json.loads(f.read())

    # Execute RAG pipeline for each QA sample
    answers = []
    for q in tqdm(_dataset):
        res = perform_rag_request(q["question"], p, _top_k=kwargs["top_k"])
        answers.append(parse_rag_results(q, res))

    results_file = kwargs["output_path_rag"]
    with open(results_file, "w", encoding="utf-8") as fp:
        json.dump(answers, fp)

    # Evaluate RAG pipeline result for each QA sample
    if kwargs["perform_traditional_eval"]:
        answers = evaluate_rag_with_traditional(answers)
    if kwargs["perform_ragas_eval_answer"] and kwargs["perform_ragas_eval_context"]:  # context and answer metrics
        answers = evaluate_rag_complete_with_ragas(results_file, answers, kwargs["eval_batch_size"])
    elif kwargs["perform_ragas_eval_answer"] and not kwargs["perform_ragas_eval_context"]:  # answer metrics only
        answers = evaluate_rag_answers_with_ragas(results_file, answers, kwargs["eval_batch_size"])
    elif kwargs["perform_ragas_eval_context"] and not kwargs["perform_ragas_eval_answer"]:  # context metrics only
        answers = evaluate_rag_retrieval_with_ragas(results_file, answers, kwargs["eval_batch_size"])
    else:
        print("No evaluation with ragas performed on RAG output")

    calculate_mean_and_std(answers)

    # Save the results to the output file (already happened in ragas
    with open(results_file, "w+", encoding="utf-8") as f:
        json.dump(answers, f)

    print(f"Your RAG results are saved in {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate your QA-Dataset with RAG and ragas")
    files = parser.add_argument_group("files")
    files.add_argument('--document_store_db', type=str, default=DOCUMENT_STORE_DB,
                       help="Path and name for the FAISS SQL database")
    files.add_argument('--document_store_index', type=str, default=DOCUMENT_STORE_INDEX,
                       help="Path and name for FAISS SQL index")
    files.add_argument('--path_to_document_store_file', type=str, default="data/Baldur_documents_128.json",
                       help="Path to FAISS document store in JSON format")
    files.add_argument('--path_to_dataset_file', type=str, default="data/qa_dataset_bg3_lp.json",
                       help="Path to your QA-Dataset file based on the document store in JSON format")
    files.add_argument('--apply_hf_dataset', type=bool, default=False,
                       help="Do you want to use a HF dataset? If true create a genertor that converts the datasets document column to 'content'")
    files.add_argument('--name_hf_dataset', type=str, default="wiki_qa",
                       help="HuggingFace dataset name. If you want use this option use generators to create a HF dataset")
    files.add_argument('--output_path_rag', type=str, default="evaluation/rag_result.json",
                       help="Output path for RAG results (without ragas) in JSON format")

    rag_details = parser.add_argument_group("RAG details")
    rag_details.add_argument('--embedding_model', type=str, default=EMBEDDING_MODEL,
                             help="Huggingface Embedding model for RAG")
    rag_details.add_argument('--synthesis_model', type=str, default=MODEL_CHATGPT_16k,
                             help="LLM to answer the questions in the dataset after retrieval")
    rag_details.add_argument('--perform_traditional_eval', type=bool, default=False,
                             help="Evaluate RAG pipeline contexts and answers with BERTScore and SourceContentAccuracy")
    rag_details.add_argument('--perform_ragas_eval_answer', type=bool, default=True,
                             help="Evaluate RAG pipeline answers with ragas")
    rag_details.add_argument('--perform_ragas_eval_context', type=bool, default=True,
                             help="Evaluate RAG pipeline contexts with ragas")
    rag_details.add_argument('--eval_batch_size', type=int, default=3, help="Batch size for ragas evaluation")
    rag_details.add_argument('--top_k', type=int, default=5, help="TopK value for RAG retrieval")
    args = parser.parse_args()

    main(path_to_document_store_file=args.path_to_document_store_file, document_store_db=args.document_store_db,
         document_store_index=args.document_store_index, path_to_dataset_file=args.path_to_dataset_file,
         apply_hf_dataset=args.apply_hf_dataset, name_hf_dataset=args.name_hf_dataset,
         output_path_rag=args.output_path_rag, perform_ragas_eval_answer=args.perform_ragas_eval_answer,
         embedding_model=args.embedding_model, perform_ragas_eval_context=args.perform_ragas_eval_context,
         synthesis_model=args.synthesis_model, perform_traditional_eval=args.perform_traditional_eval,
         top_k=args.top_k, eval_batch_size=args.eval_batch_size)
