import json
import math
import os

import datasets
import evaluate
import numpy as np
from ragas import evaluate as ragas_evaluate
from ragas.embeddings import HuggingfaceEmbeddings
from ragas.evaluation import Result
from ragas.metrics import (
    faithfulness,
    context_recall,
    context_precision,
    context_relevancy, AnswerSimilarity, AnswerRelevancy, AnswerCorrectness
)
from ragas.metrics.base import Metric

EVALUATION_FOLDER = "evaluation"

original_dataset: list[dict] = [{}]
openai_api_key = os.getenv("OPENAI_API_KEY")


def custom_huggingface_embeddings(embedding_model: str = "BAAI/bge-base-en-v1.5"):
    hf_embeddings = HuggingfaceEmbeddings()
    hf_embeddings.model_name = embedding_model
    return hf_embeddings


def custom_answer_sim_model(is_cross_encoder: bool = False):
    hf_embeddings = custom_huggingface_embeddings()
    return AnswerSimilarity(embeddings=hf_embeddings, is_cross_encoder=is_cross_encoder)


def custom_answer_rel_model():
    hf_embeddings = custom_huggingface_embeddings()
    return AnswerRelevancy(embeddings=hf_embeddings)


def custom_answer_corr_model():
    return AnswerCorrectness(answer_similarity=custom_answer_sim_model())


def parse_evaluation_entries(entry: dict) -> dict:
    return {
        "question": entry["question"],
        "contexts": [f"'{e['content']}'" for e in entry["contexts"]],
        "answer": entry["answer"],
        "ground_truths": [entry["ground_truth"]]
    }


def parse_applied_contexts_entries(entry: dict) -> dict:
    return {
        "question": entry["question"],
        "contexts": [f"'{e['content']}'" for e in entry["applied_contexts"]],
        "answer": entry["answer"],
        "ground_truths": [entry["ground_truth"]]
    }


def is_not_answered(_entry) -> bool:
    return ("answer" not in _entry or
            _entry["answer"].lower() == "answering is not possible given the available information." or
            _entry["answer"].lower().startswith("the documents do not provide"))


def generate_retrieval_test_set():
    for entry in original_dataset:
        yield parse_evaluation_entries(entry)


def generate_unanswered_test_set():
    for entry in original_dataset:
        if is_not_answered(entry) and ("scores" not in entry or "context_precision" not in entry["scores"]
                                       or "context_recall" not in entry["scores"]
                                       or "context_relevancy" not in entry["scores"]):
            yield parse_evaluation_entries(entry)


def generate_applied_context_test_set():
    for entry in original_dataset:
        if is_not_answered(entry):
            continue
        if "applied_contexts" not in entry or entry["applied_contexts"] == []:
            continue

        yield parse_applied_contexts_entries(entry)


def generate_answers_test_set():
    for entry in original_dataset:
        if is_not_answered(entry):
            continue
        yield parse_evaluation_entries(entry)


def calc_avg_scores(results: list[Result]) -> dict:
    sum_scores = {
        "context_precision": [],
        "context_recall": [],
        "context_relevancy": [],
        "context_accuracy": [],
        "applied_context_precision": [],
        "applied_context_recall": [],
        "applied_context_relevancy": [],
        "bert_recall": [],
        "bert_precision": [],
        "bert_f1": [],
        "faithfulness": [],
        "answer_relevancy": [],
        "answer_similarity": [],
        "answer_correctness": [],
    }
    for result in results:
        for i, _entry in enumerate(result.dataset):
            for (k, v) in result.scores[i].items():
                if isinstance(v, bool):
                    v = float(v)
                sum_scores[k].append(v)
    print(sum_scores)
    sum_scores = {key: np.mean(value) for (key, value) in sum_scores.items() if len(value) > 0}
    return sum_scores


def retrieve_original_entry(content: list[dict], _entry: dict) -> tuple[dict, int]:
    for j, qa_pair in enumerate(content):
        if _entry["question"].lower() == qa_pair["question"].lower():
            return qa_pair, j
    return {}, -1


def save_ragas_results(results: list[Result], file: str) -> None:
    global original_dataset
    avg_scores = calc_avg_scores(results)
    print(f"Avg Scores: {avg_scores}")
    for result in results:
        for i, entry in enumerate(result.dataset):
            orig_entry, index = retrieve_original_entry(original_dataset, entry)
            if "scores" not in orig_entry or isinstance(orig_entry["scores"], str):
                orig_entry["scores"] = {}

            for score, value in result.scores[i].items():
                orig_entry["scores"][score] = value
            original_dataset[index] = orig_entry

    with open(f"{EVALUATION_FOLDER}/{file}", "w+", encoding="utf-8") as f:
        json.dump(original_dataset, f)


def calculate_mean_and_std(file: str) -> tuple[dict, dict]:
    global original_dataset
    print(file)
    sum_scores = {
        "context_precision": [],
        "context_recall": [],
        "context_relevancy": [],
        "context_accuracy": [],
        "applied_context_precision": [],
        "applied_context_recall": [],
        "applied_context_relevancy": [],
        "bert_recall": [],
        "bert_precision": [],
        "bert_f1": [],
        "faithfulness": [],
        "answer_relevancy": [],
        "answer_similarity": [],
        "answer_correctness": [],
    }
    for _entry in original_dataset:
        if is_not_answered(_entry):
            continue
        if "scores" not in _entry or isinstance(_entry["scores"], str):
            _entry["scores"] = {}

        if _entry["scores"] != {}:
            for (k, v) in _entry["scores"].items():
                sum_scores[k].append(v)

    sum_scores_res = {key: [np.mean(value), np.std(value), np.median(value), len(value)]
                      for (key, value) in sum_scores.items()
                      if len(value) > 0}

    for key, value in sum_scores_res.items():
        print(f"{key}: {value}")
    print(" ")
    return sum_scores, sum_scores_res


def execute_metrics_evaluation(data: datasets, batch_size: int, metrics: list[Metric], failure_filename: str) \
        -> list[Result]:
    num_entries = data.shape[0]
    failure_path = f"{EVALUATION_FOLDER}/{failure_filename}"
    if not os.path.isfile(failure_path):
        with open(failure_path, "w", encoding="utf-8"):
            print("Created failure file: " + failure_path)

    collected_results = []
    for i in range(math.ceil(num_entries / batch_size)):
        if batch_size * (i + 1) < num_entries:
            _range = range(batch_size * i, batch_size * (i + 1))
        else:
            _range = range(batch_size * i, num_entries)

        subset = data.select(_range)
        try:
            result = ragas_evaluate(
                subset,
                metrics=metrics,
            )
            collected_results.append(result)
        except Exception as e:
            save_ragas_results(collected_results, failure_filename.replace("ragas_failure_", "rag_cos_"))
            print("Exception thrown as: " + str(e))

            with open(failure_path, "r", encoding="utf-8") as fr:
                try:
                    content = json.loads(fr.read())
                except json.JSONDecodeError:
                    content = []
                finally:
                    for j in range(len(subset)):
                        content.append(data[batch_size * i + j])

            with open(failure_path, "w", encoding="utf-8") as fw:
                json.dump(content, fw)
    return collected_results


def evaluate_rag_answer_similarity_with_ragas(file: str) -> None:
    batch_size = 5
    # ds = datasets.Dataset.from_generator(generate_answers_test_set, gen_kwargs={"file": file})
    ds = datasets.Dataset.from_generator(generate_answers_test_set)
    results = execute_metrics_evaluation(ds, batch_size=batch_size, metrics=[
        custom_answer_sim_model()
    ], failure_filename=file.replace("rag_cos_", "ragas_failure_"))
    save_ragas_results(results, file)


def evaluate_rag_relevancy_with_ragas(file: str) -> None:
    if "1024" in file:
        batch_size = 1
    else:
        batch_size = 2
    ds = datasets.Dataset.from_generator(generate_retrieval_test_set)
    results = execute_metrics_evaluation(ds, batch_size=batch_size, metrics=[
        context_relevancy
    ], failure_filename=file.replace("rag_cos_", "ragas_failure_").replace("ragas_eval_rag_answer_eval_",
                                                                           "ragas_failure_"))
    save_ragas_results(results, file)


def evaluate_rag_retrieval_with_ragas(file: str, batch_size: int = 1) -> None:
    if "128" in file:
        batch_size = 3
    ds = datasets.Dataset.from_generator(generate_unanswered_test_set)
    results = execute_metrics_evaluation(ds, batch_size=batch_size, metrics=[
        context_precision,
        context_recall
    ], failure_filename=file.replace("rag_cos_", "ragas_failure_").replace("ragas_eval_rag_answer_eval_",
                                                                           "ragas_failure_"))
    save_ragas_results(results, file)


def evaluate_rag_answers_with_ragas(file: str, batch_size: int = 1) -> None:
    ds = datasets.Dataset.from_generator(generate_answers_test_set)
    results = execute_metrics_evaluation(ds, batch_size=batch_size, metrics=[
        faithfulness,
        custom_answer_rel_model(),
        custom_answer_sim_model(),
        custom_answer_corr_model()
    ], failure_filename=file.replace("rag_cos_", "ragas_failure_").replace("ragas_eval_rag_answer_eval_",
                                                                           "ragas_failure_"))
    save_ragas_results(results, file)


def evaluate_rag_complete_with_ragas(file: str, batch_size: int = 3) -> None:
    batch_size = 1
    ds = datasets.Dataset.from_generator(generate_answers_test_set)
    results = execute_metrics_evaluation(ds, batch_size=batch_size, metrics=[
        context_precision,
        context_recall,
        context_relevancy,
        faithfulness,
        custom_answer_rel_model(),
        custom_answer_corr_model(),
        custom_answer_sim_model()
    ], failure_filename=file.replace("rag_cos_", "ragas_failure_"))
    save_ragas_results(results, file)


def evaluate_rag_with_traditional(file: str) -> None:
    global original_dataset
    bertscore = evaluate.load("bertscore")

    for i, entry in enumerate(original_dataset):
        retrieved_contexts = entry["contexts"]
        source_contexts = entry["meta"]["source_context"].replace(".  ", ". ").lower().strip().split(". ")
        matches = 0
        for retrieved_context in retrieved_contexts:
            retrieved_context = retrieved_context["content"].replace(".  ", ". ").lower().strip()
            if any(sentence in retrieved_context for sentence in source_contexts):
                matches += 1
                # print([sentence for sentence in source_contexts if sentence in retrieved_context])
        if "scores" not in entry or isinstance(entry["scores"], str):
            entry["scores"] = {}

        if matches == 0:
            entry["scores"]["context_accuracy"] = 0.0
        else:
            entry["scores"]["context_accuracy"] = 1.0

        if not is_not_answered(entry):
            predictions = [entry["answer"].lower()]
            references = [entry["ground_truth"].lower()]
            bert_results = bertscore.compute(predictions=predictions, references=references, lang="en",
                                             model_type="roberta-large")
            if "scores" not in entry:
                entry["scores"] = {}
            entry["scores"]["bert_recall"] = bert_results["precision"][0]
            entry["scores"]["bert_precision"] = bert_results["recall"][0]
            entry["scores"]["bert_f1"] = bert_results["f1"][0]

        original_dataset[i] = entry

    with open(f"{EVALUATION_FOLDER}/{file}", "w+", encoding="utf-8") as f:
        json.dump(original_dataset, f)


def retry_failed_evals(file: str):
    def make_dataset(file_content_fail, file_content_orig):
        yielded_entries = []
        for _entry in file_content_fail:
            orig_entry, _ = retrieve_original_entry(file_content_orig, _entry)
            if "context_relevancy" in orig_entry["scores"] and orig_entry["scores"]["context_relevancy"] > 0.0:
                continue
            for y_entry in yielded_entries:
                if _entry["question"] == y_entry["question"]:
                    continue
            yielded_entries.append(_entry)
            yield _entry

    with open(f"{EVALUATION_FOLDER}/ragas_eval_failure_subsets_ctrelevancy_{file.split('rag_answer_eval_')[1]}", "r",
              encoding="utf-8") as f:
        failed = json.loads(f.read())
    with open(f"{EVALUATION_FOLDER}/{file}", "r", encoding="utf-8") as f:
        orig = json.loads(f.read())

    batch_size = 1
    ds = datasets.Dataset.from_generator(make_dataset,
                                         gen_kwargs={"file_content_fail": failed, "file_content_orig": orig})
    results = execute_metrics_evaluation(ds, batch_size=batch_size, metrics=[
        context_relevancy
    ], failure_filename=file.split("rag_answer_eval_")[1]
                                         )
    save_ragas_results(results, file)


def compare_population_support(_files: list[str]):
    global original_dataset

    def generate_unanswered_ids():
        for _entry in original_dataset:
            if is_not_answered(_entry):
                yield _entry["meta"]["id_query"]

    def generate_answered_ids():
        for _entry in original_dataset:
            if not is_not_answered(_entry):
                yield _entry["meta"]["id_query"]

    pop = {}
    failed = []
    for file in _files:
        with open(f"{EVALUATION_FOLDER}/{file}", "r", encoding="utf-8") as fc:
            original_dataset = json.loads(fc.read())

        name = file.replace("rag_cos_", "").replace("ragas_eval_rag_answer_eval_", "")

        pop[name] = {
            "failed": list(generate_unanswered_ids()),
            "answered": list(generate_answered_ids())
        }
        if "128" in file:
            failed = pop[name]["failed"]

    # check which failed in all of them
    for entry, values in pop.items():
        print(values["failed"])
        failed = set(failed).intersection(values["failed"])

    print(failed)


if __name__ == "__main__":
    files = [
        # "ragas_eval_rag_answer_eval_128_topk5_gpt35.json",
        # "ragas_eval_rag_answer_eval_256_topk5_gpt35.json",
        # "ragas_eval_rag_answer_eval_512_topk5_gpt35.json",
        # "ragas_eval_rag_answer_eval_1024_topk5_gpt35.json",
        "rag_cos_128_topk5_gpt35.json",
        "rag_cos_256_topk5_gpt35.json",
        "rag_cos_512_topk5_gpt35.json",
        "rag_cos_1024_topk5_gpt35.json",
        "rag_cos_128_topk5_gpt4.json",
        "rag_cos_128_topk5_hybrid_gpt35.json",
        "rag_cos_128_topk5_qa_emb_gpt35.json",
        "rag_cos_128_topk5_asym_gpt35.json",
        # "ragas_eval_rag_answer_eval_128_topk10_gpt35.json",
        # "ragas_eval_rag_answer_eval_256_topk10_gpt35.json",
        # "ragas_eval_rag_answer_eval_512_topk10_gpt35.json",
        # "ragas_eval_rag_answer_eval_1024_topk10_gpt35.json"
    ]
    # compare_population_support(files)
    for _file in files:
        with open(f"{EVALUATION_FOLDER}/{_file}", "r", encoding="utf-8") as f:
            original_dataset = json.loads(f.read())
        evaluate_rag_with_traditional(_file)

    {'Q_G2_36', 'Q_G1_77', 'Q_G2_9', 'Q_G1_4', 'Q_G1_25', 'Q_G1_9', 'Q_G2_33', 'Q_G2_18', 'Q_G2_8', 'Q_G2_12', 'Q_R1_8',
     'Q_R1_11', 'Q_G1_24', 'Q_G2_20'}
    {'Q_G1_86', 'Q_G1_77', 'Q_G1_4', 'Q_G1_25', 'Q_G2_12', 'Q_G2_30', 'Q_G2_33', 'Q_G1_9'}
