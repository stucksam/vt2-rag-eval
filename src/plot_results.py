import json

import numpy as np
from matplotlib import pyplot as plt

from src.evaluate_pipeline_results import EVALUATION_FOLDER, is_not_answered

original_dataset: list[dict] = [{}]


def calculate_mean_and_std(file: str) -> tuple[dict, dict]:
    global original_dataset
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
        if "scores" not in _entry or isinstance(_entry["scores"], str):
            _entry["scores"] = {}

        if _entry["scores"] != {}:
            for (k, v) in _entry["scores"].items():
                sum_scores[k].append(v)

            if not is_not_answered(_entry) and _entry["scores"]["faithfulness"] < 0.6 and _entry["scores"]["bert_f1"] > 0.9:
                print(_entry["meta"]["id_query"])

    sum_scores_res = {key: [np.mean(value), np.std(value), np.median(value), len(value)] for (key, value) in sum_scores.items() if len(value) > 0}
    print(file)
    for key, value in sum_scores_res.items():
        print(f"{key}: {value}")
    return sum_scores, sum_scores_res


def prompt_graph(files: list[str]):
    results = []
    for file in files:
        with open(f"{EVALUATION_FOLDER}/{file}", "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())
        name = file.split("ragas_eval_rag_answer_eval_")[1].replace(".json", "")
        results.append({name: dataset[0]["ragas_avg_score"]})
    metrics = ["faithfulness", "answer_relevancy", "answer_similarity", "answer_correctness", "context_relevancy",
               "context_recall", "context_precision"]
    for metric in metrics:
        for entry in results:
            for k, v in entry.items():
                plt.scatter(v[metric], k)
        plt.title(metric)
        plt.show()


def _plot(file: str):
    def get_values(_score):
        return scores[_score], scores_res[_score][0], scores_res[_score][2]

    global original_dataset
    scores, scores_res = calculate_mean_and_std(file)
    plt.style.use('ggplot')
    _, axs = plt.subplots(1, 4, figsize=(15, 5))
    hist_scores, mean, median = get_values("context_precision")
    axs[0].hist(hist_scores, bins=20, color='#6488ea')
    axs[0].set_title("Context Precision@k")
    axs[0].axvline(median, color='orange', linestyle='dashed', linewidth=1, label=f"median: {round(median, 2)}")
    axs[0].axvline(mean, color='r', linestyle='dashed', linewidth=1, label=f"mean {round(mean, 2)}")
    axs[0].legend(loc="upper left")

    hist_scores, mean, median = get_values("context_recall")
    axs[1].hist(hist_scores, bins=20, color='#6488ea')
    axs[1].set_title('Context Recall')
    axs[1].axvline(median, color='orange', linestyle='dashed', linewidth=1, label=f"median: {round(median, 2)}")
    axs[1].axvline(mean, color='r', linestyle='dashed', linewidth=1, label=f"mean {round(mean, 2)}")
    axs[1].legend(loc="upper left")

    hist_scores, mean, median = get_values("context_relevancy")
    axs[2].hist(hist_scores, bins=20, color='#6488ea')
    axs[2].axvline(median, color='orange', linestyle='dashed', linewidth=1, label=f"median: {round(median, 2)}")
    axs[2].axvline(mean, color='r', linestyle='dashed', linewidth=1, label=f"mean {round(mean, 2)}")
    axs[2].set_title("Context Relevancy")
    axs[2].legend(loc="upper right")

    hist_scores, mean, median = get_values("context_accuracy")
    axs[3].hist(hist_scores, bins=20, color='#6488ea')
    axs[3].axvline(median, color='orange', linestyle='dashed', linewidth=1, label=f"median: {round(median, 2)}")
    axs[3].axvline(mean, color='r', linestyle='dashed', linewidth=1, label=f"mean {round(mean, 2)}")
    axs[3].set_title("Source Context Accuracy")
    axs[3].legend(loc="upper right")
    axs.set_xlabel("Scores for Metric")
    axs.set_ylabel("Number of samples")

    plt.legend()
    plt.savefig("retrieval_distro_answered.png", bbox_inches='tight')
    plt.show()


    plt.style.use('ggplot')

    _, axs = plt.subplots(2, 3, figsize=(15, 8))
    hist_scores, mean, median = get_values("faithfulness")
    axs[0, 0].hist(hist_scores, bins=20, color='#6488ea')
    axs[0, 0].set_title("Faithfulness")
    axs[0, 0].axvline(median, color='orange', linestyle='dashed', linewidth=1, label=f"median: {round(median, 2)}")
    axs[0, 0].axvline(mean, color='r', linestyle='dashed', linewidth=1, label=f"mean {round(mean, 2)}")
    axs[0, 0].legend(loc="upper left")

    hist_scores, mean, median = get_values("answer_relevancy")
    axs[0, 1].hist(hist_scores, bins=20, color='#6488ea')
    axs[0, 1].set_title('Answer Relevance')
    axs[0, 1].axvline(median, color='orange', linestyle='dashed', linewidth=1, label=f"median: {round(median, 2)}")
    axs[0, 1].axvline(mean, color='r', linestyle='dashed', linewidth=1, label=f"mean {round(mean, 2)}")
    axs[0, 1].legend(loc="upper left")

    hist_scores, mean, median = get_values("answer_correctness")
    axs[0, 2].hist(hist_scores, bins=20, color='#6488ea')
    axs[0, 2].axvline(median, color='orange', linestyle='dashed', linewidth=1, label=f"median: {round(median, 2)}")
    axs[0, 2].axvline(mean, color='r', linestyle='dashed', linewidth=1, label=f"mean {round(mean, 2)}")
    axs[0, 2].set_title("Answer Correctness")
    axs[0, 2].legend(loc="upper right")

    hist_scores, mean, median = get_values("answer_similarity")
    axs[1, 0].hist(hist_scores, bins=20, color='#6488ea')
    axs[1, 0].axvline(median, color='orange', linestyle='dashed', linewidth=1, label=f"median: {round(median, 2)}")
    axs[1, 0].axvline(mean, color='r', linestyle='dashed', linewidth=1, label=f"mean {round(mean, 2)}")
    axs[1, 0].set_title("Answer Similarity")
    axs[1, 0].legend(loc="upper left")

    hist_scores, mean, median = get_values("bert_f1")
    axs[1, 1].hist(hist_scores, bins=20, color='#6488ea')
    axs[1, 1].axvline(median, color='orange', linestyle='dashed', linewidth=1, label=f"median: {round(median, 2)}")
    axs[1, 1].axvline(mean, color='r', linestyle='dashed', linewidth=1, label=f"mean {round(mean, 2)}")
    axs[1, 1].set_title("BERTScore F1")
    axs[1, 1].legend(loc="upper right")

    axs.set_xlabel("Scores for Metric")
    axs.set_ylabel("Number of samples")
    plt.legend()
    plt.savefig("answer_distro_answered.png", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    files = [
        "rag_cos_128_topk5_gpt35.json",
    ]
    # compare_population_support(files)
    for _file in files:
        with open(f"{EVALUATION_FOLDER}/{_file}", "r", encoding="utf-8") as f:
            original_dataset = json.loads(f.read())
        _plot(_file)
