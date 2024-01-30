import json
import math
import os
import time
from pathlib import Path
from typing import List, Optional, Union

import nltk
import numpy as np
import openai
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.base import BaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from openai.types.chat import ChatCompletion
from ragas.llms import LangchainLLM
from ragas.testset import TestsetGenerator

COST_CHATGPT = 0.002 / 1000  # Cost per 1000 Token = 0.002
MODEL_CHATGPT = "gpt-3.5-turbo"
MODEL_CHATGPT_16k = "gpt-3.5-turbo-16k-0613"
MODEL_GPT4 = "gpt-4"
MODEL_GPT4_128k = "gpt-4-1106-preview"
GPT_SYSTEM = "system"
GPT_USER = "user"
GPT_ASSISTANT = "assistant"

INPUT_TEXT_INDEX = "Provide index: "
INPUT_TEXT_QUESTION = "Ask question: "
INPUT_TEXT_STOP = "stop"

DATASET_FILE = "data/qa_dataset_bg3_lp.json"

openai_api_key = os.getenv("OPENAI_API_KEY")


class JSONLoader(BaseLoader):  # see https://github.com/langchain-ai/langchain/issues/4396
    def __init__(
            self,
            file_path: Union[str, Path],
            content_key: Optional[str] = None,
    ):
        self.file_path = Path(file_path).resolve()
        self._content_key = content_key

    @property
    def load(self) -> List[Document]:
        """Load and return documents from the JSON file."""

        docs = []
        # Load JSON file
        with open(self.file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

            # Iterate through 'pages'
            for doc in data:
                id = doc["id"]
                lp_part = doc["lp_part"]
                tokens = doc["length"]
                content = doc["content"]
                metadata = dict(
                    tokens=tokens,
                    id=id,
                    lp_part=lp_part
                )
                docs.append(Document(page_content=content, metadata=metadata))
        return docs


def init_gpt():
    """
    :return:
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    msgs = [
        {"role": GPT_SYSTEM,
         "content": "You are a helpful assistant that generates questions about a given youtube transcript section that a user might"
                    "ask which has not seen the corresponding youtube video. The questions must be able to be "
                    "inferred by the provided section and be reasonably detailed. Follow the rules given in the prompt."}
    ]
    return msgs


def append_message(llm_hist: list, user: str, content: str) -> list:
    llm_hist.append({"role": user, "content": content})
    return llm_hist


def send_request(msgs, model) -> ChatCompletion:
    return openai.chat.completions.create(
        model=model,
        messages=msgs,
        temperature=0.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )


def generate_questions_with_input(llm_hist: list):
    question = input(INPUT_TEXT_QUESTION)
    model_input = "Generate a question based on the context provided below:\n\n"

    while question.lower() != INPUT_TEXT_STOP:
        llm_hist = append_message(llm_hist, GPT_USER, model_input + question)

        chat = send_request(llm_hist, MODEL_CHATGPT_16k)
        reply = chat.choices[0].message.content

        print(f"Reply: {reply}")
        print(f"Question-Tokens: {len(nltk.word_tokenize(question))}")
        print(f"Reply-Tokens: {len(nltk.word_tokenize(reply))}")
        llm_hist = append_message(llm_hist, GPT_ASSISTANT, reply)

        question = input(INPUT_TEXT_QUESTION)


def prepare_transcript(path: str, transcript: str, segment_len: int = 8192) -> None:
    def init_gpt_transcript():
        """
        :return:
        """
        openai.api_key = os.getenv("OPENAI_API_KEY")
        msgs = [
            {"role": GPT_SYSTEM,
             "content": "You are a helpful assistant that helps to transform a youtube transcript to be more readable by removing unnecessary"
                        "linebreaks, combining lines, and adding punctuation without altering, reordering or removing "
                        "any actual text content."}
        ]
        return msgs

    prompt = ("Transform  the following transcript into a coherent text by removing the newlines and using letter "
              "casing without removing, altering or reordering any content.\n\n")
    content = open(path, "r", encoding="utf-8").read()

    fixed = ""
    f = open(f"data/{transcript}", "a", encoding="utf-8")
    general_length = segment_len
    for i in range(math.ceil(len(content) / general_length)):
        try:
            content_slice = content[general_length * i: general_length * (i + 1)]
        except Exception:
            content_slice = content[general_length * i: len(content) - 1]

        hist = init_gpt_transcript()
        hist = append_message(hist, GPT_USER, prompt + content_slice)

        chat = send_request(hist)
        reply = chat.choices[0].message.content
        fixed += reply
        print(reply)
    fixed = fixed.replace("\n\n", " ").replace("Got it! Here's the transformed transcript:", "").replace('"', ' ')
    f.write(fixed)
    f.close()


def fix_transcripts_ner(content: str):
    def change_ner(cont: str, incorrect_names: list[str], correct_name: str):
        for wrong in incorrect_names:
            cont = cont.replace(wrong, correct_name)
        return cont

    content = content.replace("\n\n", " ").replace("Got it! Here's the transformed transcript:",
                                                   "").replace('"', ' ')

    content = change_ner(content, ["Sterion", "Starion", "a sterion", "a Starion", "astario", "Astarian", "Astorian",
                                   "Astorion"],
                         "Asterion")
    content = change_ner(content,
                         ["Karlak", "karlak", "carlac", "Carlak", "Carlock", "carlak", "Carlex", "Carla", "Kalak",
                          "Colac", "Harlak", "Karlax", "Karlux"], "Karlach")

    content = change_ner(content, ["tiefling", "teethling", "Deepling", "Teethlean", "tieflines", "heathling"],
                         "Tiefling")
    content = change_ner(content, ["Gail", "Gal", "Galeeeeee", "Galee"], "Gale")
    content = change_ner(content, ["Lazelle", "Lazel", "lizelle", "Lizelle", "Liesel", "Lae'zell", "Blazelle"],
                         "Lae'zel")
    content = change_ner(content, ["Geth Yankee", "get the Yankee", "githyanki", "Yankee", "Gethianki"], "Githyanki")
    content = change_ner(content, ["Kaga", "Korga", "Corga", "Koga"], "Kagha")
    content = change_ner(content, ["Holsin", "Holson", "Holston", "Helson", "Helsin", "Howsin", "Elsin"], "Halsin")
    content = change_ner(content, ["Sylvanas", "Sylvana"], "Silvanus")
    content = change_ner(content, ["Mazura", "Mazora", "Zario", "Missouri"], "Mizora")
    content = change_ner(content, ["Zario", "zaryel", "Zarielle"], "Zariel")
    content = change_ner(content, ["shadowheart", "shed Art's"], "Shadowheart")
    content = change_ner(content, ["Char", "Shark", "Shah"], "Shar")
    content = change_ner(content, ["Marion", "Larion", "larian"], "Larian")  # Developer of BG3
    content = change_ner(content, ["Zoru"], "Zorru")
    content = change_ner(content, ["Vlakis", "Vlakid", "Lackath", "Vlakith", "Vlakath", "Flakketh"], "Vlaakith")
    content = change_ner(content, ["Zevlorr", "Zebler", "Zevolor"], "Zevlor")
    content = change_ner(content, ["jurgle", "Jergel"], "Jergal")
    content = change_ner(content, ["Seluna", "Soluna", "Selenite"], "Selune")
    content = content.replace("’", "'")
    return content


def clean_transcripts():
    files = range(1, 6)
    for file in files:
        path = f"data/Baldur_LP_P{file}.txt"
        with open(path, "r", encoding="utf-8") as fr:
            content = fr.read()
        content = fix_transcripts_ner(content)

        with open(path, "w", encoding="utf-8") as fw:
            fw.write(content)

    with open(DATASET_FILE, "r", encoding="utf-8") as fr:
        content = json.loads(fr.read())

    for i, entry in enumerate(content):
        for key, value in entry.items():
            if key != "meta":
                entry[key] = fix_transcripts_ner(value)
        content[i] = entry

    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        json.dump(content, f)


def generate_questions(file: str) -> None:
    num_sentences = 5
    content = open(file, "r", encoding="utf-8").read().split(".")
    model_input = ("Generate a question based on the sentences provided below. The generated questions must be able "
                   "to be inferred from the sentences:\n\n")
    index = 0
    dataset = []
    for i in range(math.ceil(len(content) / num_sentences)):
        sentences = ".".join(content[index:index + num_sentences])

        llm_hist = init_gpt()
        llm_hist = append_message(llm_hist, GPT_USER, model_input + sentences)
        chat = send_request(llm_hist, MODEL_CHATGPT_16k)
        reply = chat.choices[0].message.content

        print(reply)

        dataset.append({"content": sentences,
                        "length": len(nltk.word_tokenize(sentences)),
                        "query": reply,
                        "id": i
                        })

        index += num_sentences

    with open(f'{file.split(".")[0]}.json', "w", encoding="utf-8") as fp:
        json.dump(dataset, fp)


def generate_rag_documents_by_length():
    len_documents = [128, 256, 512, 1024]
    for length in len_documents:
        dataset = []
        for i in range(1, 6):
            file = f"data/Baldur_LP_P{i}.txt"
            with open(file, "r", encoding="utf-8") as fp:
                content = fp.read()
            token_sum = 0
            current_doc = ""
            num_doc = 0
            sentence_start = 0
            for j, entry in enumerate(content.split(".")):
                if token_sum < length:
                    current_doc += entry + ". "
                    token_sum += len(nltk.word_tokenize(entry))
                else:
                    sentence_end = j
                    dataset.append({
                        "content": current_doc,
                        "length": token_sum,
                        "id": f"{i}.{num_doc}",
                        "start": sentence_start,
                        "end": sentence_end,
                        "lp_part": i
                    })
                    current_doc = entry + ". "
                    token_sum = len(nltk.word_tokenize(entry))
                    num_doc += 1
                    sentence_start = j + 1

            with open(f"data/Baldur_documents_{length}.json", "w", encoding="utf-8") as f:
                json.dump(dataset, f)


def check_long_segments():
    clean_transcripts()
    generate_rag_documents_by_length()
    file = "data/Baldur_documents_128.json"
    with open(file, "r", encoding="utf-8") as fr:
        content = json.loads(fr.read())
    len_min = 10000
    len_max = 0
    for entry in content:
        length = entry["length"]
        len_min = length if length < len_min else len_min
        len_max = length if length > len_max else len_max
        if length > 190:
            print(entry["content"].split("."))
            print(entry["lp_part"])
            print(entry["id"])
            print(entry["length"])
    print(f"{len_min} {len_max}")


def generate_with_ragas():
    file_path = "data/Baldur_documents_.json"
    from langchain.document_loaders import PubMedLoader

    loader = JSONLoader(file_path=file_path)
    data = loader.load
    # Add custom llms and embeddings
    generator_llm = LangchainLLM(llm=ChatOpenAI(model="gpt-3.5-turbo-16k"))
    critic_llm = LangchainLLM(llm=ChatOpenAI(model="gpt-3.5-turbo-16k"))
    embeddings_model = OpenAIEmbeddings()

    # Change resulting question type distribution
    testset_distribution = {
        "simple": 0.25,
        "reasoning": 0.5,
        "multi_context": 0.0,
        "conditional": 0.25,
    }

    # percentage of conversational question
    chat_qa = 0.2

    test_generator = TestsetGenerator(
        generator_llm=generator_llm,
        critic_llm=critic_llm,
        embeddings_model=embeddings_model,
        testset_distribution=testset_distribution,
        chat_qa=chat_qa,
        chunk_size=1024
    )

    testset = test_generator.generate(data, test_size=5)

    with open("data/testset_ragas.json", "w", encoding="utf-8") as f:
        json.dump(testset.test_data, f)


def generate_questions_from_documents(file: str):
    def generate_question(entry: dict):
        llm_hist = init_gpt()
        llm_hist = append_message(llm_hist, GPT_USER, model_input + entry["content"])
        chat = send_request(llm_hist, MODEL_GPT4)
        response = chat.choices[0].message.content

        print(response)
        return response

    def generate_context(question: str, entry: dict):
        answer_prompt = CONTEXT_FORMULATE.replace("{question}", question).replace("{context}", entry["content"])
        llm_hist = init_gpt()
        llm_hist = append_message(llm_hist, GPT_USER, answer_prompt)
        chat = send_request(llm_hist, MODEL_CHATGPT_16k)
        response = chat.choices[0].message.content

        print(f"Question: {question}")
        print(response)
        response = response.replace("Sentences: ", "")
        return response

    def generate_answer(question: str, context: str):
        answer_prompt = ANSWER_FORMULATE.replace("{question}", question).replace("{context}", context)
        llm_hist = init_gpt()
        llm_hist = append_message(llm_hist, GPT_USER, answer_prompt)
        chat = send_request(llm_hist, MODEL_CHATGPT_16k)
        response = chat.choices[0].message.content

        print(response)
        response = response.replace("Answer: ", "")
        return response

    documents = json.loads(open(file, "r", encoding="utf-8").read())
    model_input = (
        "Your task is to formulate up to 5 questions from the given context of a YouTube transcript satisfying the rules given below:\n"
        "   1.The question should make sense to humans even when read without the given context.\n"
        "   2.The question should be able to be fully answered from the given context.\n"
        "   3.The question should be framed from a part of the context that contains important information.\n"
        "   4.The question should be of moderate difficulty.\n"
        "   5.The question must be reasonable and must be understood and responded by humans.\n"
        "   6.Avoid framing questions that contain 'the speaker'.\n"
        "   7.Do not use phrases like 'provided context', 'mentioned in the transcript section' etc in the question\n"
        "   8.Avoid framing questions using the word 'and' that can be decomposed into more than one question.\n"
        "   9.The question should not contain more than 15 words, make use of abbreviations wherever possible.\n"
        "Desired format: \n"
        "Question: -||-\n\n"

        "Transcript Section: ")
    CONTEXT_FORMULATE = (
        "Please extract relevant sentences from the provided context that can potentially help answer the following question. Adhere to the following rules:\n"
        "   1.While extracting candidate sentences you're not allowed to make any changes to sentences from given "
        "   context. \n"
        "   2.The sentences should not contain 'the speaker said', 'the speaker did' etc\n"
        "   3.The sentences should be sufficient to answer the given question.\n"
        "   4.Select at most 5 sentences.\n\n"
        "Question:{question}"
        "Context:{context}\n\n"
        "Desired format: \n"
        "Sentences: -||-\n\n")

    ANSWER_FORMULATE = (
        "Answer the question using the information from the given context, satisfying the rules given below: \n"
        "   1. The answer should not contain 'the speaker said', 'the speaker did' etc\n"
        "   2. The answer should solely be generated by the given context.\n"
        "   3. The answer should not contain more than 25 words.\n"

        "Question:{question}"
        "Context:{context}"

        "Desired format: \n"
        "Answer: -||-\n")

    "   10.Extract at least three relevant sentences from the provided context that can potentially help answer the question. While "
    "   extracting candidate sentences you're not allowed to make any changes to sentences from given context. \n\n"
    dataset = []
    try:
        for i, doc in enumerate(documents):
            if doc["lp_part"] <= 2:
                continue
            time.sleep(20)
            reply = generate_question(doc)
            questions = reply.split("\n\n")
            for j, q in enumerate(questions):
                split_content = q.split("\n")
                query = split_content[0].replace("Question: ", "")
                sentences = generate_context(query, doc)
                print("")
                answer = generate_answer(query, sentences)
                print("\n")
                dataset.append({"context": sentences,
                                "context_length": len(nltk.word_tokenize(sentences)),
                                "doc_id": doc["id"],
                                "doc": doc["content"],
                                "query": query,
                                "answer": answer,
                                "id": f"QA{doc['lp_part']}.{i}.{j}"
                                })

        with open("data/qa_dataset_further.json", "w", encoding="utf-8") as fp:
            json.dump(dataset, fp)
    except Exception:
        with open("data/qa_dataset_further.json", "w", encoding="utf-8") as fp:
            json.dump(dataset, fp)


def filter_usable_questions():
    documents = json.loads(open("data/Baldur_documents.json", encoding="utf-8").read())
    qas = json.loads(open("data/qa_dataset.json", "r", encoding="utf-8").read())
    usable_dataset = []
    for qa in qas:
        print(qa["query"])
        print(qa["context"])
        res = input("Good: ")
        if res == "yes" or res == "y":
            usable_dataset.append(qa)

    with open("data/qa_dataset_usable.json", "w", encoding="utf-8") as fp:
        json.dump(usable_dataset, fp)


def define_ground_truths():
    qas = json.loads(open("data/qa_dataset_usable.json", encoding="utf-8").read())
    for i, qa in enumerate(qas):
        if "ground_truth" in qa:
            continue
        print(qa["query"])
        print(qa["context"])
        ground_truth = input("Ground Truth: ")

        if ground_truth == "stop":
            with open("data/qa_dataset_usable.json", "w", encoding="utf-8") as fp:
                json.dump(qas, fp)
            break
        if ground_truth == "skip":
            continue

        correct = input(f"Is this good: \n{ground_truth}\n\n: ")
        while correct == "n" or correct == "no":
            ground_truth = input("Ground Truth: ")
            correct = input(f"Is this good: \n{ground_truth}\n\n: ")
        qa["ground_truth"] = ground_truth
        qas[i] = qa

    with open("data/qa_dataset_usable.json", "w", encoding="utf-8") as fp:
        json.dump(qas, fp)


def improve_questions():
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        content = json.loads(f.read())

    for i, qa in enumerate(content):
        print(f"{i}: {qa['question']}")
        change = input("Change: ")
        if change == "stop":
            with open(DATASET_FILE, "w", encoding="utf-8") as f:
                json.dump(content, f)
            break
        elif change == "skip":
            continue
        elif change == "y":
            changed_q = input("New question: ")
            ok = input(f"OK: {changed_q}")
            while ok != "y":
                changed_q = input("New question: ")
                ok = input(f"OK: {changed_q}")
            qa["question"] = changed_q
            content[i] = qa

    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        json.dump(content, f)


if __name__ == "__main__":
    # prepare_transcript("C:\\Users\\Samuel\\Downloads\\Baldur's Gate 3 - Let's Play Part 3.txt", 10000)
    # generate_questions("data/Baldur_LP_P1.txt")
    # generate_questions_from_documents("data/Baldur_documents_1024.json")
    # check_long_segments()
    # generate_with_ragas()
    clean_transcripts()
    # generate_rag_documents_by_length()
    # improve_questions()
    # filter_usable_questions()
    # content = json.loads(open("data/qa_dataset_usable.json", "r").read())
    # prepare_transcript("C:\\Users\\Samuel\\Downloads\\Baldur's Gate 3 - Let's Play Part 7.txt", "Baldur_LP_P7.txt", 10000)
    # define_ground_truths()
    # for chunk in [128, 256, 512, 1024]:
    #     with open(f"data/Baldur_documents_{chunk}.json") as f:
    #         data = json.loads(f.read())
    #     _min = 10000
    #     _max = 0
    #     coll = []
    #     for entry in data:
    #         if entry["length"] < _min:
    #             _min = entry["length"]
    #         if entry["length"] > _max:
    #             _max = entry["length"]
    #         coll.append(entry["length"])
    #
    #     print(f"Max: {_max}")
    #     print(f"Min: {_min}")
    #     print(f"Avg: {np.mean(coll)}")
    #     print(f"Std: {np.std(coll)}")
    #     print(f"Num: {len(coll)}")
