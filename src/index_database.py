import json

from datasets import Dataset, load_dataset
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_MODEL_QA = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
EMBEDDING_MODEL_ASYM = "sentence-transformers/msmarco-distilbert-base-v4"
DOCUMENT_STORE_DB = "sqlite:///db/faiss_doc_store_bg3_128_cos.db"
DOCUMENT_STORE_INDEX = "db/faiss_doc_store_bg3_128_cos"
DOCUMENT_STORE_DB_QA_EMB = "sqlite:///db/faiss_doc_store_bg3_128_cos_qa_emb.db"
DOCUMENT_STORE_INDEX_QA_EMB = "db/faiss_doc_store_bg3_128_cos_qa_emb"

DOCUMENT_STORE_DB_ASYM_EMB = "sqlite:///db/faiss_doc_store_bg3_128_cos_async_emb.db"
DOCUMENT_STORE_INDEX_ASYM_EMB = "db/faiss_doc_store_bg3_128_cos_async_emb"

document_store: FAISSDocumentStore


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


def delete_document_store():
    global document_store
    document_store = load_db()
    document_store.delete_documents()


def import_dataset_from_generator(generator, document_store_db: str = DOCUMENT_STORE_DB,
                                  document_store_index: str = DOCUMENT_STORE_INDEX,
                                  embedding_model: str = EMBEDDING_MODEL, delete_store: bool = False) \
        -> FAISSDocumentStore:
    global document_store
    if delete_store:
        delete_document_store()

    dataset = Dataset.from_generator(generator)
    document_store = FAISSDocumentStore(document_store_db, similarity="cosine")
    document_store.write_documents(dataset)  # required by haystack that a dataset contains a "content" column

    retriever = EmbeddingRetriever(document_store=document_store,
                                   embedding_model=embedding_model,
                                   use_gpu=True)

    document_store.update_embeddings(retriever)

    # Save FAISS Index
    document_store.save(document_store_index)

    return document_store


def import_dataset_from_dict(documents, document_store_db: str = DOCUMENT_STORE_DB,
                             document_store_index: str = DOCUMENT_STORE_INDEX,
                             embedding_model: str = EMBEDDING_MODEL, delete_store: bool = False) -> FAISSDocumentStore:

    global document_store
    if delete_store:
        delete_document_store()

    document_store = FAISSDocumentStore(document_store_db, similarity="cosine")
    document_store.write_documents(documents)  # required by haystack that a dataset contains a "content" column

    retriever = EmbeddingRetriever(document_store=document_store,
                                   embedding_model=embedding_model,
                                   use_gpu=True)

    document_store.update_embeddings(retriever)

    # Save FAISS Index
    document_store.save(document_store_index)

    return document_store


def load_db(document_store_index: str = DOCUMENT_STORE_INDEX):
    return FAISSDocumentStore(faiss_index_path=document_store_index, faiss_config_path=f"{document_store_index}.json")


if __name__ == "__main__":
    _dataset = json.loads(open("data/Baldur_documents_128.json").read())
    import_dataset_from_dict(_dataset, document_store_db=DOCUMENT_STORE_DB_ASYM_EMB,
                             document_store_index=DOCUMENT_STORE_INDEX_ASYM_EMB, delete_store=False,
                             embedding_model=EMBEDDING_MODEL_ASYM)
