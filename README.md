# Evaluation of Retrieval Augmented Generation System utilizing Large Language Models

# Environment
Use Python 3.9 or 3.10 as haystack is not yet ready for 3.12


# Requirements
## Data Format
The document store has to be in the form of a list of entries in HuggingFace Dataset or JSON format with the following columns: <br>
```
Dataset({
    features: ['content'],
    num_rows: 153
})
dataset: Dataset
```

There may be additional key:value paris for each entry, however "content" is required containing your document chunks. Refer to one of the [chunk documents](src/data/Baldur_documents_128.json) of the thesis as a reference point.

The QA dataset has to be in the form of a list of entries in HuggingFace Dataset or JSON format with the following columns:<br>
```
Dataset({ 
    features: ['question']
    num_rows: 153
})
dataset: Dataset
```
Additional key value pairs such as ground_truth, context (for SourceContextAccuracy), or meta are optional. Refer to the [BG3 dataset](src/data/qa_dataset_bg3_lp.json) to see the reference setup.


The RAG results that is sent to Ragas for evaluation has to be in the form of a list of entries in HuggingFace Dataset or JSON format with the following columns:<br>
```
Dataset({ 
    features: ['question', 'contexts', 'answer', ground_truth'],
    num_rows: 153
})
dataset: Dataset
```
Ground truth may be empty, but some context and answer metrics will not work if that is the case. Refer to the [Ragas documentation](https://docs.ragas.io/en/latest/concepts/metrics/index.html) for help.

## Ragas Fixes
During the thesis I detected some issues inside the Ragas library which must be fixed locally to work. I'm not sure if Ragas is aware of these issues, however I have documented them here. <br><br>
Locally, go to your venv folder and then to the <code>lib/python3.10/site-packages/ragas/embeddings</code> folder and in <code>base.py</code> change 

```
def embed_documents(self, texts: List[str]) -> List[List[float]]:
    from sentence_transformers.SentenceTransformer import SentenceTransformer
    from torch import Tensor

    assert isinstance(
        self.model, SentenceTransformer
    ), "Model is not of the type Bi-encoder"
    embeddings = self.model.encode(
        texts, normalize_embeddings=True, **self.encode_kwargs
    )
    
    assert isinstance(embeddings, Tensor)
    return embeddings.tolist()
```
to 
<pre>
def embed_documents(self, texts: List[str]) -> List[List[float]]:
    from sentence_transformers.SentenceTransformer import SentenceTransformer
    from torch import Tensor, <b> from_numpy</b>  <<<-------

    assert isinstance(
        self.model, SentenceTransformer
    ), "Model is not of the type Bi-encoder"
    embeddings = self.model.encode(
        texts, normalize_embeddings=True, **self.encode_kwargs
    )
    <b> if not isinstance(embeddings, Tensor): </b>  <<<-------
    <b>     embeddings = from_numpy(embeddings) </b>  <<<-------

    assert isinstance(embeddings, Tensor)
    return embeddings.tolist()
</pre>

### JSON Datasets
Json files are loaded into dictionaries as a whole. This may work with small datasets such as this one but can become problematic with larger files due to Memory constraints. Please keep this in mind and change the code to meet your needs

# End-To-End RAG Pipeline
As outline by the thesis, the general setup is available as a parametrized Python file. The [main.py](src/main.py) file is available with various parameters which are explained here. All values next to the respective parameter are the default values.
```
files:
  --document_store_db sqlite:///db/faiss_doc_store_bg3_128_cos.db
                        Path and name for the FAISS SQL database
  --document_store_index db/faiss_doc_store_bg3_128_cos
                        Path and name for FAISS SQL index
  --path_to_document_store_file data/Baldur_documents_128.json
                        Path to FAISS document store in JSON format, only required if a new FAISS document store has to be created
  --path_to_dataset_file data/qa_dataset_bg3_lp.json
                        Path to your QA-Dataset file based on the document store in JSON format
  --apply_hf_dataset False
                        Do you want to use a HF dataset? If true create a genertor that converts the datasets document column to 'content'
  --name_hf_dataset wiki_qa
                        HuggingFace dataset name. If you want use this option use generators to create a HF dataset
  --output_path_rag evaluation/rag_result.json
                        Output path for RAG results (without ragas) in JSON format


RAG details:
  --embedding_model sentence-transformers/all-mpnet-base-v2
                        Huggingface Embedding model for RAG
  --synthesis_model gpt-3.5-turbo-16k-0613
                        LLM to answer the questions in the dataset after retrieval
  --perform_traditional_eval False
                        Evaluate RAG pipeline contexts and answers with BERTScore and SourceContentAccuracy
  --perform_ragas_eval_answer True
                        Evaluate RAG pipeline answers with ragas
  --perform_ragas_eval_context True
                        Evaluate RAG pipeline contexts with ragas
  --eval_batch_size 3
                        Batch size for ragas evaluation
  --top_k 5         
                        TopK value for RAG retrieval
```

An example of how the pipeline can be executed is found below. Please change the directory to src before executing the file, otherwise the links inside the scripts won't work :D

```
python main.py --document_store_db sqlite:///db/faiss_doc_store_bg3_128_cos.db --document_store_index db/faiss_doc_store_bg3_128_cos --path_to_dataset_file data/qa_dataset_bg3_lp.json --output_path_rag evaluation/rag_result.json --perform_ragas_eval_answer True --perform_ragas_eval_context True --top_k 5
```

# Evaluation
All experiment results that were collected during the thesis are stored in the [src/evaluation](src/evaluation) folder. A list of each experiments pipelines their responsing json file is found below:

| Experiment           | Pipeline Configs                                         | File Name                           |
|----------------------|----------------------------------------------------------|-------------------------------------|
| Benchmark            | Chunk size = 128 TopK = 5                                | rag_cos_128_topk5_gpt35.json        |
| Chunk Size & TopK    | Chunk size = 256 TopK = 5                                | rag_cos_256_topk5_gpt35.json        |
| Chunk Size & TopK    | Chunk size = 512 TopK = 5                                | rag_cos_512_topk5_gpt35.json        | 
| Chunk Size & TopK    | Chunk size = 1024 TopK = 5                               | rag_cos_1024_topk5_gpt35.json       |
| Chunk Size & TopK    | Chunk size = 128 TopK = 10                               | rag_eval_128_topk10_gpt35.json      |
| Chunk Size & TopK    | Chunk size = 256 TopK = 10                               | rag_eval_256_topk10_gpt35.json      |
| Chunk Size & TopK    | Chunk size = 512 TopK = 10                               | rag_eval_512_topk10_gpt35.json      |
| Chunk Size & TopK    | Chunk size = 1024 TopK = 10                              | rag_eval_1024_topk10_gpt35.json     |
| Distance Calculation | Chunk size = 128 TopK = 5 Dot Product                    | rag_dot_128_topk5_gpt35.json        |
| Hybrid Search        | Chunk size = 128 TopK = 5 Hybrid Search                  | rag_cos_128_topk5_hybrid_gpt35.json |
| Embedding Model      | Chunk size = 128 TopK = 5 QA-Embedding Model (symmetric) | rag_cos_128_topk5_qa_emb_gpt35.json |
| Embedding Model      | Chunk size = 128 TopK = 5 Embedding Model (asymmetric)   | rag_cos_128_topk5_asym_gpt35.json   |
| Answer LLM           | Chunk size = 128 TopK = 5 GPT-4 Answer Model             | rag_cos_128_topk5_gpt4.json         |