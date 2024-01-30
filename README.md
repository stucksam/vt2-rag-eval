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

There may be additional key:value paris for each entry, however "content" is required containing your document chunks. 

The QA dataset has to be in the form of a list of entries in HuggingFace Dataset or JSON format with the following columns:<br>
```
Dataset({ 
    features: ['question','contexts','answer','ground_truth'],
    num_rows: 153
})
dataset: Dataset
```

Ground truth may be empty but is required for certain answer metrics, please refer to the ragas documentation for further information

## JSON Datasets
Json is loaded into dictionaries as a whole. This may work with small datasets such as this one but can become problematic with larger files due to Memory cosntraints. Please keep this in mind and change the code to meet your needs

# End-To-End RAG Pipeline
As outline by the thesis, the general setup is available as a parametrized Python file. The [main.py](src/main.py) file is available with various parameters which are explained here


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