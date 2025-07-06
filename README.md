# Emissions Benchmarking for the task of medical summarization

This project benchmarks multiple large language models (LLMs) on the task of medical and general text summarization, with a focus on evaluating their energy consumption, carbon emissions, and summarization quality. The goal is to analyze the environmental impact of AI-driven summarization tasks in healthcare and other domains, helping practitioners make informed, sustainable model choices.
The repository contains scripts to generate summaries across various datasets using different models, track real-time energy usage and carbon footprint during inference, and visualize the trade-offs between model performance and environmental cost. The insights from this study enable data scientists and healthcare AI practitioners to balance model accuracy, computational efficiency, and environmental sustainability when deploying language models in practical applications.

# Repository Structure

Dataset folders (d2n/, chq/, etc.): Contain all code and results specific to that dataset.

final/: Stores the generated summaries for each model, along with recorded energy consumption and carbon emission values.

plots/: Contains scatter plots showing carbon emissions vs. input/output tokens for each model on that dataset.

modelcomparisons/: Contains candlestick plots comparing all models across datasets on energy consumption and carbon emission metrics.

# Datasets Used:

- [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA)
- [D2N](https://github.com/StanfordMIMI/clin-summ/blob/main/data/d2n/test.jsonl)
- [CHQ](https://github.com/StanfordMIMI/clin-summ/blob/main/data/chq/test.jsonl)
- [OPI](https://github.com/StanfordMIMI/clin-summ/blob/main/data/opi/test.jsonl)
- [CNN/DailyMail](https://huggingface.co/datasets/abisee/cnn_dailymail)
- [XSum](https://huggingface.co/datasets/EdinburghNLP/xsum)
- [IN-ABS](https://zenodo.org/records/7152317#.Yz6mJ9JByC0)
- [UK-ABS](https://zenodo.org/records/7152317#.Yz6mJ9JByC0)

# Models Used:

- [LLaMA 2 7B](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- [LLaMA 3.1 8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- [Qwen 2.5 7B](https://huggingface.co/Qwen/Qwen2.5-7B)
- [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
