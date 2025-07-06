# Emissions Benchmarking for the task of medical summarization

This project benchmarks multiple large language models (LLMs) on the task of medical and general text summarization, with a focus on evaluating their energy consumption, carbon emissions, and summarization quality. The goal is to analyze the environmental impact of AI-driven summarization tasks in healthcare and other domains, helping practitioners make informed, sustainable model choices.
The repository contains scripts to generate summaries across various datasets using different models, track real-time energy usage and carbon footprint during inference, and visualize the trade-offs between model performance and environmental cost. The insights from this study enable data scientists and healthcare AI practitioners to balance model accuracy, computational efficiency, and environmental sustainability when deploying language models in practical applications.

# Repository Structure

Dataset folders (d2n/, chq/, etc.): Contain all code and results specific to that dataset.

final/: Stores the generated summaries for each model, along with recorded energy consumption and carbon emission values.

plots/: Contains scatter plots showing carbon emissions vs. input/output tokens for each model on that dataset.

modelcomparisons/: Contains candlestick plots comparing all models across datasets on energy consumption and carbon emission metrics.

# Datasets Used

- [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA): A biomedical question-answering dataset derived from PubMed abstracts, focusing on clinical yes/no/maybe questions and their explanations.
- [D2N](https://github.com/StanfordMIMI/clin-summ/blob/main/data/d2n/test.jsonl): Clinical summarization dataset containing discharge notes (D) and their summaries (N), designed for hospital discharge narrative summarization.
- [CHQ](https://github.com/StanfordMIMI/clin-summ/blob/main/data/chq/test.jsonl): Consumer Health Questions dataset containing health-related questions and long-form expert answers.
- [OPI](https://github.com/StanfordMIMI/clin-summ/blob/main/data/opi/test.jsonl): Online Patient Information dataset with medical queries and publicly available online health content summaries.
- [CNN/DailyMail](https://huggingface.co/datasets/abisee/cnn_dailymail): A news summarization dataset containing news articles paired with bullet-point summaries, widely used for general-domain summarization tasks.
- [XSum](https://huggingface.co/datasets/EdinburghNLP/xsum): A highly abstractive single-sentence summarization dataset from BBC articles covering diverse topics.
- [IN-ABS](https://zenodo.org/records/7152317#.Yz6mJ9JByC0)
- [UK-ABS](https://zenodo.org/records/7152317#.Yz6mJ9JByC0)

# Models Used

- [LLaMA 2 7B](https://huggingface.co/meta-llama/Llama-2-7b-hf): A 7-billion parameter open-weight language model developed by Meta, optimized for a range of natural language understanding and generation tasks.
- [LLaMA 3.1 8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B): The next-generation 8-billion parameter model from Meta's LLaMA series, providing improved performance, context understanding, and inference efficiency.
- [Qwen 2.5 7B](https://huggingface.co/Qwen/Qwen2.5-7B): An open-weight language model from Alibaba's Qwen series, fine-tuned for instruction-following and summarization tasks with an emphasis on efficiency.
- [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1): A high-performance 7-billion parameter model by Mistral AI, designed for fast inference and strong performance on text generation benchmarks.

# Installation
## Environment Setup
Ensure you are using **Python 3.10 or higher**.  
Clone this repository and install the required libraries:
## Required libraries:
- pip install transformers
- pip install codecarbon
- pip install torch
- pip install pandas matplotlib seaborn
- pip install rouge-score

# Outputs

After running the scripts, the following files and folders will be generated:

## Summaries (`final/`):
Model-generated summaries for each dataset. These summaries are saved in **text** or **JSONL** files for every model used.

## Energy/Carbon Tracking Data:
CSV files containing:
- **Energy consumption (kWh)**
- **Carbon emissions (kg CO₂eq)**  
for each summarization task, tracked using the CodeCarbon library.

## ROUGE Scores:
ROUGE evaluation results including:
- **ROUGE-1**
- **ROUGE-2**
- **ROUGE-L**  
stored in results CSV format for summarization quality assessment.

## Visualizations:
Plots to help analyze model performance and energy usage:
- **Scatter Plots:** Showing carbon emissions vs input/output token counts for each model and dataset.
- **Candlestick Plots:** Comparing energy consumption and carbon emissions across all models for every dataset.
