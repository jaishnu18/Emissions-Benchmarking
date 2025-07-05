from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import pandas as pd
from tqdm import tqdm
import re

dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
dataset = dataset.shuffle(seed=42).select(range(100))

#D2N: Summarize the following patient/doctor conversation in 50 words or less. [Context/ Conversation] [X Question] [Answer/ ASSESSMENT AND PLAN]
#CHQ: Summarize the following patient health query in 20 words or less. [X Contex] [Qwestion / Patient Health Query] [Answer/ Summarized Question]
#OPI: Summarize the following radiology report in 10 words or less. [Context/ Radiology Report] [X Question] [Answer/ Findings]

def build_prompt(example):
    return (
        f"Context: {' '.join(example['context']['contexts'])}\n"
        f"Question: {example['question']}\n"
        f"Answer (yes/no/maybe):"
    )

prompts = [build_prompt(ex) for ex in dataset]
ground_truths = [ex['final_decision'].strip().lower() for ex in dataset]

model_checkpoints = {
    "LLaMA2_7B_Chat": "meta-llama/Llama-2-7b-chat-hf",
    "LLaMA2_13B_Chat": "meta-llama/Llama-2-13b-chat-hf",
    "LLaMA2_7B_Base": "meta-llama/Llama-2-7b-hf",
    "LLaMA3_8B": "meta-llama/Llama-3.1-8B",
    "Qwen2.5_7B": "Qwen/Qwen2.5-7B",
    "Phi-4": "microsoft/phi-4"
}

results = {}

for model_name, checkpoint in model_checkpoints.items():
    print(f"\nEvaluating {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float16, device_map="auto")

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)

    predictions = []
    for prompt in tqdm(prompts):
        try:
            output = generator(prompt, return_full_text=False)[0]["generated_text"]
        except:
            output = generator(prompt)[0]["generated_text"]
        answer_match = re.search(r"\b(yes|no|maybe)\b", output.lower())
        answer = answer_match.group(1) if answer_match else "unknown"
        predictions.append(answer)

    correct = sum([p == g for p, g in zip(predictions, ground_truths)])
    accuracy = correct / len(ground_truths)
    results[model_name] = accuracy

df = pd.DataFrame(list(results.items()), columns=["Model", "Zero-Shot Accuracy"])
print("\nAccuracy Table:")
print(df)

