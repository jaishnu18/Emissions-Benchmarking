from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import pandas as pd
from tqdm import tqdm
import re
from codecarbon import EmissionsTracker

dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
dataset = dataset.shuffle(seed=42).select(range(100))  


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

    tracker = EmissionsTracker(project_name=f"PubMedQA_{model_name}")
    tracker.start()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)

    predictions = []
    for prompt in tqdm(prompts, desc=f"Generating ({model_name})"):
        try:
            output = generator(prompt, return_full_text=False)[0]["generated_text"]
        except:
            output = generator(prompt)[0]["generated_text"]
        answer_match = re.search(r"\b(yes|no|maybe)\b", output.lower())
        answer = answer_match.group(1) if answer_match else "unknown"
        predictions.append(answer)

    emissions_kg = tracker.stop()

    correct = sum([p == g for p, g in zip(predictions, ground_truths)])
    accuracy = correct / len(ground_truths)

    results[model_name] = {
        "accuracy": round(accuracy, 4),
        "emissions_kg": round(emissions_kg, 6)
    }

df = pd.DataFrame([
    {
        "Model": model,
        "Zero-Shot Accuracy": data["accuracy"],
        "Carbon Emissions (kgCO2e)": data["emissions_kg"]
    }
    for model, data in results.items()
])

print("\nAccuracy and Emissions Table:")
print(df)
