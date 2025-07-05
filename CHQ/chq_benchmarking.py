import json
import requests
import csv
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from codecarbon import EmissionsTracker
from tqdm import tqdm

# Load dataset
url = "https://raw.githubusercontent.com/StanfordMIMI/clin-summ/main/data/chq/test.jsonl"
lines = requests.get(url).text.strip().split("\n")
data = [json.loads(line) for line in lines]
data = sorted(data[:150], key=lambda x: len(x["inputs"]))  

# Define model configs
models = {
    # "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    # "llama2-7b": "meta-llama/Llama-2-7b-hf",
     "llama3.1-8b": "meta-llama/Meta-Llama-3-8B",
    # "mistral-7b": "mistralai/Mistral-7B-v0.1"
}

def build_prompt(text):
    return f"""Summarize the following patient health query in 20 words or less, as a question:{text}\nSummary:"""

# Process for each model
for model_id, model_name in models.items():
    print(f"\nRunning inference on: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

    results = []

    with open(f"chq_tok_sum_{model_id}.txt", "w", encoding="utf-8") as summary_file:
        for record in tqdm(data, desc=f"Inference with {model_id}"):
            idx = record["idx"]
            query = record["inputs"]
            input_len = len(query)
            prompt = build_prompt(query)

            # Count input tokens
            input_tokens = len(tokenizer(prompt)["input_ids"])

            tracker = EmissionsTracker(project_name=f"CHQ_{model_id}", output_dir="./", log_level="error", save_to_file=False)
            tracker.start()

            output = generator(prompt, max_new_tokens=40, do_sample=False)[0]["generated_text"]

            emissions = tracker.stop()
            energy_kwh = tracker._total_energy.kWh

            # Extract summary
            summary = output.split("Summary:")[-1].strip().split("\n")[0]
            output_len = len(summary)

            # Count output tokens
            output_tokens = len(tokenizer(summary)["input_ids"])

            # Write summary text file
            summary_file.write(f"{idx}\t{summary}\n")

            # Collect results
            results.append({
                "idx": idx,
                "input_length": input_len,
                "output_length": output_len,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "summary_question": summary,
                "energy_kwh": round(energy_kwh, 6),
                "carbon_kg": round(emissions, 6)
            })

    # Sort results
    results_sorted = sorted(results, key=lambda x: x["input_tokens"])

    # Write to CSV
    with open(f"chq_tok_{model_id}.csv", "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "input_length", "output_length", "input_tokens", "output_tokens", "energy_kwh", "carbon_kg"])
        for r in results_sorted:
            writer.writerow([r["idx"], r["input_length"], r["output_length"], r["input_tokens"], r["output_tokens"], r["energy_kwh"], r["carbon_kg"]])

    print(f"Completed model: {model_id}")
    print("-" * 80)
