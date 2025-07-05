import json
import csv
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from codecarbon import EmissionsTracker
from rouge_score import rouge_scorer

# === Load dataset ===
with open("cnn_dailymail_random200.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()
data = [json.loads(line.strip()) for line in lines]
data = sorted(data, key=lambda x: len(x["article"]))

# === Models to evaluate ===
models = {
    #"llama2-7b": "meta-llama/Llama-2-7b-hf",
    #"llama3.1-8b": "meta-llama/Meta-Llama-3-8B",
    #"qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "mistral-7b": "mistralai/Mistral-7B-v0.1"
}

def build_prompt(article):
    return f"""Summarize the following news article in 100 words or less:\n{article}\nSummary:"""

# === For final ROUGE result aggregation ===
final_results = []

# === Run summarization + evaluation per model ===
for model_id, model_name in models.items():
    print(f"\nRunning inference on: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    results = []

    with open(f"cnn_tok_sum_{model_id}.txt", "w", encoding="utf-8") as summary_file:
        for record in tqdm(data, desc=f"Inferencing {model_id}"):
            idx = record["id"]
            article = record["article"]
            reference = record["highlights"]
            input_len = len(article)
            prompt = build_prompt(article)

            # Calculate input tokens
            input_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))

            tracker = EmissionsTracker(
                project_name=f"CNN_{model_id}",
                output_dir="./",
                log_level="error",
                save_to_file=False
            )
            tracker.start()

            output = generator(prompt, max_new_tokens=100, do_sample=False)[0]["generated_text"]

            emissions = tracker.stop()
            energy_kwh = tracker._total_energy.kWh

            # Extract summary and calculate output tokens
            summary = output.split("Summary:")[-1].strip().split("\n")[0]
            output_len = len(summary)
            output_tokens = len(tokenizer.encode(summary, add_special_tokens=False))

            summary_file.write(f"{idx}\t{summary}\n")

            # ROUGE computation
            scores = scorer.score(reference, summary)
            for metric in rouge_scores:
                rouge_scores[metric].append(scores[metric].fmeasure)

            results.append({
                "id": idx,
                "input_length": input_len,
                "output_length": output_len,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "summary": summary,
                "energy_kwh": round(energy_kwh, 6),
                "carbon_kg": round(emissions, 6)
            })

    # === Save per-record stats ===
    results_sorted = sorted(results, key=lambda x: x["input_length"])
    with open(f"cnn_tok_{model_id}.csv", "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "input_length", "output_length", "input_tokens", "output_tokens", "energy_kwh", "carbon_kg"])
        for r in results_sorted:
            writer.writerow([
                r["id"], r["input_length"], r["output_length"],
                r["input_tokens"], r["output_tokens"],
                r["energy_kwh"], r["carbon_kg"]
            ])

    # === Save average ROUGE scores for this model ===
    avg_rouge = {metric: round(sum(rouge_scores[metric]) / len(rouge_scores[metric]), 4) for metric in rouge_scores}
    final_results.append({
        "model_id": model_id,
        "rouge1_f": avg_rouge["rouge1"],
        "rouge2_f": avg_rouge["rouge2"],
        "rougeL_f": avg_rouge["rougeL"]
    })

    print(f"Completed model: {model_id}")
    print("-" * 80)

# === Save ROUGE scores to csv ===

with open("cnn_results.csv", "w", newline='', encoding="utf-8") as f: 
    writer = csv.writer(f)
    writer.writerow(["model_id", "rouge1_f", "rouge2_f", "rougeL_f"]) 
    for r in final_results:
        writer.writerow([r["model_id"], r["rouge1_f"], r["rouge2_f"], r["rougeL_f"]])

print("All models processed and ROUGE results saved.")
