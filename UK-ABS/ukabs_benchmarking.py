import json
import csv
import gc
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from codecarbon import EmissionsTracker
from rouge_score import rouge_scorer
from time import sleep
import tqdm

# === Load dataset ===
with open("/home/rs/20CS91R17/UK-ABS/pairs.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()
data = [json.loads(line.strip()) for line in lines]
data = sorted(data, key=lambda x: len(x["judgement"]))
for i in data:
    print(len(i["judgement"]))

# === Models to evaluate ===
models = {
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "llama3.1-8b": "meta-llama/Meta-Llama-3-8B",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    # "phi-4": "microsoft/phi-4",
    "mistral-7b": "mistralai/Mistral-7B-v0.1"
}

def build_prompt(article):
    return f"""You are UK legal expert. Summarize the following case court judgement into a summary of less than 500 words.
Case Court Judgement:\n{article.strip()}\nSummary:"""

# === For final ROUGE result aggregation ===
final_results = []

for model_id, model_name in models.items():
    print(f"\nRunning inference on: {model_name}")
    print(f"\nCUDA Device id: {torch.cuda.current_device()}")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    results = []

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

    with open(f"ukabs_tok_sum_{model_id}.txt", "w", encoding="utf-8") as summary_file:
        for record in tqdm.tqdm(data):
            try:
                idx = record["idx"]
                article = record["judgement"]
                reference = record["summary"]

                input_len = len(article)  # character length of input
                prompt = build_prompt(article)
                input_tokens = len(tokenizer.tokenize(prompt))  # token count of the prompt

                tracker = EmissionsTracker(
                    project_name=f"UKABS_{model_id}",
                    output_dir="./",
                    log_level="error",
                    save_to_file=False
                )
                tracker.start()

                with torch.no_grad():
                    output = generator(prompt, max_new_tokens=100, do_sample=False)[0]["generated_text"]

                emissions = tracker.stop()
                energy_kwh = tracker._total_energy.kWh

                # Extract the generated summary
                summary = output.split("Summary:")[-1].strip().split("\n")[0]
                output_len = len(summary)  # character length of output
                output_tokens = len(tokenizer.tokenize(summary))  # token count of the output

                summary_file.write(f"{idx}\t{summary}\n")

                # ROUGE score calculation
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

                # Free up GPU & memory
                del output, summary, prompt, scores
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing idx {record['idx']}: {e}")
                results.append({
                    "id": idx,
                    "input_length": len(article),
                    "output_length": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "summary": "NO SUMMARY",
                    "energy_kwh": 0,
                    "carbon_kg": 0
                })

    # === Save per-record stats ===
    results_sorted = sorted(results, key=lambda x: x["input_length"])
    with open(f"ukabs_tok_{model_id}.csv", "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "input_length", "output_length", "input_tokens", "output_tokens", "energy_kwh", "carbon_kg"])
        for r in results_sorted:
            writer.writerow([
                r["id"],
                r["input_length"],
                r["output_length"],
                r["input_tokens"],
                r["output_tokens"],
                r["energy_kwh"],
                r["carbon_kg"]
            ])

    # === Aggregate average ROUGE ===
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
with open("ukabs_tok_results.csv", "a", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["model_id", "rouge1_f", "rouge2_f", "rougeL_f"])
    for r in final_results:
        writer.writerow([r["model_id"], r["rouge1_f"], r["rouge2_f"], r["rougeL_f"]])

print("All models processed and ROUGE results saved.")
