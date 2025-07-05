import json
import requests
import csv
from codecarbon import EmissionsTracker
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm

# Model configurations
models = {
    # "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    # "llama2-7b": "meta-llama/Llama-2-7b-hf",
    # "llama3.1-8b": "meta-llama/Meta-Llama-3-8B",
     "mistral-7b": "mistralai/Mistral-7B-v0.1"
}

# Load original test data
url = "https://raw.githubusercontent.com/StanfordMIMI/clin-summ/main/data/opi/test.jsonl"
response = requests.get(url)
raw_lines = response.text.strip().split("\n")
data = [json.loads(line) for line in raw_lines]

original_data = data[:343]
sorted_data = sorted(original_data, key=lambda x: len(x["inputs"]) + len(x["target"]))

# Benchmark each model
for model_id, model_name in models.items():
    print(f"\n--- Running model: {model_name} ---")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
    summarizer = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Set output filenames
    txt_file = f"opi_tok_sum_{model_id}.txt"
    csv_file = f"opi_tok_{model_id}.csv"

    summaries = []

    # Process each input example
    with open(txt_file, "w", encoding="utf-8") as txtf:
        for entry in tqdm(original_data, desc=f"Processing {model_id}"):
            idx = entry["idx"]
            input_text = (
                "Summarize the following radiology report in 10 words or less:\n\n"
                f"Report: {entry['inputs']}\n"
                f"Impression: {entry['target']}\n\n"
                "Summary:"
            )
            input_len = len(entry["inputs"]) + len(entry["target"])

            # Token count for input
            input_tokens = len(tokenizer(input_text, return_tensors="pt")["input_ids"][0])

            # Start energy tracking
            tracker = EmissionsTracker(measure_power_secs=1, log_level="error")
            tracker.start()

            # Generate summary
            result = summarizer(input_text, max_new_tokens=20, do_sample=False)[0]["generated_text"]

            # Stop energy tracking
            carbon_kg = tracker.stop() or 0.0
            energy_kwh = tracker._total_energy.kWh or 0.0

            # Extract and tokenize the generated summary
            summary = result.split("Summary:")[-1].strip().split("\n")[0]
            output_len = len(summary)
            output_tokens = len(tokenizer(summary, return_tensors="pt")["input_ids"][0])

            # Write to text file
            txtf.write(summary + "\n" + ("-" * 80) + "\n")

            # Save all metrics
            summaries.append((idx, input_len, output_len, input_tokens, output_tokens, energy_kwh, carbon_kg))

            # Print progress info
            print(
                f"idx={idx}, input_len={input_len}, output_len={output_len}, "
                f"input_tokens={input_tokens}, output_tokens={output_tokens}, "
                f"energy={energy_kwh:.6f} kWh, carbon={carbon_kg:.6f} kg"
            )

    # Write metrics to CSV
    sorted_emissions = sorted(summaries, key=lambda x: x[1])  # Sort by input_len
    with open(csv_file, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow([
            "idx", "input_len", "output_len",
            "input_tokens", "output_tokens",
            "energy_kwh", "carbon_kg"
        ])
        for row in sorted_emissions:
            writer.writerow([
                row[0], row[1], row[2],
                row[3], row[4],
                f"{row[5]:.6f}", f"{row[6]:.6f}"
            ])
