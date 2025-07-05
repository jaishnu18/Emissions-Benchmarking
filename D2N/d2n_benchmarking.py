import requests
import json
import os
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from codecarbon import EmissionsTracker
from tqdm import tqdm

# === Load dataset ===
url = "https://raw.githubusercontent.com/StanfordMIMI/clin-summ/main/data/d2n/test.jsonl"
lines = requests.get(url).text.strip().split("\n")
data = [json.loads(line) for line in lines[:100]]
records = sorted([(idx, item) for idx, item in enumerate(data)],
                 key=lambda x: len(x[1]['inputs']) + len(x[1]['target']))


models = {
    # "t5-base": {"seq2seq": True},
     "Qwen/Qwen2.5-7B": {"causal": True},
     "meta-llama/Llama-2-7b-hf": {"causal": True},
     "meta-llama/Llama-3.1-8B": {"causal": True},
    "mistralai/Mistral-7B-v0.1": {"causal": True}
}

# === Output directory ===
os.makedirs("carbon_logs", exist_ok=True)

# === Loop over each model ===
for model_name, config in models.items():
    print(f"\n=== Running for {model_name} ===\n")

    is_seq2seq = config.get("seq2seq", False)
    is_causal = config.get("causal", False)

    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if is_seq2seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    elif is_causal:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
        summarizer = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # === Summary generation function ===
    def generate_summary(prompt, max_length=160):
        if is_seq2seq:
            result = summarizer(prompt, max_length=max_length, min_length=60, do_sample=False)[0]
            return result["summary_text"].strip()
        else:
            result = summarizer(prompt, max_new_tokens=max_length, do_sample=False, return_full_text=False)[0]
            return result["generated_text"].strip()

    results = []
    emissions_data = []

    # === Process each record ===
    for count, (idx, item) in enumerate(tqdm(records), start=1):
        convo = item["inputs"]
        target = item["target"]
        input_length = len(convo)

        # Build prompts
        prompt1 = f"Summarize the following patient-doctor conversation in 50 words or less:\n{convo}"
        prompt2 = f"Summarize the following clinical assessment and plan:\n{target}"

        # === Token counts for inputs ===
        input_tokens = len(tokenizer(prompt1 + tokenizer.eos_token, return_tensors="pt").input_ids[0]) + \
                       len(tokenizer(prompt2 + tokenizer.eos_token, return_tensors="pt").input_ids[0])

        # === Start energy tracker ===
        tracker = EmissionsTracker(
            output_dir="carbon_logs",
            output_file=f"d2n_emissions_{model_name.replace('/', '_')}.csv",
            log_level="error",
            measure_power_secs=0.5
        )
        tracker.start()

        # === Generate summaries ===
        summary = generate_summary(prompt1)
        assessment_summary = generate_summary(prompt2)

        # === Stop energy tracker ===
        emissions = tracker.stop()
        energy_kwh = tracker._total_energy.kWh
        carbon_kg = emissions

        # === Token counts for outputs ===
        output_tokens = len(tokenizer(summary + tokenizer.eos_token, return_tensors="pt").input_ids[0]) + \
                        len(tokenizer(assessment_summary + tokenizer.eos_token, return_tensors="pt").input_ids[0])

        output_length = len(summary) + len(assessment_summary)

        # === Format output ===
        formatted = (
            f"#D2N: Summary of the patient/doctor conversation in 50 words or less.\n"
            f"{summary}\n\n"
            f"Summary of assessment and plan\n"
            f"[Answer] {assessment_summary}\n"
            f"{'-'*80}\n"
        )
        results.append(formatted)

        # === Record metrics ===
        emissions_data.append({
            "Record_Idx": idx,
            "Input_Length": input_length,
            "Output_Length": output_length,
            "Input_Tokens": input_tokens,
            "Output_Tokens": output_tokens,
            "Energy_kWh": round(energy_kwh, 6),
            "Carbon_Emissions_kgCO2e": round(carbon_kg, 6)
        })

    # === Save summaries ===
    outfile_txt = f"d2n_tok_{model_name.replace('/', '_')}.txt"
    with open(outfile_txt, "w", encoding="utf-8") as f:
        f.writelines(results)

    # === Save emissions + token data ===
    df = pd.DataFrame(emissions_data)
    outfile_csv = f"d2n_tok_{model_name.replace('/', '_')}.csv"
    df.to_csv(outfile_csv, index=False)

    print(f"\nFinished model: {model_name}")
    print(f"Summary file: {outfile_txt}")
    print(f"Carbon file: {outfile_csv}")
