# NOTE: THIS FILE NEEDS TO BE RUN WITH PYTHON 3.11 OTHERWISE SCISPACY WON'T LOAD
# Use the ipynb when you want to run an interactive benchmark
# Use this file if you want to save the results as json.

from datetime import datetime
import json

import spacy
from rouge_score import rouge_scorer
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


scorer = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'],
    use_stemmer=True
)

DATASET_NAME = "test_LREC_COLING.csv"
test_df = pd.read_csv(DATASET_NAME)
DO_SAMPLE = False
MAX_NEW_TOKENS = 30

# SciSpaCy model
SCISPACY_MODEL = "en_core_sci_sm"
nlp = spacy.load(SCISPACY_MODEL)

BASE_MODEL = "HuggingFaceTB/SmolLM-135M"
ADAPTER_PATH = "../../runs/2"


def load():
    """
    Load the model, tokenizer and determine device (mps or cpu)
    """
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    device = "mps" if torch.mps.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    return model, tokenizer, device

# Initialize LM
model, tokenizer, device = load()


def predict(abstract):
    """
    Takes in the abstract and generates a title for it.
    Only returns the title! (Abstract is cutted away)
    Max number of tokens is set to 30. -> This diverges from the paper, however,
    the new trainingset includes much longer titles as well, therefore it is only
    fair to let the model "finish" generating its title.
    """
    prompt = f"{abstract}\n\nTitle: "

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = out[0][inputs['input_ids'].shape[1]:]
    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return decoded.strip()


# Entity extraction (paper-style)
def extract_entities(text):
    """
    Identifies named entities from the given text.
    What an entity is or is not, is defined by `en_core_sci_sm`.
    Returns named entities as a list.
    """
    doc = nlp(text)
    return [ent.text.lower() for ent in doc.ents]


# Partial match (paper rule)
def entity_match(e1, e2):
    tokens1 = set(e1.split())
    tokens2 = set(e2.split())
    return len(tokens1 & tokens2) > 0


def count_matches_list(x, y):
    count = 0
    for ex in x:
        for ey in y:
            if entity_match(ex, ey):
                count += 1
                break
    return count


def count_matches_set(x, y):
    x_set = set(x)
    y_set = set(y)

    count = 0
    for ex in x_set:
        for ey in y_set:
            if entity_match(ex, ey):
                count += 1
                break
    return count


# Metrics (exact paper version)

def compute_entity_metrics(Eh, Et, Es):
    """
    used to check FACTUAL CONSISTENCY

    h = hypothesis (or predicted title)
    t = truth (or actual title)
    s = source (or abstract/text)

    u = unique (multiple occurrences of a named entity count only once)
    nu = non unique (multiple occurrences of a named entity count multiple times)

    precision = Of all the items the model labeled as positive, how many were actually positive?
    recall = Of all the actual positives, how many did the model correctly identify?
    f1 = balance out of precision and recall
    """
    def safe_div(a, b):
        return a / b if b > 0 else 0.0

    # UNIQUE (U)
    Nh_u = len(set(Eh))
    Nt_u = len(set(Et))

    h_s_u = count_matches_set(Eh, Es)
    h_t_u = count_matches_set(Eh, Et)

    prec_s_u = safe_div(h_s_u, Nh_u)
    prec_t_u = safe_div(h_t_u, Nh_u)
    rec_t_u = safe_div(h_t_u, Nt_u)
    f1_t_u = safe_div(2 * prec_t_u * rec_t_u, prec_t_u + rec_t_u)

    # NON-UNIQUE (NU)
    Nh = len(Eh)
    Nt = len(Et)

    h_s = count_matches_list(Eh, Es)
    h_t = count_matches_list(Eh, Et)

    prec_s_nu = safe_div(h_s, Nh)
    prec_t_nu = safe_div(h_t, Nh)
    rec_t_nu = safe_div(h_t, Nt)
    f1_t_nu = safe_div(2 * prec_t_nu * rec_t_nu, prec_t_nu + rec_t_nu)

    return {
        "prec_s_u": prec_s_u,
        "prec_t_u": prec_t_u,
        "rec_t_u": rec_t_u,
        "f1_t_u": f1_t_u,
        "prec_s_nu": prec_s_nu,
        "prec_t_nu": prec_t_nu,
        "rec_t_nu": rec_t_nu,
        "f1_t_nu": f1_t_nu,
    }


# Run evaluation
scores1 = 0.0
scores2 = 0.0
scoresL = 0.0

entity_scores = {
    k: 0.0 for k in [
        "prec_s_u", "prec_t_u", "rec_t_u", "f1_t_u",
        "prec_s_nu", "prec_t_nu", "rec_t_nu", "f1_t_nu"
    ]
}

for row in test_df.itertuples():
    prediction = predict(row.Abstract).lower()
    ground_truth = row.Title.lower().strip()

    # Calculate rouge scores between generated title and actual title
    scores = scorer.score(ground_truth, prediction)

    s1 = scores['rouge1'].fmeasure
    s2 = scores['rouge2'].fmeasure
    sl = scores['rougeL'].fmeasure

    scores1 += s1
    scores2 += s2
    scoresL += sl

    # -------- ENTITY METRICS --------
    Eh = extract_entities(prediction)
    Et = extract_entities(ground_truth)
    Es = extract_entities(row.Abstract)

    metrics = compute_entity_metrics(Eh, Et, Es)

    for k in entity_scores:
        entity_scores[k] += metrics[k]

    print(s1, s2, sl)
    # Debug title difference if prediction does absolutely not match.
    if s1 + s2 + sl == 0:
        print("Original:", ground_truth, "\nGenerated:", prediction)


n = len(test_df)

print("\nFinal ROUGE Results:")
print("F1 Rouge 1:", scores1 / n)
print("F1 Rouge 2:", scores2 / n)
print("F1 Rouge L:", scoresL / n)

print("\nEntity-level factual consistency:")

for k, v in entity_scores.items():
    print(f"{k}: {v / n:.4f}")



aggregate_results = {
    "rouge1_f1": scores1 / n,
    "rouge2_f1": scores2 / n,
    "rougeL_f1": scoresL / n,
    "entity_metrics": {
        k: v / n
        for k, v in entity_scores.items()
    }
}

run_artifact = {
    "timestamp_utc": (
        datetime.utcnow().isoformat() + "Z"
    ),

    "parameters": {
        "dataset_name": DATASET_NAME,
        "num_examples": n,
        "base_model": BASE_MODEL,
        "adapter_path": ADAPTER_PATH,
        "finetune_type": "full" if "full" in ADAPTER_PATH else "lora",
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": DO_SAMPLE,
        "scispacy_model": SCISPACY_MODEL,
        "device": device,
        "torch_version": torch.__version__
    },

    "results": aggregate_results
}

outfile = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"

with open(outfile, "w") as f:
    json.dump(
        run_artifact,
        f,
        indent=4,
        ensure_ascii=False
    )

print(f"\nSaved results to {outfile}")