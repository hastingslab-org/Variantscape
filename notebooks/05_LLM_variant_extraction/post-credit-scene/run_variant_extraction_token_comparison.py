#!/usr/bin/env python3
"""
Run LLM-based genetic variant extraction on a CSV, then repeat inference after
adding domain-specific variant tokens to the tokenizer. Evaluate both outputs
against the Human ground-truth column and analyze tokenization for FP/FN errors.

Expected input columns:
  - PaperTitle
  - Abstract
  - Human
Optional:
  - PaperId

Example:
  python run_variant_extraction_token_comparison.py \
    --input_csv LLM_evaluation_statistics.csv \
    --model_name /path/to/local/model_or_hf_id \
    --output_dir variant_extraction_outputs \
    --batch_size 4 \
    --max_new_tokens 96
"""

from __future__ import annotations

import argparse
import ast
import gc
import json
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


NEW_TOKENS = [
    "c.", "p.", "g.", "m.", "n.", "r.",
    "del", "dup", "ins", "delins", "fs", "Ter",
    "rs",
    "Ala", "Arg", "Asn", "Asp", "Cys", "Gln", "Glu", "Gly",
    "His", "Ile", "Leu", "Lys", "Met", "Phe", "Pro", "Ser",
    "Thr", "Trp", "Tyr", "Val",
    ">", "_", ":",
    "BRCA1", "BRAF", "KRAS",
]

SYSTEM_MSG = (
    "You are a helpful medical question answering assistant. Please carefully "
    "follow the exact instructions and do not provide explanations."
)

BASELINE_COL = "NER_without_added_tokens"
ADDED_TOKEN_COL = "NER_with_added_tokens"

VARIANT_REGEX = re.compile(
    r"""
    (?:
        \brs\d+\b
        |
        \b[cpgnmr]\.\d+(?:_\d+)?(?:[ACGT]>[ACGT]|del[A-Za-z0-9]*|dup[A-Za-z0-9]*|ins[A-Za-z0-9]*|delins[A-Za-z0-9]*)\b
        |
        \bp\.[A-Z][a-z]{2}\d+[A-Z][a-z]{2}\b
        |
        \bp\.[A-Z]\d+[A-Z]\b
        |
        \bN[MR]_\d+(?:\.\d+)?:[cpgnmr]\.[A-Za-z0-9_>.+\-]+\b
        |
        \b[A-Z]{1,8}\s+[A-Z]\d+[A-Z]\b
        |
        \b[A-Z]\d+[A-Z]\b
        |
        \b\d+(?:del|ins|dup)[A-Za-z]*\b
        |
        \bIVS\d+[+\-]\d+[ACGT]>[ACGT]\b
    )
    """,
    flags=re.VERBOSE | re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM variant extraction: baseline vs tokenizer with added variant tokens."
    )
    parser.add_argument("--input_csv", required=True, type=Path, help="CSV with PaperTitle, Abstract, Human columns.")
    parser.add_argument("--model_name", required=True, help="Hugging Face model name or local checkpoint path.")
    parser.add_argument("--output_dir", default="variant_extraction_token_added_outputs", type=Path)
    parser.add_argument("--max_new_tokens", default=96, type=int)
    parser.add_argument("--batch_size", default=1, type=int, help="Generation batch size. Increase on larger GPUs.")
    parser.add_argument("--sample_n_rows", default=None, type=int, help="Optional smoke-test subset size.")
    parser.add_argument("--checkpoint_every", default=25, type=int, help="Write augmented CSV every N rows per inference pass.")
    parser.add_argument("--hf_token", default=None, help="HF token. If omitted, uses HF_TOKEN environment variable.")
    parser.add_argument("--torch_dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", default="auto", help="Use 'auto' for accelerate device_map, or 'none' to disable.")
    parser.add_argument("--trust_remote_code", action="store_true", help="Pass trust_remote_code=True to HF loaders.")
    parser.add_argument("--skip_baseline", action="store_true", help="Do not run the baseline inference pass.")
    parser.add_argument("--skip_added_tokens", action="store_true", help="Do not run the added-token inference pass.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing prediction columns instead of resuming missing values.")
    parser.add_argument("--no_eval", action="store_true", help="Only run inference; skip evaluation and tokenization analysis.")
    return parser.parse_args()


def get_hf_token(args: argparse.Namespace) -> Optional[str]:
    return args.hf_token or os.getenv("HF_TOKEN")


def dtype_from_arg(dtype_arg: str):
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if dtype_arg == "float32":
        return torch.float32
    if torch.cuda.is_available():
        # bfloat16 is safer on Ampere/Hopper; fallback to float16 on older GPUs if unsupported.
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def build_variant_prompt(title: str, abstract: str) -> str:
    return (
        "Extract only specific genetic variants from the text. Return strictly:\n"
        "- **HGVS Notation** (c., p., g.) e.g., c.2138C>G, p.Arg713Trp\n"
        "- **Protein changes** (e.g., V600E, Arg713Trp)\n"
        "- **rsIDs** (e.g., rs121913529)\n"
        "- Ignore vague terms (e.g., 'mutation found').\n\n"
        "### Format:\n"
        "- Variant: 'Variant: <mutation>, Gene: <gene>' per line\n"
        "- If none, return: 'No variant'\n"
        "- No extra text, no explanations.\n\n"
        f"Title: {title}\nAbstract: {abstract}"
    )


def load_causal_lm(args: argparse.Namespace, add_tokens: bool = False):
    token = get_hf_token(args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
        token=token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {
        "torch_dtype": dtype_from_arg(args.torch_dtype),
        "trust_remote_code": args.trust_remote_code,
        "token": token,
    }
    if args.device_map.lower() != "none":
        model_kwargs["device_map"] = args.device_map

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    n_added = 0
    if add_tokens:
        n_added = tokenizer.add_tokens(NEW_TOKENS)
        if n_added > 0:
            model.resize_token_embeddings(len(tokenizer))
            model.config.vocab_size = len(tokenizer)

    model.eval()
    return tokenizer, model, n_added


def model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_chat_prompts(prompts: Sequence[str], tokenizer) -> List[str]:
    formatted = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": prompt},
        ]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            formatted.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        else:
            formatted.append(f"System: {SYSTEM_MSG}\nUser: {prompt}\nAssistant:")
    return formatted


@torch.inference_mode()
def generate_batch(prompts: Sequence[str], tokenizer, model, max_new_tokens: int) -> List[str]:
    model_inputs = format_chat_prompts(prompts, tokenizer)
    inputs = tokenizer(model_inputs, return_tensors="pt", truncation=True, padding=True)

    # With device_map='auto', model.device may not be reliable; input IDs can be sent to the
    # embedding device, which is normally the first parameter's device.
    device = model_device(model)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
    # For left padding, generated tokens start after the full padded input length, not after
    # the unpadded attention length.
    padded_input_len = inputs["input_ids"].shape[1]
    decoded = []
    for seq in output_ids:
        new_ids = seq[padded_input_len:]
        decoded.append(tokenizer.decode(new_ids, skip_special_tokens=True).strip())
    return decoded


def is_missing_prediction(value) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip() == ""


def run_inference(
    df: pd.DataFrame,
    tokenizer,
    model,
    output_col: str,
    args: argparse.Namespace,
    checkpoint_path: Path,
) -> pd.DataFrame:
    if output_col not in df.columns or args.force:
        df[output_col] = np.nan

    pending_indices = [idx for idx, val in df[output_col].items() if is_missing_prediction(val)]
    print(f"{output_col}: {len(pending_indices):,} rows pending")

    batch_size = max(1, args.batch_size)
    processed_since_checkpoint = 0

    for start in tqdm(range(0, len(pending_indices), batch_size), desc=output_col):
        indices = pending_indices[start : start + batch_size]
        prompts = [
            build_variant_prompt(df.at[idx, "PaperTitle"], df.at[idx, "Abstract"])
            for idx in indices
        ]
        try:
            preds = generate_batch(prompts, tokenizer, model, args.max_new_tokens)
        except Exception as exc:
            preds = [f"ERROR: {type(exc).__name__}: {exc}" for _ in indices]

        for idx, pred in zip(indices, preds):
            df.at[idx, output_col] = pred

        processed_since_checkpoint += len(indices)
        if args.checkpoint_every > 0 and processed_since_checkpoint >= args.checkpoint_every:
            df.to_csv(checkpoint_path, index=False)
            processed_since_checkpoint = 0

    df.to_csv(checkpoint_path, index=False)
    return df


def human_binary_label(x) -> int:
    if pd.isna(x):
        return 0
    return 0 if str(x).strip() == "0" else 1


def prediction_binary_label(x) -> int:
    if pd.isna(x):
        return 0
    text = str(x).strip().lower()
    if text == "" or text.startswith("error:"):
        return 0
    no_variant_patterns = [
        "0", "none", "nan", "no variant", "no variants",
        "no genetic variant", "no genetic variants",
        "no genetic variant detected",
        "no genetic variant detected in this publication",
    ]
    return 0 if any(text == p or text.startswith(p + ".") for p in no_variant_patterns) else 1


def compute_binary_metrics(df: pd.DataFrame, pred_col: str, gold_col: str = "Human") -> dict:
    y_true = df[gold_col].map(human_binary_label)
    y_pred = df[pred_col].map(prediction_binary_label)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) else np.nan
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        "model": pred_col,
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "f1": f1,
    }


def add_error_type(df: pd.DataFrame, pred_col: str, gold_col: str = "Human") -> pd.DataFrame:
    out = df.copy()
    out["y_true"] = out[gold_col].map(human_binary_label)
    out["y_pred"] = out[pred_col].map(prediction_binary_label)
    conditions = [
        (out["y_true"].eq(1) & out["y_pred"].eq(1)),
        (out["y_true"].eq(0) & out["y_pred"].eq(1)),
        (out["y_true"].eq(1) & out["y_pred"].eq(0)),
        (out["y_true"].eq(0) & out["y_pred"].eq(0)),
    ]
    out["error_type"] = np.select(conditions, ["TP", "FP", "FN", "TN"], default="UNKNOWN")
    out["prediction_column"] = pred_col
    return out


def extract_candidate_variants(text) -> List[str]:
    if pd.isna(text):
        return []
    candidates = [m.group(0).strip() for m in VARIANT_REGEX.finditer(str(text))]
    return sorted({c for c in candidates if len(c) >= 3})


def maybe_parse_list_string(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return value
    text = str(value)
    if text.startswith("[") and text.endswith("]"):
        try:
            return ast.literal_eval(text)
        except Exception:
            return value
    return value


def tokenization_features(variant: str, tokenizer) -> dict:
    variant = str(variant).strip()
    result = {
        "variant": variant,
        "n_chars": len(variant),
        "has_punctuation": bool(re.search(r"[.\->_/+:]", variant)),
        "has_digit": bool(re.search(r"\d", variant)),
        "has_mixed_case": bool(re.search(r"[a-z]", variant) and re.search(r"[A-Z]", variant)),
        "has_hgvs_like_prefix": bool(re.search(r"\b[cpgnmr]\.", variant, flags=re.IGNORECASE)),
        "has_reference_sequence": bool(re.search(r"\bN[MR]_\d+(?:\.\d+)?", variant)),
        "has_rs_id": bool(re.search(r"\brs\d+\b", variant, flags=re.IGNORECASE)),
    }
    enc = tokenizer(variant, add_special_tokens=False, return_offsets_mapping=True)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])
    pieces = [variant[s:e] for s, e in enc["offset_mapping"]]
    result["n_tokens_raw"] = len(tokens)
    result["chars_per_token_raw"] = len(variant) / len(tokens) if tokens else np.nan
    result["fragmentation_ratio"] = len(tokens) / max(len(variant), 1)
    result["tokens_raw"] = json.dumps(tokens, ensure_ascii=False)
    result["pieces_raw"] = json.dumps(pieces, ensure_ascii=False)
    return result


def load_analysis_tokenizers(args: argparse.Namespace):
    token = get_hf_token(args)
    tok_base = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, trust_remote_code=args.trust_remote_code, token=token
    )
    tok_added = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, trust_remote_code=args.trust_remote_code, token=token
    )
    n_added = tok_added.add_tokens(NEW_TOKENS)
    print(f"Analysis tokenizer added tokens: {n_added}")
    return {"without_added_tokens": tok_base, "with_added_tokens": tok_added}


def build_tokenization_analysis(error_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    tokenizers = load_analysis_tokenizers(args)
    error_rows = error_df[error_df["error_type"].isin(["FP", "FN"])].copy()
    rows = []
    for _, row in tqdm(error_rows.iterrows(), total=len(error_rows), desc="tokenization analysis"):
        text = f"{row.get('PaperTitle', '')} {row.get('Abstract', '')}"
        for cand in extract_candidate_variants(text):
            for tokenizer_label, tok in tokenizers.items():
                feats = tokenization_features(cand, tok)
                feats.update({
                    "prediction_column": row["prediction_column"],
                    "error_type": row["error_type"],
                    "PaperId": row["PaperId"],
                    "candidate_variant": cand,
                    "tokenizer_version": tokenizer_label,
                    "PaperTitle": row.get("PaperTitle", None),
                })
                rows.append(feats)
    return pd.DataFrame(rows)


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["PaperTitle", "Abstract", "Human"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if "PaperId" not in df.columns:
        df.insert(0, "PaperId", np.arange(len(df)))
    return df


def cleanup_model(model=None, tokenizer=None):
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    augmented_out = args.output_dir / "LLM_evaluation_statistics_with_NER_token_added_predictions.csv"
    metrics_out = args.output_dir / "NER_without_vs_with_added_tokens_metrics.csv"
    errors_out = args.output_dir / "NER_without_vs_with_added_tokens_error_rows.csv"
    tokenization_out = args.output_dir / "NER_error_candidate_tokenization_without_vs_with_added_tokens.csv"

    if augmented_out.exists() and not args.force:
        print(f"Resuming from existing augmented CSV: {augmented_out}")
        df = pd.read_csv(augmented_out)
    else:
        print(f"Loading input CSV: {args.input_csv}")
        df = pd.read_csv(args.input_csv)

    df = validate_input(df)
    if args.sample_n_rows is not None:
        df = df.head(args.sample_n_rows).copy()
    print(f"Rows: {len(df):,}")

    if not args.skip_baseline:
        print("\n=== Baseline inference: tokenizer unchanged ===")
        tokenizer, model, n_added = load_causal_lm(args, add_tokens=False)
        print(f"Loaded model. Added tokens: {n_added}")
        df = run_inference(df, tokenizer, model, BASELINE_COL, args, augmented_out)
        cleanup_model(model, tokenizer)

    if not args.skip_added_tokens:
        print("\n=== Added-token inference ===")
        tokenizer, model, n_added = load_causal_lm(args, add_tokens=True)
        print(f"Requested {len(NEW_TOKENS)} tokens; actually added {n_added} new tokens.")
        df = run_inference(df, tokenizer, model, ADDED_TOKEN_COL, args, augmented_out)
        cleanup_model(model, tokenizer)

    # Always save predictions at this point.
    df.to_csv(augmented_out, index=False)

    if args.no_eval:
        print("Skipping evaluation because --no_eval was set.")
        print(f"Saved augmented CSV: {augmented_out.resolve()}")
        return

    comparison_cols = [c for c in [BASELINE_COL, ADDED_TOKEN_COL] if c in df.columns]
    if not comparison_cols:
        raise RuntimeError("No prediction columns found for evaluation.")

    for col in comparison_cols:
        df[f"{col}_binary"] = df[col].map(prediction_binary_label)
    df["Human_binary"] = df["Human"].map(human_binary_label)

    metrics_df = pd.DataFrame([compute_binary_metrics(df, c) for c in comparison_cols])
    metrics_df.to_csv(metrics_out, index=False)
    print("\nMetrics:")
    print(metrics_df.to_string(index=False))

    all_error_df = pd.concat([add_error_type(df, c) for c in comparison_cols], ignore_index=True)
    all_error_df.to_csv(errors_out, index=False)

    candidate_tok_df = build_tokenization_analysis(all_error_df, args)
    if not candidate_tok_df.empty:
        candidate_tok_df.to_csv(tokenization_out, index=False)

    df.to_csv(augmented_out, index=False)

    print("\nSaved outputs:")
    print(f"- {augmented_out.resolve()}")
    print(f"- {metrics_out.resolve()}")
    print(f"- {errors_out.resolve()}")
    if not candidate_tok_df.empty:
        print(f"- {tokenization_out.resolve()}")
    else:
        print("- No tokenization-analysis CSV was written because no FP/FN candidate variants were found.")


if __name__ == "__main__":
    main()
