# -*- coding: utf-8 -*-
"""semantic_similarity_evaluator.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/13B-WKC2qZRg05Kjcmu3S5E9WiD8HRD9X
"""

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from sentence_transformers import SentenceTransformer
from scipy.stats import spearmanr, pearsonr, kendalltau
import json
import re
import os
import argparse
from pathlib import Path
import logging
from typing import List, Tuple, Dict, Union, Optional
import multiprocessing as mp
from functools import partial

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Calculate semantic similarity between machine translations and reference translations')
    parser.add_argument('--input_files', nargs='+', required=True,
                        help='List of input file paths')
    parser.add_argument('--output_dir', default='./similarity_outputs',
                        help='Output directory path')
    parser.add_argument('--model_name', default='sentence-transformers/all-mpnet-base-v2',
                        help='SentenceTransformer model name')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for encoding')
    parser.add_argument('--use_token', action='store_true',
                        help='Whether to use Hugging Face token')
    parser.add_argument('--token', default=None,
                        help='Hugging Face token')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of worker processes for parallel file processing')
    return parser.parse_args()


def extract_language_pair(filename: str) -> str:
    """Extract language pair information from filename"""
    patterns = [
        r'([a-z]{2}-[a-z]{2})\.jsonl',
        r'similarity_([a-z]{2}-[a-z]{2})',
        r'([a-z]{2}-[a-z]{2})_'
    ]

    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return match.group(1)

    return Path(filename).stem


def encode_in_batches(texts: List[str], model: SentenceTransformer,
                     batch_size: int = 128, show_progress: bool = True) -> torch.Tensor:
    """Encode texts in batches to avoid memory issues"""
    all_embeddings = []

    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Encoding sentences")

    for i in iterator:
        batch = texts[i:i+batch_size]
        with torch.no_grad():
            embeddings = model.encode(batch, convert_to_tensor=True)
        all_embeddings.append(embeddings.detach())

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.cat(all_embeddings, dim=0)


def compute_similarity(mt_embeddings: torch.Tensor,
                      src_embeddings: torch.Tensor) -> np.ndarray:
    """Compute cosine similarity between two sets of embedding vectors"""
    mt_embeddings = torch.nn.functional.normalize(mt_embeddings, p=2, dim=1)
    src_embeddings = torch.nn.functional.normalize(src_embeddings, p=2, dim=1)

    similarities = torch.sum(mt_embeddings * src_embeddings, dim=1).cpu().numpy()
    return similarities


def calculate_correlations(original_scores: List[float],
                          semantic_scores: List[float]) -> Dict[str, Tuple[float, float]]:
    """Calculate correlation between original scores and semantic similarity scores"""
    valid_pairs = [(o, s) for o, s in zip(original_scores, semantic_scores)
                  if not (np.isnan(o) or np.isnan(s))]

    if not valid_pairs:
        return None

    original_filtered, semantic_filtered = zip(*valid_pairs)

    results = {}
    results['spearman'] = spearmanr(original_filtered, semantic_filtered)
    results['pearson'] = pearsonr(original_filtered, semantic_filtered)
    results['kendall'] = kendalltau(original_filtered, semantic_filtered)

    return results


def process_file(input_path: str, output_dir: str, similarity_model: SentenceTransformer,
                batch_size: int) -> Optional[str]:
    """Process a single file through the complete pipeline"""
    try:
        logger.info(f"Processing file: {input_path}")

        language_pair = extract_language_pair(input_path)

        try:
            df = pd.read_json(input_path, lines=True)
        except Exception as e:
            logger.error(f"Failed to read file: {e}")
            return None

        required_columns = ["mt", "src_translation"]
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            logger.error(f"Missing required columns: {missing}, skipping file")
            return None

        logger.info(f"Dataset size: {len(df)} samples")

        mt_sentences = df["mt"].tolist()
        src_translations = df["src_translation"].tolist()

        logger.info("Batch encoding sentences...")
        mt_embeddings = encode_in_batches(mt_sentences, similarity_model, batch_size)
        src_embeddings = encode_in_batches(src_translations, similarity_model, batch_size)

        logger.info("Computing cosine similarity...")
        similarity_scores = compute_similarity(mt_embeddings, src_embeddings)

        df["semantic_similarity"] = similarity_scores

        file_correlation_info = f"File: {input_path} (Language pair: {language_pair})\n"

        if "score" in df.columns:
            original_scores = df["score"].tolist()
            semantic_scores = df["semantic_similarity"].tolist()

            correlation_results = calculate_correlations(original_scores, semantic_scores)

            if correlation_results:
                logger.info(f"Correlation analysis:")
                for method, (coef, p) in correlation_results.items():
                    logger.info(f"  {method.capitalize()}: {coef:.4f} (p={p:.4f})")
                    file_correlation_info += f"{method.capitalize()}: {coef:.4f} (p={p:.4f})\n"
            else:
                logger.warning("No valid data pairs, skipping correlation calculation")
                file_correlation_info += "⚠️ No valid data pairs, skipping correlation calculation\n"
        else:
            logger.warning("Missing 'score' column, cannot calculate correlation")
            file_correlation_info += "⚠️ Missing 'score' column, cannot calculate correlation\n"

        filename = Path(input_path).stem

        output_path = f"{output_dir}/{filename}_scored.jsonl"
        df.to_json(output_path, orient="records", lines=True, force_ascii=False)

        readable_path = f"{output_dir}/{filename}_scored_readable.json"
        with open(readable_path, "w", encoding="utf-8") as f:
            json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

        logger.info(f"Saved: {output_path}")
        logger.info("="*60)

        return file_correlation_info

    except Exception as e:
        logger.error(f"Error processing file {input_path}: {e}")
        return f"File: {input_path} processing failed: {str(e)}\n"


def main():
    """Main function"""
    args = parse_arguments()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    correlation_summary_path = f"{args.output_dir}/correlation_summary.txt"

    logger.info(f"Loading semantic similarity model: {args.model_name}")

    token_param = {}
    if args.use_token and args.token:
        token_param['use_auth_token'] = args.token

    similarity_model = SentenceTransformer(args.model_name, **token_param)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    similarity_model = similarity_model.to(device)
    logger.info(f"Model loaded, using device: {device}")

    if args.num_workers > 1 and len(args.input_files) > 1:
        logger.info(f"Using {args.num_workers} worker processes")

        process_func = partial(
            process_file_mp,
            output_dir=args.output_dir,
            model_name=args.model_name,
            token_param=token_param,
            batch_size=args.batch_size
        )

        with mp.Pool(processes=min(args.num_workers, len(args.input_files))) as pool:
            correlation_infos = pool.map(process_func, args.input_files)

    else:
        correlation_infos = []
        for input_path in args.input_files:
            info = process_file(input_path, args.output_dir, similarity_model, args.batch_size)
            if info:
                correlation_infos.append(info)

    correlation_infos = [info for info in correlation_infos if info]

    with open(correlation_summary_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(correlation_infos))
        f.write(f"\n\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    logger.info(f"Correlation summary saved to: {correlation_summary_path}")
    logger.info("All files processed")


def process_file_mp(input_path, output_dir, model_name, token_param, batch_size):
    """Function for multiprocessing. Each process loads its own model."""
    local_model = SentenceTransformer(model_name, **token_param)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_model = local_model.to(device)

    return process_file(input_path, output_dir, local_model, batch_size)


if __name__ == "__main__":
    main()