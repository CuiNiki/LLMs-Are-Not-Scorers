# -*- coding: utf-8 -*-
"""
LLaMA-3-8B-Instruct Translation Tool for WMT Benchmark

This script processes JSONL files containing source texts in the "src" field,
translates them to English using the meta-llama/Meta-Llama-3-8B-Instruct model, and saves
the results with a new "src_translation" field to Google Drive.

Optimized for Google Colab with A100 GPU to minimize compute unit consumption.
Features batch processing for efficient inference and robust output cleaning.
"""

import os
import gc
import json
import argparse
import logging
import torch
import re
from pathlib import Path
from tqdm.notebook import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from google.colab import drive

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Translate text using LLaMA-3-8B-Instruct model and save to Google Drive'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing input JSONL files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory in Google Drive'
    )
    parser.add_argument(
        '--hf_token',
        type=str,
        required=True,
        help='Hugging Face access token for gated models'
    )
    parser.add_argument(
        '--file_batch_size',
        type=int,
        default=1,
        help='Number of files to process before clearing memory (default: 1)'
    )
    parser.add_argument(
        '--translation_batch_size',
        type=int,
        default=8,
        help='Number of translations to process at once (default: 8)'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=128,
        help='Maximum output length (default: 128)'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help='Path or name of the model to use for translation'
    )
    parser.add_argument(
        '--source_lang',
        type=str,
        default="",
        help='Source language code (optional, e.g., "de" for German)'
    )
    parser.add_argument(
        '--target_lang',
        type=str,
        default="English",
        help='Target language name (default: English)'
    )
    return parser.parse_args()

def mount_google_drive():
    """Mount Google Drive to access and save files."""
    try:
        drive.mount('/content/drive')
        logger.info("Google Drive mounted successfully")
    except Exception as e:
        logger.error(f"Failed to mount Google Drive: {e}")
        raise

def load_model(hf_token, model_path):
    """
    Load the LLaMA model and tokenizer with optimized settings for A100 GPU.

    Uses device_map="auto" for optimal placement and torch_dtype=torch.float16
    for faster and memory-efficient inference on A100 GPUs.

    Args:
        hf_token (str): Hugging Face access token
        model_path (str): Model identifier or path

    Returns:
        tuple: (tokenizer, model) with automatic device placement
    """
    logger.info(f"Loading {model_path} tokenizer and model...")

    try:
        # Load tokenizer
        logger.info("Initializing tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            token=hf_token,
            padding_side="left"
        )
        # Ensure the tokenizer has a pad token set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Log CUDA information for debugging
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            logger.warning("CUDA not available, inference will be significantly slower")

        # Load model with device_map="auto" for automatic optimal placement
        logger.info("Initializing model with device_map='auto' and float16 precision...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            token=hf_token,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )

        # Set model to evaluation mode
        model.eval()
        logger.info("Model configured for inference (evaluation mode)")

        logger.info("Model and tokenizer loaded successfully")
        return tokenizer, model

    except Exception as e:
        logger.error(f"Error loading model or tokenizer: {e}")
        raise

def build_prompt(source_text, source_lang="", target_lang="English"):
    """
    Build a prompt designed for WMT benchmark translation and semantic similarity evaluation.

    Args:
        source_text (str): The source sentence to be translated.
        source_lang (str): Ignored in this version.
        target_lang (str): Ignored in this version.

    Returns:
        str: A formatted prompt string for the LLaMA-3-8B-Instruct model.
    """
    system_prompt = (
        "You are a certified WMT benchmark translator. Translate the following sentence from the WMT22 dataset into English.\n\n"
        "Your translation will be directly compared to WMT system outputs using the 'all-mpnet-base-v2' semantic similarity model. "
        "To ensure accurate benchmarking, provide exactly one clean English sentence—no alternative translations, explanations, or additional text."
    )

    user_message = f"Sentence:\n{source_text}\n\nTranslation:"

    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>\n"
        f"<|start_header_id|>user<|end_header_id|>\n{user_message}<|eot_id|>\n"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )
    return prompt


def clean_output(output_text):
    """
    Clean the model output to extract just the translated text.

    Args:
        output_text (str): Raw model output

    Returns:
        str: Cleaned translation
    """
    # Remove any model-specific formatting
    if "<|start_header_id|>assistant<|end_header_id|>" in output_text:
        output_text = output_text.split("<|start_header_id|>assistant<|end_header_id|>", 1)[1]

    # Remove any trailing special tokens
    output_text = re.sub(r'<\|begin_of_text\|>|<\|end_of_text\|>|<\|eot_id\|>|<\|start_header_id\|>.*?<\|end_header_id\|>', '', output_text)

    # Remove any commentary the model might add
    commentary_patterns = [
        r"\n\nTranslation:",
        r"\n\n",
        r"Here's the translation:",
        r"The translation is:",
        r"In English, this means:",
        r"Translated to English:"
    ]

    for pattern in commentary_patterns:
        parts = re.split(pattern, output_text, 1)
        if len(parts) > 1 and not parts[0].strip():
            output_text = parts[1]  # Take what comes after the pattern

    # Remove any quotation marks that might wrap the translation
    output_text = re.sub(r'^[\s"\']+|[\s"\']+$', '', output_text)

    # Final cleanup
    output_text = output_text.strip()

    return output_text

def get_jsonl_files(input_dir):
    """
    Get list of JSONL files from the input directory.

    Args:
        input_dir (str): Path to input directory

    Returns:
        list: List of paths to JSONL files
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    jsonl_files = list(input_path.glob("*.jsonl"))
    if not jsonl_files:
        logger.warning(f"No JSONL files found in {input_dir}")

    return jsonl_files

def ensure_output_dir(output_dir):
    """
    Ensure the output directory exists, creating it if necessary.

    Args:
        output_dir (str): Path to output directory

    Returns:
        Path: Path object for the output directory
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        logger.info(f"Creating output directory: {output_dir}")
        output_path.mkdir(parents=True, exist_ok=True)
    return output_path

def process_file(file_path, output_path, tokenizer, model, source_lang="", target_lang="English", max_length=128, batch_size=8):
    """
    Process a single JSONL file, translating the "src" field of each entry with batched inference.

    Args:
        file_path (Path): Path to input JSONL file
        output_path (Path): Path to output directory
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        source_lang (str): Optional source language code/name
        target_lang (str): Target language name
        max_length (int): Maximum output length
        batch_size (int): Number of examples to process in a single batch

    Returns:
        int: Number of entries processed
    """
    output_file = output_path / file_path.name
    processed_count = 0

    try:
        # Load all data from the file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f if line.strip()]

        logger.info(f"Processing {file_path.name} with {len(data)} entries")

        # Filter entries with 'src' field
        valid_entries = [entry for entry in data if 'src' in entry]
        if len(valid_entries) < len(data):
            logger.warning(f"Skipped {len(data) - len(valid_entries)} entries without 'src' field")

        # Process entries in batches
        for i in range(0, len(valid_entries), batch_size):
            batch_entries = valid_entries[i:i+batch_size]
            batch_prompts = [build_prompt(entry['src'], source_lang, target_lang) for entry in batch_entries]

            # Tokenize inputs in batch with padding
            batch_inputs = tokenizer.batch_encode_plus(
                batch_prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512  # Limit input length to prevent OOM
            )

            # Generate translations
            with torch.no_grad():
                batch_outputs = model.generate(
                    input_ids=batch_inputs.input_ids.to(model.device),
                    attention_mask=batch_inputs.attention_mask.to(model.device),
                    max_length=batch_inputs.input_ids.shape[1] + max_length,
                    do_sample=False,
                    num_beams=2,  # Using beam search for better translation quality
                    pad_token_id=tokenizer.eos_token_id
                )

            # Decode outputs and clean them
            batch_texts = tokenizer.batch_decode(batch_outputs, skip_special_tokens=False)  # Keep special tokens for better parsing
            batch_translations = [clean_output(text) for text in batch_texts]

            # Add translations to entries
            for entry, translation in zip(batch_entries, batch_translations):
                entry['src_translation'] = translation
                processed_count += 1

            # Update progress
            logger.debug(f"Processed batch {i//batch_size + 1}/{(len(valid_entries) + batch_size - 1)//batch_size}")

            # Clear GPU memory after each batch
            del batch_inputs, batch_outputs, batch_texts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save the processed data
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        logger.info(f"Saved translated file to {output_file}")
        return processed_count

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        raise

def process_files_in_batches(jsonl_files, output_path, tokenizer, model, args):
    """
    Process JSONL files in batches to manage memory efficiently.

    Args:
        jsonl_files (list): List of JSONL file paths
        output_path (Path): Path to output directory
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        args: Command line arguments

    Returns:
        int: Total number of entries processed
    """
    total_processed = 0

    for i in range(0, len(jsonl_files), args.file_batch_size):
        batch_files = jsonl_files[i:i+args.file_batch_size]
        logger.info(f"Processing file batch {i//args.file_batch_size + 1}/{(len(jsonl_files) + args.file_batch_size - 1)//args.file_batch_size}")

        for file_path in batch_files:
            processed = process_file(
                file_path,
                output_path,
                tokenizer,
                model,
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                max_length=args.max_length,
                batch_size=args.translation_batch_size
            )
            total_processed += processed

        # Clear memory after each batch of files
        logger.info("Clearing memory between file batches")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"GPU memory after cleanup: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated, {torch.cuda.memory_reserved()/1e9:.2f} GB reserved")

    return total_processed

def main():
    """Main function to run the translation pipeline."""
    # Parse command line arguments
    args = parse_arguments()

    try:
        logger.info(f"Starting {args.model_path} translation tool...")
        logger.info(f"Arguments: input_dir={args.input_dir}, output_dir={args.output_dir}, "
                   f"model={args.model_path}, "
                   f"file_batch_size={args.file_batch_size}, translation_batch_size={args.translation_batch_size}, "
                   f"max_length={args.max_length}")

        # Print system information
        logger.info(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Mount Google Drive
        mount_google_drive()

        # Get JSONL files
        jsonl_files = get_jsonl_files(args.input_dir)
        logger.info(f"Found {len(jsonl_files)} JSONL files to process")

        if not jsonl_files:
            logger.error("No JSONL files found. Exiting.")
            return 1

        # Ensure output directory exists
        output_path = ensure_output_dir(args.output_dir)

        # Load model and tokenizer
        tokenizer, model = load_model(args.hf_token, args.model_path)

        # Process files in batches
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

        if start_time:
            start_time.record()

        total_processed = process_files_in_batches(
            jsonl_files,
            output_path,
            tokenizer,
            model,
            args
        )

        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
            logger.info(f"Total GPU processing time: {elapsed_time:.2f} seconds")

            if total_processed > 0:
                logger.info(f"Average time per translation: {elapsed_time / total_processed:.4f} seconds")

        logger.info(f"Translation complete. Processed {total_processed} entries across {len(jsonl_files)} files.")

        # Clean up resources
        logger.info("Cleaning up resources...")
        del tokenizer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"Final GPU memory state: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    logger.info("Script completed successfully")
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)