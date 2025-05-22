# LLMs Are Not Scorers: Rethinking MT Evaluation with Generation-Based Methods

This repository contains the source code, data, and experiment results for our paper.

## Paper Abstract

Recent work has applied large language models (LLMs) to machine translation quality estimation (MTQE) by prompting models to directly assign quality scores. However, we find that these direct scoring approaches suffer from low segment-level correlation with human judgments—especially when using decoder-only LLMs, which are not trained for regression tasks.

To address this, we propose a generation-based evaluation paradigm that leverages LLMs to generate fluent, high-quality reference translations. These references are then compared with MT outputs using semantic similarity scores from Sentence-BERT.

We conduct large-scale evaluations across 8 LLMs and 9 language pairs, and demonstrate that our method outperforms both direct LLM scoring baselines and non-LLM reference-free MTQE metrics from the MTME benchmark.

## Repository Structure

### experiment_1: Reproducing and Extending EMNLP 2024 Baseline

```
experiment_1/
├── step1_baseline_raw_data/            # Raw TSV files from Qian et al. (EMNLP 2024)
├── step1.5_unify_file_format.py        # Format converter script
├── step2_final_processed_data/         # Final JSONL input used in our method
├── step3_model_translation/            # LLM-generated reference translations
│   ├── gemma7b/, llama7b/, etc.
│   └── py/model_translation.py         # Multi-model unified translation script
├── step3.5_semantic_similarity_evaluator.py  # Computes similarity between 'src_translation' and 'mt'
├── step4_similarity_correlation/       # Final correlation results per model/language pair
```

### experiment_2: Benchmark Comparison with WMT & MTME Metrics

```
experiment_2/
├── step1_WMT_raw_data/                 # Official WMT test sets, system outputs, human scores
├── step1.5_unify_file_format.py        # Format conversion script
├── step2_final_processed_data/         # Processed JSONL files
├── step3_model_translation/            # LLM translations (same format as experiment_1)
├── step3.5_py/                         # Custom MTME-style evaluator
│   ├── mt_eval_pipeline.py             # Evaluation pipeline
│   ├── example_gpt_de_en.py/.ipynb     # Full reproducible example
├── step4_similarity_correlation/       # Similarity scores + correlation summary
├── step5_mtme/                         # Official MTME metric scores for comparison
```
### Key Differences in `experiment_2`

While `experiment_2` largely follows the same structure as `experiment_1`, it introduces three important differences tailored to the MTME benchmark setting:

1. **Different Source of Raw Data**  
   The baseline in `experiment_2` is based on MTME-provided metric scores, and the raw data is taken directly from the official WMT dataset. All raw files are stored under `step1_WMT_raw_data/`.  
   Refer to the official MTME GitHub repository for details on how to obtain and format these files:  
   https://github.com/google-research/mt-metrics-eval

2. **MTME Baseline Results**  
   The directory `step5_mtme/` contains the official evaluation scores from the MTME benchmark, including Kendall, Spearman, and Pearson correlation scores for each language pair. This serves as the comparison baseline for our similarity-based method.

3. **Specialized Similarity Evaluation Pipeline**  
   Since MTME requires scoring multiple system outputs for each language pair, we introduce a dedicated script `mt_eval_pipeline.py` under `step3.5_py/`.  
   This script ensures compatibility with MTME-style evaluations, computing similarity scores between LLM-generated references (`src_translation`) and all system outputs (`mt`).  
   Example scripts are provided to replicate the full process:
   - `example_gpt_de_en.py`
   - `example_gpt_de_en.ipynb`

   These examples demonstrate how to compute semantic similarity on the `de-en` language pair using `gpt-4-turbo`. They serve as templates for extending the evaluation to other models and language pairs.

## Reproducibility: How to Run the Experiments

### Model Translation Options

You can run translation in two ways:

#### Option 1: Run individual model script

Each folder under `step3_model_translation/` contains a dedicated script named `{model_name}.py`:

```bash
python gemma7b.py \
  --input_dir step2_final_processed_data \
  --output_dir step3_model_translation/gemma7b \
  --hf_token your_huggingface_token
```

#### Option 2: Use the unified `model_translation.py`

This wrapper script in `py/` supports all models with argument checking and auto error reporting:

```bash
python model_translation.py \
  --model gemma7b \
  --input_dir step2_final_processed_data \
  --output_dir step3_model_translation/gemma7b \
  --hf_token your_huggingface_token
```

To check available options for a specific model:

```bash
python model_translation.py --model gemma7b --help
```

#### Supported Model Name Mapping

| Model Name in CLI | Actual Model ID |
|-------------------|------------------|
| gemma7b           | google/gemma-7b |
| llama7b           | meta-llama/Llama-2-7b-chat-hf |
| llama8b           | meta-llama/Llama-3-8B-Instruct |
| qwen8b            | Qwen/Qwen3-8B |
| qwen14b           | Qwen/Qwen1.5-14B-Chat |
| openchat3_5       | openchat/openchat_3.5 |
| deepseekr1        | deepseek-r1 |
| gpt4              | gpt-4-turbo |

Note: Different models require different API tokens:

- Hugging Face models → `--hf_token`
- DeepSeek → `--deepseek_api_key`
- GPT-4 → `--openai_api_key`


### 1. Preprocess Raw Data

```bash
python step1.5_unify_file_format.py
```

### 2. Translate with LLMs

#### Option 1: Individual model script

```bash
python gemma7b.py \
  --input_dir step2_final_processed_data \
  --output_dir step3_model_translation/gemma7b \
  --hf_token your_huggingface_token
```

#### Option 2: Unified translation script

```bash
python model_translation.py \
  --model gemma7b \
  --input_dir step2_final_processed_data \
  --output_dir step3_model_translation/gemma7b \
  --hf_token your_huggingface_token
```

### 3. Compute Semantic Similarity

```bash
python step3.5_semantic_similarity_evaluator.py \
  --input_files path/to/*.jsonl \
  --output_dir path/to/output \
  --model_name sentence-transformers/all-mpnet-base-v2 \
  --token your_huggingface_token
```

### 4. Analyze Correlation

Check each model folder under `step4_similarity_correlation/` for:
- `correlation_summary.txt`
- `{lang}_scored.jsonl`
- `{lang}_scored_readable.jsonl`

## Environment Setup

Install required dependencies via pip:

```bash
pip install -r requirements.txt
```

The file includes core packages like `transformers`, `sentence-transformers`, `pandas`, and others used for model inference and evaluation.

## Citation (Placeholder)

Note: This work is currently under anonymous review for EMNLP 2025. Citation information will be added upon publication.

## Contact (Placeholder)

Note: Due to anonymous submission policy, author contact information is currently withheld. For any inquiries, please check back post-review.



---

## Additional Notes on Generation Settings

Although our codebase defaults to the following generation parameters:
- `do_sample=False`
- `num_beams=5`
- `max_length=128`

Please note that the actual experimental results reported in the paper do **not always** follow these settings exactly for all models and language pairs. The differences arise for the following reasons:

1. **Model Limitations**  
   Some models, such as `gpt-4-turbo`, do not support explicit settings for certain parameters like `do_sample`. In such cases, we substitute compatible alternatives (e.g., using `temperature=0.0` instead of `do_sample=False`) to approximate the intended behavior.

2. **Better Performance Under Alternate Settings**  
   In several instances, we observed that models achieved higher correlation scores with different parameter settings. For example:
   - `llama8b` performed better with `num_beams=2` than with `num_beams=5`.
   - For `qwen14b`, `et-en` yielded better results with `num_beams=5`, while `ne-en` performed better with `num_beams=1`.

3. **Prompt Adaptation for LLaMA-7B**  
   All models share a unified prompt format except for `llama7b`, which uses a simplified prompt due to its lower performance with longer instructions. This prompt adaptation was based on empirical comparison.

In the reported tables, we show the best-performing setting per model and language pair. The variations caused by these adjustments typically remain within a ±0.1 correlation range and do not affect the validity of our main conclusions.

For reproducibility, we recommend using the outputs provided in the `step3_model_translation/` folder, which contain the exact translations used in the final evaluation results shown in the paper.
