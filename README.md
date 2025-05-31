# SmolLM2: Benchmarking & QLoRA Finetuning

This repository explores the [`SmolLM2-1.7B-Instruct by HuggingFace`](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct), covering baseline performance, 4-bit QLoRA finetuning, and post-finetuning evaluation.

## Notebooks

### 1. Base Model Benchmark
[**`01_SmolLM2_Benchmark.ipynb`**](01_SmolLM2_Benchmark.ipynb): Establishes the baseline accuracy for the 4-bit quantized `SmolLM2-1.7B-Instruct` on the [BoolQ dataset by Google](https://huggingface.co/datasets/google/boolq).

### 2. 4bit QLoRA LLM Finetuning
[**`02_SmoLM2_4bit_Finetuning.ipynb`**](02_SmolLM2_4bit_Finetuning.ipynb): Demonstrates 4-bit LoRA (QLoRA) finetuning of `SmolLM2-1.7B-Instruct` on the BoolQ dataset.

![Loss Curve from WANDB](/img/QLORA%20Loss%20Curve%20from%20WANDB.png)

### 3. Finetuned Model Benchmark
[**`03_SmolLM2_QLORA_Benchmarking.ipynb`**](03_SmolLM2_QLoRA_Benchmarking.ipynb): Compares the performance of the baseline and QLoRA-finetuned model on the BoolQ validation set.

## Results

Performance on the BoolQ validation set after evaluating 1000 samples:

* **Baseline Model (4-bit quantized)**: **56.4% Accuracy**
* **Finetuned Model (4-bit quantized QLoRA)**: **79.4% Accuracy**

`QLoRA Finetuning enhaced LLM performance by 23% on BoolQ validation set!`

## Setup

To run the notebooks:

1.  Clone the repository.
    ```sh
    git clone https://github.com/chinmayajoshi/SmolLM2-QLoRA-BoolQ
    ```
2.  Install necessary libraries. Check notebooks for more details.
    ```sh
    pip install transformers datasets torch bitsandbytes accelerate hf_xet
    pip install peft trl wandb
    ```

## Acknowledgements 
- BoolQA Dataset: 
    - Checkout dataset on [HuggingFace](https://huggingface.co/datasets/google/boolq)
    - Checkout [github repo](https://github.com/google-research-datasets/boolean-questions)
    - Checkout the [paper by google](https://arxiv.org/abs/1905.10044)
- Base LLM Credit: [SmolLM2-1.7B-Instruct by HuggingFace](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)
- QLora Paper: [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [bitsandbytes](https://huggingface.co/docs/bitsandbytes/en/index) for accessible large language models via k-bit quantization for PyTorch