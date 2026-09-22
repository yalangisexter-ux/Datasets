#!/usr/bin/env python3
"""
============================================================
SHADOW-RONIN 4B - SINGLE FILE KAGGLE TRAINING PIPELINE
============================================================

Pipeline:
1. Install dependencies
2. Discover JSONL datasets under /kaggle/input
3. Validate + normalize + deduplicate
4. Train QLoRA/SFT once
5. Save LoRA adapter
6. Merge adapter into FP16 Transformers model
7. Convert to F16 GGUF
8. Build Shadow-Ronin calibration set
9. Build importance matrix
10. Produce:
      Q2_K
      Q3_K_M
      Q4_0
      Q4_K_M
      Q5_K_S
      Q5_K_M
      Q6_K
      Q8_0
      F16
11. Run basic GGUF smoke tests
12. Write manifest/README
13. Optionally upload to Hugging Face

IMPORTANT:
- Change HF_USERNAME below if you want automatic upload.
- HF_TOKEN should be stored in Kaggle Secrets.
- The model architecture metadata remains Qwen3-compatible
  internally so runtimes can load it correctly.
- User-facing artifact/repository names are Shadow-Ronin.
"""

import os
import sys
import json
import re
import random
import hashlib
import shutil
import subprocess
from pathlib import Path
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

BASE_MODEL = "Qwen/Qwen3-4B"

MODEL_NAME = "Shadow-Ronin-4B"

HF_USERNAME = "YOUR_HF_USERNAME"

ROOT = Path("/kaggle/working/shadow_ronin")

DATA_ROOT = Path("/kaggle/input")

DATA_DIR = ROOT / "dataset"
ADAPTER_DIR = ROOT / "adapter"
MERGED_DIR = ROOT / "merged"
GGUF_DIR = ROOT / "gguf"

CALIBRATION_FILE = ROOT / "shadow_ronin_calibration.txt"

MANIFEST_FILE = ROOT / "manifest.json"

EVAL_FILE = ROOT / "evaluation.json"

README_FILE = ROOT / "README.md"

SEED = 42

MAX_SEQ_LENGTH = 2048

NUM_EPOCHS = 3

LEARNING_RATE = 1e-4

PER_DEVICE_BATCH_SIZE = 1

GRADIENT_ACCUMULATION_STEPS = 8

LORA_R = 16

LORA_ALPHA = 32

LORA_DROPOUT = 0.05

QUANT_TYPES = [
    "Q2_K",
    "Q3_K_M",
    "Q4_0",
    "Q4_K_M",
    "Q5_K_S",
    "Q5_K_M",
    "Q6_K",
    "Q8_0",
]

INSTALL_DEPENDENCIES = True

BUILD_LLAMA_CPP = True

RUN_SMOKE_TESTS = True

UPLOAD_TO_HF = False

# ============================================================
# UTILITIES
# ============================================================

def run(command, check=True, cwd=None, capture=False):
    print()
    print("$", " ".join(map(str, command)))

    return subprocess.run(
        [str(x) for x in command],
        cwd=str(cwd) if cwd else None,
        check=check,
        text=True,
        capture_output=capture
    )


def ensure_dirs():
    for directory in [
        ROOT,
        DATA_DIR,
        ADAPTER_DIR,
        MERGED_DIR,
        GGUF_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def cleanup_memory():
    try:
        import gc
        gc.collect()
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception:
        pass


# ============================================================
# INSTALL
# ============================================================

def install_dependencies():

    if not INSTALL_DEPENDENCIES:
        return

    print("=" * 70)
    print("INSTALLING PYTHON DEPENDENCIES")
    print("=" * 70)

    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-U",
        "transformers>=4.56",
        "datasets>=3.0",
        "accelerate>=1.0",
        "peft>=0.14",
        "trl>=0.20",
        "bitsandbytes",
        "sentencepiece",
        "safetensors",
        "huggingface_hub"
    ])

    print("Installing build dependencies...")

    run([
        "apt-get",
        "-qq",
        "update"
    ])

    run([
        "apt-get",
        "-qq",
        "install",
        "-y",
        "cmake",
        "build-essential",
        "git"
    ])


# ============================================================
# GPU CHECK
# ============================================================

def gpu_check():

    import torch

    print("=" * 70)
    print("GPU CHECK")
    print("=" * 70)

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU is required for this QLoRA configuration. "
            "In Kaggle enable Notebook Settings -> Accelerator -> GPU."
        )

    for i in range(torch.cuda.device_count()):

        print(
            f"GPU {i}: "
            f"{torch.cuda.get_device_name(i)}"
        )

    print(
        "CUDA:",
        torch.version.cuda
    )

    print(
        "BF16 supported:",
        torch.cuda.is_bf16_supported()
    )

    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            2
        ),
        "GB"
    )


# ============================================================
# DATASET NORMALIZATION
# ============================================================

SYSTEM_PROMPT = (
    "You are Shadow-Ronin, an advanced technical and "
    "general-purpose AI assistant. Be accurate, direct, "
    "context-aware, and avoid inventing facts."
)


def normalize_record(obj):

    if not isinstance(obj, dict):
        return None

    messages = obj.get("messages")

    if messages is None:
        messages = obj.get("conversations")

    if messages is None:

        if (
            "prompt" in obj
            and "response" in obj
        ):

            messages = [
                {
                    "role": "user",
                    "content": str(obj["prompt"])
                },
                {
                    "role": "assistant",
                    "content": str(obj["response"])
                }
            ]

    if not isinstance(messages, list):
        return None

    if len(messages) < 2:
        return None

    cleaned = []

    for message in messages:

        if not isinstance(message, dict):
            return None

        role = message.get("role")
        content = message.get("content")

        if role not in {
            "system",
            "user",
            "assistant"
        }:
            return None

        if not isinstance(content, str):
            return None

        content = content.strip()

        if not content:
            return None

        cleaned.append({
            "role": role,
            "content": content
        })

    if not any(
        m["role"] == "user"
        for m in cleaned
    ):
        return None

    if not any(
        m["role"] == "assistant"
        for m in cleaned
    ):
        return None

    if cleaned[0]["role"] != "system":

        cleaned.insert(
            0,
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        )

    metadata = obj.get(
        "metadata",
        {}
    )

    if not isinstance(
        metadata,
        dict
    ):
        metadata = {}

    return {
        "messages": cleaned,
        "metadata": metadata
    }


def record_hash(record):

    data = json.dumps(
        record["messages"],
        ensure_ascii=False,
        sort_keys=True
    )

    return hashlib.sha256(
        data.encode("utf-8")
    ).hexdigest()


def load_dataset():

    print("=" * 70)
    print("DISCOVERING DATASETS")
    print("=" * 70)

    files = sorted(
        DATA_ROOT.rglob("*.jsonl")
    )

    if not files:

        raise FileNotFoundError(
            "No JSONL dataset found under /kaggle/input."
        )

    for path in files:

        print(
            path,
            f"({path.stat().st_size / 1024 / 1024:.2f} MB)"
        )

    raw = []

    invalid = 0

    for path in files:

        print(
            "Reading:",
            path
        )

        try:

            with open(
                path,
                "r",
                encoding="utf-8",
                errors="replace"
            ) as f:

                for line in f:

                    line = line.strip()

                    if not line:
                        continue

                    try:

                        obj = json.loads(line)

                        record = normalize_record(
                            obj
                        )

                        if record is None:
                            invalid += 1
                        else:
                            raw.append(record)

                    except json.JSONDecodeError:
                        invalid += 1

        except Exception as exc:

            print(
                "WARNING:",
                path,
                exc
            )

    unique = []

    seen = set()

    for record in raw:

        digest = record_hash(
            record
        )

        if digest not in seen:

            seen.add(digest)

            unique.append(record)

    print()
    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print("Raw:", len(raw))

    print("Invalid:", invalid)

    print(
        "Duplicates:",
        len(raw) - len(unique)
    )

    print(
        "Final:",
        len(unique)
    )

    if len(unique) < 20:

        raise RuntimeError(
            "Too few valid training examples."
        )

    output = (
        DATA_DIR /
        "shadow_ronin_training_normalized.jsonl"
    )

    with open(
        output,
        "w",
        encoding="utf-8"
    ) as f:

        for record in unique:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )

    return unique


# ============================================================
# DATASET STATISTICS
# ============================================================

def dataset_statistics(records):

    domains = Counter()
    languages = Counter()
    types = Counter()

    for record in records:

        metadata = record.get(
            "metadata",
            {}
        )

        domains[
            metadata.get(
                "domain",
                "unknown"
            )
        ] += 1

        languages[
            metadata.get(
                "language",
                "unknown"
            )
        ] += 1

        types[
            metadata.get(
                "type",
                "unknown"
            )
        ] += 1

    print()
    print("=" * 70)
    print("DOMAIN DISTRIBUTION")
    print("=" * 70)

    for key, value in domains.most_common():

        print(
            f"{key:40s} {value}"
        )

    print()
    print("=" * 70)
    print("LANGUAGE DISTRIBUTION")
    print("=" * 70)

    for key, value in languages.most_common():

        print(
            f"{key:40s} {value}"
        )

    return {
        "domains": dict(domains),
        "languages": dict(languages),
        "types": dict(types)
    }


# ============================================================
# TRAIN
# ============================================================

def train_model(records):

    import torch

    from datasets import Dataset

    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig
    )

    from peft import (
        LoraConfig,
        prepare_model_for_kbit_training
    )

    from trl import (
        SFTTrainer,
        SFTConfig
    )

    print("=" * 70)
    print("LOADING TOKENIZER")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        use_fast=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.chat_template is None:

        raise RuntimeError(
            "The selected tokenizer has no chat template."
        )

    dataset = Dataset.from_list(
        records
    )

    def format_example(example):

        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )

        return {
            "text": text
        }

    dataset = dataset.map(
        format_example,
        remove_columns=dataset.column_names
    )

    split = dataset.train_test_split(
        test_size=0.05,
        seed=SEED
    )

    train_dataset = split["train"]
    eval_dataset = split["test"]

    print(
        "Train examples:",
        len(train_dataset)
    )

    print(
        "Eval examples:",
        len(eval_dataset)
    )

    print("=" * 70)
    print("LOADING 4-BIT QLORA MODEL")
    print("=" * 70)

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model
    )

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ]
    )

    use_bf16 = (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )

    use_fp16 = (
        torch.cuda.is_available()
        and not use_bf16
    )

    training_args = SFTConfig(

        output_dir=str(
            ADAPTER_DIR
        ),

        num_train_epochs=NUM_EPOCHS,

        learning_rate=LEARNING_RATE,

        per_device_train_batch_size=(
            PER_DEVICE_BATCH_SIZE
        ),

        per_device_eval_batch_size=1,

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),

        gradient_checkpointing=True,

        optim="paged_adamw_8bit",

        logging_steps=5,

        eval_strategy="steps",

        eval_steps=50,

        save_strategy="steps",

        save_steps=50,

        save_total_limit=2,

        bf16=use_bf16,

        fp16=use_fp16,

        report_to="none",

        seed=SEED,

        max_length=MAX_SEQ_LENGTH,

        packing=True
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config
    )

    print("=" * 70)
    print("STARTING SHADOW-RONIN TRAINING")
    print("=" * 70)

    trainer.train()

    trainer.save_model(
        ADAPTER_DIR
    )

    tokenizer.save_pretrained(
        ADAPTER_DIR
    )

    print(
        "Adapter saved:",
        ADAPTER_DIR
    )

    del trainer
    del model

    cleanup_memory()

    return tokenizer


# ============================================================
# MERGE
# ============================================================

def merge_model(tokenizer):

    import torch

    from transformers import (
        AutoModelForCausalLM
    )

    from peft import (
        PeftModel
    )

    print("=" * 70)
    print("MERGING LORA INTO SHADOW-RONIN")
    print("=" * 70)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    adapter = PeftModel.from_pretrained(
        base_model,
        ADAPTER_DIR
    )

    merged = adapter.merge_and_unload()

    merged.save_pretrained(
        MERGED_DIR,
        safe_serialization=True,
        max_shard_size="4GB"
    )

    tokenizer.save_pretrained(
        MERGED_DIR
    )

    del adapter
    del base_model
    del merged

    cleanup_memory()

    print(
        "Merged model saved:",
        MERGED_DIR
    )


# ============================================================
# LLAMA.CPP
# ============================================================

def build_llama_cpp():

    llama_dir = (
        ROOT /
        "llama.cpp"
    )

    if llama_dir.exists():

        print(
            "llama.cpp already exists."
        )

    else:

        run([
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/ggml-org/llama.cpp.git",
            str(llama_dir)
        ])

    build_dir = (
        llama_dir /
        "build"
    )

    if BUILD_LLAMA_CPP:

        run([
            "cmake",
            "-S",
            str(llama_dir),
            "-B",
            str(build_dir),
            "-DGGML_CUDA=OFF"
        ])

        run([
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            "Release",
            "-j",
            "2"
        ])

    return llama_dir, build_dir


# ============================================================
# CONVERT F16
# ============================================================

def convert_to_f16_gguf(llama_dir):

    converter = (
        llama_dir /
        "convert_hf_to_gguf.py"
    )

    if not converter.exists():

        raise FileNotFoundError(
            "convert_hf_to_gguf.py not found."
        )

    output = (
        GGUF_DIR /
        f"{MODEL_NAME}-F16.gguf"
    )

    if output.exists():

        print(
            "F16 already exists. Skipping."
        )

        return output

    run([
        sys.executable,
        str(converter),
        str(MERGED_DIR),
        "--outfile",
        str(output),
        "--outtype",
        "f16"
    ])

    return output


# ============================================================
# CALIBRATION
# ============================================================

def create_calibration(records):

    texts = []

    for record in records:

        for message in record["messages"]:

            if message["role"] in {
                "user",
                "assistant"
            }:

                content = (
                    message["content"]
                    .strip()
                )

                if content:
                    texts.append(content)

    random.Random(
        SEED
    ).shuffle(texts)

    # Enough variety without creating a massive file.
    texts = texts[:2000]

    with open(
        CALIBRATION_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n\n".join(texts)
        )

    print(
        "Calibration examples:",
        len(texts)
    )


# ============================================================
# IMATRIX
# ============================================================

def create_imatrix(
    build_dir,
    f16_gguf
):

    imatrix_bin = (
        build_dir /
        "bin" /
        "llama-imatrix"
    )

    if not imatrix_bin.exists():

        raise FileNotFoundError(
            f"llama-imatrix not found: {imatrix_bin}"
        )

    output = (
        GGUF_DIR /
        f"{MODEL_NAME}-imatrix.gguf"
    )

    if output.exists():

        print(
            "imatrix already exists. Skipping."
        )

        return output

    run([
        str(imatrix_bin),
        "-m",
        str(f16_gguf),
        "-f",
        str(CALIBRATION_FILE),
        "-o",
        str(output),
        "-ngl",
        "0"
    ])

    return output


# ============================================================
# QUANTIZATION
# ============================================================

def quantize_models(
    build_dir,
    f16_gguf,
    imatrix
):

    quant_bin = (
        build_dir /
        "bin" /
        "llama-quantize"
    )

    if not quant_bin.exists():

        raise FileNotFoundError(
            f"llama-quantize not found: {quant_bin}"
        )

    for quant in QUANT_TYPES:

        output = (
            GGUF_DIR /
            f"{MODEL_NAME}-{quant}.gguf"
        )

        if output.exists():

            print(
                f"{quant} already exists. Skipping."
            )

            continue

        print()
        print("=" * 70)
        print(
            "QUANTIZING:",
            quant
        )
        print("=" * 70)

        # Current llama.cpp syntax.
        # If the checked-out build rejects this syntax,
        # inspect --help and retry with the alternate form.
        command = [
            str(quant_bin),
            "--imatrix",
            str(imatrix),
            str(f16_gguf),
            str(output),
            quant
        ]

        result = run(
            command,
            check=False,
            capture=True
        )

        if result.returncode != 0:

            print(
                result.stdout
            )

            print(
                result.stderr
            )

            # Compatibility fallback.
            command = [
                str(quant_bin),
                str(f16_gguf),
                str(output),
                quant,
                "--imatrix",
                str(imatrix)
            ]

            run(
                command,
                check=True
            )

        print(
            "Created:",
            output
        )


# ============================================================
# SMOKE TEST
# ============================================================

def smoke_tests(build_dir):

    if not RUN_SMOKE_TESTS:
        return {}

    cli = (
        build_dir /
        "bin" /
        "llama-cli"
    )

    if not cli.exists():

        print(
            "llama-cli not found; skipping tests."
        )

        return {}

    test_prompts = [
        "Explain the difference between TCP and UDP.",
        "Review this Python code conceptually and explain how you would debug it.",
        "Explain BigFix relevance and how audit logic differs from remediation.",
        "What is the purpose of CIS and DISA STIG benchmarks?",
        "Explain ransomware detection and incident response at a defensive level.",
        "Solve: If 3x + 7 = 22, what is x?",
        "Explain this in natural Telugu: Linux server security enduku important?"
    ]

    results = []

    ggufs = sorted(
        GGUF_DIR.glob(
            f"{MODEL_NAME}-*.gguf"
        )
    )

    for model in ggufs:

        if "imatrix" in model.name:
            continue

        if model.name.endswith("-F16.gguf"):
            continue

        print()
        print(
            "=" * 70
        )

        print(
            "SMOKE TEST:",
            model.name
        )

        print(
            "=" * 70
        )

        prompt = test_prompts[
            len(results) % len(test_prompts)
        ]

        command = [
            str(cli),
            "-m",
            str(model),
            "-n",
            "128",
            "-p",
            prompt
        ]

        result = run(
            command,
            check=False,
            capture=True
        )

        ok = (
            result.returncode == 0
        )

        results.append({
            "model": model.name,
            "prompt": prompt,
            "success": ok,
            "output": (
                result.stdout[-4000:]
                if result.stdout
                else result.stderr[-4000:]
            )
        })

        print(
            "SUCCESS" if ok else "FAILED"
        )

    return results


# ============================================================
# README / MANIFEST
# ============================================================

def write_metadata(
    records,
    stats,
    eval_results
):

    model_files = []

    for path in sorted(
        GGUF_DIR.glob(
            f"{MODEL_NAME}-*.gguf"
        )
    ):

        if "imatrix" in path.name:
            continue

        model_files.append({
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path)
        })

    manifest = {
        "model_name": MODEL_NAME,
        "base_model": BASE_MODEL,
        "training_examples": len(records),
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "max_sequence_length": MAX_SEQ_LENGTH,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "quantizations": QUANT_TYPES,
        "statistics": stats,
        "models": model_files,
        "smoke_tests": eval_results
    }

    with open(
        MANIFEST_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False
        )

    with open(
        EVAL_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            eval_results,
            f,
            indent=2,
            ensure_ascii=False
        )

    readme = f"""# {MODEL_NAME}

Shadow-Ronin is a Qwen3-4B-derived model fine-tuned using
the Shadow-Ronin training dataset.

## Model

Name: {MODEL_NAME}

Training base:
{BASE_MODEL}

## Training

QLoRA / SFT

Epochs:
{NUM_EPOCHS}

Maximum sequence length:
{MAX_SEQ_LENGTH}

LoRA rank:
{LORA_R}

## Capabilities targeted

- Telugu
- Roman Telugu
- Telugu-English mixed language
- English
- Mathematics
- Programming
- Software architecture
- Cloud architecture
- DevOps
- Linux administration
- Networking
- Cybersecurity
- Authorized security testing
- Malware detection and analysis
- Ransomware defense
- RDP security
- Code review
- Debugging
- Repository intelligence
- BigFix
- CIS
- DISA STIG
- SCAP
- XCCDF
- OVAL
- CDML
- Technical writing
- Philosophy
- Geography
- Science

## GGUF variants

"""

    for quant in QUANT_TYPES:

        readme += (
            f"- `{MODEL_NAME}-{quant}.gguf`\\n"
        )

    readme += (
        f"- `{MODEL_NAME}-F16.gguf`\\n"
    )

    readme += """
## Runtime compatibility

The exported files are named Shadow-Ronin, while internal
architecture metadata remains compatible with the Qwen3
architecture so Transformers/llama.cpp can load the model.

## Security

Security training is intended for authorized testing,
defensive analysis, detection, hardening, and controlled
laboratory environments.
"""

    with open(
        README_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(readme)


# ============================================================
# OPTIONAL HF UPLOAD
# ============================================================

def upload_to_huggingface():

    if not UPLOAD_TO_HF:

        print(
            "Hugging Face upload disabled."
        )

        return

    if HF_USERNAME == "YOUR_HF_USERNAME":

        raise ValueError(
            "Set HF_USERNAME before enabling upload."
        )

    token = os.environ.get(
        "HF_TOKEN"
    )

    if not token:

        raise RuntimeError(
            "HF_TOKEN not found in environment/Kaggle Secrets."
        )

    from huggingface_hub import (
        login,
        HfApi
    )

    login(
        token=token
    )

    api = HfApi()

    transformers_repo = (
        f"{HF_USERNAME}/{MODEL_NAME}"
    )

    gguf_repo = (
        f"{HF_USERNAME}/{MODEL_NAME}-GGUF"
    )

    api.create_repo(
        repo_id=transformers_repo,
        repo_type="model",
        exist_ok=True
    )

    api.create_repo(
        repo_id=gguf_repo,
        repo_type="model",
        exist_ok=True
    )

    api.upload_folder(
        folder_path=str(
            MERGED_DIR
        ),
        repo_id=transformers_repo,
        repo_type="model",
        commit_message=(
            "Upload Shadow-Ronin-4B"
        )
    )

    for model in sorted(
        GGUF_DIR.glob(
            f"{MODEL_NAME}-*.gguf"
        )
    ):

        if "imatrix" in model.name:
            continue

        api.upload_file(
            path_or_fileobj=str(model),
            path_in_repo=model.name,
            repo_id=gguf_repo,
            repo_type="model",
            commit_message=(
                f"Upload {model.name}"
            )
        )

    api.upload_file(
        path_or_fileobj=str(
            README_FILE
        ),
        path_in_repo="README.md",
        repo_id=gguf_repo,
        repo_type="model",
        commit_message="Add Shadow-Ronin README"
    )

    print(
        "Hugging Face upload complete."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("        SHADOW-RONIN 4B BUILD PIPELINE")
    print("=" * 70)
    print()

    random.seed(
        SEED
    )

    ensure_dirs()

    install_dependencies()

    # Imports after pip installation.
    import torch

    gpu_check()

    records = load_dataset()

    stats = dataset_statistics(
        records
    )

    # --------------------------------------------------------
    # TRAIN ONCE
    # --------------------------------------------------------

    tokenizer = train_model(
        records
    )

    # --------------------------------------------------------
    # MERGE ONCE
    # --------------------------------------------------------

    merge_model(
        tokenizer
    )

    # --------------------------------------------------------
    # LLAMA.CPP
    # --------------------------------------------------------

    llama_dir, build_dir = (
        build_llama_cpp()
    )

    # --------------------------------------------------------
    # F16
    # --------------------------------------------------------

    f16_gguf = convert_to_f16_gguf(
        llama_dir
    )

    # --------------------------------------------------------
    # CALIBRATION
    # --------------------------------------------------------

    create_calibration(
        records
    )

    # --------------------------------------------------------
    # IMATRIX
    # --------------------------------------------------------

    imatrix = create_imatrix(
        build_dir,
        f16_gguf
    )

    # --------------------------------------------------------
    # ALL QUANTS
    # --------------------------------------------------------

    quantize_models(
        build_dir,
        f16_gguf,
        imatrix
    )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    eval_results = smoke_tests(
        build_dir
    )

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    write_metadata(
        records,
        stats,
        eval_results
    )

    # --------------------------------------------------------
    # OPTIONAL UPLOAD
    # --------------------------------------------------------

    upload_to_huggingface()

    print()
    print("=" * 70)
    print("        SHADOW-RONIN BUILD COMPLETE")
    print("=" * 70)
    print()

    print(
        "Output directory:",
        ROOT
    )

    print()
    print("GGUF files:")

    for model in sorted(
        GGUF_DIR.glob(
            f"{MODEL_NAME}-*.gguf"
        )
    ):

        if "imatrix" in model.name:
            continue

        size_gb = (
            model.stat().st_size
            / (1024 ** 3)
        )

        print(
            f"  {model.name:40s}"
            f"{size_gb:.3f} GB"
        )

    print()
    print(
        "Manifest:",
        MANIFEST_FILE
    )

    print(
        "README:",
        README_FILE
    )

    print()
    print(
        "Shadow-Ronin is ready."
    )


if __name__ == "__main__":
    main()
