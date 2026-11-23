"""Hugging Face runner: immutable model/config/data, fixed batches, resumable output."""
from __future__ import annotations

import csv
import gc
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

from protocol import (atomic_json, audit_jobs, audit_prompt_templates,
                      baseline_jobs, digest, followup_jobs,
                      int_seed, lock_json, make_pool, parse_answer, question_text, read_json)


def code_fingerprint():
    root = Path(__file__).parent
    return {name: digest((root / name).read_text(encoding="utf-8"))
            for name in ["protocol.py", "engine.py", "analysis.py"]}


def prepare(root, config):
    """Fetch pinned public parquet, then persist a shared item pool."""
    from huggingface_hub import hf_hub_download
    import pandas as pd
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    sampling = {k: config[k] for k in ["seed", "dataset", "dataset_revision", "subjects", "main_per_subject"]}
    lock_json(root / "sampling_lock.json", sampling)
    pool_path = root / "item_pool.json"
    if not pool_path.exists():
        src = hf_hub_download(repo_id=config["dataset"], repo_type="dataset", revision=config["dataset_revision"],
                              filename="data/test-00000-of-00001.parquet")
        pool = make_pool(pd.read_parquet(src).to_dict("records"), config)
        atomic_json(pool_path, pool)
    pool = read_json(pool_path)
    if pool["dataset_revision"] != config["dataset_revision"]:
        raise RuntimeError("dataset revision mismatch")
    phase_dir = root / config["phase"]
    phase_dir.mkdir(exist_ok=True)
    return phase_dir, pool["pools"][config["phase"]]


def freeze(phase_dir, raw_items, config):
    phase_dir = Path(phase_dir)
    # Core study scores against the published key. No invented or certified evidence.
    if "R2" in config["conditions"]:
        raise ValueError("Core protocol does not support evidence conditions.")
    items = [dict(item) for item in raw_items]
    payload = {"config": config, "items": items, "code": code_fingerprint(),
               "prompt_repetition_audit": audit_prompt_templates()}
    payload["study_hash"] = digest(payload)
    lock_json(phase_dir / "study_lock.json", payload)
    return payload




def runtime_environment():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Colab: 런타임 → 런타임 유형 변경 → L4 또는 A100 GPU를 선택하세요.")
    name = torch.cuda.get_device_name(0)
    if not ("L4" in name or "A100" in name):
        raise RuntimeError(f"GPU {name}: 이 연구는 L4/A100 BF16 설정으로 고정돼 있습니다. T4 자동 대체를 하지 않습니다.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 support required")
    packages = {}
    for p in ["torch", "transformers", "accelerate", "bitsandbytes", "huggingface-hub", "tokenizers", "numpy", "pandas", "scipy"]:
        packages[p] = importlib.metadata.version(p)
    driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()
    return {"gpu": name, "driver": driver, "compute_capability": list(torch.cuda.get_device_capability(0)),
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "cuda": torch.version.cuda, "python": platform.python_version(), "packages": packages}


class HFBackend:
    def __init__(self, model_key, config):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
        self.environment = runtime_environment()
        self.config = config
        self.device = "cuda:0"
        spec = config["models"][model_key]
        self.tokenizer = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"], trust_remote_code=False)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(
            spec["repo"], revision=spec["revision"], trust_remote_code=False,
            quantization_config=quant, torch_dtype=torch.bfloat16, device_map={"": 0},
            attn_implementation=config["attention"], low_cpu_mem_usage=True)
        self.model.eval()
        # Prevent mutable model defaults from introducing hidden sampling parameters.
        self.generation = GenerationConfig(
            do_sample=config["do_sample"], temperature=config["temperature"], top_p=config["top_p"],
            top_k=config["top_k"], repetition_penalty=1.0, max_new_tokens=config["max_new_tokens"],
            pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.model.generation_config.eos_token_id,
            use_cache=True, num_beams=1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        self.environment["chat_template_hash"] = digest(self.tokenizer.chat_template)
        self.environment["generation_config"] = self.generation.to_dict()

    def render(self, job):
        return self.tokenizer.apply_chat_template(job["messages"], tokenize=False, add_generation_prompt=True,
                                                   enable_thinking=self.config["enable_thinking"])

    def preflight(self, jobs):
        """Refuse over-length inputs, rather than silently truncating selected conditions."""
        lengths = [len(self.tokenizer.encode(self.render(j), add_special_tokens=False)) for j in jobs]
        if lengths and max(lengths) > self.config["max_input_tokens"]:
            raise RuntimeError(f"입력 길이 {max(lengths)} > {self.config['max_input_tokens']}. 자동 절단하지 않습니다. 파일럿에서 공통 한도를 정하고 새 study로 실행하세요.")
        return {"count": len(jobs), "max_input_tokens_observed": max(lengths, default=0)}

    def generate(self, jobs, seed):
        import torch
        from transformers import set_seed
        set_seed(seed)
        rendered = [self.render(j) for j in jobs]
        inputs = self.tokenizer(rendered, padding=True, truncation=False, add_special_tokens=False,
                                return_token_type_ids=False, return_tensors="pt").to(self.device)
        if inputs.input_ids.shape[1] > self.config["max_input_tokens"]:
            raise RuntimeError("input too long; no truncation allowed")
        start = time.monotonic()
        with torch.inference_mode():
            out = self.model.generate(**inputs, generation_config=self.generation)
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        result = []
        eos = self.generation.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        for j, (sequence, prompt) in enumerate(zip(out, rendered)):
            ids = sequence[inputs.input_ids.shape[1]:].tolist()
            stop = next((k for k, token in enumerate(ids) if token in eos), None)
            truncated = stop is None and len(ids) >= self.config["max_new_tokens"]
            if stop is not None:
                ids = ids[:stop]
            result.append({"text": self.tokenizer.decode(ids, skip_special_tokens=True),
                           "input_tokens": int(inputs.attention_mask[j].sum().item()), "output_tokens": len(ids),
                           "truncated": truncated, "rendered_prompt_hash": digest(prompt),
                           "seconds_per_batch": elapsed})
        return result

    def close(self):
        import torch
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


def execute_batches(jobs, directory, backend, config, stop_after_batches=None):
    """Write complete batches atomically; never re-batch only the missing rows."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock_json(directory / "schedule.json", {"jobs_hash": digest(jobs), "task_ids": [j["task_id"] for j in jobs],
                                            "batch_size": config["batch_size"]})
    all_records, fresh = [], 0
    total = (len(jobs) + config["batch_size"] - 1) // config["batch_size"]
    for index, offset in enumerate(range(0, len(jobs), config["batch_size"])):
        batch = jobs[offset:offset + config["batch_size"]]
        path = directory / f"batch_{index:06d}.json"
        batch_hash = digest(batch)
        if path.exists():
            saved = read_json(path)
            if saved["batch_hash"] != batch_hash or len(saved["records"]) != len(batch) or saved["records_hash"] != digest(saved["records"]):
                raise RuntimeError(f"Corrupt or mismatched checkpoint: {path}")
            all_records.extend(saved["records"])
            continue
        if stop_after_batches is not None and fresh >= stop_after_batches:
            return all_records, False
        # Hash excludes model's previous answer so the randomized schedule/seed scheme is common across models.
        seed = int_seed(config["seed"], directory.name, index, [j["task_id"] for j in batch])
        outputs = backend.generate(batch, seed)
        if len(outputs) != len(batch):
            raise RuntimeError("backend returned wrong batch size")
        records = [{**job, **output, "batch_index": index, "batch_seed": seed} for job, output in zip(batch, outputs)]
        atomic_json(path, {"batch_hash": batch_hash, "records_hash": digest(records), "records": records})
        all_records.extend(records)
        fresh += 1
        if fresh == 1 or fresh % 10 == 0:
            print(f"{directory.name}: {index + 1}/{total} batches saved", flush=True)
    return all_records, True


def run_model(phase_dir, frozen, model_key, run_tag="primary", backend=None, stop_after_batches=None):
    """One run tag locks a hardware/software regime across *both* models."""
    config, items = frozen["config"], frozen["items"]
    if model_key not in config["models"]:
        raise ValueError("unknown model")
    phase_dir = Path(phase_dir)
    if digest(read_json(phase_dir / "study_lock.json")) != digest(frozen):
        raise RuntimeError("study_lock differs")
    owned = backend is None
    backend = backend or HFBackend(model_key, config)
    run_dir = phase_dir / "runs" / run_tag
    model_dir = run_dir / model_key
    try:
        common_env = {k: v for k, v in backend.environment.items() if k not in {"chat_template_hash", "generation_config"}}
        lock_json(run_dir / "environment_lock.json", common_env)
        lock_json(model_dir / "run_lock.json", {"study_hash": frozen["study_hash"], "model": config["models"][model_key],
                                                "environment": backend.environment})
        first_jobs = baseline_jobs(items)
        backend.preflight(first_jobs)
        base, complete = execute_batches(first_jobs, model_dir / "baseline", backend, config, stop_after_batches)
        if not complete:
            return {"complete": False, "stage": "baseline", "path": str(model_dir)}
        base_by_id = {r["question_id"]: r for r in base}
        jobs = followup_jobs(items, base_by_id, config)
        audit = audit_jobs(items, jobs, config)
        audit["token_check"] = backend.preflight(jobs)
        atomic_json(model_dir / "design_audit.json", audit)
        follow, complete = execute_batches(jobs, model_dir / "followup", backend, config, stop_after_batches)
        status = {"complete": complete, "baseline_count": len(base), "followup_count": len(follow),
                  "expected_followup_count": len(jobs), "study_hash": frozen["study_hash"], "model_key": model_key}
        atomic_json(model_dir / "status.json", status)
        return status
    finally:
        if owned:
            backend.close()




def read_records(model_dir, stage):
    result = []
    for path in sorted((Path(model_dir) / stage).glob("batch_*.json")):
        batch = read_json(path)
        if batch["records_hash"] != digest(batch["records"]):
            raise ValueError(f"Corrupt checkpoint: {path}")
        result.extend(batch["records"])
    return result
