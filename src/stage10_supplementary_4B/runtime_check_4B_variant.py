"""Check real imports before downloading weights; no model/data changes."""
import importlib.metadata
import os
import platform
import sys

PINS = {
    "transformers": "4.57.6", "accelerate": "1.12.0",
    "bitsandbytes": "0.48.2", "huggingface-hub": "0.36.0",
    "tokenizers": "0.22.2",
    "numpy": "2.2.6", "pandas": "2.2.3", "pyarrow": "20.0.0",
    "matplotlib": "3.10.3", "scipy": "1.15.3",
}


def check_runtime():
    # This experiment uses PyTorch only. Set before importing transformers.
    os.environ["USE_TF"] = "0"
    os.environ["USE_FLAX"] = "0"
    versions = {name: importlib.metadata.version(name) for name in PINS}
    mismatch = {name: (versions[name], want) for name, want in PINS.items()
                if versions[name] != want}
    if mismatch:
        raise RuntimeError(f"설치 버전 불일치: {mismatch}. 1번 설치 → 세션 재시작 → 2번 순서로 실행하세요.")
    import torch
    import transformers
    from transformers.generation import GenerationMixin
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    import accelerate
    import bitsandbytes
    import numpy
    import pandas
    import pyarrow
    import scipy.stats
    import matplotlib.pyplot
    if transformers.__version__ != PINS["transformers"]:
        raise RuntimeError("메모리와 설치 파일의 transformers 버전이 다릅니다. 세션을 재시작하세요.")
    report = {"python": platform.python_version(), "executable": sys.executable,
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "transformers_file": transformers.__file__, "packages": versions}
    print("환경 점검 통과: GenerationMixin / Qwen3 / 양자화 라이브러리 / 분석 패키지 import OK", flush=True)
    print(f"Python {report['python']} | torch {report['torch']} | transformers {transformers.__version__}", flush=True)
    return report


if __name__ == "__main__":
    import json
    print(json.dumps(check_runtime(), ensure_ascii=False, indent=2))
