"""Fixed-batch Drive checkpoints and progress based on verified saved responses."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread
import time
import zipfile

from engine import HFBackend, runtime_environment
from progress_ui import activity
from protocol import atomic_json, digest, int_seed, lock_json, read_json
from ablation_protocol import MODEL, NEW_DEFENSES, verify_frozen


def timestamp():
    return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds")


def validate_source_environment(current, source):
    expected = source["run"]["environment"]
    differences = {}
    for key in ["gpu", "compute_capability", "cuda"]:
        if current.get(key) != expected[key]:
            differences[key] = {"source": expected[key], "current": current.get(key)}
    for key in ["torch", "transformers", "accelerate", "bitsandbytes", "huggingface-hub", "tokenizers"]:
        actual = current.get("packages", {}).get(key)
        if actual != expected["packages"][key]:
            differences[key] = {"source": expected["packages"][key], "current": actual}
    if differences:
        raise RuntimeError(f"A100/핵심 생성 환경이 원본과 다릅니다. 자동 변경하지 않습니다. 이 출력을 공유하세요: {differences}")
    return {key: {"source": expected[key], "current": current.get(key)}
            for key in ["python", "driver", "gpu_memory_bytes"] if expected[key] != current.get(key)}


def generation_environment(env):
    return {"gpu": env["gpu"], "compute_capability": env["compute_capability"], "cuda": env["cuda"],
            "packages": env["packages"], "chat_template_hash": env["chat_template_hash"],
            "generation_config": env["generation_config"], "synthetic_test": env.get("synthetic_test", False)}


def bind_environment(root, frozen, source, backend):
    """Permit recorded driver/Python patch differences, never generation-environment drift."""
    root = Path(root)
    validate_source_environment(backend.environment, source)
    for key in ["chat_template_hash", "generation_config"]:
        if backend.environment[key] != source["run"]["environment"][key]:
            raise RuntimeError(f"원본과 {key} 불일치")
    lock_json(root / "run_lock.json", {"study_hash": frozen["study_hash"], "source_hash": source["source_hash"],
                                       "generation_environment": generation_environment(backend.environment)})
    if not (root / "environment_lock.json").exists():
        atomic_json(root / "environment_lock.json", backend.environment)
    else:
        if generation_environment(read_json(root / "environment_lock.json")) != generation_environment(backend.environment):
            raise RuntimeError("재개 환경의 핵심 패키지·모델·GPU 설정이 달라졌습니다.")
    name = digest(backend.environment)[:20] + ".json"
    lock_json(root / "session_environments" / name, backend.environment)


@contextmanager
def batch_wait_notice(label):
    """Only print long-wait notices; the saved-progress bar handles normal batches."""
    done, start = Event(), time.monotonic()
    def pulse():
        while not done.wait(30):
            print(f"{label}: {time.monotonic() - start:.0f}초 경과; 생성 완료 신호 대기 중", flush=True)
    thread = Thread(target=pulse, daemon=True)
    thread.start()
    try:
        yield
    finally:
        done.set()
        thread.join(timeout=1)


class AblationBackend(HFBackend):
    def __init__(self, source, cfg):
        current = runtime_environment()
        self.source_environment_notes = validate_source_environment(current, source)
        super().__init__(MODEL, cfg)
        for key in ["chat_template_hash", "generation_config"]:
            if self.environment[key] != source["run"]["environment"][key]:
                self.close()
                raise RuntimeError(f"원본과 {key}가 달라졌습니다. 같은 모델/라이브러리 설정을 사용해야 합니다.")

    def generate(self, jobs, seed):
        if any(j["phase"] != "followup" or j["defense"] not in NEW_DEFENSES for j in jobs):
            raise RuntimeError("최초 답변 및 S0/S1 재생성은 허용하지 않습니다.")
        import torch
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        outputs = super().generate(jobs, seed)
        self.last_batch_resource = {
            "generation_seconds": float(outputs[0]["seconds_per_batch"]),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "scope": "Whole mixed-condition batch; includes resident weights, excludes download and Drive writes."}
        return outputs


def bind_backend(root, frozen, source, backend, jobs):
    root = Path(root)
    bind_environment(root, frozen, source, backend)
    atomic_json(root / "source_environment_comparison.json", {
        "generation_environment_matches_original": True,
        "other_environment_differences": getattr(backend, "source_environment_notes", {}),
        "scope": "Only SC/SI/SR are generated; original S0/S1 are reused. Pinned generation environment is checked against the source."})
    with activity("10,800개 입력 길이 점검 (문제·보기 자동 절단 없음)"):
        audit = backend.preflight(jobs)
    if audit.get("count") != len(jobs):
        raise RuntimeError("입력 점검 건수가 다릅니다.")
    audit["jobs_hash"] = digest(jobs)
    lock_json(root / "token_preflight.json", audit)


def batch_seed(cfg, index, batch):
    return int_seed(cfg["seed"], "ablation_followup", index, [j["task_id"] for j in batch])


def validate_checkpoint(path, batch, cfg, index):
    saved = read_json(path)
    rows = saved.get("records", [])
    if saved.get("batch_hash") != digest(batch) or saved.get("records_hash") != digest(rows) or len(rows) != len(batch):
        raise RuntimeError(f"체크포인트 손상/설정 불일치: {path}. 파일을 임의 삭제하지 마세요.")
    if saved.get("batch_index") != index or saved.get("batch_seed") != batch_seed(cfg, index, batch):
        raise RuntimeError(f"배치 번호/시드 불일치: {path}")
    for row, job in zip(rows, batch):
        if any(row.get(k) != v for k, v in job.items()):
            raise RuntimeError(f"문항/이전 답변/후속 프롬프트 불일치: {path}")
        if row.get("batch_index") != index or row.get("batch_seed") != saved["batch_seed"]:
            raise RuntimeError(f"응답 배치 메타데이터 불일치: {path}")
        if not isinstance(row.get("text"), str) or not isinstance(row.get("truncated"), bool):
            raise RuntimeError(f"응답 필드 손상: {path}")
        if not all(isinstance(row.get(k), int) and row[k] >= 0 for k in ["input_tokens", "output_tokens"]):
            raise RuntimeError(f"토큰 필드 손상: {path}")
    return saved


def scan_saved(root, jobs, cfg):
    folder = Path(root) / "checkpoints"
    folder.mkdir(parents=True, exist_ok=True)
    size = cfg["batch_size"]
    allowed = {f"batch_{i:06d}.json" for i in range((len(jobs) + size - 1) // size)}
    if any(p.name not in allowed for p in folder.glob("batch_*.json")):
        raise RuntimeError("다른 실행의 배치 파일이 섞였습니다.")
    result = {}
    for i, offset in enumerate(range(0, len(jobs), size)):
        path = folder / f"batch_{i:06d}.json"
        if path.exists():
            result[i] = validate_checkpoint(path, jobs[offset:offset + size], cfg, i)
    return result


def write_status(root, saved, jobs, cfg, state):
    n = sum(len(b["records"]) for b in saved.values())
    status = {"state": state, "complete": n == len(jobs), "saved_responses": n,
              "expected_responses": len(jobs), "saved_batches": len(saved),
              "expected_batches": (len(jobs) + cfg["batch_size"] - 1) // cfg["batch_size"],
              "original_baselines_reused": 300, "new_baselines_generated": 0,
              "original_followups_reused": 7200, "new_controls_generated": 0, "new_arms": NEW_DEFENSES,
              "percent": 100 * n / len(jobs), "updated_at_kst": timestamp(),
              "save_path": str(Path(root).resolve())}
    atomic_json(Path(root) / "status.json", status)
    return status


def run_followups(root, frozen, source, cfg, jobs, backend, max_new_batches=None):
    root = Path(root)
    if any(j["defense"] not in NEW_DEFENSES for j in jobs):
        raise RuntimeError("이 노트북은 SC/SI/SR만 생성합니다. 기존 S0/S1을 재생성하지 않습니다.")
    actual = verify_frozen(root, source, cfg, jobs)
    if actual != frozen:
        raise RuntimeError("메모리의 동결 설정이 파일과 다릅니다.")
    bind_environment(root, frozen, source, backend)
    if not (root / "token_preflight.json").exists():
        raise RuntimeError("4번 모델 준비·입력 점검부터 실행하세요.")
    if read_json(root / "token_preflight.json").get("jobs_hash") != digest(jobs):
        raise RuntimeError("입력 길이 점검 작업 목록 불일치")
    if max_new_batches is not None and (not isinstance(max_new_batches, int) or max_new_batches < 1):
        raise ValueError("max_new_batches must be a positive integer or None")
    with activity("Drive의 저장 배치와 해시 확인"):
        saved = scan_saved(root, jobs, cfg)
    status = write_status(root, saved, jobs, cfg, "complete" if len(saved) == 675 else "ready")
    print(f"검증된 저장 응답 {status['saved_responses']:,}/{len(jobs):,} | 최초 답변 재생성 0개", flush=True)
    if status["complete"]:
        return status
    try:
        from tqdm.auto import tqdm
        bar = tqdm(total=len(jobs), initial=status["saved_responses"], desc="Drive 저장 완료", unit="응답")
    except ImportError:
        bar = None
    fresh = 0
    session_start = time.monotonic()
    try:
        for index, offset in enumerate(range(0, len(jobs), cfg["batch_size"])):
            if index in saved:
                continue
            if max_new_batches is not None and fresh >= max_new_batches:
                break
            batch = jobs[offset:offset + cfg["batch_size"]]
            seed = batch_seed(cfg, index, batch)
            with batch_wait_notice(f"배치 {index + 1}/675"):
                outputs = backend.generate(batch, seed)
                if len(outputs) != len(batch):
                    raise RuntimeError("생성된 응답 수가 배치와 다릅니다.")
                rows = [{**job, **out, "batch_index": index, "batch_seed": seed}
                        for job, out in zip(batch, outputs)]
                payload = {"batch_index": index, "batch_seed": seed, "batch_hash": digest(batch),
                           "records_hash": digest(rows), "records": rows,
                           "resource": getattr(backend, "last_batch_resource", {}),
                           "environment_hash": digest(backend.environment),
                           "saved_at_kst": timestamp()}
                path = root / "checkpoints" / f"batch_{index:06d}.json"
                if path.exists():
                    raise RuntimeError("다른 노트북에서 같은 배치를 저장했습니다. 동시 실행을 중단하세요.")
                atomic_json(path, payload)
                saved[index] = validate_checkpoint(path, batch, cfg, index)
            fresh += 1
            status = write_status(root, saved, jobs, cfg, "running")
            remaining = status["expected_batches"] - status["saved_batches"]
            eta_minutes = (time.monotonic() - session_start) / fresh * remaining / 60
            if bar is not None:
                bar.update(len(batch))
                bar.set_postfix(saved=f"{status['saved_batches']}/675", ETA_min=f"{eta_minutes:.1f}")
            if fresh == 1 or status["saved_batches"] % 10 == 0 or status["complete"]:
                print(f"저장 확인 {status['saved_responses']:,}/10,800 ({status['percent']:.1f}%) | "
                      f"배치 {status['saved_batches']}/675 | 남은 실행 약 {eta_minutes:.1f}분", flush=True)
        state = "complete" if len(saved) == 675 else "paused_after_batch_limit"
        return write_status(root, saved, jobs, cfg, state)
    except BaseException:
        # Never mark unsaved outputs as complete; even a partially written final file is audited on resume.
        try:
            write_status(root, saved, jobs, cfg, "interrupted")
        except Exception:
            pass
        print("중단됐습니다. 같은 A100·설정으로 재개하면 검증된 저장 배치를 건너뜁니다. "
              "미저장 배치만 원래 순서·시드로 다시 생성합니다.", flush=True)
        raise
    finally:
        if bar is not None:
            bar.close()


def load_complete(root, frozen, source, cfg, jobs):
    actual = verify_frozen(root, source, cfg, jobs)
    if actual != frozen:
        raise RuntimeError("동결 설정 불일치")
    lock = read_json(Path(root) / "run_lock.json")
    if lock["study_hash"] != frozen["study_hash"] or lock["source_hash"] != source["source_hash"]:
        raise RuntimeError("실행 잠금 불일치")
    env = read_json(Path(root) / "environment_lock.json")
    if lock["generation_environment"] != generation_environment(env):
        raise RuntimeError("저장된 생성 환경 잠금 불일치")
    saved = scan_saved(root, jobs, cfg)
    status = write_status(root, saved, jobs, cfg, "complete" if len(saved) == 675 else "incomplete")
    if not status["complete"]:
        raise RuntimeError(f"아직 {status['saved_responses']:,}/10,800개입니다. 부분 결과로 최종 비교하지 않습니다.")
    rows = [r for i in sorted(saved) for r in saved[i]["records"]]
    return rows, [saved[i].get("resource", {}) for i in sorted(saved)]


def export_results(root):
    root = Path(root)
    archive = root.parent / (root.name + "_results.zip")
    temp = archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(root.rglob("*")):
            if path.is_file() and not path.name.endswith(".tmp") and "__pycache__" not in path.parts:
                z.write(path, Path(root.name) / path.relative_to(root))
    temp.replace(archive)
    print("결과 ZIP:", archive)
    return archive
