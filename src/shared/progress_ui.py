"""Read-only progress UI around the unchanged engine and backend calls.

Counts after generate are generated responses, NOT guaranteed disk writes.
Saved totals are checked again when run_model returns. UI never seeds RNGs.
"""
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread
import time

from engine import HFBackend, read_records, run_model


@contextmanager
def activity(label, interval=30):
    start = time.monotonic()
    done = Event()
    print(f"▶ {label}", flush=True)

    def pulse():
        while not done.wait(interval):
            print(f"  … {label} 대기 중 | {time.monotonic() - start:.0f}초 경과 "
                  "(완료 신호를 기다리는 중이며 정상 작동 보장은 아닙니다)", flush=True)

    thread = Thread(target=pulse, daemon=True)
    thread.start()
    try:
        yield
    except BaseException:
        print(f"✗ {label} 중단 | {time.monotonic() - start:.1f}초", flush=True)
        raise
    else:
        print(f"✓ {label} 완료 | {time.monotonic() - start:.1f}초", flush=True)
    finally:
        done.set()
        thread.join(timeout=1)


class ProgressBackend:
    def __init__(self, backend, model_dir, totals):
        self.backend = backend
        self.environment = backend.environment
        self.model_dir = Path(model_dir)
        self.totals = totals
        self.counts = {stage: 0 for stage in totals}
        self.stage = None
        self.fresh_count = 0
        self.generation_seconds = 0.0

    def preflight(self, jobs):
        stage = jobs[0]["phase"] if jobs else "followup"
        self.stage = stage
        self.fresh_count = 0
        self.generation_seconds = 0.0
        with activity(f"{stage}: 입력 길이 점검 {len(jobs):,}개"):
            result = self.backend.preflight(jobs)
        # Count records with valid hashes. Engine also verifies schedule and jobs.
        with activity(f"{stage}: 기존 저장 응답 확인"):
            cached = read_records(self.model_dir, stage)
            self.counts[stage] = len(cached)
        print(f"{stage}: 기존 저장 {len(cached):,}/{len(jobs):,}개 "
              "(재사용 가능 여부는 엔진이 추가 검사)", flush=True)
        return result

    def generate(self, jobs, seed):
        stage = self.stage
        completed = self.counts[stage]
        with activity(f"{stage}: 새 배치 {len(jobs)}응답 생성 "
                      f"[현재 {completed:,}/{self.totals[stage]:,}]"):
            start = time.monotonic()
            outputs = self.backend.generate(jobs, seed)
            elapsed = time.monotonic() - start
        if len(outputs) != len(jobs):
            raise RuntimeError("backend returned wrong batch size")
        self.counts[stage] += len(outputs)
        self.fresh_count += len(outputs)
        self.generation_seconds += elapsed
        count, total = self.counts[stage], self.totals[stage]
        remaining = max(0, total - count)
        eta = self.generation_seconds / self.fresh_count * remaining
        overall, expected = sum(self.counts.values()), sum(self.totals.values())
        print(f"{stage}: 생성·기존 저장 {count:,}/{total:,} ({count / total:.1%}) | "
              f"모델 전체 {overall:,}/{expected:,} | 이 단계 남은 생성 시간 약 {eta / 60:.1f}분\n"
              "  이번 배치는 생성 완료; 파일 저장은 이어서 엔진이 수행합니다. "
              "예상 시간에는 다운로드·입력 점검·Drive 저장 시간이 제외됩니다.", flush=True)
        return outputs


def run_with_progress(phase_dir, frozen, model_key, run_tag="primary",
                      stop_after_batches=None, backend=None):
    config = frozen["config"]
    n = len(frozen["items"])
    totals = {"baseline": n, "followup": n * len(config["conditions"]) *
              len(config["defenses"]) * len(config["variants"])}
    model_dir = Path(phase_dir) / "runs" / run_tag / model_key
    print(f"실행: {model_key} | 첫 답변 {totals['baseline']:,} + "
          f"후속 {totals['followup']:,} = {sum(totals.values()):,}응답\n저장 위치: {model_dir}", flush=True)
    owned = backend is None
    try:
        if owned:
            with activity("모델 준비: 라이브러리 로드 / 가중치 다운로드 / GPU 적재"):
                backend = HFBackend(model_key, config)
        wrapped = ProgressBackend(backend, model_dir, totals)
        status = run_model(phase_dir, frozen, model_key, run_tag,
                           backend=wrapped, stop_after_batches=stop_after_batches)
        saved = {stage: len(read_records(model_dir, stage)) for stage in totals}
        print(f"{'완료' if status['complete'] else '일시 중단'}: "
              f"실제 저장 {sum(saved.values()):,}/{sum(totals.values()):,}응답 | "
              f"baseline={saved['baseline']}, followup={saved['followup']}", flush=True)
        return status
    except BaseException:
        print("실행이 중단됐습니다. 완료로 처리하지 않습니다. "
              "이미 저장된 배치는 같은 환경·설정으로 6번을 재실행하면 재사용합니다.", flush=True)
        raise
    finally:
        if owned and backend is not None:
            backend.close()
