"""Fail-safe performance profiling for the MiniMax H3 Serverless worker.

这个模块只观察运行过程，不改变 H3 workflow 或推理参数。
所有 profiling 异常都会被吞掉，避免监控故障影响视频生成。
"""

from __future__ import annotations

import datetime as _datetime
import importlib.metadata
import json
import os
import pathlib
import shutil
import subprocess
import threading
import time
from typing import Any, Optional


PERSISTENT_PROFILE_DIR = pathlib.Path("/runpod-volume/profiling")
FALLBACK_PROFILE_DIR = pathlib.Path("/tmp/h3-profiles")


def _now() -> str:
    """返回带毫秒的墙上时钟时间，便于和 RunPod 日志对齐。"""
    return _datetime.datetime.now().astimezone().isoformat(timespec="milliseconds")


def _safe_float(value: Any) -> Optional[float]:
    try:
        return round(float(value), 6)
    except Exception:
        return None


class H3Profiler:
    """记录 H3 Job 的真实可观察阶段，并尽量采集系统指标。"""

    def __init__(self, job_id: str, params: Optional[dict[str, Any]] = None) -> None:
        self.job_id = str(job_id or "unknown")
        self.params = dict(params or {})
        self.started_at = _now()
        self._perf_start = time.perf_counter()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._monitor: Optional[threading.Thread] = None
        self._samples: list[dict[str, Any]] = []
        self._phases: dict[str, dict[str, Any]] = {}
        self._events: list[dict[str, Any]] = []
        self._report_path: Optional[pathlib.Path] = None
        self._psutil_available = False
        self._nvidia_smi_available = shutil.which("nvidia-smi") is not None
        self._torch_available = False
        self._runtime_info: dict[str, Any] = {}

        # psutil 不是强制依赖；缺失时报告 unavailable，不能让 profiling 破坏推理。
        try:
            import psutil  # type: ignore
            self._psutil = psutil
            self._psutil_available = True
        except Exception:
            self._psutil = None

        # CUDA 与 PyTorch API 也采用可选导入，CPU 或导入失败时继续运行。
        try:
            import torch  # type: ignore
            self._torch = torch
            self._torch_available = True
        except Exception:
            self._torch = None

    def _log(self, message: str) -> None:
        try:
            print(f"[{_now()}] [H3-PROFILE] {message}", flush=True)
        except Exception:
            pass

    def _record(self, message: str, **fields: Any) -> None:
        with self._lock:
            self._events.append({"time": _now(), "message": message, **fields})
        self._log(message)

    def _memory(self) -> dict[str, Any]:
        """使用 PyTorch 官方 API 记录显存；失败时返回 unavailable。"""
        result: dict[str, Any] = {
            "memory_allocated_bytes": None,
            "memory_reserved_bytes": None,
            "free_vram_bytes": None,
            "total_vram_bytes": None,
            "cuda_available": False,
        }
        try:
            if not self._torch_available or not self._torch.cuda.is_available():
                return result
            result["cuda_available"] = True
            result["memory_allocated_bytes"] = int(self._torch.cuda.memory_allocated())
            result["memory_reserved_bytes"] = int(self._torch.cuda.memory_reserved())
            free_bytes, total_bytes = self._torch.cuda.mem_get_info()
            result["free_vram_bytes"] = int(free_bytes)
            result["total_vram_bytes"] = int(total_bytes)
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    def _nvidia_smi(self) -> dict[str, Any]:
        """用 nvidia-smi 获取硬件遥测；不可用时记录 unavailable，不抛错。"""
        result = {
            "available": self._nvidia_smi_available,
            "gpu_utilization_percent": None,
            "memory_utilization_percent": None,
            "power_watts": None,
            "temperature_c": None,
            "vram_used_bytes": None,
        }
        if not self._nvidia_smi_available:
            return result
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,power.draw,temperature.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ]
            line = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=2).splitlines()[0]
            values = [x.strip() for x in line.split(",")]
            if len(values) >= 5:
                result["gpu_utilization_percent"] = _safe_float(values[0])
                result["memory_utilization_percent"] = _safe_float(values[1])
                result["power_watts"] = _safe_float(values[2])
                result["temperature_c"] = _safe_float(values[3])
                # nvidia-smi reports MiB; normalize to bytes without estimating usage.
                result["vram_used_bytes"] = int(float(values[4]) * 1024 * 1024)
        except Exception as exc:
            result["available"] = False
            result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    def _system(self) -> dict[str, Any]:
        if not self._psutil_available:
            return {"psutil": "unavailable"}
        try:
            memory = self._psutil.virtual_memory()
            return {
                "psutil": "available",
                "cpu_percent": _safe_float(self._psutil.cpu_percent(interval=None)),
                "ram_used_bytes": int(memory.used),
                "ram_available_bytes": int(memory.available),
            }
        except Exception as exc:
            return {"psutil": "unavailable", "error": f"{type(exc).__name__}: {exc}"}

    def _runtime(self) -> dict[str, Any]:
        """读取版本和 GPU 信息；这些信息用于解释性能，不参与推理。"""
        info: dict[str, Any] = {
            "gpu": "unavailable",
            "gpu_count": None,
            "cuda": "unavailable",
            "pytorch": "unavailable",
            "comfyui": "unavailable",
            "comfy_kitchen": "unavailable",
            "comfy_aimdo": "unavailable",
        }
        try:
            if self._torch_available:
                info["pytorch"] = str(self._torch.__version__)
                info["cuda"] = str(self._torch.version.cuda or "unavailable")
                if self._torch.cuda.is_available():
                    info["gpu_count"] = int(self._torch.cuda.device_count())
                    info["gpu"] = str(self._torch.cuda.get_device_name(0))
        except Exception:
            pass
        for key, package in (("comfyui", "comfyui"), ("comfy_kitchen", "comfy-kitchen"), ("comfy_aimdo", "comfy-aimdo")):
            try:
                info[key] = importlib.metadata.version(package)
            except Exception:
                pass
        return info

    def sample(self, label: str) -> None:
        """记录一次显存、nvidia-smi 和 CPU/RAM 快照。"""
        try:
            sample = {"time": _now(), "label": label, "perf_seconds": time.perf_counter() - self._perf_start}
            sample.update(self._memory())
            sample.update({"nvidia_smi": self._nvidia_smi(), "system": self._system()})
            with self._lock:
                self._samples.append(sample)
            memory = sample.get("memory_allocated_bytes")
            self._log(f"📊 {label} | memory_allocated={memory if memory is not None else 'unavailable'} bytes")
        except Exception as exc:
            self._log(f"⚠️ 显存/系统采样失败（不影响推理）：{type(exc).__name__}: {exc}")

    def start(self) -> None:
        """开始 Job 计时和周期性采样。"""
        self._record("🚀 Job 开始", job_id=self.job_id)
        self._runtime_info = self._runtime()
        self._record("🖥️ 运行环境", **self._runtime_info)
        self._record("⚙️ H3 配置", model_profile=os.getenv("MODEL_PROFILE", "unavailable"), steps=self.params.get("steps", "unavailable"), resolution=f"{self.params.get('width', 'unavailable')}x{self.params.get('height', 'unavailable')}", fps=self.params.get("fps", "unavailable"), frames=self.params.get("frames", "unavailable"), attention=os.getenv("ENABLE_ATTENTION", "unavailable"))
        self._record("ℹ️ 逐步采样计时", status="step-level timing unavailable；当前 handler 未获得真实 sampler callback")
        self.sample("job_start")
        self._monitor = threading.Thread(target=self._monitor_loop, name="h3-profiler", daemon=True)
        self._monitor.start()

    def _monitor_loop(self) -> None:
        # 每秒采样一次，足以观察长时间推理，又避免高频监控干扰 GPU。
        while not self._stop.wait(1.0):
            self.sample("periodic")

    def begin(self, name: str, message: str) -> None:
        try:
            self._phases[name] = {"start": _now(), "start_perf": time.perf_counter()}
            self._record(message, phase=name)
            self.sample(f"{name}_start")
        except Exception as exc:
            self._log(f"⚠️ 阶段计时失败（不影响推理）：{type(exc).__name__}: {exc}")

    def end(self, name: str, message: str) -> None:
        try:
            phase = self._phases.setdefault(name, {})
            phase["end"] = _now()
            start_perf = phase.get("start_perf")
            if start_perf is not None:
                phase["duration_seconds"] = round(time.perf_counter() - start_perf, 6)
            self._record(message, phase=name, duration_seconds=phase.get("duration_seconds"))
            self.sample(f"{name}_end")
        except Exception as exc:
            self._log(f"⚠️ 阶段结束计时失败（不影响推理）：{type(exc).__name__}: {exc}")

    def note(self, message: str, **fields: Any) -> None:
        self._record(message, **fields)

    def _choose_report_path(self) -> pathlib.Path:
        """优先测试并使用 Network Volume；不可写时才降级到 /tmp。"""
        try:
            PERSISTENT_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
            probe = PERSISTENT_PROFILE_DIR / f".write-test-{os.getpid()}"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return PERSISTENT_PROFILE_DIR / f"h3-performance-{self.job_id}.json"
        except Exception as exc:
            self._log(f"⚠️ Network Volume 不可写，profiling 已降级保存到临时目录 | {type(exc).__name__}: {exc}")
            FALLBACK_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
            return FALLBACK_PROFILE_DIR / f"h3-performance-{self.job_id}.json"

    def finish(self, result: Optional[dict[str, Any]] = None, error: Optional[str] = None) -> None:
        """停止监控、输出中文报告并保存 JSON；保存失败绝不影响结果返回。"""
        try:
            self.sample("job_end")
            self._stop.set()
            if self._monitor is not None:
                self._monitor.join(timeout=0.2)
            total = round(time.perf_counter() - self._perf_start, 6)
            with self._lock:
                phases = json.loads(json.dumps(self._phases, ensure_ascii=False, default=str))
                events = list(self._events)
                samples = list(self._samples)
            # 性能报告只保存结果元数据，不保存 MP4 Base64，避免报告本身膨胀成视频文件。
            result_summary = {
                key: value for key, value in (result or {}).items()
                if key not in {"video", "audio"}
            }
            report = {
                "schema_version": 1,
                "task": {
                    "job_id": self.job_id,
                    "started_at": self.started_at,
                    "finished_at": _now(),
                    "worker_execution_seconds_observed": total,
                    "result": result_summary,
                    "error": error,
                },
                "configuration": {
                    "model_profile": os.getenv("MODEL_PROFILE", "unavailable"),
                    "model_name": "minimax_h3_fl2va_mxfp8.safetensors" if "mxfp8" in os.getenv("MODEL_PROFILE", "").lower() else "unavailable",
                    "model_quantization": "MXFP8 (profile declaration; per-kernel utilization unavailable)",
                    "steps": self.params.get("steps", "unavailable"),
                    "width": self.params.get("width", "unavailable"),
                    "height": self.params.get("height", "unavailable"),
                    "fps": self.params.get("fps", "unavailable"),
                    "frames": self.params.get("frames", "unavailable"),
                    "duration_seconds": self.params.get("duration", "unavailable"),
                    "turbo_lora": self.params.get("lora_strength", "unavailable"),
                    "attention": os.getenv("ENABLE_ATTENTION", "unavailable"),
                },
                "runtime": self._runtime_info,
                "model_loading": {
                    "text_encoder": "timing unavailable from handler; ComfyUI internal loader does not expose callback",
                    "video_vae": "timing unavailable from handler; ComfyUI internal loader does not expose callback",
                    "h3_dit": "timing unavailable from handler; ComfyUI internal loader does not expose callback",
                    "audio_vae": "timing unavailable from handler; ComfyUI internal loader does not expose callback",
                    "staged_mb": "底层未提供",
                    "patches": "底层未提供",
                    "dynamic_vram_loading": "已在 ComfyUI 日志中声明时才可确认；本模块不猜测",
                    "async_weight_offloading": "已在 ComfyUI 日志中声明时才可确认；本模块不猜测",
                    "offload_transfer_statistics": "底层未提供",
                },
                "sampling": {
                    "start": "unavailable",
                    "end": "unavailable",
                    "duration_seconds": "unavailable",
                    "steps": self.params.get("steps", "unavailable"),
                    "step_level_timing": "step-level timing unavailable",
                    "reason": "当前 ComfyUI/H3 sampler 未向 Serverless handler 暴露真实 step callback；禁止用总耗时除以 steps 推算",
                    "comfyui_execution_window": phases.get("comfyui_execution", {}),
                },
                "post_processing": {
                    "video_vae": "unavailable separately",
                    "audio_vae": "unavailable separately",
                    "mp4_encoding": "unavailable separately",
                    "output_file": (result or {}).get("filename"),
                },
                "phases": phases,
                "gpu_samples": samples,
                "events": events,
            }
            path = self._choose_report_path()
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            self._report_path = path
            self._log("✅ 中文性能报告已保存")
            self._log(f"📄 报告路径：{path}")
            self._log(f"⏱️ Worker 观测总耗时：{total:.3f}s")
            self._log("⚠️ step-level timing unavailable；未伪造逐步耗时")
        except Exception as exc:
            self._log(f"⚠️ profiling 保存失败（不影响 H3 结果）：{type(exc).__name__}: {exc}")


def start_profiler(job_id: str, params: Optional[dict[str, Any]] = None) -> H3Profiler:
    """安全创建 profiler；构造失败时返回仍可安全调用的对象。"""
    try:
        return H3Profiler(job_id, params)
    except Exception:
        return H3Profiler.__new__(H3Profiler)  # pragma: no cover
