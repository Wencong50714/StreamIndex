import importlib.util
import os
import sys
import time

from model.modelclass import Model


_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, "..", "..", "..", ".."))


def _load_stream_infer_module():
    sglang_python_path = os.path.join(_REPO_ROOT, "sglang", "python")
    stream_infer_path = os.path.join(sglang_python_path, "stream_infer.py")

    if not os.path.exists(stream_infer_path):
        raise FileNotFoundError(f"Cannot find stream_infer.py at: {stream_infer_path}")

    if sglang_python_path not in sys.path:
        sys.path.insert(0, sglang_python_path)

    spec = importlib.util.spec_from_file_location("stream_infer_module", stream_infer_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load stream_infer.py from {stream_infer_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StreamQwenVL(Model):
    def __init__(
        self,
        model_name="stream_qwen2vl",
        model_path=None,
        tp_size=1,
        sample_fps=1.0,
        temperature=0.01,
        max_tokens=256,
        mode="offline",
        enable_motion=True,
        enable_cache=True,
        resize=True,
        cluster_frame_topk=None,
        frame_select_strategy="low_motion",
        cache_strategy="cold",
        cold_token_ratio=None,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.tp_size = max(1, int(tp_size or 1))
        self.sample_fps = max(0.01, float(sample_fps or 1.0))
        self.temperature = float(temperature or 0.01)
        self.max_tokens = int(max_tokens or 256)
        self.mode = mode or "offline"
        self.enable_motion = bool(enable_motion)
        self.enable_cache = bool(enable_cache)
        self.resize = bool(resize)
        self.cluster_frame_topk = cluster_frame_topk
        self.frame_select_strategy = frame_select_strategy or "low_motion"
        self.cache_strategy = cache_strategy or "cold"
        self.cold_token_ratio = cold_token_ratio
        self.stream_infer = _load_stream_infer_module()
        self._stream_state = {}
        self._model_init()

    def _resolve_default_model_path(self):
        normalized_name = (
            (self.model_name or "")
            .strip()
            .lower()
            .replace("-", "")
            .replace("_", "")
            .replace(".", "")
        )

        if normalized_name in {"streamqwen25vl32b", "qwen25vl32b"}:
            return "Qwen/Qwen2.5-VL-32B-Instruct"
        if normalized_name in {"streamqwen25vl", "qwen25vl"}:
            return "Qwen/Qwen2.5-VL-7B-Instruct"
        if normalized_name in {"streamqwen3vl", "qwen3vl"}:
            return "Qwen/Qwen3-VL-8B-Instruct"
        if normalized_name in {"streamllava7b", "llavaonevisionqwen27bov"}:
            return "lmms-lab/llava-onevision-qwen2-7b-ov"
        if normalized_name in {"streamllava05b", "streamllava05bov", "llavaonevisionqwen205bov"}:
            return "lmms-lab/llava-onevision-qwen2-0.5b-ov"

        if self.model_name == "StreamModel" and self.model_path is not None:
            print(f"Load {self.model_path}")
            return self.model_path

        return "Qwen/Qwen2-VL-7B-Instruct"

    def _model_init(self):
        if self.model_path is None:
            self.model_path = self._resolve_default_model_path()

        self.stream_infer.Config.model_path = self.model_path
        self.stream_infer.Config.tp_size = self.tp_size
        self.stream_infer.StreamEngineConfig.enable_motion_frame_select = self.enable_motion
        self.stream_infer.StreamEngineConfig.enable_embedding_cache = self.enable_cache
        self.stream_infer.StreamEngineConfig.enable_frame_resize = self.resize
        self.stream_infer.StreamEngineConfig.frame_select_strategy = self.frame_select_strategy
        self.stream_infer.StreamEngineConfig.token_reuse_strategy = self.cache_strategy

        if self.cluster_frame_topk is not None:
            self.stream_infer.StreamEngineConfig.motion_select_topk = max(
                2, int(self.cluster_frame_topk)
            )
        if self.cold_token_ratio is not None:
            self.stream_infer.StreamEngineConfig.cold_token_ratio = float(self.cold_token_ratio)

        self.stream_infer.get_engine(
            model_path=self.model_path,
            tp_size=self.tp_size,
        )

    @staticmethod
    def _extract_metric(stats, primary_key, fallback_key, fallback_value):
        if not isinstance(stats, dict):
            return fallback_value
        value = stats.get(primary_key, stats.get(fallback_key, fallback_value))
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback_value

    def _normalize_query_result(self, query_result, fallback_fwd_frames, fallback_total_frames):
        if isinstance(query_result, dict):
            response = query_result.get("text", "")
            stats = query_result.get("stats", {})
        else:
            response = str(query_result)
            stats = {}

        fwd_encoded_frames = self._extract_metric(
            stats,
            "fwd_encoded_frames",
            "encoded_vision_tokens",
            float(fallback_fwd_frames),
        )
        total_encoded_frames = self._extract_metric(
            stats,
            "total_encoded_frames",
            "cache_total_tokens",
            float(fallback_total_frames),
        )
        return response, fwd_encoded_frames, total_encoded_frames

    def Run(self, file, inp):
        return self.stream_infer.run_infer(
            video_file_name=file,
            prompt=inp,
            sample_fps=self.sample_fps,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            mode=self.mode,
            resize=self.resize,
        )

    def open_stream(self, video_path):
        stream_id = self.stream_infer.open_stream(
            video_path=video_path,
            sample_fps=self.sample_fps,
        )
        self._stream_state[stream_id] = {
            "video_path": video_path,
            "last_submitted_frames": 0,
            "total_submitted_frames": 0,
        }
        return stream_id

    def submit_frames_until(self, stream_id, video_path, start_time, end_time, base_frame_id=0):
        next_frame_id = self.stream_infer.submit_frames_until(
            stream_id=stream_id,
            video_path=video_path,
            end_time=end_time,
            start_time=start_time,
            sample_fps=self.sample_fps,
            base_frame_id=base_frame_id,
            mode=self.mode,
        )

        submitted_frames = max(0, int(next_frame_id - base_frame_id))
        state = self._stream_state.setdefault(
            stream_id,
            {
                "video_path": video_path,
                "last_submitted_frames": 0,
                "total_submitted_frames": 0,
            },
        )
        state["video_path"] = video_path
        state["last_submitted_frames"] = submitted_frames
        state["total_submitted_frames"] += submitted_frames
        return next_frame_id

    def query(self, stream_id, prompt):
        engine = self.stream_infer.get_engine()
        state = self._stream_state.get(
            stream_id,
            {"last_submitted_frames": 0, "total_submitted_frames": 0},
        )

        start_time = time.perf_counter()
        query_result = engine.submit_query(
            stream_id,
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            return_stats=True,
        )
        end_to_end_latency = time.perf_counter() - start_time

        response, fwd_encoded_frames, total_encoded_frames = self._normalize_query_result(
            query_result,
            fallback_fwd_frames=state["last_submitted_frames"],
            fallback_total_frames=state["total_submitted_frames"],
        )
        return response, fwd_encoded_frames, end_to_end_latency, total_encoded_frames

    def close_stream(self, stream_id):
        try:
            self.stream_infer.close_stream(stream_id)
        finally:
            self._stream_state.pop(stream_id, None)

    def name(self):
        return self.model_name

    def __del__(self):
        try:
            self.stream_infer.shutdown_engine()
        except Exception:
            pass
