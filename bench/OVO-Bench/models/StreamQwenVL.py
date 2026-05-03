import importlib.util
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
ovobench_root = os.path.abspath(os.path.join(current_dir, ".."))
if ovobench_root not in sys.path:
    sys.path.insert(0, ovobench_root)

from utils.OVOBench import OVOBenchOffline


def _load_stream_infer_module():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
    sglang_python_path = os.path.join(repo_root, "sglang", "python")
    stream_infer_path = os.path.join(repo_root, "sglang", "python", "stream_infer.py")

    if not os.path.exists(stream_infer_path):
        raise FileNotFoundError(f"Cannot find stream_infer.py at: {stream_infer_path}")

    if os.path.isdir(sglang_python_path) and sglang_python_path not in sys.path:
        sys.path.insert(0, sglang_python_path)

    spec = importlib.util.spec_from_file_location("stream_infer_module", stream_infer_path)
    stream_infer_module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError("Failed to load stream_infer.py module loader")
    spec.loader.exec_module(stream_infer_module)
    return stream_infer_module


class EvalStreamQwenVL(OVOBenchOffline):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.stream_infer = _load_stream_infer_module()
        self._model_init()

    def _resolve_default_model_path(self):
        if self.args.model == "stream_qwen2_5vl_32b":
            return "Qwen/Qwen2.5-VL-32B-Instruct"

        if self.args.model == "stream_qwen2_5vl":
            return "Qwen/Qwen2.5-VL-7B-Instruct"

        if self.args.model == "stream_qwen3vl":
            return "Qwen/Qwen3-VL-8B-Instruct"

        if self.args.model == "stream_llava7b":
            return "lmms-lab/llava-onevision-qwen2-7b-ov"

        if self.args.model == "stream_llava0.5b":
            return "lmms-lab/llava-onevision-qwen2-0.5b-ov"

        if self.args.model == "StreamModel" and self.args.model_path is not None:
            print(self.args.model_path)
            return self.args.model_path

        return "Qwen/Qwen2.5-VL-7B-Instruct"

    def _model_init(self):
        if self.args.model_path is None:
            self.args.model_path = self._resolve_default_model_path()

        self.stream_infer.Config.model_path = self.args.model_path
        self.stream_infer.Config.tp_size = self.args.tp_size
        self.stream_infer.StreamEngineConfig.enable_motion_frame_select = getattr(
            self.args, "enable_motion", True
        )
        self.stream_infer.StreamEngineConfig.enable_embedding_cache = getattr(
            self.args, "enable_cache", True
        )
        self.stream_infer.StreamEngineConfig.frame_select_strategy = getattr(
            self.args, "frame_select_strategy", "low_motion"
        )
        self.stream_infer.StreamEngineConfig.token_reuse_strategy = (
            getattr(self.args, "cache_strategy", None) or "cold"
        )
        self.stream_infer.StreamEngineConfig.enable_frame_resize = getattr(
            self.args, "resize", True
        )
        self.stream_infer.StreamEngineConfig.use_abs_mem = getattr(
            self.args, "use_abs_mem", True
        )
        if getattr(self.args, "cluster_frame_topk", None) is not None:
            self.stream_infer.StreamEngineConfig.motion_select_topk = max(
                2, int(self.args.cluster_frame_topk)
            )
        if hasattr(self.args, "cold_token_ratio"):
            self.stream_infer.StreamEngineConfig.cold_token_ratio = float(
                self.args.cold_token_ratio
            )
        self.stream_infer.get_engine(
            model_path=self.args.model_path,
            tp_size=self.args.tp_size,
        )

    def inference(self, video_file_name, prompt):
        response, fwd_encoded_frames, end_to_end_latency, total_encoded_frames = self.stream_infer.run_infer(
            video_file_name=video_file_name,
            prompt=prompt,
            sample_fps=self.args.sample_fps,
            temperature=self.args.temperature,
            max_tokens=self.args.max_tokens,
            mode=self.args.mode,
            resize=getattr(self.args, "resize", True),
        )
        return response, fwd_encoded_frames, end_to_end_latency, total_encoded_frames

    def __del__(self):
        try:
            self.stream_infer.shutdown_engine()
        except Exception:
            pass
