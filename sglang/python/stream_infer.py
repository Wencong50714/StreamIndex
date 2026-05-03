
import atexit
import argparse
import logging
import math
import time
import warnings
from typing import Optional, Tuple

import cv2
import numpy as np

from sglang.srt.stream_engine.stream_engine import StreamEngine
from sglang.srt.stream_engine.config import Config as StreamEngineConfig
from sglang.srt.stream_engine.types import Frame

# ---------------------------------------------------------------------------
# Suppress multiprocessing resource-tracker spurious warnings / errors.
#
# When sglang's Engine.shutdown() calls kill_process_tree(), child processes
# are terminated before they can unregister shared-memory segments.  The
# resource tracker then complains with:
#   UserWarning: resource_tracker: process died unexpectedly …
#   KeyError: '/mp-…'
#
# We (1) silence the UserWarning and (2) monkey-patch the resource tracker's
# ``main`` function so that UNREGISTER for unknown names uses ``discard``
# instead of ``remove``, preventing the ``KeyError`` traceback entirely.
# This must happen *before* any multiprocessing resource is created so that
# the forked resource-tracker process inherits the patch.
# ---------------------------------------------------------------------------

warnings.filterwarnings(
    "ignore",
    message=r"resource_tracker:.*",
    category=UserWarning,
    module=r"multiprocessing\.resource_tracker",
)

try:
    import multiprocessing.resource_tracker as _rt

    # Patch the ``main`` function used by the resource-tracker process.
    # We replace ``cache[rtype].remove(name)`` with a safe ``discard``.
    import textwrap, types

    _original_main = _rt.main

    def _patched_main(fd):
        """Wrap resource_tracker.main() to tolerate missing entries."""
        import sys
        import traceback as _tb

        # Override warnings.warn inside the tracker process to silence
        # the "process died unexpectedly" message.
        _orig_warn = warnings.warn

        def _quiet_warn(msg, *a, **kw):
            if "resource_tracker" in str(msg):
                return
            _orig_warn(msg, *a, **kw)

        warnings.warn = _quiet_warn

        try:
            _original_main(fd)
        except KeyError:
            # Swallow KeyError from ``cache[rtype].remove(name)``
            pass
        except Exception:
            pass

    _rt.main = _patched_main
except Exception:
    pass  # If patching fails, proceed without it

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Config:
    """Default configuration for stream inference."""
    min_pixels: int = 256 * 28 * 28       # Qwen2-VL default
    max_pixels: int = 1280 * 28 * 28      # Qwen2-VL default
    model_path: str = "Qwen/Qwen2-VL-7B-Instruct"
    tp_size: int = 1


def str2bool(value):
    """Parse common string/int representations into bool."""
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Invalid boolean value: {value}. Use true/false or 1/0."
    )


def parse_cold_token_ratio(value):
    """Parse cold-token ratio, where negative means disabling ratio mode."""
    if isinstance(value, (int, float)):
        ratio = float(value)
    else:
        normalized = str(value).strip().lower()
        if normalized in {"off", "disable", "disabled"}:
            return -1.0
        ratio = float(normalized)
    if ratio < -1.0 or ratio > 1.0:
        raise argparse.ArgumentTypeError(
            f"Invalid ratio: {value}. Expected a value in [-1, 1]."
        )
    return ratio


def parse_mem_fraction_static(value):
    """Parse SGLang static GPU memory fraction / KV cache ratio."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        ratio = float(value)
    else:
        normalized = str(value).strip().lower()
        if normalized in {"auto", "default", "none", "unset"}:
            return None
        ratio = float(normalized)
    if ratio <= 0.0 or ratio >= 1.0:
        raise argparse.ArgumentTypeError(
            f"Invalid mem_fraction_static: {value}. Expected a value in (0, 1)."
        )
    return ratio


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_FACTOR = 28
MAX_RATIO = 200

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = None,
    max_pixels: int = None,
) -> Tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if min_pixels is None:
        min_pixels = Config.min_pixels
    if max_pixels is None:
        max_pixels = Config.max_pixels
        
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def resize_frame(frame: np.ndarray, factor: int = IMAGE_FACTOR) -> np.ndarray:
    """Resize a single frame to dimensions divisible by factor."""
    h, w = frame.shape[:2]
    new_h, new_w = smart_resize(h, w, factor=factor)
    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    return resized_frame


class VideoFrameIterator:
    """
    Iterator for streaming frames from a video file.

    Supports both online (real-time simulated) and offline (batch) modes.
    """

    def __init__(
        self,
        video_path: str,
        sample_fps: int = 1,
        max_frames: Optional[int] = None,
        mode: str = "offline",
        repeat: bool = False,
        clip_start: float = 0.0,
        clip_end: Optional[float] = None,
    ):
        """
        Initialize video frame iterator.

        Args:
            video_path: Path to the video file.
            sample_fps: Target frame rate for sampling.
            max_frames: Optional cap on total frames extracted.
            mode: "online" for real-time simulation, "offline" for batch mode.
            repeat: Whether to loop the video indefinitely (online mode only).
            clip_start: Start reading from just after this time (seconds). 0 means
                from the very beginning of the video.
            clip_end: Stop reading after this time (seconds, inclusive). None means
                read until end of video.
        """
        self.video_path = video_path
        self.sample_fps = sample_fps
        self.max_frames = max_frames
        self.mode = mode
        self.repeat = repeat

        # Video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_interval = max(1, int(round(self.video_fps / sample_fps)))

        # State tracking
        self.frame_idx = 0
        self.extracted_count = 0
        self.start_time = None  # used for online-mode wall-clock pacing

        # Clip window: _end_frame_idx is the last raw frame index to include.
        self._end_frame_idx: Optional[int] = (
            int(clip_end * self.video_fps) if clip_end is not None else None
        )
        # Seek past clip_start so we don't re-submit already-seen frames.
        # We advance one raw frame beyond clip_start to avoid re-yielding a
        # frame that was already submitted at exactly clip_start seconds.
        if clip_start > 0.0:
            seek_idx = int(clip_start * self.video_fps) + 1
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, seek_idx)
            self.frame_idx = seek_idx

    def __iter__(self):
        """Return self as an iterator."""
        return self

    def __next__(self) -> Tuple[int, np.ndarray]:
        while True:
            if self._end_frame_idx is not None and self.frame_idx > self._end_frame_idx:
                raise StopIteration

            if self.max_frames is not None and self.extracted_count >= self.max_frames:
                raise StopIteration

            ret, frame = self.cap.read()
            if not ret:
                if self.handle_eof():
                    continue
                else:
                    raise StopIteration

            # Sample frames at the specified interval
            if self.frame_idx % self.frame_interval == 0:
                if self.mode == "online":
                    if self.start_time is None:
                        self.start_time = time.monotonic()

                    expected_time = self.extracted_count / self.sample_fps
                    elapsed = time.monotonic() - self.start_time

                    wait_time = expected_time - elapsed
                    if wait_time > 0:
                        time.sleep(wait_time)

                frame_id = self.extracted_count
                self.extracted_count += 1
                self.frame_idx += 1
                return frame_id, frame
            else:
                self.frame_idx += 1

    def handle_eof(self) -> bool:
        """Handle end of file, optionally repeating for online mode."""
        if self.repeat and self.mode == "online":
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_idx = 0
            self.start_time = time.monotonic()
            self.extracted_count = 0
            return True
        return False

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

    def reset(self):
        """Reset iterator to beginning of video."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_idx = 0
        self.extracted_count = 0
        self.start_time = None

    @property
    def fps(self) -> float:
        """Return the actual FPS of the video."""
        return self.video_fps

    @property
    def duration(self) -> float:
        """Return the duration of the video in seconds."""
        return self.total_frames / self.video_fps if self.video_fps > 0 else 0

    def __repr__(self) -> str:
        return (
            f"VideoFrameIterator("
            f"path='{self.video_path}', "
            f"mode='{self.mode}', "
            f"sample_fps={self.sample_fps})"
        )
        
# ---------------------------------------------------------------------------
# Global engine singleton
# ---------------------------------------------------------------------------

_engine: Optional[StreamEngine] = None


def get_engine(
    model_path: Optional[str] = None,
    tp_size: Optional[int] = None,
    **kwargs,
) -> StreamEngine:
    """Get or create a global :class:`StreamEngine` instance.

    On the first call the engine is initialised with the given (or default)
    parameters.  Subsequent calls return the cached instance.
    """
    global _engine
    if _engine is None:
        _engine = StreamEngine(
            model_path=model_path or Config.model_path,
            tp_size=tp_size or Config.tp_size,
            **kwargs,
        )
    return _engine


def shutdown_engine() -> None:
    """Shutdown and clear the global :class:`StreamEngine` instance."""
    global _engine
    if _engine is not None:
        try:
            _engine.shutdown()
        except Exception:
            # Suppress errors during shutdown (e.g. already-terminated
            # child processes, resource tracker KeyErrors, etc.)
            pass
        finally:
            _engine = None


atexit.register(shutdown_engine)


def run_infer(
    video_file_name: str,
    prompt: str,
    sample_fps: int = 1,
    temperature: float = 0.01,
    max_tokens: int = 256,
    mode: str = "offline",
    resize: bool = True,
) -> Tuple[str, float, float, float]:
    """
    Perform streaming inference on a video with the given prompt.

    Args:
        video_file_name: Path to the video file
        prompt: The prompt/question to answer
        sample_fps: Frames per second to sample from video
        temperature: Sampling temperature
        max_tokens: Maximum number of tokens to generate
        mode: "offline" or "online" inference mode
        resize: Whether to apply stream-engine frame resize policy
            (max/min pixels in stream_engine config)

    Returns:
        A tuple containing:
            - str: The model's response
            - float: Forward encoded frames for this query
            - float: Total time taken for inference in seconds
            - float: Stream-level total encoded frames

    Raises:
        RuntimeError: If StreamEngine is not initialized
    """

    StreamEngineConfig.enable_frame_resize = bool(resize)

    # Initialize engine
    engine = get_engine()

    # Create a stream for this video
    stream_id = engine.create_stream(
        video_path=video_file_name,
        sample_fps=sample_fps,
    )

    try:
        # Extract frames via the iterator
        frame_iter = VideoFrameIterator(
            video_path=video_file_name,
            sample_fps=sample_fps,
            mode=mode,
        )

        # Iterate and add each sampled frame to the engine
        for frame_id, frame_data in frame_iter:
            # resized = resize_frame(frame_data)
            frame = Frame(frame_id=frame_id, image=frame_data)
            engine.submit_frame(stream_id, frame)

        logger.info(
            "Submitted %d frames from '%s' to stream '%s'",
            frame_iter.extracted_count, video_file_name, stream_id,
        )

        # Submit query and get response, measure the inference time
        start_time = time.time()
        query_result = engine.submit_query(
            stream_id, prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            return_stats=True,
        )
        elapsed = time.time() - start_time

        response = query_result["text"] if isinstance(query_result, dict) else str(query_result)

        # Prefer frame-level stats; fallback to legacy token-level fields.
        fwd_encoded_frames = 0.0
        total_encoded_frames = 0.0
        if isinstance(query_result, dict):
            stats = query_result.get("stats", {})
            fwd_encoded_frames = float(
                stats.get("fwd_encoded_frames", stats.get("encoded_vision_tokens", 0.0))
            )
            total_encoded_frames = float(
                stats.get("total_encoded_frames", stats.get("cache_total_tokens", fwd_encoded_frames))
            )
        else:
            total_encoded_frames = fwd_encoded_frames

        logger.info(
            (
                "Inference done in %.2fs  |  fwd_encoded_frames=%.3f  |  "
                "total_encoded_frames=%.3f  |  response_len=%d"
            ),
            elapsed, fwd_encoded_frames, total_encoded_frames, len(response),
        )

        return response, fwd_encoded_frames, elapsed, total_encoded_frames

    finally:
        # Always clean up the stream
        engine.close_stream(stream_id)


# ---------------------------------------------------------------------------
# Persistent-stream API (one stream, many incremental submits + queries)
# ---------------------------------------------------------------------------

def open_stream(
    video_path: str,
    sample_fps: int = 1,
) -> str:
    """Create a persistent stream for a video and return its stream_id.

    The stream stays alive until :func:`close_stream` is called, so multiple
    :func:`submit_frames_until` / :func:`query_stream` calls can share the
    same ``StreamingMemory`` without rebuilding it each time.
    """
    engine = get_engine()
    stream_id = engine.create_stream(video_path=video_path, sample_fps=sample_fps)
    logger.info("Opened persistent stream '%s' for video: %s", stream_id, video_path)
    return stream_id


def submit_frames_until(
    stream_id: str,
    video_path: str,
    end_time: float,
    start_time: float = 0.0,
    sample_fps: int = 1,
    base_frame_id: int = 0,
    mode: str = "offline",
) -> int:
    """Submit frames from (start_time, end_time] of a video into an existing stream.

    Args:
        stream_id: Target stream identifier returned by :func:`open_stream`.
        video_path: Path to the original (full-length) video file.
        end_time: Submit frames up to and including this timestamp (seconds).
        start_time: Resume from just after this timestamp (seconds). Pass 0
            on the first call to start from the beginning.
        sample_fps: Frame sampling rate (frames per second).
        base_frame_id: Global frame-ID offset so that frame IDs are
            monotonically increasing across successive calls.
        mode: ``"offline"`` or ``"online"``.

    Returns:
        The next ``base_frame_id`` to use in the following call
        (i.e. ``base_frame_id + number_of_frames_submitted``).
    """
    engine = get_engine()
    frame_iter = VideoFrameIterator(
        video_path=video_path,
        sample_fps=sample_fps,
        mode=mode,
        clip_start=start_time,
        clip_end=end_time,
    )
    submitted = 0
    for local_id, frame_data in frame_iter:
        frame = Frame(frame_id=base_frame_id + local_id, image=frame_data)
        engine.submit_frame(stream_id, frame)
        submitted += 1

    logger.info(
        "submit_frames_until: %d frames [%.2fs → %.2fs] → stream '%s'",
        submitted, start_time, end_time, stream_id,
    )
    return base_frame_id + submitted


def query_stream(
    stream_id: str,
    prompt: str,
    temperature: float = 0.01,
    max_tokens: int = 256,
) -> Tuple[str, int, float]:
    """Query an existing stream and return ``(response, vision_tokens, elapsed)``.

    Only the current *prompt* is sent — no chat history is carried over.
    """
    engine = get_engine()
    t0 = time.time()
    query_result = engine.submit_query(
        stream_id, prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        return_stats=True,
    )
    elapsed = time.time() - t0

    response = query_result["text"] if isinstance(query_result, dict) else str(query_result)
    total_vision_tokens = 0
    if isinstance(query_result, dict):
        stats = query_result.get("stats", {})
        total_vision_tokens = int(stats.get("encoded_vision_tokens", 0))

    logger.info(
        "query_stream: stream='%s' elapsed=%.2fs vision_tokens=%d response_len=%d",
        stream_id, elapsed, total_vision_tokens, len(response),
    )
    return response, total_vision_tokens, elapsed


def close_stream(stream_id: str) -> None:
    """Close and release a persistent stream created by :func:`open_stream`."""
    engine = get_engine()
    engine.close_stream(stream_id)
    logger.info("Closed persistent stream '%s'", stream_id)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for stream inference."""
    parser = argparse.ArgumentParser(
        description="Run streaming video inference with SGLang StreamEngine.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        required=True,
        help="Prompt / question to ask about the video.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=Config.model_path,
        help=f"Path or HuggingFace ID of the model (default: {Config.model_path}).",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=Config.tp_size,
        help=f"Tensor-parallel size (default: {Config.tp_size}).",
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=parse_mem_fraction_static,
        default=StreamEngineConfig.mem_fraction_static,
        help=(
            "Fraction of GPU memory reserved by SGLang for static allocations, "
            "including the KV cache budget. Use a value in (0, 1), or "
            "'auto'/'default' to keep SGLang's upstream default."
        ),
    )
    parser.add_argument(
        "--sample-fps",
        type=int,
        default=1,
        help="Frames per second to sample from the video (default: 1).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.01,
        help="Sampling temperature (default: 0.01).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate (default: 256).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["offline", "online"],
        default="offline",
        help="Inference mode: 'offline' (batch) or 'online' (real-time) (default: offline).",
    )
    parser.add_argument(
        "--resize",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help=(
            "Enable stream-engine frame resize policy based on min/max pixels "
            "from stream_engine config (default: true)."
        ),
    )
    parser.add_argument(
        "--enable_motion",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help=(
            "Enable motion-based frame selection "
            "(maps to enable_motion_frame_select, default: true)."
        ),
    )
    parser.add_argument(
        "--enable_cache",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help=(
            "Enable embedding cache "
            "(maps to enable_embedding_cache, default: true)."
        ),
    )
    parser.add_argument(
        "--cluster-frame-topk",
        type=int,
        default=StreamEngineConfig.motion_select_topk,
        help=(
            "Number of frames to keep per retrieved cluster. "
            "Selection fixes the first/last frame and fills the remaining slots "
            "with the highest adjacent-frame motion scores "
            f"(default: {StreamEngineConfig.motion_select_topk})."
        ),
    )
    parser.add_argument(
        "--frame-select-strategy",
        type=str,
        choices=["motion", "uniform", "random", "low_motion", "hybrid"],
        default=StreamEngineConfig.frame_select_strategy,
        help=(
            "Frame selection strategy for retrieved clusters "
            f"(default: {StreamEngineConfig.frame_select_strategy})."
        ),
    )
    parser.add_argument(
        "--cache-strategy",
        type=str,
        choices=["cold", "random", "hot"],
        default=StreamEngineConfig.token_reuse_strategy,
        help=(
            "Token reuse strategy under the reuse-token budget: "
            "cold reuses low-motion tokens, hot reuses high-motion tokens, "
            "and random reuses a seeded random subset "
            f"(default: {StreamEngineConfig.token_reuse_strategy})."
        ),
    )
    parser.add_argument(
        "--cold-token-ratio",
        type=parse_cold_token_ratio,
        default=StreamEngineConfig.cold_token_ratio,
        help=(
            "Ratio of visual tokens to reuse from the representative frame. "
            "Use a value in [0, 1], or a negative value to fall back to threshold-based reuse "
            "for the cold strategy "
            f"(default: {StreamEngineConfig.cold_token_ratio})."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry-point for stream inference."""
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    # Override global config from CLI args
    Config.model_path = args.model_path
    Config.tp_size = args.tp_size
    StreamEngineConfig.mem_fraction_static = args.mem_fraction_static
    StreamEngineConfig.enable_motion_frame_select = args.enable_motion
    StreamEngineConfig.enable_embedding_cache = args.enable_cache
    StreamEngineConfig.enable_frame_resize = args.resize
    StreamEngineConfig.frame_select_strategy = args.frame_select_strategy
    StreamEngineConfig.token_reuse_strategy = args.cache_strategy
    StreamEngineConfig.motion_select_topk = max(2, int(args.cluster_frame_topk))
    StreamEngineConfig.cold_token_ratio = args.cold_token_ratio
    print(
        f"Frame Select Strategy: {args.frame_select_strategy}; "
        f"Cache Strategy: {args.cache_strategy}"
    )

    try:
        response, fwd_encoded_frames, elapsed, total_encoded_frames = run_infer(
            video_file_name=args.video,
            prompt=args.prompt,
            sample_fps=args.sample_fps,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            mode=args.mode,
            resize=args.resize,
        )

        print("\n" + "=" * 60)
        print(f"Response:\n{response}")
        print(f"\nFwd encoded frames: {fwd_encoded_frames:.3f}")
        print(f"Total encoded frames: {total_encoded_frames:.3f}")
        print(f"Inference time: {elapsed:.2f}s")
        print("=" * 60)
    finally:
        shutdown_engine()


if __name__ == "__main__":
    main()
