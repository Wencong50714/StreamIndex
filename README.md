<h2><a href=""> StreamIndex: Decoupling Indexing from Encoding for Scalable Streaming Video Understanding</a></h2>

This repository is the official implementation of **StreamIndex**, a streaming video understanding system built on top of SGLang.


## Highlights

- **Streaming memory**: incrementally indexes video frames into stream-local memory instead of repeatedly encoding the full context.
- **Query-time retrieval**: retrieves relevant clusters and recent frames for each incoming question.
- **Motion-aware frame selection**: selects representative frames from retrieved clusters using configurable motion strategies.
- **Embedding and token reuse**: supports cluster-level cache reuse and stream-level CPU cache reuse to reduce repeated visual encoding.
- **Benchmark adapters**: integrates StreamIndex with OVO-Bench and StreamingBench wrappers.
- **Interactive demo**: provides a FastAPI demo for multiple simultaneous video streams on a single GPU.

## Repository Layout

```text
.
|-- sglang/python/stream_infer.py
|   `-- Public CLI and Python wrapper for stream inference.
|-- sglang/python/sglang/srt/stream_engine/
|   |-- stream_engine.py       # Top-level streaming runtime.
|   |-- stream_memory.py       # Streaming memory and retrieval.
|   |-- frame_selector.py      # Motion/uniform/random frame selection.
|   |-- motion_analyzer.py     # CPU motion analysis.
|   |-- embedding_cache.py     # Visual embedding cache.
|   |-- encoding_mgr.py        # Visual encoding orchestration.
|   |-- model_adapter.py       # VLM-specific adapter logic.
|   `-- config.py              # Stream engine defaults.
`-- pyproject.toml
    `-- Python environment and editable package configuration.
```

## Installation

The project is configured with `pyproject.toml`. The versions were derived from the working `.venv` environment used for development.

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .
```

## Model Preparation

StreamIndex currently supports Qwen-VL and LLaVA-OneVision style model adapters through the local SGLang runtime. Common model IDs include:

- `Qwen/Qwen2-VL-7B-Instruct`
- `Qwen/Qwen2.5-VL-7B-Instruct`
- `Qwen/Qwen2.5-VL-32B-Instruct`
- `Qwen/Qwen3-VL-8B-Instruct`
- `lmms-lab/llava-onevision-qwen2-7b-ov`
- `lmms-lab/llava-onevision-qwen2-0.5b-ov`

Models are downloaded by the underlying Hugging Face / ModelScope tooling when first used, unless already cached locally.

## Quick Start

Run local inference on a single video:

```bash
python sglang/python/stream_infer.py \
  --video test_data/real12_chunk3.mp4 \
  --prompt "What color shirt is the man wearing when he holds the white plate with a piece of raw steak? Options: A. Dark gray. B. Green. C. Blue. D. White. Output A, B, C, or D." \
  --model-path Qwen/Qwen2.5-VL-7B-Instruct \
  --tp-size 1 \
  --sample-fps 1 \
  --mode offline \
  --enable_motion true \
  --enable_cache true \
  --temperature 0.01 \
  --max-tokens 256
```

The command prints the model response, forward encoded frames, total encoded frames, and end-to-end inference time.

You can also use the installed console script:

```bash
stream-infer \
  --video test_data/real12_chunk3.mp4 \
  --prompt "Describe the key event in this video." \
  --model-path Qwen/Qwen2.5-VL-7B-Instruct
```

## Core Configuration

Important runtime switches:

| Argument | Default | Description |
| --- | --- | --- |
| `--sample-fps` | `1` | Frame sampling rate for the video stream. |
| `--mode` | `offline` | `offline` batch-style streaming or `online` real-time simulation. |
| `--enable_motion` | `true` | Enables motion-aware cluster frame selection. |
| `--enable_cache` | `true` | Enables embedding / token reuse cache. |
| `--frame-select-strategy` | `low_motion` | One of `motion`, `uniform`, `random`, `low_motion`, `hybrid`. |
| `--cluster-frame-topk` | `4` | Number of frames retained from each retrieved cluster. |
| `--cache-strategy` | `cold` | Token reuse strategy: `cold`, `random`, or `hot`. |
| `--cold-token-ratio` | `0.8` | Ratio of representative-frame visual tokens to reuse. |
| `--resize` | `true` | Applies the stream engine frame resize policy. |
| `--mem-fraction-static` | `0.85` | Static GPU memory fraction for SGLang allocations. |

Defaults live in:

```text
sglang/python/sglang/srt/stream_engine/config.py
```

## Evaluation

1. Clone [OVO-Bench](https://github.com/joeleelyf/ovo-bench) and [StreamingBench](https://github.com/THUNLP-MT/StreamingBench) to bench directory.
2. Add this repository's `bench/OVO-Bench/models/StreamQwenVL.py` and `bench/StreamingBench/src/model/StreamQwenVL.py` to benchmark model directories. 

Layout like
```
bench
|-- OVO-Bench
|   |-- ...
|-- StreamingBench
|   |-- ...
```


### OVO-Bench

Prepare the OVO-Bench data under `bench/OVO-Bench/data/`, then run:

```bash
cd bench/OVO-Bench

CUDA_VISIBLE_DEVICES=0 python inference.py \
  --anno_path data/ovo_bench_new.json \
  --video_dir data/ \
  --chunked_dir data/chunked_videos \
  --result_dir results_streamindex \
  --mode offline \
  --task all \
  --model stream_qwen2vl \
  --model_path Qwen/Qwen2-VL-7B-Instruct \
  --tp-size 1 \
  --sample-fps 1 \
  --enable_motion true \
  --enable_cache true \
  --temperature 0.01 \
  --max-tokens 256

python score.py \
  --result_dir results_streamindex \
  --model stream_qwen2vl \
  --mode offline
```

Supported StreamIndex model names include `stream_qwen2vl`, `stream_qwen2_5vl`, `stream_qwen2_5vl_32b`, `stream_qwen3vl`, `stream_llava7b`, and `stream_llava0.5b`.

### StreamingBench

Prepare the StreamingBench data following `bench/StreamingBench/README.md`, then run:

```bash
cd bench/StreamingBench/src

MODEL_NAME="qwen2vl"

CUDA_VISIBLE_DEVICES=0 python eval.py \
  --model_name ${MODEL_NAME} \
  --benchmark_name Streaming \
  --data_file ./data/questions_real.json \
  --output_file ./data/real_output_${MODEL_NAME}.json \
  --context_time -1 \
  --tp_size 1 \
  --sample_fps 1 \
  --enable_motion true \
  --enable_cache true \
  --temperature 0.01 \
  --max_tokens 256 \
  --mode offline

cd data
python count.py \
  --model ${MODEL_NAME} \
  --task real \
  --src ./real_output_${MODEL_NAME}.json
```
