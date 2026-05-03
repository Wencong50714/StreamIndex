import asyncio
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers.models.auto.processing_auto import (
    PROCESSOR_MAPPING_NAMES as HF_MAPPING_NAMES,
)

import sglang.srt.managers.multimodal_processor as sgl_mm_processor_utils
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputFormat,
)
from sglang.srt.models.llava import (
    LlavaForConditionalGeneration,
    LlavaLlamaForCausalLM,
    LlavaMistralForCausalLM,
    LlavaQwenForCausalLM,
)
from sglang.srt.models.llavavid import LlavaVidForCausalLM
from sglang.srt.models.mistral import Mistral3ForConditionalGeneration
from sglang.srt.multimodal.mm_utils import expand2square, process_anyres_image
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.utils import ImageData, load_image, logger
from sglang.utils import get_exception_traceback


class LlavaImageProcessor(BaseMultimodalProcessor):
    models = [
        LlavaLlamaForCausalLM,
        LlavaVidForCausalLM,
        LlavaQwenForCausalLM,
        LlavaMistralForCausalLM,
    ]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.IM_TOKEN_ID = getattr(hf_config, "image_token_index", 151646)
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=getattr(_processor, "image_token", "<image>"),
            image_token_id=self.IM_TOKEN_ID,
        ).build(_processor)

    @staticmethod
    def _process_single_image_task(
        image_data: Union[str, bytes, ImageData],
        image_aspect_ratio: Optional[str] = None,
        image_grid_pinpoints: Optional[str] = None,
        processor=None,
    ):
        image_processor = (
            processor.image_processor if hasattr(processor, "image_processor") else processor
        )

        try:
            url = image_data.url if isinstance(image_data, ImageData) else image_data
            image, image_size = load_image(url)
            if image_size is not None:
                # It is a video with multiple images
                image_hash = hash(url)
                pixel_values = image_processor(image)["pixel_values"]
                for _ in range(len(pixel_values)):
                    pixel_values[_] = pixel_values[_].astype(np.float16)
                pixel_values = np.stack(pixel_values, axis=0)
                return pixel_values, image_hash, image_size
            else:
                # It is an image
                image_hash = hash(url)
                if image_aspect_ratio == "pad":
                    image = expand2square(
                        image,
                        tuple(int(x * 255) for x in image_processor.image_mean),
                    )
                    pixel_values = image_processor(image.convert("RGB"))[
                        "pixel_values"
                    ][0]
                elif image_aspect_ratio == "anyres" or (
                    image_aspect_ratio is not None
                    and "anyres_max" in image_aspect_ratio
                ):
                    pixel_values = process_anyres_image(
                        image, image_processor, image_grid_pinpoints
                    )
                else:
                    pixel_values = image_processor(image)["pixel_values"][0]

                if isinstance(pixel_values, np.ndarray):
                    pixel_values = pixel_values.astype(np.float16)

                return pixel_values, image_hash, image.size
        except Exception:
            logger.error("Exception in TokenizerManager:\n" + get_exception_traceback())

    async def _process_single_image(
        self,
        image_data: Union[bytes, str, ImageData],
        aspect_ratio: str,
        grid_pinpoints: str,
    ):
        if self.cpu_executor is not None:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self.cpu_executor,
                LlavaImageProcessor._process_single_image_task,
                image_data,
                aspect_ratio,
                grid_pinpoints,
                self._processor,
            )
        else:
            return self._process_single_image_task(
                image_data,
                aspect_ratio,
                grid_pinpoints,
                self._processor.image_processor,
            )

    def _normalize_precomputed_image(
        self, item: Dict
    ) -> Tuple[torch.Tensor, Optional[Tuple[int, int]]]:
        feature = item.get("feature")
        if feature is None:
            raise ValueError(
                "`precomputed_embedding` input requires a `feature` field."
            )
        if not isinstance(feature, (torch.Tensor, np.ndarray)):
            raise TypeError(
                "`precomputed_embedding.feature` must be a torch.Tensor or np.ndarray, "
                f"got {type(feature)}"
            )

        feature_tensor = torch.as_tensor(feature)
        if feature_tensor.ndim < 2:
            raise ValueError(
                "`precomputed_embedding.feature` must be at least 2D, "
                f"got shape={tuple(feature_tensor.shape)}"
            )
        feature_tensor = feature_tensor.reshape(-1, feature_tensor.shape[-1])

        image_sizes = item.get("image_sizes")
        normalized_size: Optional[Tuple[int, int]] = None
        if image_sizes is not None:
            flattened = image_sizes
            if isinstance(flattened, torch.Tensor):
                flattened = flattened.tolist()
            if isinstance(flattened, np.ndarray):
                flattened = flattened.tolist()
            if isinstance(flattened, list) and flattened:
                candidate = flattened[0]
                if isinstance(candidate, torch.Tensor):
                    candidate = candidate.tolist()
                if isinstance(candidate, np.ndarray):
                    candidate = candidate.tolist()
                if isinstance(candidate, (list, tuple)) and len(candidate) == 2:
                    normalized_size = (int(candidate[0]), int(candidate[1]))

        return feature_tensor, normalized_size

    def _process_precomputed_image_data(
        self,
        image_data: List[Dict],
        input_text,
    ) -> Dict:
        if isinstance(input_text, str):
            input_ids = self._processor.tokenizer(
                input_text, return_tensors="pt", add_special_tokens=True
            ).input_ids.flatten().tolist()
        elif isinstance(input_text, list):
            input_ids = list(input_text)
        else:
            raise ValueError(
                "LLaVA precomputed image inputs require string prompt text or input_ids."
            )

        image_positions = [
            idx for idx, token_id in enumerate(input_ids) if token_id == self.IM_TOKEN_ID
        ]

        concatenated_features: List[torch.Tensor] = []
        expanded_offsets: List[Tuple[int, int]] = []
        image_feature_lens: List[int] = []
        image_sizes: List[Tuple[int, int]] = []
        running_shift = 0
        logical_image_specs: List[Tuple[torch.Tensor, Optional[Tuple[int, int]], List[int]]] = []

        for item in image_data:
            feature_tensor, normalized_size = self._normalize_precomputed_image(item)
            raw_feature_lens = item.get("image_feature_lens")
            if raw_feature_lens is None:
                per_image_feature_lens = [int(feature_tensor.shape[0])]
            else:
                if isinstance(raw_feature_lens, torch.Tensor):
                    raw_feature_lens = raw_feature_lens.tolist()
                if isinstance(raw_feature_lens, np.ndarray):
                    raw_feature_lens = raw_feature_lens.tolist()
                per_image_feature_lens = [int(length) for length in raw_feature_lens]
                if not per_image_feature_lens or any(length <= 0 for length in per_image_feature_lens):
                    raise ValueError(
                        "`image_feature_lens` must contain one or more positive integers."
                    )
                if sum(per_image_feature_lens) != int(feature_tensor.shape[0]):
                    raise ValueError(
                        "Packed precomputed embeddings must match the concatenated "
                        f"feature length. feature_len={int(feature_tensor.shape[0])} "
                        f"image_feature_lens={per_image_feature_lens}"
                    )

            logical_image_specs.append(
                (feature_tensor, normalized_size, per_image_feature_lens)
            )

        logical_image_count = sum(len(spec[2]) for spec in logical_image_specs)
        if len(image_positions) != logical_image_count:
            raise ValueError(
                "The number of `<image>` placeholders must match the number of logical "
                "precomputed images. "
                f"placeholders={len(image_positions)} image_data={len(image_data)} "
                f"logical_images={logical_image_count}"
            )

        image_position_iter = iter(image_positions)
        for feature_tensor, normalized_size, per_image_feature_lens in logical_image_specs:
            concatenated_features.append(feature_tensor)
            for feature_len in per_image_feature_lens:
                position = next(image_position_iter)
                start = int(position + running_shift)
                end = int(start + feature_len - 1)

                expanded_offsets.append((start, end))
                image_feature_lens.append(feature_len)
                if normalized_size is not None:
                    image_sizes.append(normalized_size)

                running_shift += feature_len - 1

        mm_item = MultimodalDataItem(
            modality=Modality.IMAGE,
            offsets=expanded_offsets,
            precomputed_embeddings=torch.cat(concatenated_features, dim=0),
            format=MultimodalInputFormat.PRECOMPUTED_EMBEDDING,
            model_specific_data={
                "image_feature_lens": image_feature_lens,
                **({"image_sizes": image_sizes} if image_sizes else {}),
            },
        )
        return {
            "mm_items": [mm_item],
            "im_token_id": self.IM_TOKEN_ID,
        }

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes, ImageData]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        # FIX: Handle precomputed embeddings (dictionaries)
        # If the input is already a dictionary, we skip the CPU image processor.
        # We also need to infer 'image_sizes' from 'pixel_values' if missing,
        # because pad_input_ids requires it.
        if (
            isinstance(image_data, list)
            and len(image_data) > 0
            and isinstance(image_data[0], dict)
        ):
            return self._process_precomputed_image_data(image_data, input_text)

        modalities = request_obj.modalities or ["image"]
        aspect_ratio = getattr(self.hf_config, "image_aspect_ratio", None)
        grid_pinpoints = (
            self.hf_config.image_grid_pinpoints
            if hasattr(self.hf_config, "image_grid_pinpoints")
            and "anyres" in aspect_ratio
            else None
        )

        if isinstance(image_data, list) and len(image_data) > 0:
            if "multi-images" in modalities or "video" in modalities:
                # Multiple images
                aspect_ratio = "pad"  # LLaVA OneVision Handling: more than one image --> interleaved image mode or video mode. We do not use anyres
                pixel_values, data_hashes, image_sizes = [], [], []
                res = []
                for img_data in image_data:
                    res.append(
                        self._process_single_image(
                            img_data, aspect_ratio, grid_pinpoints
                        )
                    )

                res = await asyncio.gather(*res)
                for pixel_v, image_h, image_s in res:
                    pixel_values.append(pixel_v)
                    data_hashes.append(image_h)
                    image_sizes.append(image_s)

                if isinstance(pixel_values[0], np.ndarray):
                    pixel_values = np.stack(pixel_values, axis=0)
            else:
                # A single image
                pixel_values, image_hash, image_size = await self._process_single_image(
                    image_data[0], aspect_ratio, grid_pinpoints
                )
                image_sizes = [image_size]
        else:
            raise ValueError(f"Invalid image data: {image_data}")
        modality = Modality.IMAGE
        if isinstance(request_obj.modalities, list):
            if request_obj.modalities[0] == "multi-images":
                modality = Modality.MULTI_IMAGES
            elif request_obj.modalities[0] == "video":
                modality = Modality.VIDEO

        return {
            "mm_items": [
                MultimodalDataItem(
                    feature=pixel_values,
                    model_specific_data={
                        "image_sizes": image_sizes,
                    },
                    modality=modality,
                )
            ],
            "im_token_id": self.IM_TOKEN_ID,
        }


class LlavaMultimodalProcessor(BaseMultimodalProcessor):
    """
    This is a wrapper class used to identify the multimodal processor for Llava architectures' vision model.
    """

    models = [LlavaForConditionalGeneration, Mistral3ForConditionalGeneration]

    def _get_sgl_processor_cls(self, model_type: str):
        if model_type == "clip_vision_model":
            return LlavaImageProcessor
        if hf_name := HF_MAPPING_NAMES.get(model_type):
            sgl_mm_processor_set = sgl_mm_processor_utils.PROCESSOR_MAPPING.values()
            sgl_processor_cls = list(
                filter(lambda p: p.__name__ == hf_name, sgl_mm_processor_set)
            )
            if sgl_processor_cls:
                return sgl_processor_cls[0]
        raise ValueError(
            f"Cannot find corresponding multimodal processor registered in sglang for model type `{model_type}`"
        )

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        assert hasattr(hf_config, "vision_config")
        assert hasattr(hf_config, "text_config")
        self.vision_config = hf_config.vision_config
        self.text_config = hf_config.text_config
        self.hf_config = hf_config

        if vision_type := getattr(self.vision_config, "model_type"):
            self.inner = self._get_sgl_processor_cls(vision_type)(
                hf_config, server_args, _processor, *args, **kwargs
            )
        else:
            raise ValueError(
                f"Required `vision_config.model_type` is not found in hf_config: `{hf_config}`"
            )

    async def process_mm_data_async(self, *args, **kwargs):
        return await self.inner.process_mm_data_async(*args, **kwargs)
