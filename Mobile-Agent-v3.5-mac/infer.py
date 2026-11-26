"""Serve local GUI-Owl inference over stdin/stdout for the OSWorld runner."""

import json
import os
import sys
import time
import traceback
from contextlib import redirect_stdout
from functools import partial
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent


def screenshot_placeholder_mask(model, original, input_ids, inputs_embeds, image_features=None, video_features=None):
    if input_ids is None or image_features is None or video_features is not None:
        return original(input_ids, inputs_embeds, image_features, video_features)

    # In Transformers 4.57, the original check indexes embeddings with an
    # expanded boolean mask. MPS can return an incorrect element count.
    image_mask = input_ids == model.config.image_token_id
    video_mask = input_ids == model.config.video_token_id
    placeholder_count = int(image_mask.sum().item())
    if placeholder_count * inputs_embeds.shape[-1] != image_features.numel():
        raise ValueError(
            f"Image features and image tokens do not match: tokens: {placeholder_count}, "
            f"features: {image_features.shape[0]}"
        )
    return tuple(mask.unsqueeze(-1).expand_as(inputs_embeds) for mask in (image_mask, video_mask))


class LocalModel:
    def __init__(self, device, max_tokens, weights=ROOT / "weights"):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable in this process.")
        self.processor = AutoProcessor.from_pretrained(weights, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            weights, local_files_only=True, device_map={"": device},
            dtype=torch.float16 if device == "mps" else "auto",
            attn_implementation="eager" if device == "mps" else "sdpa",
        ).eval()
        if device == "mps":
            inner_model = self.model.model
            inner_model.get_placeholder_mask = partial(
                screenshot_placeholder_mask, inner_model, inner_model.get_placeholder_mask
            )
        self.device = device
        self.max_tokens = max_tokens
        self.torch = torch

    def predict(self, messages):
        # Open explicitly here to avoid OSWorld's OpenAI/Qwen image-loading wrappers.
        chat = []
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            blocks = []
            for block in content:
                if block["type"] == "image":
                    with Image.open(block["image"]) as image:
                        blocks.append({"type": "image", "image": image.convert("RGB")})
                else:
                    blocks.append(block)
            chat.append({"role": message["role"], "content": blocks})

        inputs = self.processor.apply_chat_template(
            chat, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self.device)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.max_tokens)
        input_tokens = inputs.input_ids.shape[-1]
        generated = output[0, input_tokens:]
        # Qwen's tool tags are special tokens and are essential for parsing the action.
        text = self.processor.tokenizer.decode(generated, skip_special_tokens=False)
        for marker in ("<|im_end|>", "<|endoftext|>"):
            text = text.removesuffix(marker).rstrip()
        return text.strip(), input_tokens, generated.numel()



def send(response):
    print(json.dumps(response, ensure_ascii=False), flush=True)


def main():
    device, max_tokens, *weight_args = sys.argv[1:]
    weights = Path(weight_args[0]) if weight_args else ROOT / "weights"
    if device == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    started = time.perf_counter()
    with redirect_stdout(sys.stderr):
        model = LocalModel(device, int(max_tokens), weights)
    send({"model_load_seconds": round(time.perf_counter() - started, 3)})
    for line in sys.stdin:
        try:
            with redirect_stdout(sys.stderr):
                text, input_tokens, output_tokens = model.predict(json.loads(line))
            send({"text": text, "input_tokens": input_tokens, "output_tokens": output_tokens})
        except Exception as error:
            traceback.print_exc()
            send({"error": str(error)})


if __name__ == "__main__":
    main()
