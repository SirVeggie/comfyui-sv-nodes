from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from comfy_api.latest import io

PACK_DIR = Path(__file__).resolve().parent
PROVIDERS_PATH = PACK_DIR / "llm_providers.json"
EXAMPLE_PROVIDERS_PATH = PACK_DIR / "llm_providers.example.json"


def _providers_file() -> Path:
    if PROVIDERS_PATH.is_file():
        return PROVIDERS_PATH
    return EXAMPLE_PROVIDERS_PATH


def load_providers() -> list[dict]:
    path = _providers_file()
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("providers", [])
    if not isinstance(data, list):
        raise ValueError(f"Invalid provider file format in {path}")
    providers: list[dict] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        base_url = entry.get("base_url")
        if not name or not base_url:
            continue
        providers.append({
            "name": str(name),
            "base_url": str(base_url).rstrip("/"),
            "api_key": str(entry.get("api_key", "")),
        })
    return providers


def provider_names() -> list[str]:
    names = [p["name"] for p in load_providers()]
    return names or ["(configure llm_providers.json)"]


def get_provider(name: str) -> dict:
    for provider in load_providers():
        if provider["name"] == name:
            return provider
    raise ValueError(f"Unknown LLM provider: {name!r}")


def build_llm_args(
    temperature: float = 0.0,
    top_p: float = 0.0,
    top_k: int = 0,
    min_p: float = 0.0,
    adaptive_p_target: float = 0.0,
    adaptive_p_decay: float = 0.0,
    thinking: bool = False,
) -> str:
    payload: dict = {}

    if temperature:
        payload["temperature"] = temperature
    if top_p:
        payload["top_p"] = top_p
    if top_k:
        payload["top_k"] = int(top_k)
    if min_p:
        payload["min_p"] = min_p

    use_adaptive = bool(adaptive_p_target or adaptive_p_decay)
    if adaptive_p_target:
        payload["adaptive_target"] = adaptive_p_target
    if adaptive_p_decay:
        payload["adaptive_decay"] = adaptive_p_decay

    if use_adaptive:
        samplers: list[str] = []
        if top_k:
            samplers.append("top_k")
        if top_p:
            samplers.append("top_p")
        samplers.append("min_p")
        if temperature:
            samplers.append("temperature")
        samplers.append("adaptive_p")
        payload["samplers"] = samplers

    if thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": True}

    return json.dumps(payload)


def parse_llm_args(llm_args: str | None) -> dict:
    if not llm_args or not str(llm_args).strip():
        return {}
    extra = json.loads(llm_args)
    if not isinstance(extra, dict):
        raise ValueError("llm args must be a JSON object")
    return extra


def _image_tensor_to_data_url(image: torch.Tensor) -> str:
    arr = image[0].detach().cpu().numpy()
    arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    pil = Image.fromarray(arr, mode="RGB")
    buf = BytesIO()
    pil.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _build_user_content(prompt: str, image_1: torch.Tensor | None, image_2: torch.Tensor | None):
    parts: list[dict] = []
    if image_1 is not None:
        parts.append({
            "type": "image_url",
            "image_url": {"url": _image_tensor_to_data_url(image_1)},
        })
    if image_2 is not None:
        parts.append({
            "type": "image_url",
            "image_url": {"url": _image_tensor_to_data_url(image_2)},
        })
    parts.append({"type": "text", "text": prompt})
    if len(parts) == 1:
        return prompt
    return parts


def _merge_request_body(model: str, prompt: str, image_1, image_2, seed, llm_args: str | None) -> dict:
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": _build_user_content(prompt, image_1, image_2)}],
        "stream": True,
    }
    if seed is not None:
        body["seed"] = int(seed)
    body.update(parse_llm_args(llm_args))
    return body


def _format_stats(model: str, duration: float, prompt_tokens: int, completion_tokens: int) -> str:
    speed = completion_tokens / duration if duration > 0 else 0.0
    return (
        f"model: {model}\n"
        f"duration: {duration:.2f}s\n"
        f"prompt tokens: {prompt_tokens}\n"
        f"completion tokens: {completion_tokens}\n"
        f"speed: {speed:.2f} tok/s"
    )


def _stream_chat_completion(url: str, api_key: str, body: dict) -> tuple[str, str, int, int, float]:
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    model_used = body.get("model", "")
    prompt_tokens = 0
    completion_tokens = 0
    content_parts: list[str] = []
    start = time.perf_counter()

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                if chunk.get("model"):
                    model_used = chunk["model"]
                usage = chunk.get("usage") or {}
                if usage:
                    prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens) or prompt_tokens)
                    completion_tokens = int(usage.get("completion_tokens", completion_tokens) or completion_tokens)
                choices = chunk.get("choices") or []
                if choices:
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        content_parts.append(delta["content"])
                    message = choice.get("message") or {}
                    if message.get("content"):
                        content_parts.append(message["content"])
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM request failed ({e.code}): {detail}") from e

    duration = time.perf_counter() - start
    return "".join(content_parts), model_used, prompt_tokens, completion_tokens, duration


class LLMArgs(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SV-LLMArgs",
            display_name="LLM Args",
            category="SV Nodes/LLM",
            inputs=[
                io.Float.Input("temperature", default=0.0, min=0.0, max=2.0, step=0.01),
                io.Float.Input("top_p", default=0.0, min=0.0, max=1.0, step=0.01),
                io.Int.Input("top_k", default=0, min=0, max=1000, step=1),
                io.Float.Input("min_p", default=0.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("adaptive_p_target", default=0.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("adaptive_p_decay", default=0.0, min=0.0, max=1.0, step=0.01),
                io.Boolean.Input("thinking", default=False),
            ],
            outputs=[
                io.String.Output(display_name="llm_args"),
            ],
        )

    @classmethod
    def execute(
        cls,
        temperature,
        top_p,
        top_k,
        min_p,
        adaptive_p_target,
        adaptive_p_decay,
        thinking,
    ) -> io.NodeOutput:
        return io.NodeOutput(build_llm_args(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            adaptive_p_target=adaptive_p_target,
            adaptive_p_decay=adaptive_p_decay,
            thinking=thinking,
        ),)


class LLMRequest(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SV-LLMRequest",
            display_name="LLM Request",
            category="SV Nodes/LLM",
            inputs=[
                io.Combo.Input("provider", options=provider_names()),
                io.String.Input("model", default="", multiline=False),
                io.String.Input("prompt", multiline=True),
                io.Image.Input("image_1", optional=True),
                io.Image.Input("image_2", optional=True),
                io.Int.Input("seed", optional=True),
                io.String.Input("llm_args", multiline=True, optional=True),
            ],
            outputs=[
                io.String.Output(display_name="result"),
                io.String.Output(display_name="generation_stats"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, provider, model, prompt, image_1=None, image_2=None, seed=None, llm_args=None) -> str:
        return json.dumps({
            "provider": provider,
            "model": model,
            "prompt": prompt,
            "seed": seed,
            "llm_args": llm_args,
            "image_1": image_1.shape if isinstance(image_1, torch.Tensor) else None,
            "image_2": image_2.shape if isinstance(image_2, torch.Tensor) else None,
        }, sort_keys=True)

    @classmethod
    def execute(
        cls,
        provider,
        model,
        prompt,
        image_1=None,
        image_2=None,
        seed=None,
        llm_args=None,
    ) -> io.NodeOutput:
        if provider == "(configure llm_providers.json)":
            raise ValueError(
                f"Copy {EXAMPLE_PROVIDERS_PATH.name} to {PROVIDERS_PATH.name} and add your providers."
            )
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Model is required")

        profile = get_provider(provider)
        body = _merge_request_body(model.strip(), prompt, image_1, image_2, seed, llm_args)
        url = f"{profile['base_url']}/chat/completions"
        result, model_used, prompt_tokens, completion_tokens, duration = _stream_chat_completion(
            url,
            profile["api_key"],
            body,
        )
        stats = _format_stats(model_used, duration, prompt_tokens, completion_tokens)
        return io.NodeOutput(result, stats)
