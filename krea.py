from __future__ import annotations

import math
import sys

import comfy.utils
import torch

from comfy_api.latest import io

KREA2_IMAGE_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, quantity, "
    "text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n"
    "<|vision_start|><|image_pad|><|vision_end|>\n"
    "{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

KREA2_CONDITIONING_DIM = 12 * 2560
IMAGE_PIXEL_OPTIONS = ("384x384", "512x512", "768x768", "1024x1024")
MAX_SEED = sys.maxsize


def _resize_image_to_pixels(image, target_pixels):
    samples = image[:, :, :, :3].movedim(-1, 1)
    _, _, height, width = samples.shape
    scale_by = math.sqrt(target_pixels / (width * height))
    new_width = max(16, round(width * scale_by))
    new_height = max(16, round(height * scale_by))
    resized = comfy.utils.common_upscale(samples, new_width, new_height, "area", "disabled")
    return resized.movedim(1, -1)


def _scale_conditioning(conditioning, strength):
    out = []
    for cond, extras in conditioning:
        out.append([cond * strength, extras.copy()])
    return out


def _manual_seed(seed, offset=0):
    return (int(seed) + offset) % MAX_SEED


def _rand(shape, seed, offset, device, dtype):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_manual_seed(seed, offset))
    values = torch.rand(shape, generator=generator, device="cpu", dtype=torch.float32)
    return values.to(device=device, dtype=dtype)


def _randn(shape, seed, offset, device, dtype):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_manual_seed(seed, offset))
    values = torch.randn(shape, generator=generator, device="cpu", dtype=torch.float32)
    return values.to(device=device, dtype=dtype)


def _conditioning_rms(cond):
    return cond.float().pow(2).mean().sqrt().to(device=cond.device, dtype=cond.dtype)


def _match_rms(source, target):
    source_rms = _conditioning_rms(source)
    target_rms = _conditioning_rms(target).clamp(min=1e-6)
    return target * (source_rms / target_rms)


def _append_conditioning(conditioning_to, conditioning_from):
    if len(conditioning_from) == 0:
        return conditioning_to

    cond_from = conditioning_from[0][0]
    out = []
    for cond_to, extras in conditioning_to:
        ref = cond_from
        if ref.device != cond_to.device:
            ref = ref.to(cond_to.device)
        out.append([torch.cat((cond_to, ref), dim=1), extras.copy()])
    return out


def _encode_reference(clip, images, prompt):
    images = [image for image in images if image is not None]
    if not images:
        return None

    tokens = clip.tokenize(
        prompt or "",
        images=images,
        llama_template=KREA2_IMAGE_TEMPLATE,
    )
    reference = clip.encode_from_tokens_scheduled(tokens)

    if len(reference) == 0:
        raise RuntimeError("Krea 2 Reference did not produce conditioning.")

    cond_dim = reference[0][0].shape[-1]
    if cond_dim != KREA2_CONDITIONING_DIM:
        raise ValueError(
            "Krea 2 Reference expects a Krea 2 CLIP/text encoder "
            f"(conditioning dim 30720), but got dim {cond_dim}."
        )

    return reference


def _apply_reference(clip, images, prompt, strength, conditioning=None):
    if conditioning is not None and strength == 0:
        return conditioning

    reference = _encode_reference(clip, images, prompt)
    if reference is None:
        return conditioning

    reference = _scale_conditioning(reference, strength)
    if conditioning is None:
        return reference
    return _append_conditioning(conditioning, reference)


def _apply_references_individually(clip, references, prompt, conditioning=None):
    result = conditioning
    for image, strength in references:
        if image is None or strength == 0:
            continue

        reference = _encode_reference(clip, [image], prompt)
        if reference is None:
            continue

        reference = _scale_conditioning(reference, strength)
        if result is None:
            result = reference
        else:
            result = _append_conditioning(result, reference)
    return result


class ScaleConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SV-ScaleConditioning",
            display_name="Scale Conditioning",
            category="SV Nodes/Processing",
            inputs=[
                io.Conditioning.Input("conditioning", optional=True),
                io.Float.Input("strength", default=1.0, min=-5.0, max=5.0, step=0.05),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
            ],
        )

    @classmethod
    def execute(cls, conditioning=None, strength=1.0) -> io.NodeOutput:
        if conditioning is None:
            return io.NodeOutput(None,)
        return io.NodeOutput(_scale_conditioning(conditioning, strength),)


class Krea2ResizeImage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SV-Krea2ResizeImage",
            display_name="Krea 2 Resize Image",
            category="SV Nodes/image",
            inputs=[
                io.Image.Input("image", optional=True),
                io.Combo.Input("image_pixels", options=list(IMAGE_PIXEL_OPTIONS), default="384x384"),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(cls, image=None, image_pixels="384x384") -> io.NodeOutput:
        if image is None:
            return io.NodeOutput(None,)

        side = int(image_pixels.split("x", 1)[0])
        return io.NodeOutput(_resize_image_to_pixels(image, side * side),)


class ConditioningNoiseNudge(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SV-ConditioningNoiseNudge",
            display_name="Conditioning Noise Nudge",
            category="SV Nodes/Processing",
            inputs=[
                io.Conditioning.Input("conditioning", optional=True),
                io.Float.Input("strength", default=0.03, min=0.0, max=1.0, step=0.01),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED, step=1),
                io.Combo.Input("scale_mode", options=["rms", "std", "absolute"], default="rms"),
                io.Combo.Input("distribution", options=["gaussian", "uniform"], default="gaussian"),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
            ],
        )

    @classmethod
    def execute(
        cls,
        conditioning=None,
        strength=0.03,
        seed=0,
        scale_mode="rms",
        distribution="gaussian",
    ) -> io.NodeOutput:
        if conditioning is None:
            return io.NodeOutput(None,)

        out = []
        for index, (cond, extras) in enumerate(conditioning):
            if strength == 0:
                out.append([cond.clone(), extras.copy()])
                continue

            if distribution == "uniform":
                noise = _rand(cond.shape, seed, index, cond.device, cond.dtype) * 2 - 1
            else:
                noise = _randn(cond.shape, seed, index, cond.device, cond.dtype)

            if scale_mode == "std":
                scale = cond.float().std().to(device=cond.device, dtype=cond.dtype).clamp(min=1e-6)
            elif scale_mode == "absolute":
                scale = torch.tensor(1.0, device=cond.device, dtype=cond.dtype)
            else:
                scale = _conditioning_rms(cond).clamp(min=1e-6)

            out.append([cond + noise * scale * strength, extras.copy()])
        return io.NodeOutput(out,)


class ConditioningTokenDropoutNudge(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SV-ConditioningTokenDropoutNudge",
            display_name="Conditioning Token Dropout Nudge",
            category="SV Nodes/Processing",
            inputs=[
                io.Conditioning.Input("conditioning", optional=True),
                io.Float.Input("probability", default=0.10, min=0.0, max=1.0, step=0.01),
                io.Float.Input("strength", default=0.50, min=0.0, max=1.0, step=0.01),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED, step=1),
                io.Combo.Input("drop_to", options=["mean", "zero"], default="mean"),
                io.Boolean.Input("protect_first_token", default=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
            ],
        )

    @classmethod
    def execute(
        cls,
        conditioning=None,
        probability=0.10,
        strength=0.50,
        seed=0,
        drop_to="mean",
        protect_first_token=True,
    ) -> io.NodeOutput:
        if conditioning is None:
            return io.NodeOutput(None,)

        out = []
        for index, (cond, extras) in enumerate(conditioning):
            if probability == 0 or strength == 0:
                out.append([cond.clone(), extras.copy()])
                continue

            if cond.ndim >= 2:
                mask_shape = (*cond.shape[:-1], 1)
                mask = _rand(mask_shape, seed, index, cond.device, cond.dtype) < probability
                if protect_first_token and cond.shape[-2] > 0:
                    mask[..., :1, :] = False
                if drop_to == "zero":
                    target = torch.zeros_like(cond)
                else:
                    target = cond.mean(dim=-2, keepdim=True)
            else:
                mask = _rand(cond.shape, seed, index, cond.device, cond.dtype) < probability
                target = torch.zeros_like(cond) if drop_to == "zero" else cond.mean()

            nudged = cond * (1 - strength) + target * strength
            out.append([torch.where(mask, nudged, cond), extras.copy()])
        return io.NodeOutput(out,)


class ConditioningScaleJitter(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SV-ConditioningScaleJitter",
            display_name="Conditioning Scale Jitter",
            category="SV Nodes/Processing",
            inputs=[
                io.Conditioning.Input("conditioning", optional=True),
                io.Float.Input("amount", default=0.05, min=0.0, max=1.0, step=0.01),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED, step=1),
                io.Combo.Input("granularity", options=["tokens", "channels", "elements", "global"], default="tokens"),
                io.Boolean.Input("preserve_rms", default=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
            ],
        )

    @staticmethod
    def _jitter_shape(cond, granularity):
        if granularity == "global":
            return (1,) * cond.ndim
        if granularity == "channels" and cond.ndim >= 1:
            return (*((1,) * (cond.ndim - 1)), cond.shape[-1])
        if granularity == "tokens" and cond.ndim >= 2:
            return (*cond.shape[:-1], 1)
        return cond.shape

    @classmethod
    def execute(
        cls,
        conditioning=None,
        amount=0.05,
        seed=0,
        granularity="tokens",
        preserve_rms=True,
    ) -> io.NodeOutput:
        if conditioning is None:
            return io.NodeOutput(None,)

        out = []
        for index, (cond, extras) in enumerate(conditioning):
            if amount == 0:
                out.append([cond.clone(), extras.copy()])
                continue

            jitter_shape = cls._jitter_shape(cond, granularity)
            jitter = _rand(jitter_shape, seed, index, cond.device, cond.dtype) * 2 - 1
            nudged = cond * (1 + jitter * amount)
            if preserve_rms:
                nudged = _match_rms(cond, nudged)
            out.append([nudged, extras.copy()])
        return io.NodeOutput(out,)


class Krea2Reference(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SV-Krea2Reference",
            display_name="Krea 2 Reference",
            category="SV Nodes/Processing",
            inputs=[
                io.Clip.Input("clip"),
                io.Conditioning.Input("conditioning", optional=True),
                io.String.Input("prompt", force_input=True, multiline=True, optional=True),
                io.Float.Input("strength", default=1.0, min=-5.0, max=5.0, step=0.05),
                io.Image.Input("image", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip,
        conditioning=None,
        prompt=None,
        strength=1.0,
        image=None,
    ) -> io.NodeOutput:
        result = _apply_reference(clip, [image], prompt, strength, conditioning)
        return io.NodeOutput(result,)


class Krea2LocalStyleReferenceOld(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SV-Krea2LocalStyleReferenceOld",
            display_name="Krea 2 Local Style Reference Old",
            category="SV Nodes/Processing",
            inputs=[
                io.Clip.Input("clip"),
                io.Image.Input("image"),
                io.Float.Input("strength", default=1.0, min=-5.0, max=5.0, step=0.05),
                io.String.Input(
                    "reference_instruction",
                    default=(
                        "Use this reference image for visual style, palette, lighting, texture, "
                        "medium, rendering, and composition language. Do not copy the subject, "
                        "identity, pose, layout, or text unless the main prompt asks for it."
                    ),
                    multiline=True,
                ),
                io.Combo.Input("image_pixels", options=list(IMAGE_PIXEL_OPTIONS), default="384x384"),
                io.Conditioning.Input("conditioning", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip,
        image,
        strength,
        reference_instruction,
        image_pixels,
        conditioning=None,
    ) -> io.NodeOutput:
        if conditioning is not None and strength == 0:
            return io.NodeOutput(conditioning)

        side = int(image_pixels.split("x", 1)[0])
        image_ref = _resize_image_to_pixels(image, side * side)
        reference = _encode_reference(clip, [image_ref], reference_instruction)

        reference = _scale_conditioning(reference, strength)
        if conditioning is None:
            return io.NodeOutput(reference)
        return io.NodeOutput(_append_conditioning(conditioning, reference))


class Krea2ReferencePlus(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SV-Krea2ReferencePlus",
            display_name="Krea 2 Reference Plus",
            category="SV Nodes/Processing",
            inputs=[
                io.Clip.Input("clip"),
                io.Conditioning.Input("conditioning", optional=True),
                io.String.Input("prompt", force_input=True, multiline=True, optional=True),
                io.Float.Input("strength", default=1.0, min=-5.0, max=5.0, step=0.05),
                io.Image.Input("image_1", optional=True),
                io.Image.Input("image_2", optional=True),
                io.Image.Input("image_3", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip,
        conditioning=None,
        prompt=None,
        strength=1.0,
        image_1=None,
        image_2=None,
        image_3=None,
    ) -> io.NodeOutput:
        result = _apply_reference(
            clip, [image_1, image_2, image_3], prompt, strength, conditioning,
        )
        return io.NodeOutput(result,)


class Krea2ReferenceAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SV-Krea2ReferenceAdvanced",
            display_name="Krea 2 Reference Advanced",
            category="SV Nodes/Processing",
            inputs=[
                io.Clip.Input("clip"),
                io.Conditioning.Input("conditioning", optional=True),
                io.String.Input("prompt", force_input=True, multiline=True, optional=True),
                io.Float.Input("strength_1", default=1.0, min=-5.0, max=5.0, step=0.05),
                io.Float.Input("strength_2", default=1.0, min=-5.0, max=5.0, step=0.05),
                io.Float.Input("strength_3", default=1.0, min=-5.0, max=5.0, step=0.05),
                io.Image.Input("image_1", optional=True),
                io.Image.Input("image_2", optional=True),
                io.Image.Input("image_3", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip,
        conditioning=None,
        prompt=None,
        strength_1=1.0,
        strength_2=1.0,
        strength_3=1.0,
        image_1=None,
        image_2=None,
        image_3=None,
    ) -> io.NodeOutput:
        result = _apply_references_individually(
            clip,
            (
                (image_1, strength_1),
                (image_2, strength_2),
                (image_3, strength_3),
            ),
            prompt,
            conditioning,
        )
        return io.NodeOutput(result,)
