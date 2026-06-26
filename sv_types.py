from comfy_api.latest import io

Wildcards = io.Custom("wildcards")
SvPrompt = io.Custom("sv_prompt")
CondList = io.Custom("cond_list")
RsOutput = io.Custom("RS_OUTPUT")
PpmOutput = io.Custom("PPM_OUTPUT")
BpOutput = io.Custom("BP_OUTPUT")
Sigmas = io.Custom("SIGMAS")
SvSampler = io.Custom("SAMPLER")
FlowControl = io.Custom("FLOW_CONTROL")
SvPipe = io.Custom("sv_pipe")
Accumulation = io.Custom("ACCUMULATION")
Signal = io.Custom("signal")
Timer = io.Custom("TIMER")
Curve = io.Custom("CURVE")

V1_TYPE_MAP = {
    "STRING": io.String,
    "INT": io.Int,
    "FLOAT": io.Float,
    "BOOLEAN": io.Boolean,
    "IMAGE": io.Image,
    "MASK": io.Mask,
    "LATENT": io.Latent,
    "CONDITIONING": io.Conditioning,
    "MODEL": io.Model,
    "VAE": io.Vae,
    "CLIP": io.Clip,
    "wildcards": Wildcards,
    "sv_prompt": SvPrompt,
    "cond_list": CondList,
    "RS_OUTPUT": RsOutput,
    "PPM_OUTPUT": PpmOutput,
    "BP_OUTPUT": BpOutput,
    "SIGMAS": Sigmas,
    "SAMPLER": SvSampler,
    "FLOW_CONTROL": FlowControl,
    "sv_pipe": SvPipe,
    "ACCUMULATION": Accumulation,
    "signal": Signal,
    "TIMER": Timer,
    "CURVE": Curve,
}

V1_HIDDEN_MAP = {
    "UNIQUE_ID": "io.Hidden.unique_id",
    "PROMPT": "io.Hidden.prompt",
    "EXTRA_PNGINFO": "io.Hidden.extra_pnginfo",
    "DYNPROMPT": "io.Hidden.dynprompt",
}

V1_OPTION_REMAP = {
    "forceInput": "force_input",
    "rawLink": "raw_link",
    "defaultInput": "force_input",
    "multiline": "multiline",
    "default": "default",
    "min": "min",
    "max": "max",
    "step": "step",
    "lazy": "lazy",
    "label_on": "label_on",
    "label_off": "label_off",
}
