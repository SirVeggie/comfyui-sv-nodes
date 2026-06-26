import os
import time
import copy
import comfy.samplers
import folder_paths
from .logic import approx_index, calculate_sigma_range, calculate_sigma_range_percent, clean_prompt, default, find_percent, get_sigmas, get_skimming_mask, interpolated_scales, needs_seed, normalize_adjust, process, process_advanced, process_simple, process_control, process_vars, process_wildcards, remove_comments, separate_lora, separate_lora_advanced, unescape_prompt
from .sv_types import (
    Wildcards, SvPrompt, CondList, RsOutput, PpmOutput, BpOutput, Sigmas, SvSampler,
    FlowControl, SvPipe, Accumulation, Signal, Timer, Curve,
)
import node_helpers
import hashlib
import math
import random as _random
import json
import re
import torch
import sys
import time
from functools import partial
from comfy_execution.graph_utils import GraphBuilder, is_link
from comfy_execution.graph import ExecutionBlocker
from comfy_api.latest import ComfyExtension, io, ui

#-------------------------------------------------------------------------------#
# Constants

NUM_FLOW_SOCKETS = 4


#-------------------------------------------------------------------------------#

class SimpleText(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-SimpleText',
            display_name='Simple Text',
            category='SV Nodes/Input',
            inputs=[
            io.String.Input('text', multiline=True),
            ],
            outputs=[
            io.String.Output(display_name='text'),
            ]
        )

    
    @classmethod
    def execute(cls, text) -> io.NodeOutput:
        if not isinstance(text, str):
            raise TypeError("Invalid text input type")
        return io.NodeOutput(text,)

#-------------------------------------------------------------------------------#

class WildcardProcessing(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-WildcardProcessing',
            display_name='Wildcard Processing',
            category='SV Nodes/Processing',
            inputs=[
            io.String.Input('text', force_input=True),
            Wildcards.Input('wildcards'),
            io.Int.Input('seed', force_input=True),
            ],
            outputs=[
            io.String.Output(display_name='output'),
            ]
        )

    
    @classmethod
    def check_lazy_status(cls, text, **kwargs):
        return ["wildcards"] if "__" in text else []
    @classmethod
    def execute(cls, text, wildcards, seed) -> io.NodeOutput:
        return io.NodeOutput(process_wildcards(text, wildcards, seed, 5),)
    @classmethod
    def fingerprint_inputs(cls, text, wildcards, seed):
        return f"{text} {seed}"
    

#-------------------------------------------------------------------------------#

class WildcardLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-WildcardLoader',
            display_name='Load Wildcards',
            category='SV Nodes/IO',
            inputs=[
            io.String.Input('dir_path'),
            ],
            outputs=[
            Wildcards.Output(display_name='wildcards'),
            ]
        )

    
    @classmethod
    def execute(cls, dir_path) -> io.NodeOutput:
        wildcards = {}
        if not os.path.isdir(dir_path):
            return io.NodeOutput(wildcards,)
        dirs = [dir_path]
        while dirs:
            dir_path = dirs.pop()
            for filename in os.listdir(dir_path):
                filepath = os.path.join(dir_path, filename)
                if os.path.isdir(filepath):
                    dirs.append(filepath)
                    continue
                if not os.path.isfile(filepath):
                    continue
                key = os.path.splitext(filename)[0].lower()
                if key in wildcards:
                    raise ValueError(f"Duplicate wildcard '{key}' found in '{filepath}'")
                with open(filepath, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                    wildcards[key] = lines
        return io.NodeOutput(wildcards,)
    
    @classmethod
    def fingerprint_inputs(cls, dir_path):
        return os.path.getmtime(dir_path)
    

#-------------------------------------------------------------------------------#

class WildcardString(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-WildcardString',
            display_name='String Wildcards',
            category='SV Nodes/Input',
            inputs=[
            io.String.Input('input', multiline=True),
            Wildcards.Input('wildcards', optional=True),
            ],
            outputs=[
            Wildcards.Output(display_name='wildcards'),
            ]
        )

    
    key_regex = re.compile(r"^\[[\w -]+\]$")
    @classmethod
    def execute(cls, input: str, wildcards: dict[str: list[str]] = None) -> io.NodeOutput:
        result = copy.deepcopy(wildcards) if wildcards is not None else {}
        lines = [line.strip() for line in input.splitlines() if line.strip()]
        current_key = None
        for line in lines:
            if line.startswith("#") or line.startswith("//"):
                continue
            elif WildcardString.key_regex.match(line):
                current_key = line[1:-1].strip().lower()
                if current_key not in result:
                    result[current_key] = []
            elif current_key is not None:
                result[current_key].append(line)
        return io.NodeOutput(result,)

#-------------------------------------------------------------------------------#

class PromptProcessing(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-PromptProcessing',
            display_name='Prompt Processing',
            category='SV Nodes/Processing',
            inputs=[
            io.String.Input('text', force_input=True),
            io.String.Input('variables', force_input=True, optional=True),
            io.Int.Input('seed', force_input=True, optional=True),
            ],
            outputs=[
            io.String.Output(display_name='1st pass'),
            io.String.Output(display_name='2nd pass'),
            io.String.Output(display_name='3rd pass'),
            ]
        )

    
    @classmethod
    def execute(cls, text, variables="", seed=1) -> io.NodeOutput:
        text = remove_comments(text)
        return process(text, 0, variables, seed), process(text, 1, variables, seed), process(text, 2, variables, seed)
    
    @classmethod
    def fingerprint_inputs(cls, text, variables, seed):
        return f"{text} {variables} {seed}"


#-------------------------------------------------------------------------------#

class PromptProcessingRecursive(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-PromptProcessingRecursive',
            display_name='Recursive Processing',
            category='SV Nodes/Processing',
            inputs=[
            io.String.Input('text', force_input=True),
            io.Int.Input('step', force_input=True),
            io.Float.Input('progress', force_input=True),
            io.String.Input('variables', force_input=True, optional=True),
            io.Int.Input('seed', force_input=True, lazy=True, optional=True),
            ],
            outputs=[
            io.String.Output(display_name='prompt'),
            io.String.Output(display_name='lora'),
            ]
        )

    
    @classmethod
    def check_lazy_status(cls, text, **kwargs):
        if needs_seed(text):
            return ["seed"]
        return []
    
    @classmethod
    def execute(cls, text, step, progress, variables="", seed=1) -> io.NodeOutput:
        text = remove_comments(text)
        return LoraSeparator.execute( process_advanced(text, variables, seed, step, progress))


#-------------------------------------------------------------------------------#

class PromptProcessingAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-PromptProcessingAdvanced',
            display_name='Advanced Processing',
            category='SV Nodes/Processing',
            inputs=[
            io.String.Input('prompt', force_input=True),
            io.Int.Input('steps', force_input=True),
            io.Int.Input('phase', default=1, min=1, max=100, step=1, optional=True),
            io.String.Input('variables', force_input=True, optional=True),
            io.Int.Input('seed', force_input=True, lazy=True, optional=True),
            ],
            outputs=[
            SvPrompt.Output(display_name='prompt'),
            io.String.Output(display_name='lora'),
            ]
        )

    
    @classmethod
    def check_lazy_status(cls, prompt, **kwargs):
        if needs_seed(prompt):
            return ["seed"]
        return []
    
    @classmethod
    def execute(cls, prompt, steps, phase=1, variables="", seed=1) -> io.NodeOutput:
        prompt = remove_comments(prompt)
        parts = re.split(r"[\n\r]+[\s]*-+[\s]*[\n\r]+", prompt, 1)
        full_positive = parts[0]
        full_negative = parts[1] if len(parts) > 1 else ""
        variables += "\npositive=" + clean_prompt(separate_lora(full_positive)[0])
        
        result = []
        
        for i in range(1, steps + 1):
            progress = 1 if steps == 1 else (i - 1) / (steps - 1)
            progress *= 0.999
            progress += phase - 1
            pos = process_advanced(full_positive, variables, seed, i, progress)
            neg = process_advanced(full_negative, variables, seed, i, progress)
            result.append((pos, neg))
        
        return separate_lora_advanced(result)


#-------------------------------------------------------------------------------#

class PromptProcessingEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-PromptProcessingEncode',
            display_name='Encode Prompt',
            category='SV Nodes/Processing',
            inputs=[
            io.Clip.Input('clip'),
            SvPrompt.Input('prompt'),
            ],
            outputs=[
            io.Conditioning.Output(display_name='positive'),
            io.Conditioning.Output(display_name='negative'),
            ]
        )

    cache = {}
    
    @classmethod
    def execute(cls, clip, prompt) -> io.NodeOutput:
        steps = len(prompt)
        pconds = []
        nconds = []
        
        for i in range(1, steps + 1):
            start = (i - 1) / steps
            end = i / steps
            pos, neg = prompt[i - 1]
            
            if not cls.cacheHas(pos):
                cls.cacheSet(pos, encode(clip, pos))
            pcond = node_helpers.conditioning_set_values(cls.cacheGet(pos), {"start_percent": start, "end_percent": end})
            pconds += pcond
            if not cls.cacheHas(neg):
                cls.cacheSet(neg, encode(clip, neg))
            ncond = node_helpers.conditioning_set_values(cls.cacheGet(neg), {"start_percent": start, "end_percent": end})
            nconds += ncond
        
        cls.cacheClean()
        return io.NodeOutput(pconds, nconds)
    
    @classmethod
    def cacheHas(cls, prompt):
        return prompt in PromptProcessingEncode.cache
    @classmethod
    def cacheSet(cls, prompt, encoded):
        PromptProcessingEncode.cache[prompt] = [prompt, encoded, time.time_ns() // 1000000]
    @classmethod
    def cacheGet(cls, prompt):
        return PromptProcessingEncode.cache[prompt][1]
    def cacheClean(self):
        if len(PromptProcessingEncode.cache) < 200:
            return
        for item in sorted(list(PromptProcessingEncode.cache.values()), key=lambda x: x[2])[:100]:
            del PromptProcessingEncode.cache[item[0]]

def encode(clip, text):
    text = unescape_prompt(text)
    tokens = clip.tokenize(text)
    output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
    cond = output.pop("cond")
    return [[cond, output]]

#-------------------------------------------------------------------------------#

class PromptProcessingEncodeList(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-PromptProcessingEncodeList',
            display_name='Encode Prompt List',
            category='SV Nodes/Processing',
            inputs=[
            io.Clip.Input('clip'),
            SvPrompt.Input('prompt'),
            ],
            outputs=[
            CondList.Output(display_name='conds'),
            ]
        )

    
    @classmethod
    def execute(cls, clip, prompt) -> io.NodeOutput:
        steps = len(prompt)
        cache = {}
        conds = []
        
        for i in range(1, steps + 1):
            pos, neg = prompt[i - 1]
            
            if pos not in cache:
                cache[pos] = encode(clip, pos)
            pcond = cache[pos]
            if neg not in cache:
                cache[neg] = encode(clip, neg)
            ncond = cache[neg]
            conds.append((pcond, ncond))
        
        return io.NodeOutput(conds,)

#-------------------------------------------------------------------------------#

class PromptProcessingGetCond(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-PromptProcessingGetCond',
            display_name='Get Conditioning',
            category='SV Nodes/Processing',
            inputs=[
            CondList.Input('conds'),
            io.Int.Input('step', force_input=True),
            ],
            outputs=[
            io.Conditioning.Output(display_name='positive'),
            io.Conditioning.Output(display_name='negative'),
            ]
        )

    
    @classmethod
    def execute(cls, conds, step) -> io.NodeOutput:
        return conds[step - 1]


#-------------------------------------------------------------------------------#

class PromptProcessingSimple(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-PromptProcessingSimple',
            display_name='Simple Processing',
            category='SV Nodes/Processing',
            inputs=[
            io.String.Input('prompt', force_input=True),
            io.String.Input('variables', force_input=True, optional=True),
            io.Int.Input('seed', force_input=True, lazy=True, optional=True),
            ],
            outputs=[
            io.String.Output(display_name='pos 1'),
            io.String.Output(display_name='neg 1'),
            io.String.Output(display_name='pos 2'),
            io.String.Output(display_name='neg 2'),
            ]
        )

    
    @classmethod
    def check_lazy_status(cls, prompt, **kwargs):
        if needs_seed(prompt):
            return ["seed"]
        return []
    
    @classmethod
    def execute(cls, prompt, variables="", seed=1) -> io.NodeOutput:
        prompt = remove_comments(prompt)
        parts = re.split(r"[\n\r]+[\s]*-+[\s]*[\n\r]+", prompt, 1)
        full_positive = parts[0]
        full_negative = parts[1] if len(parts) > 1 else ""
        variables += "\npositive=" + clean_prompt(separate_lora(full_positive)[0])
        
        pos1 = process_simple(full_positive, variables, seed, False)
        neg1 = process_simple(full_negative, variables, seed, False)
        pos2 = process_simple(full_positive, variables, seed, True)
        neg2 = process_simple(full_negative, variables, seed, True)
        
        return io.NodeOutput(pos1, neg1, pos2, neg2)

#-------------------------------------------------------------------------------#

class PromptProcessingPromptControl(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-PromptProcessingPromptControl',
            display_name='Control Processing',
            category='SV Nodes/Processing',
            inputs=[
            io.String.Input('prompt', force_input=True),
            io.Int.Input('steps', force_input=True),
            io.Int.Input('phase', default=1, min=1, max=100, step=1, optional=True),
            io.String.Input('variables', force_input=True, optional=True),
            io.Int.Input('seed', force_input=True, lazy=True, optional=True),
            ],
            outputs=[
            io.String.Output(display_name='positive'),
            io.String.Output(display_name='negative'),
            ]
        )

    
    @classmethod
    def check_lazy_status(cls, prompt, **kwargs):
        if needs_seed(prompt):
            return ["seed"]
        return []
    
    @classmethod
    def execute(cls, prompt, steps, phase=1, variables="", seed=1) -> io.NodeOutput:
        prompt = remove_comments(prompt)
        parts = re.split(r"[\n\r]+[\s]*-+[\s]*[\n\r]+", prompt, 1)
        full_positive = parts[0]
        full_negative = parts[1] if len(parts) > 1 else ""
        variables += "\npositive=" + clean_prompt(separate_lora(full_positive)[0])
        
        pos = process_control(full_positive, steps, phase, variables, seed)
        neg = process_control(full_negative, steps, phase, variables, seed)
        
        return io.NodeOutput(pos, neg)

#-------------------------------------------------------------------------------#

class PromptProcessingVars(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-PromptProcessingVars',
            display_name='Variable Processing',
            category='SV Nodes/Processing',
            inputs=[
            io.String.Input('prompt', force_input=True),
            io.String.Input('variables', force_input=True, optional=True),
            io.Int.Input('seed', force_input=True, optional=True),
            ],
            outputs=[
            io.String.Output(display_name='output'),
            ]
        )

    
    @classmethod
    def execute(cls, prompt, variables="", seed=1) -> io.NodeOutput:
        prompt = remove_comments(prompt)
        parts = re.split(r"[\n\r]+[\s]*-+[\s]*[\n\r]+", prompt, 1)
        full_positive = parts[0]
        variables += "\npositive=" + clean_prompt(separate_lora(full_positive)[0])
        
        if len(parts) <= 1:
            return io.NodeOutput(process_vars(parts[0], variables, seed),)
        parts[0] = process_vars(parts[0], variables, seed)
        parts[1] = process_vars(parts[1], variables, seed)
        
        return io.NodeOutput(f"{parts[0]}\n---\n{parts[1]}",)

#-------------------------------------------------------------------------------#

class ResolutionSelector(io.ComfyNode):
    _match_template = io.MatchType.Template('ResolutionSelector')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-ResolutionSelector',
            display_name='Resolution Selector',
            category='SV Nodes/Input',
            inputs=[
            io.Int.Input('base', default=768, min=64, max=4096, step=64),
            io.Combo.Input('ratio', options=['1:1', '5:4', '4:3', '3:2', '16:9', '21:9']),
            io.Boolean.Input('orientation', default=False, label_on='portrait', label_off='landscape'),
            io.MatchType.Input('seed', template=cls._match_template, optional=True),
            io.MatchType.Input('random', template=cls._match_template, optional=True),
            ],
            outputs=[
            io.Int.Output(display_name='width'),
            io.Int.Output(display_name='height'),
            ]
        )

    RATIOS = ["1:1", "5:4", "4:3", "3:2", "16:9", "21:9"]
    
    @classmethod
    def execute(cls, base, ratio, orientation, seed=-1, random="") -> io.NodeOutput:
        if not isinstance(seed, int):
            raise TypeError("Invalid seed input type")
        if not isinstance(random, str) and not None:
            raise TypeError("Invalid random input type")
        
        ratio = ratio.split(":")
        if len(ratio) != 2:
            raise ValueError("Invalid ratio")
        
        if random == "orientation":
            rand = _random.Random(seed)
            orientation = rand.choice([True, False])
        elif random and seed and seed >= 1:
            random = random.replace(" ", "").split(",")
            rand = _random.Random(seed)
            ratio = rand.choice(random).split(":")
        ratio = math.sqrt(float(ratio[0]) / float(ratio[1]))
        
        width = math.floor(base * ratio / 64) * 64 
        height = math.floor(base / ratio / 64) * 64
        
        if orientation:
            width, height = height, width
        return width, height


#-------------------------------------------------------------------------------#

class ResolutionSelector2(io.ComfyNode):
    _match_template = io.MatchType.Template('ResolutionSelector2')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-ResolutionSelector2',
            display_name='Resolution Selector 2',
            category='SV Nodes/Input',
            inputs=[
            io.Int.Input('base', default=768, min=64, max=4096, step=64),
            io.Combo.Input('ratio', options=['1:1', '5:4', '4:3', '3:2', '16:9', '21:9']),
            io.Boolean.Input('orientation', default=False, label_on='portrait', label_off='landscape'),
            io.Float.Input('hires', min=1, max=4, step=0.1, default=1.5),
            io.Int.Input('batch', min=1, max=32, step=1, default=1),
            io.MatchType.Input('seed', template=cls._match_template, optional=True),
            io.MatchType.Input('random', template=cls._match_template, optional=True),
            ],
            outputs=[
            RsOutput.Output(display_name='packet'),
            ]
        )

    
    @classmethod
    def execute(cls, base, ratio, orientation, hires, batch, seed=-1, random="") -> io.NodeOutput:
        result = ResolutionSelector.execute( base, ratio, orientation, seed, random)
        return io.NodeOutput((result[0], result[1], hires, batch),)

#-------------------------------------------------------------------------------#

class ResolutionSelector2Output(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-ResolutionSelector2Output',
            display_name='Selector Output',
            category='SV Nodes/Input',
            inputs=[
            RsOutput.Input('packet'),
            ],
            outputs=[
            io.Int.Output(display_name='width'),
            io.Int.Output(display_name='height'),
            io.Float.Output(display_name='hires ratio'),
            io.Int.Output(display_name='batch size'),
            ]
        )

    
    @classmethod
    def execute(cls, packet) -> io.NodeOutput:
        if not isinstance(packet, tuple):
            raise TypeError("Invalid packet input type")
        if len(packet) != 4:
            raise ValueError("Invalid packet length")
        return packet


#-------------------------------------------------------------------------------#

class NormalizeImageSize(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-NormalizeImageSize',
            display_name='Normalize Size',
            category='SV Nodes/Input',
            inputs=[
            io.Int.Input('width', force_input=True),
            io.Int.Input('height', force_input=True),
            io.Int.Input('size', min=64, max=4096, step=64, default=768),
            ],
            outputs=[
            io.Int.Output(display_name='width'),
            io.Int.Output(display_name='height'),
            ]
        )

    
    @classmethod
    def execute(cls, width, height, size) -> io.NodeOutput:
        ratio = math.sqrt(float(width) / float(height))
        width = math.floor(size * ratio / 64) * 64
        height = math.floor(size / ratio / 64) * 64
        return width, height


#-------------------------------------------------------------------------------#

class NormalizeImageSize64(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-NormalizeImageSize64',
            display_name='Normalize Size (64)',
            category='SV Nodes/Input',
            inputs=[
            io.Int.Input('width', force_input=True),
            io.Int.Input('height', force_input=True),
            ],
            outputs=[
            io.Int.Output(display_name='width'),
            io.Int.Output(display_name='height'),
            ]
        )

    
    @classmethod
    def execute(cls, width, height) -> io.NodeOutput:
        width = math.floor(width / 64) * 64
        height = math.floor(height / 64) * 64
        return width, height


#-------------------------------------------------------------------------------#

class BasicParams(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-BasicParams',
            display_name='Params',
            category='SV Nodes/Input',
            inputs=[
            io.Float.Input('cfg', min=0, max=20, step=0.1, default=8.0),
            io.Int.Input('steps', min=1, max=100, step=1, default=10),
            io.Float.Input('denoise', min=0, max=1, step=0.01, default=1.0),
            io.Combo.Input('sampler', options=['euler', 'euler_ancestral', 'heun', 'dpm_2', 'dpm_2_ancestral']),
            ],
            outputs=[
            BpOutput.Output(display_name='packet'),
            ]
        )

    
    @classmethod
    def execute(cls, cfg, steps, denoise, sampler) -> io.NodeOutput:
        if not isinstance(cfg, float) and not isinstance(cfg, int):
            raise TypeError("Invalid cfg input type")
        if not isinstance(steps, int):
            raise TypeError("Invalid steps input type")
        if not isinstance(denoise, float) and not isinstance(denoise, int):
            raise TypeError("Invalid denoise input type")
        if not isinstance(sampler, str):
            raise TypeError("Invalid sampler input type")
        return io.NodeOutput((cfg, steps, denoise, sampler, None, None, None, None),)

#-------------------------------------------------------------------------------#

class BasicParamsPlus(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-BasicParamsPlus',
            display_name='Params Plus',
            category='SV Nodes/Input',
            inputs=[
            io.Float.Input('cfg', min=0, max=20, step=0.1, default=8.0),
            io.Int.Input('steps', min=1, max=100, step=1, default=10),
            io.Float.Input('denoise', min=0, max=1, step=0.01, default=1.0),
            io.Combo.Input('sampler', options=['euler', 'euler_ancestral', 'heun', 'dpm_2', 'dpm_2_ancestral']),
            io.Combo.Input('scheduler', options=['normal', 'karras', 'exponential', 'sgm_uniform', 'simple', 'ddim_uniform', 'ays']),
            ],
            outputs=[
            BpOutput.Output(display_name='packet'),
            ]
        )

    
    @classmethod
    def execute(cls, cfg, steps, denoise, sampler, scheduler) -> io.NodeOutput:
        if not isinstance(cfg, float) and not isinstance(cfg, int):
            raise TypeError("Invalid cfg input type")
        if not isinstance(steps, int):
            raise TypeError("Invalid steps input type")
        if not isinstance(denoise, float) and not isinstance(denoise, int):
            raise TypeError("Invalid denoise input type")
        if not isinstance(sampler, str):
            raise TypeError("Invalid sampler input type")
        return io.NodeOutput((cfg, steps, denoise, sampler, scheduler, scheduler == "ays", None, None),)

#-------------------------------------------------------------------------------#

class BasicParamsStartEnd(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-BasicParamsStartEnd',
            display_name='Params Start/End',
            category='SV Nodes/Input',
            inputs=[
            io.Float.Input('cfg', min=0, max=20, step=0.1, default=8.0),
            io.Int.Input('steps', min=1, max=100, step=1, default=10),
            io.Combo.Input('sampler', options=['euler', 'euler_ancestral', 'heun', 'dpm_2', 'dpm_2_ancestral']),
            io.Combo.Input('scheduler', options=['normal', 'karras', 'exponential', 'sgm_uniform', 'simple', 'ddim_uniform', 'ays']),
            io.Float.Input('start', min=0, max=1, step=0.01, default=0.0),
            io.Float.Input('end', min=0, max=1, step=0.01, default=1.0),
            ],
            outputs=[
            BpOutput.Output(display_name='packet'),
            ]
        )

    
    @classmethod
    def execute(cls, cfg, steps, start, end, sampler, scheduler) -> io.NodeOutput:
        if not isinstance(cfg, float) and not isinstance(cfg, int):
            raise TypeError("Invalid cfg input type")
        if not isinstance(steps, int):
            raise TypeError("Invalid steps input type")
        if not isinstance(start, float) and not isinstance(start, int):
            raise TypeError("Invalid start input type")
        if not isinstance(end, float) and not isinstance(end, int):
            raise TypeError("Invalid end input type")
        if not isinstance(sampler, str):
            raise TypeError("Invalid sampler input type")
        return io.NodeOutput((cfg, steps, None, sampler, scheduler, scheduler == "ays", start, end),)

#-------------------------------------------------------------------------------#

class BasicParamsCustom(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-BasicParamsCustom',
            display_name='Params Custom',
            category='SV Nodes/Input',
            inputs=[
            io.Float.Input('cfg', min=0, max=20, step=0.1, default=8.0, optional=True),
            io.Int.Input('steps', min=1, max=100, step=1, default=10, optional=True),
            io.Combo.Input('sampler', options=['euler', 'euler_ancestral', 'heun', 'dpm_2', 'dpm_2_ancestral'], optional=True),
            io.Combo.Input('scheduler', options=['normal', 'karras', 'exponential', 'sgm_uniform', 'simple', 'ddim_uniform', 'ays'], optional=True),
            ],
            outputs=[
            BpOutput.Output(display_name='packet'),
            ]
        )

    
    @classmethod
    def execute(cls, cfg=8.0, steps=10, sampler="euler", scheduler="normal") -> io.NodeOutput:
        if not isinstance(cfg, float) and not isinstance(cfg, int):
            raise TypeError("Invalid cfg input type")
        if not isinstance(steps, int):
            raise TypeError("Invalid steps input type")
        if not isinstance(sampler, str):
            raise TypeError("Invalid sampler input type")
        if not isinstance(scheduler, str):
            raise TypeError("Invalid scheduler input type")
        return io.NodeOutput((cfg, steps, None, sampler, scheduler, scheduler == "ays", None, None),)

#-------------------------------------------------------------------------------#

class BasicParamsOutput(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-BasicParamsOutput',
            display_name='Params Output',
            category='SV Nodes/Input',
            inputs=[
            BpOutput.Input('packet'),
            ],
            outputs=[
            io.Float.Output(display_name='cfg'),
            io.Int.Output(display_name='steps'),
            io.Float.Output(display_name='denoise'),
            io.String.Output(display_name='sampler'),
            io.String.Output(display_name='scheduler'),
            io.Boolean.Output(display_name='ays'),
            SvSampler.Output(display_name='SAMPLER'),
            io.Float.Output(display_name='start'),
            io.Float.Output(display_name='end'),
            ]
        )

    
    @classmethod
    def execute(cls, packet) -> io.NodeOutput:
        if not isinstance(packet, tuple):
            raise TypeError("Invalid packet input type")
        if len(packet) != 8:
            raise ValueError("Invalid packet length")
        cfg = packet[0] or 8.0
        steps = packet[1] or 10
        denoise = packet[2] or 1.0
        sampler = packet[3] or comfy.samplers.SAMPLER_NAMES[0]
        sampler2 = comfy.samplers.sampler_object(sampler)
        scheduler = comfy.samplers.SCHEDULER_NAMES[0] if packet[4] in [None, "ays"] else packet[4]
        ays = packet[5] or False
        start = packet[6] or (1.0 - denoise)
        end = packet[7] or 1.0
        return cfg, steps, denoise, sampler, scheduler, ays, sampler2, start, end


#-------------------------------------------------------------------------------#

class SamplerNameToSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-SamplerNameToSampler',
            display_name='Sampler Converter',
            category='SV Nodes/Input',
            inputs=[
            io.Combo.Input('name', options=['euler', 'euler_ancestral', 'heun', 'dpm_2', 'dpm_2_ancestral']),
            ],
            outputs=[
            SvSampler.Output(display_name='sampler'),
            ]
        )

    
    @classmethod
    def execute(cls, name) -> io.NodeOutput:
        if not isinstance(name, str):
            raise TypeError("Invalid name input type")
        if name not in comfy.samplers.SAMPLER_NAMES:
            raise ValueError("Invalid name")
        return io.NodeOutput(comfy.samplers.sampler_object(name),)

#-------------------------------------------------------------------------------#

class StringSeparator(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-StringSeparator',
            display_name='String Separator',
            category='SV Nodes/Processing',
            inputs=[
            io.String.Input('text', force_input=True),
            io.String.Input('separator', default='\\n---\\n'),
            ],
            outputs=[
            io.String.Output(display_name='part1'),
            io.String.Output(display_name='part2'),
            ]
        )

    
    @classmethod
    def execute(cls, text, separator="\\n---\\n") -> io.NodeOutput:
        if not isinstance(text, str):
            raise TypeError("Invalid text input type")
        if not isinstance(separator, str):
            raise TypeError("Invalid separator input type")
        separator = separator.replace("\\n", "\n").replace("\\t", "\t")
        parts = text.split(separator, 1)
        return parts[0], parts[1] if len(parts) > 1 else ""


#-------------------------------------------------------------------------------#

class LoraSeparator(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-LoraSeparator',
            display_name='Lora Separator',
            category='SV Nodes/Processing',
            inputs=[
            io.String.Input('text', force_input=True),
            ],
            outputs=[
            io.String.Output(display_name='prompt'),
            io.String.Output(display_name='lora'),
            ]
        )

    
    @classmethod
    def execute(cls, text) -> io.NodeOutput:
        return separate_lora(text)


#-------------------------------------------------------------------------------#

class StringCombine(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-StringCombine',
            display_name='String Combine',
            category='SV Nodes/Processing',
            inputs=[
            io.String.Input('part1', force_input=True),
            io.String.Input('part2', force_input=True),
            io.String.Input('separator', default='\\n'),
            ],
            outputs=[
            io.String.Output(display_name='text'),
            ]
        )

    
    @classmethod
    def execute(cls, part1, part2, separator="\\n") -> io.NodeOutput:
        if not isinstance(part1, str) or not isinstance(part2, str):
            raise TypeError("Invalid part input type")
        if not isinstance(separator, str):
            raise TypeError("Invalid separator input type")
        separator = separator.replace("\\n", "\n").replace("\\t", "\t")
        return io.NodeOutput(part1 + separator + part2,)

#-------------------------------------------------------------------------------#

class LoadTextFile(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-LoadTextFile',
            display_name='Load Text File',
            category='SV Nodes/IO',
            inputs=[
            io.String.Input('path', default='', multiline=False),
            ],
            outputs=[
            io.String.Output(display_name='content'),
            io.Boolean.Output(display_name='success'),
            ]
        )

    
    @classmethod
    def fingerprint_inputs(cls, path):
        if os.path.exists(path):
            return os.path.getmtime(path)
        return ""
    
    @classmethod
    def execute(cls, path) -> io.NodeOutput:
        if not isinstance(path, str):
            raise TypeError("Invalid path input type")
        try:
            with open(path, "r", encoding="utf-8") as file:
                return io.NodeOutput(file.read(), True)
        except Exception as e:
            print(e)
            return io.NodeOutput("", False)

#-------------------------------------------------------------------------------#

class SaveTextFile(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-SaveTextFile',
            display_name='Save Text File',
            category='SV Nodes/IO',
            inputs=[
            io.String.Input('path', default='', multiline=False),
            io.String.Input('content', force_input=True),
            ],
            outputs=[
            io.Boolean.Output(display_name='success'),
            ],
            is_output_node=True
        )

    
    @classmethod
    def fingerprint_inputs(cls, path, content):
        return path + " " + content
    
    @classmethod
    def execute(cls, path, content) -> io.NodeOutput:
        if not isinstance(path, str):
            raise TypeError("Invalid path input type")
        if not isinstance(content, (str, int, float, bool)):
            raise TypeError("Invalid content input type")
        try:
            with open(path, "w") as file:
                file.write(str(content))
            return io.NodeOutput(True,)
        except:
            return io.NodeOutput(False,)

#-------------------------------------------------------------------------------#

class BooleanNot(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-BooleanNot',
            display_name='Boolean Not',
            category='SV Nodes/Logic',
            inputs=[
            io.Boolean.Input('value', force_input=True),
            ],
            outputs=[
            io.Boolean.Output(display_name='bool'),
            ]
        )

    
    @classmethod
    def execute(cls, value) -> io.NodeOutput:
        return io.NodeOutput(not value,)

#-------------------------------------------------------------------------------#

class MathAddInt(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-MathAddInt',
            display_name='Add Int',
            category='SV Nodes/Logic',
            inputs=[
            io.Int.Input('int', force_input=True),
            io.Int.Input('add', default=1, min=-sys.maxsize, max=sys.maxsize, step=1),
            ],
            outputs=[
            io.Int.Output(display_name='int'),
            ]
        )

    
    @classmethod
    def execute(cls, int, add) -> io.NodeOutput:
        return io.NodeOutput(int + add,)

#-------------------------------------------------------------------------------#

class MathCompare(io.ComfyNode):
    _match_template = io.MatchType.Template('MathCompare')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-MathCompare',
            display_name='Simple Compare',
            category='SV Nodes/Logic',
            inputs=[
            io.MatchType.Input('number', template=cls._match_template),
            io.Combo.Input('operator', options=['>', '<', '>=', '<=', '==', '!=']),
            io.Float.Input('other', default=0, min=-sys.float_info.max, max=sys.float_info.max, step=0.01),
            ],
            outputs=[
            io.Boolean.Output(display_name='bool'),
            ]
        )

    
    @classmethod
    def execute(cls, number, other, operator) -> io.NodeOutput:
        if operator == "<":
            return io.NodeOutput(number < other,)
        if operator == ">":
            return io.NodeOutput(number > other,)
        if operator == "<=":
            return io.NodeOutput(number <= other,)
        if operator == ">=":
            return io.NodeOutput(number >= other,)
        if operator == "==":
            return io.NodeOutput(number == other,)
        if operator == "!=":
            return io.NodeOutput(number != other,)
        return io.NodeOutput(False,)

#-------------------------------------------------------------------------------#

class EquationCompare(io.ComfyNode):
    _match_template = io.MatchType.Template('EquationCompare')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-EquationCompare',
            display_name='Equation Compare',
            category='SV Nodes/Logic',
            inputs=[
            io.String.Input('equation', multiline=False, default=''),
            io.MatchType.Input('a', template=cls._match_template, optional=True),
            io.MatchType.Input('b', template=cls._match_template, optional=True),
            ],
            outputs=[
            io.Boolean.Output(display_name='bool'),
            ]
        )

    
    @classmethod
    def execute(cls, equation, a=None, b=None) -> io.NodeOutput:
        equation = re.sub(r"\s+", "", equation)
        equation = re.sub(r"(?<=\d)a(?!nd)", "*" + str(a), equation)
        equation = re.sub(r"a(?!nd)", str(a), equation)
        equation = re.sub(r"(?<=\d)b", "*" + str(b), equation)
        equation = re.sub(r"b", str(b), equation)
        return io.NodeOutput(evaluateComparison(equation, 0),)

#-------------------------------------------------------------------------------#

class SigmaOneStep(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-SigmaOneStep',
            display_name='Sigmas One Step',
            category='SV Nodes/Sigmas',
            inputs=[
            Sigmas.Input('sigmas'),
            ],
            outputs=[
            Sigmas.Output(display_name='sigmas'),
            ]
        )

    
    @classmethod
    def execute(cls, sigmas) -> io.NodeOutput:
        lastSigma = sigmas[-1].item()
        return io.NodeOutput(torch.FloatTensor([lastSigma, 0]).cpu(),)

#-------------------------------------------------------------------------------#

class SigmaRange(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-SigmaRange',
            display_name='Sigma Range',
            category='SV Nodes/Sigmas',
            inputs=[
            Sigmas.Input('sigmas'),
            io.Int.Input('start', default=0, min=0, max=100),
            io.Int.Input('end', default=0, min=0, max=100),
            ],
            outputs=[
            Sigmas.Output(display_name='sigmas'),
            ]
        )

    
    @classmethod
    def execute(cls, sigmas, start, end) -> io.NodeOutput:
        return io.NodeOutput(sigmas[start:end + 1],)

#-------------------------------------------------------------------------------#

class SigmaContinue(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-SigmaContinue',
            display_name='Sigma Continue',
            category='SV Nodes/Sigmas',
            inputs=[
            Sigmas.Input('source'),
            Sigmas.Input('imitate'),
            io.Int.Input('steps', min=1, max=100, step=1, default=1),
            ],
            outputs=[
            Sigmas.Output(display_name='sigmas'),
            ]
        )

    
    @classmethod
    def execute(cls, source, imitate, steps) -> io.NodeOutput:
        if steps < 1 or len(source) < 1:
            return io.NodeOutput(torch.FloatTensor([]).cpu(),)
        lastSigma = source[-1].item()
        if lastSigma < 0.0001:
            return io.NodeOutput(torch.FloatTensor([]).cpu(),)
        return io.NodeOutput(torch.FloatTensor(calculate_sigma_range(imitate.tolist(), lastSigma, 0, steps)).cpu(),)

#-------------------------------------------------------------------------------#

class SigmaContinueLinear(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-SigmaContinueLinear',
            display_name='Sigma Linear',
            category='SV Nodes/Sigmas',
            inputs=[
            Sigmas.Input('source'),
            io.Int.Input('steps', min=1, max=100, step=1, default=1),
            ],
            outputs=[
            Sigmas.Output(display_name='sigmas'),
            ]
        )

    
    @classmethod
    def execute(cls, source, steps) -> io.NodeOutput:
        if steps < 1 or len(source) < 1:
            return io.NodeOutput(torch.FloatTensor([]).cpu(),)
        lastSigma = source[-1].item()
        if lastSigma < 0.0001:
            return io.NodeOutput(torch.FloatTensor([]).cpu(),)
        step = lastSigma / steps
        return io.NodeOutput(torch.FloatTensor([step * i for i in reversed(range(0, steps + 1))]).cpu(),)

#-------------------------------------------------------------------------------#

class SigmaRemap(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-SigmaRemap',
            display_name='Sigma Remap',
            category='SV Nodes/Sigmas',
            inputs=[
            Sigmas.Input('sigmas'),
            io.Float.Input('start', default=0, min=0, max=1, step=0.01),
            io.Float.Input('end', default=1, min=0, max=1, step=0.01),
            io.Int.Input('steps', min=1, max=100, step=1, default=1),
            ],
            outputs=[
            Sigmas.Output(display_name='sigmas'),
            ]
        )

    
    @classmethod
    def execute(cls, sigmas, start, end, steps) -> io.NodeOutput:
        return io.NodeOutput(torch.FloatTensor(calculate_sigma_range_percent(sigmas.tolist(), start, end, steps)).cpu(),)

#-------------------------------------------------------------------------------#

class SigmaRescale(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-SigmaRescale',
            display_name='Sigma Rescale',
            category='SV Nodes/Sigmas',
            inputs=[
            Sigmas.Input('sigmas'),
            io.Float.Input('scale', default=2, min=0, max=10, step=0.01),
            ],
            outputs=[
            Sigmas.Output(display_name='sigmas'),
            ]
        )

    
    @classmethod
    def execute(cls, sigmas, scale) -> io.NodeOutput:
        return io.NodeOutput(torch.FloatTensor([s * scale for s in sigmas.tolist()]).cpu(),)

#-------------------------------------------------------------------------------#

class SigmaConcat(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-SigmaConcat',
            display_name='Sigma Concat',
            category='SV Nodes/Sigmas',
            inputs=[
            Sigmas.Input('sigmas1'),
            Sigmas.Input('sigmas2'),
            ],
            outputs=[
            Sigmas.Output(display_name='sigmas'),
            ]
        )

    
    @classmethod
    def execute(cls, sigmas1, sigmas2) -> io.NodeOutput:
        list1 = sigmas1.tolist()
        list2 = sigmas2.tolist()
        if len(list1) < 1:
            return io.NodeOutput(torch.FloatTensor(list2).cpu(),)
        if len(list2) < 1:
            return io.NodeOutput(torch.FloatTensor(list1).cpu(),)
        if list1[-1] == list2[0]:
            list2 = list2[1:]
        if len(list2) < 1:
            return io.NodeOutput(torch.FloatTensor(list1).cpu(),)
        return io.NodeOutput(torch.FloatTensor(list1 + list2).cpu(),)

#-------------------------------------------------------------------------------#

class SigmaEmpty(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-SigmaEmpty',
            display_name='Sigma Empty',
            category='SV Nodes/Sigmas',
            inputs=[
            
            ],
            outputs=[
            Sigmas.Output(display_name='sigmas'),
            ]
        )

    
    @classmethod
    def execute(cls):
        return io.NodeOutput(torch.FloatTensor([]).cpu(),)

#-------------------------------------------------------------------------------#

class SigmaAsFloat(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-SigmaAsFloat',
            display_name='Sigma As Float',
            category='SV Nodes/Sigmas',
            inputs=[
            Sigmas.Input('sigmas'),
            ],
            outputs=[
            io.Float.Output(display_name='float'),
            ],
            is_deprecated=True
        )

    
    @classmethod
    def execute(cls, sigmas) -> io.NodeOutput:
        if len(sigmas) < 1:
            raise ValueError("Invalid sigmas length")
        return io.NodeOutput(sigmas[0].item(),)

#-------------------------------------------------------------------------------#

class SigmaStartEnd(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-SigmaStartEnd',
            display_name='Sigma Start/End',
            category='SV Nodes/Sigmas',
            inputs=[
            Sigmas.Input('sigmas'),
            ],
            outputs=[
            io.Float.Output(display_name='start'),
            io.Float.Output(display_name='end'),
            ]
        )

    
    @classmethod
    def execute(cls, sigmas) -> io.NodeOutput:
        if len(sigmas) < 1:
            raise ValueError("Invalid sigmas length")
        return io.NodeOutput(sigmas[0].item(), sigmas[-1].item())

#-------------------------------------------------------------------------------#

class SigmaLength(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-SigmaLength',
            display_name='Sigma Length',
            category='SV Nodes/Sigmas',
            inputs=[
            Sigmas.Input('sigmas'),
            ],
            outputs=[
            io.Int.Output(display_name='length'),
            ]
        )

    
    @classmethod
    def execute(cls, sigmas) -> io.NodeOutput:
        return io.NodeOutput(len(sigmas),)

#-------------------------------------------------------------------------------#

class SigmaStrength(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-SigmaStrength',
            display_name='Sigma Strength',
            category='SV Nodes/Sigmas',
            inputs=[
            io.Model.Input('model'),
            Sigmas.Input('sigmas'),
            ],
            outputs=[
            io.Float.Output(display_name='strength'),
            ]
        )

    
    @classmethod
    def execute(cls, model, sigmas) -> io.NodeOutput:
        sigma = (sigmas[0] - sigmas[-1]) / model.model.latent_format.scale_factor
        return io.NodeOutput(sigma.item(),)

#-------------------------------------------------------------------------------#

class SigmaReverse(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-SigmaReverse',
            display_name='Sigma Reverse',
            category='SV Nodes/Sigmas',
            inputs=[
            Sigmas.Input('sigmas'),
            ],
            outputs=[
            Sigmas.Output(display_name='sigmas'),
            ]
        )

    
    @classmethod
    def execute(cls, sigmas) -> io.NodeOutput:
        out = sigmas.tolist()
        if out[-1] == 0:
            out[-1] = out[-1] + 0.0001
        out.reverse()
        return io.NodeOutput(torch.FloatTensor(out).cpu(),)

#-------------------------------------------------------------------------------#

class NormalizeSamples(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-NormalizeSamples',
            display_name='Normalize Samples',
            category='SV Nodes/Sampling',
            inputs=[
            io.Latent.Input('latent'),
            ],
            outputs=[
            io.Latent.Output(display_name='normalized'),
            ]
        )

    
    @classmethod
    def execute(cls, latent) -> io.NodeOutput:
        samples = latent["samples"]
        out = latent.copy()
        out["samples"] = (samples - samples.mean()) / samples.std()
        return io.NodeOutput(out,)

#-------------------------------------------------------------------------------#

class ModelName(io.ComfyNode):
    _match_template = io.MatchType.Template('ModelName')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-ModelName',
            display_name='Model Name',
            category='SV Nodes/Input',
            inputs=[
            io.Combo.Input('model', options=['model_a.safetensors', 'model_b.safetensors']),
            ],
            outputs=[
            io.MatchType.Output(template=cls._match_template, display_name='model name'),
            ]
        )

    
    @classmethod
    def execute(cls, model) -> io.NodeOutput:
        if not isinstance(model, str):
            raise TypeError("Invalid model input type")
        return io.NodeOutput(model,)

#-------------------------------------------------------------------------------#

class PromptPlusModel(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-PromptPlusModel',
            display_name='Prompt + Model',
            category='SV Nodes/Input',
            inputs=[
            io.Combo.Input('model', options=['model_a.safetensors', 'model_b.safetensors']),
            io.String.Input('prompt', multiline=True),
            ],
            outputs=[
            PpmOutput.Output(display_name='packet'),
            ]
        )

    
    @classmethod
    def execute(cls, model, prompt) -> io.NodeOutput:
        if not isinstance(model, str):
            raise TypeError("Invalid model input type")
        if not isinstance(prompt, str):
            raise TypeError("Invalid prompt input type")
        return io.NodeOutput((model, prompt),)

#-------------------------------------------------------------------------------#

class PromptPlusModelOutput(io.ComfyNode):
    _match_template = io.MatchType.Template('PromptPlusModelOutput')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-PromptPlusModelOutput',
            display_name='P+M Output',
            category='SV Nodes/Input',
            inputs=[
            PpmOutput.Input('packet'),
            ],
            outputs=[
            io.MatchType.Output(template=cls._match_template, display_name='model name'),
            io.String.Output(display_name='prompt'),
            ]
        )

    
    @classmethod
    def execute(cls, packet) -> io.NodeOutput:
        if not isinstance(packet, tuple):
            raise TypeError("Invalid output input type")
        if len(packet) != 2:
            raise ValueError("Invalid output length")
        return packet


#-------------------------------------------------------------------------------#

class InputSelect(io.ComfyNode):
    _match_template = io.MatchType.Template('InputSelect')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-InputSelect',
            display_name='Input Select',
            category='SV Nodes/Flow',
            inputs=[
            io.Int.Input('select', min=1, max=5, step=1, default=1),
            io.MatchType.Input('_1_', template=cls._match_template, lazy=True, optional=True),
            io.MatchType.Input('_2_', template=cls._match_template, lazy=True, optional=True),
            io.MatchType.Input('_3_', template=cls._match_template, lazy=True, optional=True),
            io.MatchType.Input('_4_', template=cls._match_template, lazy=True, optional=True),
            io.MatchType.Input('_5_', template=cls._match_template, lazy=True, optional=True),
            ],
            outputs=[
            io.MatchType.Output(template=cls._match_template, display_name='out'),
            ]
        )

    
    @classmethod
    def check_lazy_status(cls, select, **kwargs):
        if select == 1:
            return ["_1_"]
        if select == 2:
            return ["_2_"]
        if select == 3:
            return ["_3_"]
        if select == 4:
            return ["_4_"]
        if select == 5:
            return ["_5_"]
        return []
    
    @classmethod
    def execute(cls, select, _1_=None, _2_=None, _3_=None, _4_=None, _5_=None) -> io.NodeOutput:
        if select == 1:
            return io.NodeOutput(_1_,)
        if select == 2:
            return io.NodeOutput(_2_,)
        if select == 3:
            return io.NodeOutput(_3_,)
        if select == 4:
            return io.NodeOutput(_4_,)
        if select == 5:
            return io.NodeOutput(_5_,)
        return io.NodeOutput(None,)

#-------------------------------------------------------------------------------#

class InputSelectBoolean(io.ComfyNode):
    _match_template = io.MatchType.Template('InputSelectBoolean')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-InputSelectBoolean',
            display_name='Boolean Select',
            category='SV Nodes/Flow',
            inputs=[
            io.Boolean.Input('select'),
            io.MatchType.Input('on', template=cls._match_template, lazy=True, optional=True),
            io.MatchType.Input('off', template=cls._match_template, lazy=True, optional=True),
            ],
            outputs=[
            io.MatchType.Output(template=cls._match_template, display_name='out'),
            ]
        )

    
    @classmethod
    def check_lazy_status(cls, select, **kwargs):
        if select:
            return ["on"]
        return ["off"]
    
    @classmethod
    def execute(cls, select, on=None, off=None) -> io.NodeOutput:
        if select:
            return io.NodeOutput(on,)
        return io.NodeOutput(off,)

#-------------------------------------------------------------------------------#

class InputSelectCompare(io.ComfyNode):
    _match_template = io.MatchType.Template('InputSelectCompare')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-InputSelectCompare',
            display_name='Comparison Select',
            category='SV Nodes/Flow',
            inputs=[
            io.String.Input('equation', multiline=False, default=''),
            io.MatchType.Input('true', template=cls._match_template, lazy=True, optional=True),
            io.MatchType.Input('false', template=cls._match_template, lazy=True, optional=True),
            io.MatchType.Input('x', template=cls._match_template, optional=True),
            ],
            outputs=[
            io.MatchType.Output(template=cls._match_template, display_name='out'),
            ]
        )

    
    @classmethod
    def check_lazy_status(cls, x, equation, **kwargs):
        if evaluateComparison(equation, x):
            return ["true"]
        return ["false"]
    
    @classmethod
    def execute(cls, x, equation: str, true=None, false=None) -> io.NodeOutput:
        if evaluateComparison(equation, x):
            return io.NodeOutput(true,)
        return io.NodeOutput(false,)
def evaluateComparison(op: str, var):
    while "(" in op or ")" in op:
        op = re.sub(r"\([^()]+\)", lambda x : str(evaluateComparison(x.group(0)[1:-1], var)), op)
    op = re.sub(r"\s+", "", op.lower())
    op = re.sub(r"(?<=\d)x(?!or)", "*" + str(var), op)
    op = re.sub(r"x(?!or)", str(var), op)
    
    operators = ["and", "or", "xor"]
    operators = "|".join(operators)
    parts = re.sub(f"({operators})", r"<|>\1<|>", op).split("<|>")
    parts[0] = evaluatePart(parts[0])
    for i in range(1, len(parts), 2):
        parts[i+1] = evaluatePart(parts[i+1])
        p = parts[i]
        if parts[i-1] != True and parts[i-1] != False:
            raise SyntaxError(f"operand {parts[i-1]} of {p} wasn't a boolean")
        if parts[i+1] != True and parts[i+1] != False:
            raise SyntaxError(f"operand {parts[i+1]} of {p} wasn't a boolean")
        if p == "and":
            parts[i+1] = parts[i-1] and parts[i+1]
        elif p == "or":
            parts[i+1] = parts[i-1] or parts[i+1]
        elif p == "xor":
            parts[i+1] = parts[i-1] != parts[i+1]
    return parts[-1]

def evaluatePart(part: str):
    part = part.replace("==", "=")
    operators = ["=", "!=", "<=", ">=", "<", ">"]
    operators = "|".join(operators)
    parts = re.sub(f"({operators})", r"<|>\1<|>", part).split("<|>")
    
    if len(parts) == 1:
        return evaluateSubPart(parts[0])
    parts[0] = evaluateSubPart(parts[0])
    for i in range(1, len(parts), 2):
        parts[i+1] = evaluateSubPart(parts[i+1])
        p = parts[i]
        if p == "=" and not (parts[i-1] == parts[i+1]):
            return False
        if p == "!=" and not (parts[i-1] != parts[i+1]):
            return False
        if p == "<" and not (parts[i-1] < parts[i+1]):
            return False
        if p == ">" and not (parts[i-1] > parts[i+1]):
            return False
        if p == "<=" and not (parts[i-1] <= parts[i+1]):
            return False
        if p == ">=" and not (parts[i-1] >= parts[i+1]):
            return False
    return True

def evaluateSubPart(part: str):
    if part.lower() == "true":
        return True
    if part.lower() == "false":
        return False
    if part.lower() == "none":
        return None
    if re.sub(r"^[+-]", "", part).isdigit():
        return int(part)
    if re.sub(r"^[+-]", "", part).replace('.', '', 1).isdigit():
        return float(part)
    return parseCurve(part)(0)


#-------------------------------------------------------------------------------#

class FlowBlocker(io.ComfyNode):
    _match_template = io.MatchType.Template('FlowBlocker')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-FlowBlocker',
            display_name='Blocker',
            category='SV Nodes/Flow',
            inputs=[
            io.MatchType.Input('input', template=cls._match_template, lazy=True),
            io.Boolean.Input('block'),
            ],
            outputs=[
            io.MatchType.Output(template=cls._match_template, display_name='output'),
            ]
        )

    
    @classmethod
    def check_lazy_status(cls, block, input=None):
        if not block:
            return ["input"]
        return []

    @classmethod
    def execute(cls, input, block) -> io.NodeOutput:
        if block:
            return io.NodeOutput(ExecutionBlocker(None),)
        return io.NodeOutput(input,)

#-------------------------------------------------------------------------------#

class FlowGate(io.ComfyNode):
    _match_template = io.MatchType.Template('FlowGate')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-FlowGate',
            display_name='Gate',
            category='SV Nodes/Flow',
            inputs=[
            io.MatchType.Input('input', template=cls._match_template, lazy=True),
            io.Boolean.Input('open'),
            ],
            outputs=[
            io.MatchType.Output(template=cls._match_template, display_name='output'),
            ]
        )

    
    @classmethod
    def check_lazy_status(cls, open, input=None):
        if open:
            return ["input"]
        return []

    @classmethod
    def execute(cls, input, open) -> io.NodeOutput:
        if not open:
            return io.NodeOutput(ExecutionBlocker(None),)
        return io.NodeOutput(input,)

#-------------------------------------------------------------------------------#

class FlowGateMulti(io.ComfyNode):
    _match_template = io.MatchType.Template('FlowGateMulti')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-FlowGateMulti',
            display_name='Multi Gate',
            category='SV Nodes/Flow',
            inputs=[
            io.MatchType.Input('input', template=cls._match_template, lazy=True),
            io.Boolean.Input('open 1'),
            io.Boolean.Input('open 2'),
            io.Boolean.Input('open 3'),
            io.Boolean.Input('open 4'),
            io.Boolean.Input('open 5'),
            ],
            outputs=[
            io.MatchType.Output(template=cls._match_template, display_name='1'),
            io.MatchType.Output(template=cls._match_template, display_name='2'),
            io.MatchType.Output(template=cls._match_template, display_name='3'),
            io.MatchType.Output(template=cls._match_template, display_name='4'),
            io.MatchType.Output(template=cls._match_template, display_name='5'),
            ]
        )

    
    @classmethod
    def check_lazy_status(cls, **kwargs):
        if kwargs.get("open 1") or kwargs.get("open 2") or kwargs.get("open 3") or kwargs.get("open 4") or kwargs.get("open 5"):
            return ["input"]
        return []

    @classmethod
    def execute(cls, input, **kwargs) -> io.NodeOutput:
        block = ExecutionBlocker(None)
        out1 = input if kwargs.get("open 1", False) else block
        out2 = input if kwargs.get("open 2", False) else block
        out3 = input if kwargs.get("open 3", False) else block
        out4 = input if kwargs.get("open 4", False) else block
        out5 = input if kwargs.get("open 5", False) else block
        return out1, out2, out3, out4, out5


#-------------------------------------------------------------------------------#

class IfBranch(io.ComfyNode):
    _match_template = io.MatchType.Template('IfBranch')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-IfBranch',
            display_name='If Branch',
            category='SV Nodes/Flow',
            inputs=[
            io.MatchType.Input('input', template=cls._match_template),
            io.Boolean.Input('condition'),
            ],
            outputs=[
            io.MatchType.Output(template=cls._match_template, display_name='if'),
            io.MatchType.Output(template=cls._match_template, display_name='else'),
            ]
        )

    
    @classmethod
    def execute(cls, input, condition) -> io.NodeOutput:
        if condition:
            return io.NodeOutput(input, ExecutionBlocker(None))
        return io.NodeOutput(ExecutionBlocker(None), input)

#-------------------------------------------------------------------------------#

class ForLoopOpen(io.ComfyNode):
    _match_template = io.MatchType.Template('ForLoopOpen')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-ForLoopOpen',
            display_name='For Loop Open',
            category='InversionDemo Nodes/Flow',
            inputs=[
            io.Int.Input('remaining', default=1, min=0, max=100000, step=1),
            io.MatchType.Input('initial_value1', template=cls._match_template, optional=True),
            io.MatchType.Input('initial_value2', template=cls._match_template, optional=True),
            io.MatchType.Input('initial_value3', template=cls._match_template, optional=True),
            ],
            outputs=[
            FlowControl.Output(display_name='flow_control'),
            io.Int.Output(display_name='remaining'),
            io.MatchType.Output(template=cls._match_template, display_name='value1'),
            io.MatchType.Output(template=cls._match_template, display_name='value2'),
            io.MatchType.Output(template=cls._match_template, display_name='value3'),
            ],
            enable_expand=True
        )

    
    @classmethod
    def execute(cls, remaining, **kwargs) -> io.NodeOutput:
        graph = GraphBuilder()
        if "initial_value0" in kwargs:
            remaining = kwargs["initial_value0"]
        while_open = graph.node("SV-WhileLoopOpen", condition=remaining, initial_value0=remaining, **{("initial_value%d" % i): kwargs.get("initial_value%d" % i, None) for i in range(1, NUM_FLOW_SOCKETS)})
        outputs = [kwargs.get("initial_value%d" % i, None) for i in range(1, NUM_FLOW_SOCKETS)]
        return {
            "result": tuple(["stub", remaining] + outputs),
            "expand": graph.finalize(),
        }


#-------------------------------------------------------------------------------#

class ForLoopClose(io.ComfyNode):
    _match_template = io.MatchType.Template('ForLoopClose')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-ForLoopClose',
            display_name='For Loop Close',
            category='InversionDemo Nodes/Flow',
            inputs=[
            FlowControl.Input('flow_control', raw_link=True),
            io.MatchType.Input('initial_value1', template=cls._match_template, raw_link=True, optional=True),
            io.MatchType.Input('initial_value2', template=cls._match_template, raw_link=True, optional=True),
            io.MatchType.Input('initial_value3', template=cls._match_template, raw_link=True, optional=True),
            ],
            outputs=[
            io.MatchType.Output(template=cls._match_template, display_name='value1'),
            io.MatchType.Output(template=cls._match_template, display_name='value2'),
            io.MatchType.Output(template=cls._match_template, display_name='value3'),
            ],
            enable_expand=True
        )

    
    @classmethod
    def execute(cls, flow_control, **kwargs) -> io.NodeOutput:
        graph = GraphBuilder()
        while_open = flow_control[0]
        # TODO - Requires WAS-ns. Will definitely want to solve before merging
        sub = graph.node("SV-IntMathOperation", operation="subtract", a=[while_open,1], b=1)
        cond = graph.node("SV-ToBoolNode", value=sub.out(0))
        input_values = {("initial_value%d" % i): kwargs.get("initial_value%d" % i, None) for i in range(1, NUM_FLOW_SOCKETS)}
        while_close = graph.node("SV-WhileLoopClose",
                flow_control=flow_control,
                condition=cond.out(0),
                initial_value0=sub.out(0),
                **input_values)
        return {
            "result": tuple([while_close.out(i) for i in range(1, NUM_FLOW_SOCKETS)]),
            "expand": graph.finalize(),
        }


#-------------------------------------------------------------------------------#

class WhileLoopOpen(io.ComfyNode):
    _match_template = io.MatchType.Template('WhileLoopOpen')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-WhileLoopOpen',
            display_name='While Loop Open',
            category='InversionDemo Nodes/Flow',
            inputs=[
            io.Boolean.Input('condition', default=True),
            io.MatchType.Input('initial_value0', template=cls._match_template, optional=True),
            io.MatchType.Input('initial_value1', template=cls._match_template, optional=True),
            io.MatchType.Input('initial_value2', template=cls._match_template, optional=True),
            io.MatchType.Input('initial_value3', template=cls._match_template, optional=True),
            ],
            outputs=[
            FlowControl.Output(display_name='FLOW_CONTROL'),
            io.MatchType.Output(template=cls._match_template, display_name='value0'),
            io.MatchType.Output(template=cls._match_template, display_name='value1'),
            io.MatchType.Output(template=cls._match_template, display_name='value2'),
            io.MatchType.Output(template=cls._match_template, display_name='value3'),
            ]
        )

    
    @classmethod
    def execute(cls, condition, **kwargs) -> io.NodeOutput:
        values = []
        for i in range(NUM_FLOW_SOCKETS):
            values.append(kwargs.get("initial_value%d" % i, None))
        return tuple(["stub"] + values)


#-------------------------------------------------------------------------------#

class WhileLoopClose(io.ComfyNode):
    _match_template = io.MatchType.Template('WhileLoopClose')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-WhileLoopClose',
            display_name='While Loop Close',
            category='InversionDemo Nodes/Flow',
            inputs=[
            FlowControl.Input('flow_control', raw_link=True),
            io.Boolean.Input('condition', force_input=True),
            io.MatchType.Input('initial_value0', template=cls._match_template, optional=True),
            io.MatchType.Input('initial_value1', template=cls._match_template, optional=True),
            io.MatchType.Input('initial_value2', template=cls._match_template, optional=True),
            io.MatchType.Input('initial_value3', template=cls._match_template, optional=True),
            ],
            outputs=[
            io.MatchType.Output(template=cls._match_template, display_name='value0'),
            io.MatchType.Output(template=cls._match_template, display_name='value1'),
            io.MatchType.Output(template=cls._match_template, display_name='value2'),
            io.MatchType.Output(template=cls._match_template, display_name='value3'),
            ],
            hidden=[
            io.Hidden.dynprompt,
            io.Hidden.unique_id,
            ],
            enable_expand=True
        )

    
    @classmethod
    def explore_dependencies(cls, node_id, dynprompt, upstream):
        node_info = dynprompt.get_node(node_id)
        if "inputs" not in node_info:
            return
        for k, v in node_info["inputs"].items():
            if is_link(v):
                parent_id = v[0]
                if parent_id not in upstream:
                    upstream[parent_id] = []
                    cls.explore_dependencies(parent_id, dynprompt, upstream)
                upstream[parent_id].append(node_id)

    @classmethod
    def collect_contained(cls, node_id, upstream, contained):
        if node_id not in upstream:
            return
        for child_id in upstream[node_id]:
            if child_id not in contained:
                contained[child_id] = True
                cls.collect_contained(child_id, upstream, contained)


    @classmethod
    def execute(cls, flow_control, condition, dynprompt=None, unique_id=None, **kwargs) -> io.NodeOutput:
        if not condition:
            # We're done with the loop
            values = []
            for i in range(NUM_FLOW_SOCKETS):
                values.append(kwargs.get("initial_value%d" % i, None))
            return io.NodeOutput(*values)

        # We want to loop
        this_node = cls.hidden.dynprompt.get_node(cls.hidden.unique_id)
        upstream = {}
        # Get the list of all nodes between the open and close nodes
        cls.explore_dependencies(cls.hidden.unique_id, cls.hidden.dynprompt, upstream)

        contained = {}
        open_node = flow_control[0]
        cls.collect_contained(open_node, upstream, contained)
        contained[cls.hidden.unique_id] = True
        contained[open_node] = True

        # We'll use the default prefix, but to avoid having node names grow exponentially in size,
        # we'll use "Recurse" for the name of the recursively-generated copy of this node.
        graph = GraphBuilder()
        for node_id in contained:
            original_node = cls.hidden.dynprompt.get_node(node_id)
            node = graph.node(original_node["class_type"], "Recurse" if node_id == cls.hidden.unique_id else node_id)
            node.set_override_display_id(node_id)
        for node_id in contained:
            original_node = cls.hidden.dynprompt.get_node(node_id)
            node = graph.lookup_node("Recurse" if node_id == cls.hidden.unique_id else node_id)
            for k, v in original_node["inputs"].items():
                if is_link(v) and v[0] in contained:
                    parent = graph.lookup_node(v[0])
                    node.set_input(k, parent.out(v[1]))
                else:
                    node.set_input(k, v)
        new_open = graph.lookup_node(open_node)
        for i in range(NUM_FLOW_SOCKETS):
            key = "initial_value%d" % i
            new_open.set_input(key, kwargs.get(key, None))
        my_clone = graph.lookup_node("Recurse")
        result = map(lambda x: my_clone.out(x), range(NUM_FLOW_SOCKETS))
        return io.NodeOutput(tuple(result), expand=graph.finalize(),
        )


#-------------------------------------------------------------------------------#

class IntMathOperation(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-IntMathOperation',
            display_name='Int Math Operation',
            category='InversionDemo Nodes/Logic',
            inputs=[
            io.Int.Input('a', default=0, min=-sys.maxsize, max=sys.maxsize, step=1),
            io.Int.Input('b', default=0, min=-sys.maxsize, max=sys.maxsize, step=1),
            io.Combo.Input('operation', options=['add', 'subtract', 'multiply', 'divide', 'modulo', 'power']),
            ],
            outputs=[
            io.Int.Output(),
            ]
        )

    
    @classmethod
    def execute(cls, a, b, operation) -> io.NodeOutput:
        if operation == "add":
            return io.NodeOutput(a + b,)
        elif operation == "subtract":
            return io.NodeOutput(a - b,)
        elif operation == "multiply":
            return io.NodeOutput(a * b,)
        elif operation == "divide":
            return io.NodeOutput(a // b,)
        elif operation == "modulo":
            return io.NodeOutput(a % b,)
        elif operation == "power":
            return io.NodeOutput(a ** b,)

#-------------------------------------------------------------------------------#

class ToBoolNode(io.ComfyNode):
    _match_template = io.MatchType.Template('ToBoolNode')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-ToBoolNode',
            display_name='To Bool',
            category='InversionDemo Nodes/Logic',
            inputs=[
            io.MatchType.Input('value', template=cls._match_template),
            io.Boolean.Input('invert', default=False, optional=True),
            ],
            outputs=[
            io.Boolean.Output(),
            ]
        )

    
    @classmethod
    def execute(cls, value, invert = False) -> io.NodeOutput:
        if isinstance(value, torch.Tensor):
            if value.max().item() == 0 and value.min().item() == 0:
                result = False
            else:
                result = True
        else:
            try:
                result = bool(value)
            except:
                # Can't convert it? Well then it's something or other. I dunno, I'm not a Python programmer.
                result = True

        if invert:
            result = not result

        return io.NodeOutput(result,)

#-------------------------------------------------------------------------------#

class AccumulateNode(io.ComfyNode):
    _match_template = io.MatchType.Template('AccumulateNode')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-AccumulateNode',
            display_name='Accumulate',
            category='InversionDemo Nodes/Lists',
            inputs=[
            io.MatchType.Input('to_add', template=cls._match_template),
            Accumulation.Input('accumulation', optional=True),
            ],
            outputs=[
            Accumulation.Output(),
            ]
        )

    
    @classmethod
    def execute(cls, to_add, accumulation = None) -> io.NodeOutput:
        if accumulation is None:
            value = [to_add]
        else:
            value = accumulation["accum"] + [to_add]
        return io.NodeOutput({"accum": value},)

#-------------------------------------------------------------------------------#

class AccumulationHeadNode(io.ComfyNode):
    _match_template = io.MatchType.Template('AccumulationHeadNode')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-AccumulationHeadNode',
            display_name='Accumulation Head',
            category='InversionDemo Nodes/Lists',
            inputs=[
            Accumulation.Input('accumulation'),
            ],
            outputs=[
            Accumulation.Output(),
            io.MatchType.Output(template=cls._match_template),
            ]
        )

    
    @classmethod
    def execute(cls, accumulation) -> io.NodeOutput:
        accum = accumulation["accum"]
        if len(accum) == 0:
            return io.NodeOutput(accumulation, None)
        else:
            return io.NodeOutput({"accum": accum[1:]}, accum[0])

#-------------------------------------------------------------------------------#

class AccumulationTailNode(io.ComfyNode):
    _match_template = io.MatchType.Template('AccumulationTailNode')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-AccumulationTailNode',
            display_name='Accumulation Tail',
            category='InversionDemo Nodes/Lists',
            inputs=[
            Accumulation.Input('accumulation'),
            ],
            outputs=[
            Accumulation.Output(),
            io.MatchType.Output(template=cls._match_template),
            ]
        )

    
    @classmethod
    def execute(cls, accumulation) -> io.NodeOutput:
        accum = accumulation["accum"]
        if len(accum) == 0:
            return io.NodeOutput(None, accumulation)
        else:
            return io.NodeOutput({"accum": accum[:-1]}, accum[-1])

#-------------------------------------------------------------------------------#

class AccumulationToListNode(io.ComfyNode):
    _match_template = io.MatchType.Template('AccumulationToListNode')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-AccumulationToListNode',
            display_name='Accumulation To List',
            category='InversionDemo Nodes/Lists',
            inputs=[
            Accumulation.Input('accumulation'),
            ],
            outputs=[
            io.MatchType.Output(template=cls._match_template),
            ]
        )

    
    OUTPUT_IS_LIST = (True,)

    @classmethod
    def execute(cls, accumulation) -> io.NodeOutput:
        return io.NodeOutput(accumulation["accum"],)

#-------------------------------------------------------------------------------#

class ListToAccumulationNode(io.ComfyNode):
    _match_template = io.MatchType.Template('ListToAccumulationNode')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-ListToAccumulationNode',
            display_name='List To Accumulation',
            category='InversionDemo Nodes/Lists',
            inputs=[
            io.MatchType.Input('list', template=cls._match_template),
            ],
            outputs=[
            Accumulation.Output(),
            ]
        )

    
    INPUT_IS_LIST = (True,)

    @classmethod
    def execute(cls, list) -> io.NodeOutput:
        return io.NodeOutput({"accum": list},)

#-------------------------------------------------------------------------------#

class AccumulationGetLengthNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-AccumulationGetLengthNode',
            display_name='Accumulation Get Length',
            category='InversionDemo Nodes/Lists',
            inputs=[
            Accumulation.Input('accumulation'),
            ],
            outputs=[
            io.Int.Output(),
            ]
        )

    
    @classmethod
    def execute(cls, accumulation) -> io.NodeOutput:
        return io.NodeOutput(len(accumulation['accum']),)

#-------------------------------------------------------------------------------#

class AccumulationGetItemNode(io.ComfyNode):
    _match_template = io.MatchType.Template('AccumulationGetItemNode')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-AccumulationGetItemNode',
            display_name='Accumulation Get Item',
            category='InversionDemo Nodes/Lists',
            inputs=[
            Accumulation.Input('accumulation'),
            io.Int.Input('index', default=0, step=1),
            ],
            outputs=[
            io.MatchType.Output(template=cls._match_template),
            ]
        )

    
    @classmethod
    def execute(cls, accumulation, index) -> io.NodeOutput:
        return io.NodeOutput(accumulation['accum'][index],)

#-------------------------------------------------------------------------------#

class AccumulationSetItemNode(io.ComfyNode):
    _match_template = io.MatchType.Template('AccumulationSetItemNode')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-AccumulationSetItemNode',
            display_name='Accumulation Set Item',
            category='InversionDemo Nodes/Lists',
            inputs=[
            Accumulation.Input('accumulation'),
            io.Int.Input('index', default=0, step=1),
            io.MatchType.Input('value', template=cls._match_template),
            ],
            outputs=[
            Accumulation.Output(),
            ]
        )

    
    @classmethod
    def execute(cls, accumulation, index, value) -> io.NodeOutput:
        new_accum = accumulation['accum'][:]
        new_accum[index] = value
        return io.NodeOutput({"accum": new_accum},)

#-------------------------------------------------------------------------------#

class HashObject(io.ComfyNode):
    _match_template = io.MatchType.Template('HashObject')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-HashObject',
            display_name='Hash Object',
            category='SV Nodes/Flow',
            inputs=[
            io.MatchType.Input('obj', template=cls._match_template),
            ],
            outputs=[
            io.String.Output(display_name='hash'),
            ]
        )

    def __init__(s):
        pass
    
    
    @classmethod
    
    
    def execute(cls, obj) -> io.NodeOutput:
        return io.NodeOutput(hash_item(obj),)
def hash_item(item):
    if isinstance(item, (tuple, list)):
        temp = ""
        for item in item:
            temp += hash_item(item)
        return hashlib.md5(temp.encode()).hexdigest()
    elif isinstance(item, (str, int, float, bool, type(None))):
        return hashlib.md5(str(item).encode()).hexdigest()
    elif hasattr(item, "model") and hasattr(item.model, "state_dict"):
        # Hashing a model takes too long to be worth it
        return ""
    elif isinstance(item, (list, tuple, dict)):
        return hashlib.md5(json.dumps(item, sort_keys=True).encode()).hexdigest()
    else:
        if hasattr(item, "__dict__"):
            return hashlib.md5(json.dumps(item.__dict__, sort_keys=True).encode()).hexdigest()
        else:
            return hashlib.md5(repr(item).encode()).hexdigest()


#-------------------------------------------------------------------------------#

class HashItems(io.ComfyNode):
    _match_template = io.MatchType.Template('HashItems')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-HashItems',
            display_name='Hash Items',
            category='SV Nodes/Flow',
            inputs=[
            io.MatchType.Input('_1_', template=cls._match_template, optional=True),
            io.MatchType.Input('_2_', template=cls._match_template, optional=True),
            io.MatchType.Input('_3_', template=cls._match_template, optional=True),
            io.MatchType.Input('_4_', template=cls._match_template, optional=True),
            io.MatchType.Input('_5_', template=cls._match_template, optional=True),
            io.MatchType.Input('_6_', template=cls._match_template, optional=True),
            io.MatchType.Input('_7_', template=cls._match_template, optional=True),
            io.MatchType.Input('_8_', template=cls._match_template, optional=True),
            io.MatchType.Input('_9_', template=cls._match_template, optional=True),
            io.MatchType.Input('_10_', template=cls._match_template, optional=True),
            ],
            outputs=[
            io.String.Output(display_name='hash'),
            ]
        )

    def __init__(s):
        pass
    
    
    @classmethod
    
    
    def execute(cls, _1_=None, _2_=None, _3_=None, _4_=None, _5_=None, _6_=None, _7_=None, _8_=None, _9_=None, _10_=None) -> io.NodeOutput:
        args = [_1_, _2_, _3_, _4_, _5_, _6_, _7_, _8_, _9_, _10_]
        return io.NodeOutput(hash_item(args),)

#-------------------------------------------------------------------------------#

class HashModel(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-HashModel',
            display_name='Hash Model',
            category='SV Nodes/Flow',
            inputs=[
            io.Model.Input('model'),
            ],
            outputs=[
            io.String.Output(display_name='hash'),
            ]
        )

    
    @classmethod
    def execute(cls, model) -> io.NodeOutput:
        return io.NodeOutput(hashlib.md5(str(model.model.state_dict()).encode()).hexdigest(),)

#-------------------------------------------------------------------------------#

class CacheObject(io.ComfyNode):
    _match_template = io.MatchType.Template('CacheObject')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-CacheObject',
            display_name='Cache Object',
            category='SV Nodes/Flow',
            inputs=[
            io.MatchType.Input('any', template=cls._match_template, lazy=True),
            io.String.Input('hash', force_input=True),
            io.String.Input('id'),
            ],
            outputs=[
            io.MatchType.Output(template=cls._match_template, display_name='cache'),
            ],
            not_idempotent=True
        )

    _objs: dict = {}
    hashes = {}
    obj = None
    
    
    @classmethod
    def check_lazy_status(cls, hash, id, **kwargs):
        if CacheObject.hashes.get(id) is not None and CacheObject.hashes.get(id) == hash:
            return []
        return ["any"]
    
    @classmethod
    def execute(cls, any, hash, id) -> io.NodeOutput:
        if CacheObject._objs.get(cls.hidden.unique_id) is not None and CacheObject.hashes.get(id) is not None and CacheObject.hashes.get(id) == hash:
            return io.NodeOutput(CacheObject._objs.get(cls.hidden.unique_id),)
        CacheObject.hashes[id] = hash
        CacheObject._objs[cls.hidden.unique_id] = any
        return io.NodeOutput(any,)

#-------------------------------------------------------------------------------#

class ManualCache(io.ComfyNode):
    _match_template = io.MatchType.Template('ManualCache')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-ManualCache',
            display_name='Manual Cache',
            category='SV Nodes/Flow',
            inputs=[
            io.MatchType.Input('any', template=cls._match_template, lazy=True),
            io.Boolean.Input('enable', force_input=True),
            io.String.Input('id'),
            ],
            outputs=[
            io.MatchType.Output(template=cls._match_template, display_name='cache'),
            ],
            not_idempotent=True
        )

    _objs: dict = {}
    hasValue = {}
    obj = None
    
    
    @classmethod
    def check_lazy_status(cls, enable, id, **kwargs):
        if (id is None or id == "") and enable:
            return []
        if enable and ManualCache.hasValue.get(id):
            return []
        return ["any"]
    
    @classmethod
    def execute(cls, any, enable, id) -> io.NodeOutput:
        if enable and cls.hidden.unique_id in ManualCache._objs:
            return io.NodeOutput(ManualCache._objs.get(cls.hidden.unique_id),)
        if any is None:
            del ManualCache.hasValue[id]
        ManualCache._objs[cls.hidden.unique_id] = any
        ManualCache.hasValue[id] = True
        return io.NodeOutput(any,)

#-------------------------------------------------------------------------------#

class ClearCustomCaches(io.ComfyNode):
    _id = None
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-ClearCustomCaches',
            display_name='Clear Caches',
            category='SV Nodes/Flow',
            inputs=[
            
            ],
            outputs=[
            
            ],
            is_output_node=True,
            not_idempotent=True
        )

    _id = None
    id = None
    
    
    @classmethod
    
    
    def execute(cls):
        if ClearCustomCaches._id is None:
            ClearCustomCaches._id = round(time.time() * 1000)
            VariableSet.storage = {}
            CacheObject.hashes = {}
            ManualCache.hasValue = {}
        return io.NodeOutput(None,)

#-------------------------------------------------------------------------------#

class FlowNode(io.ComfyNode):
    _match_template = io.MatchType.Template('FlowNode')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-FlowNode',
            display_name='Flow Node',
            category='SV Nodes/Flow',
            inputs=[
            io.MatchType.Input('_1_', template=cls._match_template, optional=True),
            io.MatchType.Input('_2_', template=cls._match_template, optional=True),
            io.MatchType.Input('_3_', template=cls._match_template, optional=True),
            io.MatchType.Input('_4_', template=cls._match_template, optional=True),
            io.MatchType.Input('_5_', template=cls._match_template, optional=True),
            ],
            outputs=[
            io.MatchType.Output(template=cls._match_template, display_name='_1_'),
            io.MatchType.Output(template=cls._match_template, display_name='_2_'),
            io.MatchType.Output(template=cls._match_template, display_name='_3_'),
            io.MatchType.Output(template=cls._match_template, display_name='_4_'),
            io.MatchType.Output(template=cls._match_template, display_name='_5_'),
            ]
        )

    
    @classmethod
    def execute(cls, _1_=None, _2_=None, _3_=None, _4_=None, _5_=None) -> io.NodeOutput:
        return io.NodeOutput(_1_, _2_, _3_, _4_, _5_)

#-------------------------------------------------------------------------------#

class FlowPipeInput(io.ComfyNode):
    _match_template = io.MatchType.Template('FlowPipeInput')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-FlowPipeInput',
            display_name='Pipe In',
            category='SV Nodes/Pipes',
            inputs=[
            io.Int.Input('start', min=1, max=100, step=1, default=1),
            SvPipe.Input('pipe', optional=True),
            io.MatchType.Input('_1_', template=cls._match_template, optional=True),
            io.MatchType.Input('_2_', template=cls._match_template, optional=True),
            io.MatchType.Input('_3_', template=cls._match_template, optional=True),
            io.MatchType.Input('_4_', template=cls._match_template, optional=True),
            io.MatchType.Input('_5_', template=cls._match_template, optional=True),
            ],
            outputs=[
            SvPipe.Output(display_name='pipe'),
            ]
        )

    
    @classmethod
    def execute(cls, start, pipe=None, **kwargs) -> io.NodeOutput:
        if not isinstance(pipe, (dict, type(None))):
            raise TypeError("Invalid pipe input type")
        if not isinstance(start, int):
            raise TypeError("Invalid start input type")
        if start < 1:
            raise ValueError("Invalid start value")
        pipe = {**pipe} if pipe else {}
        if kwargs.get("_1_", None) is not None:
            pipe[f"_{0 + start}_"] = kwargs.get("_1_", None)
        if kwargs.get("_2_", None) is not None:
            pipe[f"_{1 + start}_"] = kwargs.get("_2_", None)
        if kwargs.get("_3_", None) is not None:
            pipe[f"_{2 + start}_"] = kwargs.get("_3_", None)
        if kwargs.get("_4_", None) is not None:
            pipe[f"_{3 + start}_"] = kwargs.get("_4_", None)
        if kwargs.get("_5_", None) is not None:
            pipe[f"_{4 + start}_"] = kwargs.get("_5_", None)
        return io.NodeOutput(pipe,)

#-------------------------------------------------------------------------------#

class FlowPipeOutput(io.ComfyNode):
    _match_template = io.MatchType.Template('FlowPipeOutput')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-FlowPipeOutput',
            display_name='Pipe Out',
            category='SV Nodes/Pipes',
            inputs=[
            SvPipe.Input('pipe'),
            io.Int.Input('index', min=1, max=100, step=1, default=1),
            ],
            outputs=[
            SvPipe.Output(display_name='pipe'),
            io.MatchType.Output(template=cls._match_template, display_name='_1_'),
            io.MatchType.Output(template=cls._match_template, display_name='_2_'),
            io.MatchType.Output(template=cls._match_template, display_name='_3_'),
            io.MatchType.Output(template=cls._match_template, display_name='_4_'),
            io.MatchType.Output(template=cls._match_template, display_name='_5_'),
            ]
        )

    
    @classmethod
    def execute(cls, pipe, index) -> io.NodeOutput:
        if not isinstance(pipe, dict):
            raise TypeError("Invalid pipe input type")
        if not isinstance(index, int):
            raise TypeError("Invalid index input type")
        if index < 1:
            raise ValueError("Invalid index value")
        return io.NodeOutput(pipe, pipe.get(f"_{0 + index}_", None), pipe.get(f"_{1 + index}_", None), pipe.get(f"_{2 + index}_", None), pipe.get(f"_{3 + index}_", None), pipe.get(f"_{4 + index}_", None))

#-------------------------------------------------------------------------------#

class FlowPipeInputLarge(io.ComfyNode):
    _match_template = io.MatchType.Template('FlowPipeInputLarge')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-FlowPipeInputLarge',
            display_name='Pipe In Large',
            category='SV Nodes/Pipes',
            inputs=[
            io.Int.Input('start', min=1, max=100, step=1, default=1),
            SvPipe.Input('pipe', optional=True),
            io.MatchType.Input('_1_', template=cls._match_template, optional=True),
            io.MatchType.Input('_2_', template=cls._match_template, optional=True),
            io.MatchType.Input('_3_', template=cls._match_template, optional=True),
            io.MatchType.Input('_4_', template=cls._match_template, optional=True),
            io.MatchType.Input('_5_', template=cls._match_template, optional=True),
            io.MatchType.Input('_6_', template=cls._match_template, optional=True),
            io.MatchType.Input('_7_', template=cls._match_template, optional=True),
            io.MatchType.Input('_8_', template=cls._match_template, optional=True),
            io.MatchType.Input('_9_', template=cls._match_template, optional=True),
            io.MatchType.Input('_10_', template=cls._match_template, optional=True),
            ],
            outputs=[
            SvPipe.Output(display_name='pipe'),
            ]
        )

    
    @classmethod
    def execute(cls, start, pipe=None, **kwargs) -> io.NodeOutput:
        if not isinstance(pipe, (dict, type(None))):
            raise TypeError("Invalid pipe input type")
        if not isinstance(start, int):
            raise TypeError("Invalid start input type")
        if start < 1:
            raise ValueError("Invalid start value")
        pipe = {**pipe} if pipe else {}
        if kwargs.get("_1_", None) is not None:
            pipe[f"_{0 + start}_"] = kwargs.get("_1_", None)
        if kwargs.get("_2_", None) is not None:
            pipe[f"_{1 + start}_"] = kwargs.get("_2_", None)
        if kwargs.get("_3_", None) is not None:
            pipe[f"_{2 + start}_"] = kwargs.get("_3_", None)
        if kwargs.get("_4_", None) is not None:
            pipe[f"_{3 + start}_"] = kwargs.get("_4_", None)
        if kwargs.get("_5_", None) is not None:
            pipe[f"_{4 + start}_"] = kwargs.get("_5_", None)
        if kwargs.get("_6_", None) is not None:
            pipe[f"_{5 + start}_"] = kwargs.get("_6_", None)
        if kwargs.get("_7_", None) is not None:
            pipe[f"_{6 + start}_"] = kwargs.get("_7_", None)
        if kwargs.get("_8_", None) is not None:
            pipe[f"_{7 + start}_"] = kwargs.get("_8_", None)
        if kwargs.get("_9_", None) is not None:
            pipe[f"_{8 + start}_"] = kwargs.get("_9_", None)
        if kwargs.get("_10_", None) is not None:
            pipe[f"_{9 + start}_"] = kwargs.get("_10_", None)
        return io.NodeOutput(pipe,)

#-------------------------------------------------------------------------------#

class FlowPipeOutputLarge(io.ComfyNode):
    _match_template = io.MatchType.Template('FlowPipeOutputLarge')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-FlowPipeOutputLarge',
            display_name='Pipe Out Large',
            category='SV Nodes/Pipes',
            inputs=[
            SvPipe.Input('pipe'),
            io.Int.Input('index', min=1, max=100, step=1, default=1),
            ],
            outputs=[
            SvPipe.Output(display_name='pipe'),
            io.MatchType.Output(template=cls._match_template, display_name='_1_'),
            io.MatchType.Output(template=cls._match_template, display_name='_2_'),
            io.MatchType.Output(template=cls._match_template, display_name='_3_'),
            io.MatchType.Output(template=cls._match_template, display_name='_4_'),
            io.MatchType.Output(template=cls._match_template, display_name='_5_'),
            io.MatchType.Output(template=cls._match_template, display_name='_6_'),
            io.MatchType.Output(template=cls._match_template, display_name='_7_'),
            io.MatchType.Output(template=cls._match_template, display_name='_8_'),
            io.MatchType.Output(template=cls._match_template, display_name='_9_'),
            io.MatchType.Output(template=cls._match_template, display_name='_10_'),
            ]
        )

    
    @classmethod
    def execute(cls, pipe, index) -> io.NodeOutput:
        if not isinstance(pipe, dict):
            raise TypeError("Invalid pipe input type")
        if not isinstance(index, int):
            raise TypeError("Invalid index input type")
        if index < 1:
            raise ValueError("Invalid index value")
        return io.NodeOutput(pipe, pipe.get(f"_{0 + index}_", None), pipe.get(f"_{1 + index}_", None), pipe.get(f"_{2 + index}_", None), pipe.get(f"_{3 + index}_", None), pipe.get(f"_{4 + index}_", None), pipe.get(f"_{5 + index}_", None), pipe.get(f"_{6 + index}_", None), pipe.get(f"_{7 + index}_", None), pipe.get(f"_{8 + index}_", None), pipe.get(f"_{9 + index}_", None))

#-------------------------------------------------------------------------------#

class FlowPipeInputIndex(io.ComfyNode):
    _match_template = io.MatchType.Template('FlowPipeInputIndex')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-FlowPipeInputIndex',
            display_name='Pipe In Index',
            category='SV Nodes/Pipes',
            inputs=[
            io.Int.Input('index', min=1, max=100, step=1, default=1),
            SvPipe.Input('pipe', optional=True),
            io.MatchType.Input('value', template=cls._match_template, optional=True),
            ],
            outputs=[
            SvPipe.Output(display_name='pipe'),
            ]
        )

    
    @classmethod
    def execute(cls, index, pipe=None, value=None) -> io.NodeOutput:
        if not isinstance(pipe, (dict, type(None))):
            raise TypeError("Invalid pipe input type")
        if not isinstance(index, int):
            raise TypeError("Invalid index input type")
        if index < 1:
            raise ValueError("Invalid index value")
        if value is None:
            return io.NodeOutput(pipe,)
        pipe = {**pipe} if pipe else {}
        pipe[f"_{index}_"] = value
        return io.NodeOutput(pipe,)

#-------------------------------------------------------------------------------#

class FlowPipeOutputIndex(io.ComfyNode):
    _match_template = io.MatchType.Template('FlowPipeOutputIndex')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-FlowPipeOutputIndex',
            display_name='Pipe Out Index',
            category='SV Nodes/Pipes',
            inputs=[
            SvPipe.Input('pipe'),
            io.Int.Input('index', min=1, max=100, step=1, default=1),
            ],
            outputs=[
            SvPipe.Output(display_name='pipe'),
            io.MatchType.Output(template=cls._match_template, display_name='value'),
            ]
        )

    
    @classmethod
    def execute(cls, pipe, index) -> io.NodeOutput:
        if not isinstance(pipe, dict):
            raise TypeError("Invalid pipe input type")
        if not isinstance(index, int):
            raise TypeError("Invalid index input type")
        if index < 1:
            raise ValueError("Invalid index value")
        return io.NodeOutput(pipe, pipe.get(f"_{index}_", None))

#-------------------------------------------------------------------------------#

class FlowPipeInputKey(io.ComfyNode):
    _match_template = io.MatchType.Template('FlowPipeInputKey')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-FlowPipeInputKey',
            display_name='Pipe In Key',
            category='SV Nodes/Pipes',
            inputs=[
            io.String.Input('key', multiline=False),
            SvPipe.Input('pipe', optional=True),
            io.MatchType.Input('value', template=cls._match_template, optional=True),
            ],
            outputs=[
            SvPipe.Output(display_name='pipe'),
            ]
        )

    
    @classmethod
    def execute(cls, key, pipe=None, value=None) -> io.NodeOutput:
        if not isinstance(pipe, (dict, type(None))):
            raise TypeError("Invalid pipe input type")
        if not isinstance(key, str):
            raise TypeError("Invalid key input type")
        if value is None:
            return io.NodeOutput(pipe,)
        pipe = {**pipe} if pipe else {}
        pipe[key] = value
        return io.NodeOutput(pipe,)

#-------------------------------------------------------------------------------#

class FlowPipeOutputKey(io.ComfyNode):
    _match_template = io.MatchType.Template('FlowPipeOutputKey')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-FlowPipeOutputKey',
            display_name='Pipe Out Key',
            category='SV Nodes/Pipes',
            inputs=[
            SvPipe.Input('pipe'),
            io.String.Input('key', multiline=False),
            ],
            outputs=[
            SvPipe.Output(display_name='pipe'),
            io.MatchType.Output(template=cls._match_template, display_name='value'),
            ]
        )

    
    @classmethod
    def execute(cls, pipe, key) -> io.NodeOutput:
        if not isinstance(pipe, dict):
            raise TypeError("Invalid pipe input type")
        if not isinstance(key, str):
            raise TypeError("Invalid key input type")
        return io.NodeOutput(pipe, pipe.get(key, None))

#-------------------------------------------------------------------------------#

class FlowPipeInputKeyTuple(io.ComfyNode):
    _match_template = io.MatchType.Template('FlowPipeInputKeyTuple')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-FlowPipeInputKeyTuple',
            display_name='Pipe In Tuple',
            category='SV Nodes/Pipes',
            inputs=[
            io.String.Input('key', multiline=False),
            SvPipe.Input('pipe', optional=True),
            io.MatchType.Input('_1_', template=cls._match_template, optional=True),
            io.MatchType.Input('_2_', template=cls._match_template, optional=True),
            io.MatchType.Input('_3_', template=cls._match_template, optional=True),
            io.MatchType.Input('_4_', template=cls._match_template, optional=True),
            io.MatchType.Input('_5_', template=cls._match_template, optional=True),
            ],
            outputs=[
            SvPipe.Output(display_name='pipe'),
            ]
        )

    
    @classmethod
    def execute(cls, key, pipe=None, _1_=None, _2_=None, _3_=None, _4_=None, _5_=None) -> io.NodeOutput:
        if not isinstance(pipe, (dict, type(None))):
            raise TypeError("Invalid pipe input type")
        if not isinstance(key, str):
            raise TypeError("Invalid key input type")
        pipe = {**pipe} if pipe else {}
        old = pipe.get(key, (None, None, None, None, None))
        value1 = _1_ if _1_ is not None else old[0]
        value2 = _2_ if _2_ is not None else old[1]
        value3 = _3_ if _3_ is not None else old[2]
        value4 = _4_ if _4_ is not None else old[3]
        value5 = _5_ if _5_ is not None else old[4]
        pipe[key] = (value1, value2, value3, value4, value5)
        return io.NodeOutput(pipe,)

#-------------------------------------------------------------------------------#

class FlowPipeOutputKeyTuple(io.ComfyNode):
    _match_template = io.MatchType.Template('FlowPipeOutputKeyTuple')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-FlowPipeOutputKeyTuple',
            display_name='Pipe Out Tuple',
            category='SV Nodes/Pipes',
            inputs=[
            SvPipe.Input('pipe'),
            io.String.Input('key', multiline=False),
            ],
            outputs=[
            SvPipe.Output(display_name='pipe'),
            io.MatchType.Output(template=cls._match_template, display_name='_1_'),
            io.MatchType.Output(template=cls._match_template, display_name='_2_'),
            io.MatchType.Output(template=cls._match_template, display_name='_3_'),
            io.MatchType.Output(template=cls._match_template, display_name='_4_'),
            io.MatchType.Output(template=cls._match_template, display_name='_5_'),
            ]
        )

    
    @classmethod
    def execute(cls, pipe, key) -> io.NodeOutput:
        empty = (None, None, None, None, None)
        if key not in pipe or key is None or len(key) == 0:
            return io.NodeOutput(pipe, *empty)
        if not isinstance(pipe, dict):
            raise TypeError(f"Invalid pipe input type with key '{key}'")
        if not isinstance(key, str):
            raise TypeError(f"Invalid key input type with key '{key}'")
        value = pipe.get(key, None)
        if value is None:
            return io.NodeOutput(pipe, *empty)
        if not isinstance(value, (tuple, list)):
            raise ValueError(f"Invalid value type with key '{key}'")
        return io.NodeOutput(pipe, *value)

#-------------------------------------------------------------------------------#

class FlowPipeInputModel(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-FlowPipeInputModel',
            display_name='Pipe In Model',
            category='SV Nodes/Pipes',
            inputs=[
            io.String.Input('key', multiline=False, default=''),
            SvPipe.Input('pipe', optional=True),
            io.Model.Input('model', optional=True),
            io.Clip.Input('clip', optional=True),
            io.Vae.Input('vae', optional=True),
            ],
            outputs=[
            SvPipe.Output(display_name='pipe'),
            ]
        )

    
    @classmethod
    def execute(cls, key, pipe=None, model=None, clip=None, vae=None) -> io.NodeOutput:
        if not isinstance(pipe, (dict, type(None))):
            raise TypeError("Invalid pipe input type")
        if pipe is None:
            pipe = {}
        key = f"__model[{key}]__"
        old = pipe.get(key, (None, None, None))
        model = model if model is not None else old[0]
        clip = clip if clip is not None else old[1]
        vae = vae if vae is not None else old[2]
        pipe[key] = (model, clip, vae)
        return io.NodeOutput(pipe,)

#-------------------------------------------------------------------------------#

class FlowPipeOutputModel(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-FlowPipeOutputModel',
            display_name='Pipe Out Model',
            category='SV Nodes/Pipes',
            inputs=[
            SvPipe.Input('pipe'),
            io.String.Input('key', multiline=False, default=''),
            ],
            outputs=[
            SvPipe.Output(display_name='pipe'),
            io.Model.Output(display_name='model'),
            io.Clip.Output(display_name='clip'),
            io.Vae.Output(display_name='vae'),
            ]
        )

    
    @classmethod
    def execute(cls, pipe, key) -> io.NodeOutput:
        if not isinstance(pipe, dict):
            raise TypeError("Invalid pipe input type")
        key = f"__model[{key}]__"
        value = pipe.get(key, None)
        if value is None:
            return io.NodeOutput(pipe, None, None, None)
        if not isinstance(value, (tuple, list)):
            raise ValueError("Invalid value type")
        if len(value) != 3:
            raise ValueError("Invalid value length")
        return io.NodeOutput(pipe, *value)

#-------------------------------------------------------------------------------#

class FlowPipeInputParams(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-FlowPipeInputParams',
            display_name='Pipe In Params',
            category='SV Nodes/Pipes',
            inputs=[
            io.String.Input('key', multiline=False, default=''),
            SvPipe.Input('pipe', optional=True),
            io.Conditioning.Input('positive', optional=True),
            io.Conditioning.Input('negative', optional=True),
            io.Latent.Input('latent', optional=True),
            io.Int.Input('seed', force_input=True, optional=True),
            BpOutput.Input('params', optional=True),
            io.Float.Input('cfg', force_input=True, optional=True),
            io.Int.Input('steps', force_input=True, optional=True),
            io.Float.Input('denoise', force_input=True, optional=True),
            io.Combo.Input('sampler name', options=['euler', 'euler_ancestral', 'heun', 'dpm_2', 'dpm_2_ancestral'], optional=True),
            io.Combo.Input('scheduler', options=['normal', 'karras', 'exponential', 'sgm_uniform', 'simple', 'ddim_uniform'], optional=True),
            io.Boolean.Input('ays', force_input=True, optional=True),
            ],
            outputs=[
            SvPipe.Output(display_name='pipe'),
            ]
        )

    
    @classmethod
    def execute(cls, key, pipe=None, positive=None, negative=None, latent=None, seed=None, params=None, cfg=None, steps=None, denoise=None, sampler=None, scheduler=None, ays=None) -> io.NodeOutput:
        if not isinstance(pipe, (dict, type(None))):
            raise TypeError("Invalid pipe input type")
        if pipe is None:
            pipe = {}
        key = f"__params[{key}]__"
        _cfg, _steps, _denoise, _sampler, _scheduler, _ays, _sampler2 = None, None, None, None, None, None, None
        if params is not None:
            _cfg, _steps, _denoise, _sampler, _scheduler, _ays, _sampler2 = BasicParamsOutput.run(None, params)
        _cfg = default(cfg, _cfg)
        _steps = default(steps, _steps)
        _denoise = default(denoise, _denoise)
        _sampler = comfy.samplers.sampler_object(sampler) if sampler is not None else _sampler
        _scheduler = comfy.samplers.scheduler_object(scheduler) if scheduler is not None else _scheduler
        _ays = default(ays, _ays)
        _sampler2 = comfy.samplers.sampler_object(sampler) if sampler is not None else _sampler2
        
        old = pipe.get(key, (None, None, None, None, None, None, None, None, None, None, None))
        positive = positive if positive is not None else old[0]
        negative = negative if negative is not None else old[1]
        latent = latent if latent is not None else old[2]
        seed = seed if seed is not None else old[3]
        cfg = _cfg if _cfg is not None else old[4]
        steps = _steps if _steps is not None else old[5]
        denoise = _denoise if _denoise is not None else old[6]
        sampler = _sampler if _sampler is not None else old[7]
        sampler2 = _sampler2 if _sampler2 is not None else old[8]
        scheduler = _scheduler if _scheduler is not None else old[9]
        ays = _ays if _ays is not None else old[10]
        
        pipe[key] = (positive, negative, latent, seed, cfg, steps, denoise, sampler, sampler2, scheduler, ays)
        return io.NodeOutput(pipe,)

#-------------------------------------------------------------------------------#

class FlowPipeOutputParams(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-FlowPipeOutputParams',
            display_name='Pipe Out Params',
            category='SV Nodes/Pipes',
            inputs=[
            SvPipe.Input('pipe'),
            io.String.Input('key', multiline=False, default=''),
            ],
            outputs=[
            SvPipe.Output(display_name='pipe'),
            io.Conditioning.Output(display_name='positive'),
            io.Conditioning.Output(display_name='negative'),
            io.Latent.Output(display_name='latent'),
            io.Int.Output(display_name='seed'),
            io.Float.Output(display_name='cfg'),
            io.Int.Output(display_name='steps'),
            io.Float.Output(display_name='denoise'),
            io.String.Output(display_name='sampler name'),
            SvSampler.Output(display_name='sampler'),
            io.String.Output(display_name='scheduler'),
            io.Boolean.Output(display_name='ays'),
            ]
        )

    
    @classmethod
    def execute(cls, pipe, key) -> io.NodeOutput:
        if not isinstance(pipe, dict):
            raise TypeError("Invalid pipe input type")
        key = f"__params[{key}]__"
        value = pipe.get(key, None)
        if value is None:
            return io.NodeOutput(pipe, None, None, None, None, None, None, None, None, None, None, None)
        if not isinstance(value, (tuple, list)):
            raise ValueError("Invalid value type")
        if len(value) != 11:
            raise ValueError("Invalid value length")
        return io.NodeOutput(pipe, *value)

#-------------------------------------------------------------------------------#

class FlowPipeCombine(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-FlowPipeCombine',
            display_name='Pipe Combine',
            category='SV Nodes/Pipes',
            inputs=[
            SvPipe.Input('pipe1'),
            SvPipe.Input('pipe2'),
            ],
            outputs=[
            SvPipe.Output(display_name='pipe'),
            ]
        )

    
    @classmethod
    def execute(cls, pipe1, pipe2) -> io.NodeOutput:
        if not isinstance(pipe1, (dict, type(None))) or not isinstance(pipe2, (dict, type(None))):
            raise TypeError("Invalid pipe input type")
        if pipe1 is None:
            pipe1 = {}
        if pipe2 is None:
            pipe2 = {}
        # remove None values
        pipe1 = {k: v for k, v in pipe1.items() if v is not None}
        pipe2 = {k: v for k, v in pipe2.items() if v is not None}
        return io.NodeOutput({**pipe1, **pipe2},)

#-------------------------------------------------------------------------------#

class CheckNone(io.ComfyNode):
    _match_template = io.MatchType.Template('CheckNone')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-CheckNone',
            display_name='Check None',
            category='SV Nodes/Logic',
            inputs=[
            io.MatchType.Input('any', template=cls._match_template),
            ],
            outputs=[
            io.Boolean.Output(display_name='bool'),
            ]
        )

    
    @classmethod
    def execute(cls, any) -> io.NodeOutput:
        return io.NodeOutput(any is None,)

#-------------------------------------------------------------------------------#

class CheckNoneNot(io.ComfyNode):
    _match_template = io.MatchType.Template('CheckNoneNot')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-CheckNoneNot',
            display_name='Check Not None',
            category='SV Nodes/Logic',
            inputs=[
            io.MatchType.Input('any', template=cls._match_template),
            ],
            outputs=[
            io.Boolean.Output(display_name='bool'),
            ]
        )

    
    @classmethod
    def execute(cls, any) -> io.NodeOutput:
        return io.NodeOutput(any is not None,)

#-------------------------------------------------------------------------------#

class DefaultInt(io.ComfyNode):
    _match_template = io.MatchType.Template('DefaultInt')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-DefaultInt',
            display_name='Default Int',
            category='SV Nodes/Logic',
            inputs=[
            io.MatchType.Input('any', template=cls._match_template),
            io.Int.Input('default', min=-sys.maxsize, max=sys.maxsize, step=1, default=0, lazy=True),
            ],
            outputs=[
            io.Int.Output(display_name='int'),
            ]
        )

    
    @classmethod
    def check_lazy_status(cls, any, **kwargs):
        if any is None:
            return ["default"]
        return []
    
    @classmethod
    def execute(cls, any, default) -> io.NodeOutput:
        if any is None or not isinstance(any, int):
            return io.NodeOutput(default,)
        return io.NodeOutput(any,)

#-------------------------------------------------------------------------------#

class DefaultFloat(io.ComfyNode):
    _match_template = io.MatchType.Template('DefaultFloat')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-DefaultFloat',
            display_name='Default Float',
            category='SV Nodes/Logic',
            inputs=[
            io.MatchType.Input('any', template=cls._match_template),
            io.Float.Input('default', min=-sys.float_info.max, max=sys.float_info.max, step=0.01, default=0.0, lazy=True),
            ],
            outputs=[
            io.Float.Output(display_name='float'),
            ]
        )

    
    @classmethod
    def check_lazy_status(cls, any, **kwargs):
        if any is None:
            return ["default"]
        return []
    
    @classmethod
    def execute(cls, any, default) -> io.NodeOutput:
        if any is None or not isinstance(any, float):
            return io.NodeOutput(default,)
        return io.NodeOutput(any,)

#-------------------------------------------------------------------------------#

class DefaultString(io.ComfyNode):
    _match_template = io.MatchType.Template('DefaultString')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-DefaultString',
            display_name='Default String',
            category='SV Nodes/Logic',
            inputs=[
            io.MatchType.Input('any', template=cls._match_template),
            io.String.Input('default', multiline=False, default='', lazy=True),
            ],
            outputs=[
            io.String.Output(display_name='string'),
            ]
        )

    
    @classmethod
    def check_lazy_status(cls, any, **kwargs):
        if any is None:
            return ["default"]
        return []
    
    @classmethod
    def execute(cls, any, default) -> io.NodeOutput:
        if any is None or not isinstance(any, str):
            return io.NodeOutput(default,)
        return io.NodeOutput(any,)

#-------------------------------------------------------------------------------#

class DefaultBoolean(io.ComfyNode):
    _match_template = io.MatchType.Template('DefaultBoolean')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-DefaultBoolean',
            display_name='Default Boolean',
            category='SV Nodes/Logic',
            inputs=[
            io.MatchType.Input('any', template=cls._match_template),
            io.Boolean.Input('default', default=False, lazy=True),
            ],
            outputs=[
            io.Boolean.Output(display_name='bool'),
            ]
        )

    
    @classmethod
    def check_lazy_status(cls, any, **kwargs):
        if any is None:
            return ["default"]
        return []
    
    @classmethod
    def execute(cls, any, default) -> io.NodeOutput:
        if any is None or not isinstance(any, bool):
            return io.NodeOutput(default,)
        return io.NodeOutput(any,)

#-------------------------------------------------------------------------------#

class DefaultValue(io.ComfyNode):
    _match_template = io.MatchType.Template('DefaultValue')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-DefaultValue',
            display_name='Default Value',
            category='SV Nodes/Logic',
            inputs=[
            io.MatchType.Input('any', template=cls._match_template),
            io.MatchType.Input('default', template=cls._match_template, lazy=True),
            ],
            outputs=[
            io.MatchType.Output(template=cls._match_template, display_name='value'),
            ]
        )

    
    @classmethod
    def check_lazy_status(cls, any, **kwargs):
        if any is None:
            return ["default"]
        return []
    
    @classmethod
    def execute(cls, any, default) -> io.NodeOutput:
        if any is None:
            return io.NodeOutput(default,)
        return io.NodeOutput(any,)

#-------------------------------------------------------------------------------#

class AnyToAny(io.ComfyNode):
    _match_template = io.MatchType.Template('AnyToAny')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-AnyToAny',
            display_name='Any to Any',
            category='SV Nodes',
            inputs=[
            io.MatchType.Input('input', template=cls._match_template),
            ],
            outputs=[
            io.MatchType.Output(template=cls._match_template, display_name='output'),
            ]
        )

    
    @classmethod
    def execute(cls, input) -> io.NodeOutput:
        return io.NodeOutput(input,)

#-------------------------------------------------------------------------------#

class ConsolePrint(io.ComfyNode):
    _match_template = io.MatchType.Template('ConsolePrint')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-ConsolePrint',
            display_name='Console Print',
            category='SV Nodes/Debug',
            inputs=[
            io.String.Input('text', multiline=True),
            io.MatchType.Input('signal', template=cls._match_template, optional=True),
            ],
            outputs=[
            
            ],
            is_output_node=True
        )

    
    @classmethod
    def execute(cls, text, signal=None) -> io.NodeOutput:
        print(text.replace("_signal_", str(signal)))
        return io.NodeOutput()


#-------------------------------------------------------------------------------#

class ConsolePrintMulti(io.ComfyNode):
    _match_template = io.MatchType.Template('ConsolePrintMulti')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-ConsolePrintMulti',
            display_name='Console Print Multi',
            category='SV Nodes/Debug',
            inputs=[
            io.String.Input('text', multiline=True),
            io.MatchType.Input('signal1', template=cls._match_template),
            io.MatchType.Input('signal2', template=cls._match_template, optional=True),
            io.MatchType.Input('signal3', template=cls._match_template, optional=True),
            ],
            outputs=[
            
            ],
            is_output_node=True
        )

    
    @classmethod
    def execute(cls, text, signal1, signal2=None, signal3=None) -> io.NodeOutput:
        print(text.replace("_signal1_", str(signal1)).replace("_signal2_", str(signal2)).replace("_signal3_", str(signal3)))
        return io.NodeOutput()


#-------------------------------------------------------------------------------#

class ConsolePrintLoop(io.ComfyNode):
    _match_template = io.MatchType.Template('ConsolePrintLoop')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-ConsolePrintLoop',
            display_name='Console Print Loop',
            category='SV Nodes/Debug',
            inputs=[
            io.String.Input('text', multiline=True),
            io.MatchType.Input('signal1', template=cls._match_template, optional=True),
            io.MatchType.Input('signal2', template=cls._match_template, optional=True),
            io.MatchType.Input('signal3', template=cls._match_template, optional=True),
            ],
            outputs=[
            io.MatchType.Output(template=cls._match_template, display_name='signal1'),
            io.MatchType.Output(template=cls._match_template, display_name='signal2'),
            io.MatchType.Output(template=cls._match_template, display_name='signal3'),
            ],
            is_output_node=True
        )

    
    @classmethod
    def execute(cls, text, signal1=None, signal2=None, signal3=None) -> io.NodeOutput:
        print(text.replace("_signal1_", str(signal1)).replace("_signal2_", str(signal2)).replace("_signal3_", str(signal3)))
        return io.NodeOutput(signal1, signal2, signal3)

#-------------------------------------------------------------------------------#

class AssertNotNone(io.ComfyNode):
    _match_template = io.MatchType.Template('AssertNotNone')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-AssertNotNone',
            display_name='Assert Not None',
            category='SV Nodes/Debug',
            inputs=[
            io.MatchType.Input('any', template=cls._match_template),
            ],
            outputs=[
            
            ],
            is_output_node=True
        )

    
    @classmethod
    def execute(cls, any) -> io.NodeOutput:
        if any is None:
            raise ValueError("AssertNotNone: Value is None")
        return io.NodeOutput()


#-------------------------------------------------------------------------------#

class TimerStart(io.ComfyNode):
    _match_template = io.MatchType.Template('TimerStart')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-TimerStart',
            display_name='Timer Start',
            category='SV Nodes/Debug',
            inputs=[
            io.MatchType.Input('any', template=cls._match_template),
            ],
            outputs=[
            io.MatchType.Output(template=cls._match_template, display_name='any'),
            Timer.Output(display_name='timestamp'),
            ],
            is_output_node=True
        )

    
    @classmethod
    def execute(cls, any) -> io.NodeOutput:
        return io.NodeOutput(any, time.time())

#-------------------------------------------------------------------------------#

class TimerEnd(io.ComfyNode):
    _match_template = io.MatchType.Template('TimerEnd')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-TimerEnd',
            display_name='Timer End',
            category='SV Nodes/Debug',
            inputs=[
            io.MatchType.Input('any', template=cls._match_template),
            Timer.Input('timestamp'),
            ],
            outputs=[
            io.MatchType.Output(template=cls._match_template, display_name='any'),
            io.Float.Output(display_name='time'),
            ],
            is_output_node=True
        )

    
    @classmethod
    def execute(cls, any, timestamp) -> io.NodeOutput:
        return io.NodeOutput(any, time.time() - timestamp)

#-------------------------------------------------------------------------------#

class CurveFromEquation(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-CurveFromEquation',
            display_name='Curve from Equation',
            category='SV Nodes/Logic',
            inputs=[
            io.String.Input('fn', multiline=False, default='x^2 + 1'),
            ],
            outputs=[
            Curve.Output(display_name='curve'),
            ]
        )

    
    @classmethod
    def execute(cls, fn) -> io.NodeOutput:
        return io.NodeOutput(parseCurve(fn),)
def parseCurve(curve):
    def f(curve, t):
        curve = re.sub(r"\s+", "", curve)
        curve = collapseSigns(curve)
        while "(" in curve or ")" in curve:
            curve = re.sub(r"\w+\([^()]+\)", lambda x : str(parseCurveFunction(x.group(0), t)), curve)
            curve = re.sub(r"(?<!\w)\([^(,)]+\)", lambda x : str(parseCurve(x.group(0)[1:-1])(t)), curve)
            curve = collapseSigns(curve)
        parts = [x for x in filter(lambda x : len(x), re.split("(?<!\^)(?<!\*|/|%)(?=[-+])", curve))]
        if len(parts) == 0:
            raise ValueError("Invalid curve: No parts found")
        sum = 0
        for part in parts:
            subparts = re.split(r"(?<=[*/%])|(?=[*/%])", part)
            for i in range(len(subparts)):
                if subparts[i] != "*" and subparts[i] != "/" and subparts[i] != "%":
                    subparts[i] = parseCurvePart(subparts[i], t)
            for i in range(len(subparts)):
                if subparts[i] == "*":
                    subparts[i+1] = subparts[i-1] * subparts[i+1]
                if subparts[i] == "/":
                    subparts[i+1] = subparts[i-1] / subparts[i+1]
                if subparts[i] == "%":
                    subparts[i+1] = subparts[i-1] % subparts[i+1]
            sum += subparts[-1]
        return sum
    return partial(f, curve)

def parseCurveFunction(func, t):
    name = re.search(r"^\w+", func).group(0).lower()
    func = re.sub(r"^\w+\(|\)", "", func)
    parts: list[float] = [parseCurve(x)(t) for x in func.split(",")]
    
    if name == "min":
        return min(parts)
    if name == "max":
        return max(parts)
    if name == "floor":
        return math.floor(parts[0])
    if name == "ceil":
        return math.ceil(parts[0])
    if name == "round":
        return round(parts[0])
    if name == "sqrt":
        return math.sqrt(parts[0])
    if name == "clamp":
        if len(parts) < 3:
            raise ValueError("Clamp requires 3 arguments: clamp(x, min, max)")
        if parts[1] > parts[2]:
            parts[1], parts[2] = parts[2], parts[1]
        return max(parts[1], min(parts[0], parts[2]))
    if name == "closest":
        if len(parts) < 2:
            raise ValueError("Closest requires at least 2 arguments: closest(x, args...)")
        mapped = map(lambda x: { "i": x, "abs": abs(parts[0] - parts[x]) }, range(1, len(parts)))
        return parts[min(mapped, key=lambda x: x["abs"])["i"]]
    raise ValueError("Invalid math function")

def collapseSigns(curve):
    curve = re.sub(r"\+\+", "+", curve)
    curve = re.sub(r"\-\-", "+", curve)
    curve = re.sub(r"\+\-", "-", curve)
    curve = re.sub(r"\-\+", "-", curve)
    return curve

def parseCurvePart(part, x):
    if part == "+" or part == "-":
        raise ValueError("Invalid curve: Operator without operand")
    if "^" in part:
        base, power = part.split("^")
    else:
        base = part
        power = "1"
    if power == "x" or power == "-x":
        power = str(float(power.replace("x", "1")) * x)
    elif "x" in power:
        power = str(float(power.replace("x", "")) * x)
    if not re.sub(r"[+-]", "", power).replace('.', '', 1).isdigit():
        raise ValueError(f"Invalid curve: Power is invalid in {part}")
    if re.sub(r"[+-]", "", base).replace('.', '', 1).isdigit():
        return float(base) ** float(power)
    if "x" not in base:
        raise ValueError(f"Invalid curve: Base must contain 'x' in {part}")
    base = base.replace("x", "")
    if base == "" or base == "-":
        base = base + "1"
    multiplier = float(base)
    return multiplier * (x ** float(power))


#-------------------------------------------------------------------------------#

class ApplyCurve(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-ApplyCurve',
            display_name='Apply Curve',
            category='SV Nodes/Logic',
            inputs=[
            Curve.Input('curve'),
            io.Float.Input('t', force_input=True),
            ],
            outputs=[
            io.Float.Output(display_name='value'),
            ]
        )

    
    @classmethod
    def execute(cls, curve, t) -> io.NodeOutput:
        return io.NodeOutput(curve(t),)

#-------------------------------------------------------------------------------#

class ApplyCurveFromStep(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-ApplyCurveFromStep',
            display_name='Apply Curve from Step',
            category='SV Nodes/Logic',
            inputs=[
            Curve.Input('curve'),
            io.Int.Input('step', force_input=True),
            io.Int.Input('steps', min=1, max=100, step=1, default=10),
            ],
            outputs=[
            io.Float.Output(display_name='value'),
            ]
        )

    
    @classmethod
    def execute(cls, curve, step, steps) -> io.NodeOutput:
        if step < 1 or step > steps:
            raise ValueError("Invalid step value")
        if steps < 1:
            raise ValueError("Invalid steps value")
        if steps == 1:
            return io.NodeOutput(curve(1),)
        return io.NodeOutput(curve((step - 1) / (steps - 1)),)

#-------------------------------------------------------------------------------#

class MathOperation(io.ComfyNode):
    _match_template = io.MatchType.Template('MathOperation')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-MathOperation',
            display_name='Math Operation',
            category='SV Nodes/Logic',
            inputs=[
            io.String.Input('op', multiline=False, default=''),
            io.MatchType.Input('a', template=cls._match_template, optional=True),
            io.MatchType.Input('b', template=cls._match_template, optional=True),
            ],
            outputs=[
            io.Int.Output(display_name='int'),
            io.Float.Output(display_name='float'),
            ]
        )

    
    @classmethod
    def execute(cls, op: str, a=None, b=None) -> io.NodeOutput:
        op = op.lower()
        if a is None and 'a' in op:
            raise ValueError("Invalid operation: Missing 'a' value")
        if b is None and 'b' in op:
            raise ValueError("Invalid operation: Missing 'b' value")
        op = re.sub(r"\s+", "", op)
        op = re.sub(r"(?<=\d)a", "*" + str(a), op)
        op = re.sub(r"\ba\b", str(a), op)
        op = re.sub(r"(?<=\d)b", "*" + str(b), op)
        op = re.sub(r"\bb\b", str(b), op)
        result = parseCurve(op)(0)
        return math.floor(result), result


#-------------------------------------------------------------------------------#

class FloatRerouteForSubnodes(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-FloatRerouteForSubnodes',
            display_name='Float Reroute',
            category='SV Nodes/Flow',
            inputs=[
            io.Float.Input('float', min=0, max=1, step=0.01, default=0.0, force_input=True),
            ],
            outputs=[
            io.Float.Output(display_name='float'),
            ]
        )

    
    @classmethod
    def execute(cls, float) -> io.NodeOutput:
        return io.NodeOutput(float,)

#-------------------------------------------------------------------------------#

class ModelReroute(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-ModelReroute',
            display_name='Model Reroute',
            category='SV Nodes/Flow',
            inputs=[
            io.Model.Input('model'),
            ],
            outputs=[
            io.Model.Output(display_name='model'),
            ]
        )

    
    @classmethod
    def execute(cls, model) -> io.NodeOutput:
        return io.NodeOutput(model,)

#-------------------------------------------------------------------------------#

class SigmaReroute(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-SigmaReroute',
            display_name='Sigmas Reroute',
            category='SV Nodes/Flow',
            inputs=[
            Sigmas.Input('sigmas'),
            ],
            outputs=[
            Sigmas.Output(display_name='sigmas'),
            ]
        )

    
    @classmethod
    def execute(cls, sigmas) -> io.NodeOutput:
        return io.NodeOutput(sigmas,)

#-------------------------------------------------------------------------------#

class ConditioningReroute(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-ConditioningReroute',
            display_name='Conditioning Reroute',
            category='SV Nodes/Flow',
            inputs=[
            io.Conditioning.Input('conditioning'),
            ],
            outputs=[
            io.Conditioning.Output(display_name='conditioning'),
            ]
        )

    
    @classmethod
    def execute(cls, conditioning) -> io.NodeOutput:
        return io.NodeOutput(conditioning,)

#-------------------------------------------------------------------------------#

class ImageReroute(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-ImageReroute',
            display_name='Image Reroute',
            category='SV Nodes/Flow',
            inputs=[
            io.Image.Input('image'),
            ],
            outputs=[
            io.Image.Output(display_name='image'),
            ]
        )

    
    @classmethod
    def execute(cls, image) -> io.NodeOutput:
        return io.NodeOutput(image,)

#-------------------------------------------------------------------------------#

class SwapValues(io.ComfyNode):
    _match_template = io.MatchType.Template('SwapValues')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-SwapValues',
            display_name='Swap',
            category='SV Nodes/Logic',
            inputs=[
            io.MatchType.Input('_1_', template=cls._match_template),
            io.MatchType.Input('_2_', template=cls._match_template),
            io.Boolean.Input('swap', default=True),
            ],
            outputs=[
            io.MatchType.Output(template=cls._match_template, display_name='_2_'),
            io.MatchType.Output(template=cls._match_template, display_name='_1_'),
            ]
        )

    
    @classmethod
    def execute(cls, _1_, _2_, swap) -> io.NodeOutput:
        if swap:
            return io.NodeOutput(_2_, _1_)
        return io.NodeOutput(_1_, _2_)

#-------------------------------------------------------------------------------#

class VariableSet(io.ComfyNode):
    _match_template = io.MatchType.Template('VariableSet')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-VariableSet',
            display_name='Var Set',
            category='SV Nodes/Logic',
            inputs=[
            io.String.Input('key'),
            Signal.Input('signal', optional=True),
            io.MatchType.Input('value', template=cls._match_template, optional=True),
            io.Boolean.Input('set', force_input=True, default=True, optional=True),
            ],
            outputs=[
            Signal.Output(display_name='signal'),
            io.String.Output(display_name='value'),
            ],
            is_output_node=True
        )

    storage = {}
    
    @classmethod
    def execute(cls, value, key, set) -> io.NodeOutput:
        if set == False:
            return io.NodeOutput(None, None)
        if value is None:
            del VariableSet.storage[key]
            return io.NodeOutput(None, None)
        VariableSet.storage[key] = value
        return io.NodeOutput(None, value)

#-------------------------------------------------------------------------------#

class VariableGet(io.ComfyNode):
    _match_template = io.MatchType.Template('VariableGet')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-VariableGet',
            display_name='Var Get',
            category='SV Nodes/Logic',
            inputs=[
            io.String.Input('key'),
            Signal.Input('signal', optional=True),
            ],
            outputs=[
            Signal.Output(display_name='signal'),
            io.MatchType.Output(template=cls._match_template, display_name='value'),
            ]
        )

    
    @classmethod
    def execute(cls, key, signal) -> io.NodeOutput:
        if key not in VariableSet.storage:
            return io.NodeOutput(None, None)
        return io.NodeOutput(None, VariableSet.storage[key])

#-------------------------------------------------------------------------------#

class VariableClear(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-VariableClear',
            display_name='Var Clear',
            category='SV Nodes/Logic',
            inputs=[
            Signal.Input('signal', optional=True),
            ],
            outputs=[
            Signal.Output(display_name='signal'),
            ]
        )

    
    @classmethod
    def execute(cls, signal) -> io.NodeOutput:
        VariableSet.storage = {}
        return io.NodeOutput(None,)

#-------------------------------------------------------------------------------#

class EmptyValue:
    pass

class ValueRepeater(io.ComfyNode):
    _match_template = io.MatchType.Template('ValueRepeater')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-ValueRepeater',
            display_name='Value Repeater',
            category='SV Nodes/Logic',
            inputs=[
            io.MatchType.Input('value', template=cls._match_template, lazy=True),
            io.Boolean.Input('repeat', force_input=True),
            ],
            outputs=[
            io.MatchType.Output(template=cls._match_template, display_name='value'),
            ]
        )

    _saved: dict = {}
    
    @classmethod
    def check_lazy_status(cls, value, repeat):
        if repeat and type(ValueRepeater._saved.get(cls.hidden.unique_id, EmptyValue())) is not EmptyValue:
            return []
        return ["value"]
    
    @classmethod
    def execute(cls, value, repeat) -> io.NodeOutput:
        if repeat and type(ValueRepeater._saved.get(cls.hidden.unique_id, EmptyValue())) is not EmptyValue:
            return io.NodeOutput(ValueRepeater._saved.get(cls.hidden.unique_id, EmptyValue()),)
        ValueRepeater._saved[cls.hidden.unique_id] = value
        return io.NodeOutput(value,)

#-------------------------------------------------------------------------------#

class ValueGate(io.ComfyNode):
    _match_template = io.MatchType.Template('ValueGate')

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-ValueGate',
            display_name='Value Gate',
            category='SV Nodes/Logic',
            inputs=[
            io.MatchType.Input('value', template=cls._match_template, lazy=True),
            io.Boolean.Input('gate', force_input=True),
            ],
            outputs=[
            io.MatchType.Output(template=cls._match_template, display_name='value'),
            ]
        )

    @classmethod
    def check_lazy_status(cls, gate, value=None):
        if gate:
            return ["value"]
        return []

    @classmethod
    def execute(cls, value, gate) -> io.NodeOutput:
        if not gate:
            return io.NodeOutput(None,)
        return io.NodeOutput(value,)

#-------------------------------------------------------------------------------#

class PrimitiveFloat(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-PrimitiveFloat',
            display_name='Primitive Float',
            category='SV Nodes/Input',
            inputs=[
            io.Float.Input('float', min=-sys.float_info.max, max=sys.float_info.max, step=0.5, default=0.0),
            ],
            outputs=[
            io.Float.Output(display_name='float'),
            ]
        )

    
    @classmethod
    def execute(cls, float) -> io.NodeOutput:
        return io.NodeOutput(float,)

#-------------------------------------------------------------------------------#

class UnitFloat(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-UnitFloat',
            display_name='Unit Float',
            category='SV Nodes/Input',
            inputs=[
            io.Float.Input('float', min=0, max=1, step=0.01, default=0.0),
            ],
            outputs=[
            io.Float.Output(display_name='float'),
            ]
        )

    
    @classmethod
    def execute(cls, float) -> io.NodeOutput:
        return io.NodeOutput(float,)

#-------------------------------------------------------------------------------#

class MetadataJson(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-MetadataJson',
            display_name='Metadata Json',
            category='SV Nodes/IO',
            inputs=[
            io.String.Input('prompt', force_input=True, optional=True),
            ],
            outputs=[
            io.String.Output(display_name='json'),
            ]
        )

    
    @classmethod
    def execute(cls, prompt) -> io.NodeOutput:
        return io.NodeOutput(json.dumps({"extra": {
            "prompt": prompt,
        }}),)

#-------------------------------------------------------------------------------#

class PadImage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-PadImage',
            display_name='Pad Image',
            category='SV Nodes/image',
            inputs=[
            io.Image.Input('image'),
            io.Int.Input('padding', default=32, min=0, max=9999, step=1),
            io.Float.Input('color', default=0.5, min=0, max=1, step=0.01),
            ],
            outputs=[
            io.Image.Output(display_name='image'),
            ]
        )

    
    @classmethod
    def execute(cls, image, padding, color) -> io.NodeOutput:
        d1, d2, d3, d4 = image.size()

        new_image = torch.ones(
            (d1, d2 + padding * 2, d3 + padding * 2, d4),
            dtype=torch.float32,
        ) * color

        new_image[:, padding:padding + d2, padding:padding + d3, :] = image

        return io.NodeOutput(new_image,)

#-------------------------------------------------------------------------------#

class SV_SkimmedCFGDifference(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-SkimmedCFGDifference',
            display_name='Skimmed Diff CFG',
            category='SV Nodes/Model Patching',
            inputs=[
            io.Model.Input('model'),
            io.Float.Input('reference_CFG', default=3.0, min=0.0, max=25, step=0.5),
            io.Combo.Input('method', options=['linear_distance', 'squared_distance', 'root_distance', 'absolute_sum']),
            io.Float.Input('start_at', default=0.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input('end_at', default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[
            io.Model.Output(),
            ]
        )

    
    @classmethod
    def execute(cls, model, reference_CFG, method, start_at, end_at) -> io.NodeOutput:
        
        @torch.no_grad()
        def pre_cfg_patch(args):
            conds_out = args["conds_out"]
            cond_scale = args["cond_scale"]
            x_orig = args['input']
            sigma = args["sigma"][0]
            
            sigmas = get_sigmas(args)
            start_at_sigma = find_percent(sigmas, start_at)
            end_at_sigma = find_percent(sigmas, end_at)

            if not torch.any(conds_out[1]) or sigma <= end_at_sigma or sigma > start_at_sigma:
                return conds_out

            if method == "absolute_sum":
                ref_norm = (conds_out[0] * reference_CFG - conds_out[1] * (reference_CFG - 1)).norm(p=1)
                cfg_norm = (conds_out[0] * cond_scale - conds_out[1] * (cond_scale - 1)).norm(p=1)
                new_scale = cond_scale * ref_norm / cfg_norm
                fallback_weight = (new_scale - 1) / (cond_scale - 1)
                conds_out[1] = conds_out[0] * (1 - fallback_weight) + conds_out[1] * fallback_weight
            elif method in ["linear_distance","squared_distance","root_distance"]:
                conds_out[1] = interpolated_scales(x_orig,conds_out[0],conds_out[1],cond_scale,reference_CFG,method=="squared_distance",method=="root_distance")
            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(pre_cfg_patch)
        return io.NodeOutput(m,)

#-------------------------------------------------------------------------------#

class SV_SkimmedCFGDual(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-SkimmedCFGDual',
            display_name='Skimmed Dual CFG',
            category='SV Nodes/Model Patching',
            inputs=[
            io.Model.Input('model'),
            io.Float.Input('positive_cfg', default=3.0, min=0.0, max=25, step=0.5),
            io.Float.Input('negative_cfg', default=3.0, min=0.0, max=25, step=0.5),
            io.Float.Input('start_at', default=0.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input('end_at', default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[
            io.Model.Output(),
            ]
        )

    
    @classmethod
    def execute(cls, model, positive_cfg, negative_cfg, start_at, end_at) -> io.NodeOutput:

        @torch.no_grad()
        def pre_cfg_patch(args):
            conds_out = args["conds_out"]
            cond_scale = args["cond_scale"]
            x_orig = args['input']
            sigma = args["sigma"][0]
            
            sigmas = get_sigmas(args)
            start_at_sigma = find_percent(sigmas, start_at)
            end_at_sigma = find_percent(sigmas, end_at)

            if not torch.any(conds_out[1]) or sigma <= end_at_sigma or sigma > start_at_sigma:
                return conds_out

            fallback_weight_positive = (positive_cfg - 1) / (cond_scale - 1)
            fallback_weight_negative = (negative_cfg - 1) / (cond_scale - 1)

            skim_mask = get_skimming_mask(x_orig, conds_out[1], conds_out[0], cond_scale)
            conds_out[1][skim_mask] = conds_out[0][skim_mask] * (1 - fallback_weight_negative) + conds_out[1][skim_mask] * fallback_weight_negative

            skim_mask = get_skimming_mask(x_orig, conds_out[0], conds_out[1], cond_scale)
            conds_out[1][skim_mask] = conds_out[0][skim_mask] * (1 - fallback_weight_positive) + conds_out[1][skim_mask] * fallback_weight_positive

            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(pre_cfg_patch)
        return io.NodeOutput(m,)

#-------------------------------------------------------------------------------#

class SV_CondDiffSharpening(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-CondDiffSharpening',
            display_name='CFG Sharpening',
            category='SV Nodes/Model Patching',
            inputs=[
            io.Model.Input('model'),
            io.Combo.Input('do_on', options=['both', 'cond', 'uncond'], default='both'),
            io.Float.Input('scale', default=0.5, min=-10.0, max=10.0, step=0.05),
            io.Float.Input('start_at', default=0.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input('end_at', default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[
            io.Model.Output(),
            ]
        )

    
    @classmethod
    def execute(cls, model, do_on, scale, start_at, end_at) -> io.NodeOutput:
        prev_cond = None
        prev_uncond = None

        @torch.no_grad()
        def sharpen_conds_pre_cfg(args):
            nonlocal prev_cond, prev_uncond
            conds_out = args["conds_out"]
            uncond = torch.any(conds_out[1])
            sigma = args["sigma"][0].item()
            
            sigmas = get_sigmas(args)
            start_at_sigma = find_percent(sigmas, start_at)
            end_at_sigma = find_percent(sigmas, end_at)
            
            first_step = sigmas[0] == sigma
            # last_step = sigmas[-1] == sigma
            if first_step:
                prev_cond = None
                prev_uncond = None

            for b in range(len(conds_out[0])):
                for c in range(len(conds_out[0][b])):
                    if not first_step and sigma > end_at_sigma and sigma <= start_at_sigma:
                        if prev_cond is not None and do_on in ['both','cond']:
                            conds_out[0][b][c] = normalize_adjust(conds_out[0][b][c], prev_cond[b][c], scale)
                        if prev_uncond is not None and uncond and do_on in ['both','uncond']:
                            conds_out[1][b][c] = normalize_adjust(conds_out[1][b][c], prev_uncond[b][c], scale)

            prev_cond = conds_out[0]
            if uncond:
                prev_uncond = conds_out[1]

            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(sharpen_conds_pre_cfg)
        return io.NodeOutput(m,)

#-------------------------------------------------------------------------------#

class SV_CFGVariableScale(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-CFGVariableScale',
            display_name='CFG Variable Scale',
            category='model_patches/Pre CFG',
            inputs=[
            io.Model.Input('model'),
            io.Float.Input('start_mult', default=1.0, min=0.0, max=10.0, step=0.01),
            io.Float.Input('end_mult', default=1.5, min=0.0, max=10.0, step=0.01),
            io.Combo.Input('mode', options=['sigma', 'steps']),
            ],
            outputs=[
            io.Model.Output(),
            ]
        )

    
    @classmethod
    def execute(cls, model, start_mult, end_mult, mode) -> io.NodeOutput:

        @torch.no_grad()
        def variable_scale_pre_cfg_patch(args):
            conds_out = args["conds_out"]
            args['cond_scale'] = args['cond_scale'] * 2
            uncond = torch.any(conds_out[1])

            if not uncond:
                return conds_out
            
            sigmas = get_sigmas(args)
            sigma_max = sigmas[0]

            sigma = args["sigma"][0].item()
            if mode == "steps":
                progression = approx_index(sigmas, sigma) / (len(sigmas) - 1)
            else:
                progression = 1 - sigma / sigma_max

            progression = max(min(progression, 1), 0)
            current_multiplier = start_mult * (1 - progression) + end_mult * progression

            conds_out[0] = conds_out[0] * current_multiplier
            conds_out[1] = conds_out[1] * current_multiplier

            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(variable_scale_pre_cfg_patch)
        return io.NodeOutput(m,)

#-------------------------------------------------------------------------------#

class CompressConds(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-CompressConds',
            display_name='Compress Conds',
            category='SV Nodes/Experiments',
            inputs=[
            io.Conditioning.Input('positive'),
            io.Conditioning.Input('negative', optional=True),
            ],
            outputs=[
            io.Conditioning.Output(display_name='positive'),
            io.Conditioning.Output(display_name='negative'),
            ]
        )

    
    @classmethod
    def execute(cls, positive, negative=None) -> io.NodeOutput:
        pos = cls.compress(positive)
        neg = cls.compress(negative) if negative is not None else None
        return io.NodeOutput(pos, neg)
        
    @classmethod
    def compress(cls, conditioning):
        out = copy.deepcopy(conditioning)
        for o in out:
            o[0] = cls.reduc(o[0])
        return out
    
    @classmethod
    def reduc(cls, o):
        u,s,v = torch.svd(o)
        return (u[0,:75,:75]@s[0,:75].diag()@v[0,:,:75].T)[None]


#-------------------------------------------------------------------------------#

class GetImageSize(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='SV-GetImageSize',
            display_name='Image Size',
            category='SV Nodes/Logic',
            inputs=[
            io.Image.Input('image'),
            ],
            outputs=[
            io.Int.Output(display_name='width'),
            io.Int.Output(display_name='height'),
            ]
        )

    
    @classmethod
    def execute(cls, image) -> io.NodeOutput:
        samples = image.movedim(-1,1)
        size_w = samples.shape[3]
        size_h = samples.shape[2]

        return io.NodeOutput(size_w, size_h,)


NODE_LIST = [
    SimpleText,
    WildcardProcessing,
    WildcardLoader,
    WildcardString,
    PromptProcessing,
    PromptProcessingRecursive,
    PromptProcessingAdvanced,
    PromptProcessingEncode,
    PromptProcessingEncodeList,
    PromptProcessingGetCond,
    PromptProcessingSimple,
    PromptProcessingPromptControl,
    PromptProcessingVars,
    ResolutionSelector,
    ResolutionSelector2,
    ResolutionSelector2Output,
    NormalizeImageSize,
    NormalizeImageSize64,
    BasicParams,
    BasicParamsPlus,
    BasicParamsStartEnd,
    BasicParamsCustom,
    BasicParamsOutput,
    SamplerNameToSampler,
    StringSeparator,
    LoraSeparator,
    StringCombine,
    LoadTextFile,
    SaveTextFile,
    BooleanNot,
    MathAddInt,
    MathCompare,
    EquationCompare,
    SigmaOneStep,
    SigmaRange,
    SigmaContinue,
    SigmaContinueLinear,
    SigmaRemap,
    SigmaRescale,
    SigmaConcat,
    SigmaEmpty,
    SigmaAsFloat,
    SigmaStartEnd,
    SigmaLength,
    SigmaStrength,
    SigmaReverse,
    NormalizeSamples,
    ModelName,
    PromptPlusModel,
    PromptPlusModelOutput,
    InputSelect,
    InputSelectBoolean,
    InputSelectCompare,
    FlowBlocker,
    FlowGate,
    FlowGateMulti,
    IfBranch,
    ForLoopOpen,
    ForLoopClose,
    WhileLoopOpen,
    WhileLoopClose,
    IntMathOperation,
    ToBoolNode,
    AccumulateNode,
    AccumulationHeadNode,
    AccumulationTailNode,
    AccumulationToListNode,
    ListToAccumulationNode,
    AccumulationGetLengthNode,
    AccumulationGetItemNode,
    AccumulationSetItemNode,
    HashObject,
    HashItems,
    HashModel,
    CacheObject,
    ManualCache,
    ClearCustomCaches,
    FlowNode,
    FlowPipeInput,
    FlowPipeOutput,
    FlowPipeInputLarge,
    FlowPipeOutputLarge,
    FlowPipeInputIndex,
    FlowPipeOutputIndex,
    FlowPipeInputKey,
    FlowPipeOutputKey,
    FlowPipeInputKeyTuple,
    FlowPipeOutputKeyTuple,
    FlowPipeInputModel,
    FlowPipeOutputModel,
    FlowPipeInputParams,
    FlowPipeOutputParams,
    FlowPipeCombine,
    CheckNone,
    CheckNoneNot,
    DefaultInt,
    DefaultFloat,
    DefaultString,
    DefaultBoolean,
    DefaultValue,
    AnyToAny,
    ConsolePrint,
    ConsolePrintMulti,
    ConsolePrintLoop,
    AssertNotNone,
    TimerStart,
    TimerEnd,
    CurveFromEquation,
    ApplyCurve,
    ApplyCurveFromStep,
    MathOperation,
    FloatRerouteForSubnodes,
    ModelReroute,
    SigmaReroute,
    ConditioningReroute,
    ImageReroute,
    SwapValues,
    VariableSet,
    VariableGet,
    VariableClear,
    ValueRepeater,
    ValueGate,
    PrimitiveFloat,
    UnitFloat,
    MetadataJson,
    PadImage,
    SV_SkimmedCFGDifference,
    SV_SkimmedCFGDual,
    SV_CondDiffSharpening,
    SV_CFGVariableScale,
    CompressConds,
    GetImageSize,
]


#-------------------------------------------------------------------------------#
# Extension

class SVNodesExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return NODE_LIST

async def comfy_entrypoint() -> SVNodesExtension:
    return SVNodesExtension()
