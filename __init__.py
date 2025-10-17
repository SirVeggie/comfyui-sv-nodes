import os
import time
import copy
import comfy.samplers
import folder_paths
from .logic import calculate_sigma_range, calculate_sigma_range_percent, clean_prompt, default, needs_seed, process, process_advanced, process_simple, process_control, process_vars, process_wildcards, remove_comments, separate_lora, separate_lora_advanced, unescape_prompt
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

#-------------------------------------------------------------------------------#
# Helper classes

def MakeSmartType(t):
    if isinstance(t, str):
        return SmartType(t)
    return t

class SmartType(str):
    def __ne__(self, other):
        if self == "*" or other == "*":
            return False
        selfset = set(self.split(','))
        otherset = set(other.split(','))
        return not selfset.issubset(otherset)

def VariantSupport():
    def decorator(cls):
        if hasattr(cls, "INPUT_TYPES"):
            old_input_types = getattr(cls, "INPUT_TYPES")
            def new_input_types(*args, **kwargs):
                types = old_input_types(*args, **kwargs)
                for category in ["required", "optional"]:
                    if category not in types:
                        continue
                    for key, value in types[category].items():
                        if isinstance(value, tuple):
                            types[category][key] = (MakeSmartType(value[0]),) + value[1:]
                return types
            setattr(cls, "INPUT_TYPES", new_input_types)
        if hasattr(cls, "RETURN_TYPES"):
            old_return_types = cls.RETURN_TYPES
            setattr(cls, "RETURN_TYPES", tuple(MakeSmartType(x) for x in old_return_types))
        if hasattr(cls, "VALIDATE_INPUTS"):
            # Reflection is used to determine what the function signature is, so we can't just change the function signature
            raise NotImplementedError("VariantSupport does not support VALIDATE_INPUTS yet")
        else:
            def validate_inputs(input_types):
                inputs = cls.INPUT_TYPES()
                for key, value in input_types.items():
                    if isinstance(value, SmartType):
                        continue
                    if "required" in inputs and key in inputs["required"]:
                        expected_type = inputs["required"][key][0]
                    elif "optional" in inputs and key in inputs["optional"]:
                        expected_type = inputs["optional"][key][0]
                    else:
                        expected_type = None
                    if expected_type is not None and MakeSmartType(value) != expected_type:
                        return f"Invalid type of {key}: {value} (expected {expected_type})"
                return True
            setattr(cls, "VALIDATE_INPUTS", validate_inputs)
        return cls
    return decorator

#-------------------------------------------------------------------------------#
# Mappings

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

#-------------------------------------------------------------------------------#
# Classes

class SimpleText:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Input"
    
    def run(self, text):
        if not isinstance(text, str):
            raise TypeError("Invalid text input type")
        return (text,)

NODE_CLASS_MAPPINGS["SV-SimpleText"] = SimpleText
NODE_DISPLAY_NAME_MAPPINGS["SV-SimpleText"] = "Simple Text"

#-------------------------------------------------------------------------------#

class WildcardProcessing:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "wildcards": ("wildcards",),
                "seed": ("INT", {"forceInput": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Processing"
    
    def check_lazy_status(self, text, **kwargs):
        return ["wildcards"] if "__" in text else []
    def run(self, text, wildcards, seed):
        return process_wildcards(text, wildcards, seed, 5)
    
    @classmethod
    def IS_CACHED(s, text, wildcards, seed):
        return f"{text} {seed}"
    
NODE_CLASS_MAPPINGS["SV-WildcardProcessing"] = WildcardProcessing
NODE_DISPLAY_NAME_MAPPINGS["SV-WildcardProcessing"] = "Wildcard Processing"

#-------------------------------------------------------------------------------#

class WildcardLoader:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dir_path": ("STRING",)
            }
        }
    
    RETURN_TYPES = ("wildcards",)
    RETURN_NAMES = ("wildcards",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/IO"
    
    def run(self, dir_path):
        wildcards = {}
        if not os.path.isdir(dir_path):
            return (wildcards,)
        
        for filename in os.listdir(dir_path):
            filepath = os.path.join(dir_path, filename)
            if not os.path.isfile(filepath):
                continue
            key = os.path.splitext(filename)[0]
            with open(filepath, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                wildcards[key] = lines
        return (wildcards,)
    
    @classmethod
    def IS_CACHED(s, dir_path):
        return os.path.getmtime(dir_path)
    
NODE_CLASS_MAPPINGS["SV-WildcardLoader"] = WildcardLoader
NODE_DISPLAY_NAME_MAPPINGS["SV-WildcardLoader"] = "Load Wildcards"

#-------------------------------------------------------------------------------#

class WildcardTemp:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("STRING", {"multiline": True}),
            },
            "optional": {
                "wildcards": ("wildcards",),
            }
        }
    
    RETURN_TYPES = ("wildcards",)
    RETURN_NAMES = ("wildcards",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Input"
    
    key_regex = re.compile(r"^\[[\w -]+\]$")
    def run(self, input: str, wildcards: dict[str: list[str]] = None):
        result = copy.deepcopy(wildcards) if wildcards is not None else {}
        lines = [line.strip() for line in input.splitlines() if line.strip()]
        current_key = None
        for line in lines:
            if line.startswith("#") or line.startswith("//"):
                continue
            elif WildcardTemp.key_regex.match(line):
                current_key = line[1:-1].strip()
                if current_key not in result:
                    result[current_key] = []
            elif current_key is not None:
                result[current_key].append(line)
        return (result,)

#-------------------------------------------------------------------------------#

class PromptProcessing:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True})
            },
            "optional": {
                "variables": ("STRING", {"forceInput": True}),
                "seed": ("INT", {"forceInput": True})
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("1st pass", "2nd pass", "3rd pass")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Processing"
    
    def run(self, text, variables="", seed=1):
        text = remove_comments(text)
        return process(text, 0, variables, seed), process(text, 1, variables, seed), process(text, 2, variables, seed)
    
    @classmethod
    def IS_CACHED(s, text, variables, seed):
        return f"{text} {variables} {seed}"

NODE_CLASS_MAPPINGS["SV-PromptProcessing"] = PromptProcessing
NODE_DISPLAY_NAME_MAPPINGS["SV-PromptProcessing"] = "Prompt Processing"

#-------------------------------------------------------------------------------#

class PromptProcessingRecursive:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "step": ("INT", {"forceInput": True}),
                "progress": ("FLOAT", {"forceInput": True}),
            },
            "optional": {
                "variables": ("STRING", {"forceInput": True}),
                "seed": ("INT", {"forceInput": True, "lazy": True}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "lora")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Processing"
    
    def check_lazy_status(self, text, **kwargs):
        if needs_seed(text):
            return ["seed"]
        return []
    
    def run(self, text, step, progress, variables="", seed=1):
        text = remove_comments(text)
        return LoraSeparator.run(self, process_advanced(text, variables, seed, step, progress))

NODE_CLASS_MAPPINGS["SV-PromptProcessingRecursive"] = PromptProcessingRecursive
NODE_DISPLAY_NAME_MAPPINGS["SV-PromptProcessingRecursive"] = "Recursive Processing"

#-------------------------------------------------------------------------------#

class PromptProcessingAdvanced:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True}),
                "steps": ("INT", {"forceInput": True}),
            },
            "optional": {
                "phase": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "variables": ("STRING", {"forceInput": True}),
                "seed": ("INT", {"forceInput": True, "lazy": True}),
            }
        }
    
    RETURN_TYPES = ("sv_prompt", "STRING")
    RETURN_NAMES = ("prompt", "lora")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Processing"
    
    def check_lazy_status(self, prompt, **kwargs):
        if needs_seed(prompt):
            return ["seed"]
        return []
    
    def run(self, prompt, steps, phase=1, variables="", seed=1):
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

NODE_CLASS_MAPPINGS["SV-PromptProcessingAdvanced"] = PromptProcessingAdvanced
NODE_DISPLAY_NAME_MAPPINGS["SV-PromptProcessingAdvanced"] = "Advanced Processing"

#-------------------------------------------------------------------------------#

class PromptProcessingEncode:
    cache = {}
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("sv_prompt",),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Processing"
    
    def run(self, clip, prompt):
        steps = len(prompt)
        pconds = []
        nconds = []
        
        for i in range(1, steps + 1):
            start = (i - 1) / steps
            end = i / steps
            pos, neg = prompt[i - 1]
            
            if not self.cacheHas(pos):
                self.cacheSet(pos, encode(clip, pos))
            pcond = node_helpers.conditioning_set_values(self.cacheGet(pos), {"start_percent": start, "end_percent": end})
            pconds += pcond
            if not self.cacheHas(neg):
                self.cacheSet(neg, encode(clip, neg))
            ncond = node_helpers.conditioning_set_values(self.cacheGet(neg), {"start_percent": start, "end_percent": end})
            nconds += ncond
        
        self.cacheClean()
        return pconds, nconds
    
    def cacheHas(self, prompt):
        return prompt in PromptProcessingEncode.cache
    def cacheSet(self, prompt, encoded):
        PromptProcessingEncode.cache[prompt] = [prompt, encoded, time.time_ns() // 1000000]
    def cacheGet(self, prompt):
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

NODE_CLASS_MAPPINGS["SV-PromptProcessingEncode"] = PromptProcessingEncode
NODE_DISPLAY_NAME_MAPPINGS["SV-PromptProcessingEncode"] = "Encode Prompt"

#-------------------------------------------------------------------------------#

class PromptProcessingEncodeList:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("sv_prompt",),
            }
        }
    
    RETURN_TYPES = ("cond_list",)
    RETURN_NAMES = ("conds",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Processing"
    
    def run(self, clip, prompt):
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
        
        return (conds,)

NODE_CLASS_MAPPINGS["SV-PromptProcessingEncodeList"] = PromptProcessingEncodeList
NODE_DISPLAY_NAME_MAPPINGS["SV-PromptProcessingEncodeList"] = "Encode Prompt List"

#-------------------------------------------------------------------------------#

class PromptProcessingGetCond:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conds": ("cond_list",),
                "step": ("INT", {"defaultInput": True}),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Processing"
    
    def run(self, conds, step):
        return conds[step - 1]

NODE_CLASS_MAPPINGS["SV-PromptProcessingGetCond"] = PromptProcessingGetCond
NODE_DISPLAY_NAME_MAPPINGS["SV-PromptProcessingGetCond"] = "Get Conditioning"

#-------------------------------------------------------------------------------#

class PromptProcessingSimple:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "variables": ("STRING", {"forceInput": True}),
                "seed": ("INT", {"forceInput": True, "lazy": True}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("pos 1", "neg 1", "pos 2", "neg 2")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Processing"
    
    def check_lazy_status(self, prompt, **kwargs):
        if needs_seed(prompt):
            return ["seed"]
        return []
    
    def run(self, prompt, variables="", seed=1):
        prompt = remove_comments(prompt)
        parts = re.split(r"[\n\r]+[\s]*-+[\s]*[\n\r]+", prompt, 1)
        full_positive = parts[0]
        full_negative = parts[1] if len(parts) > 1 else ""
        variables += "\npositive=" + clean_prompt(separate_lora(full_positive)[0])
        
        pos1 = process_simple(full_positive, variables, seed, False)
        neg1 = process_simple(full_negative, variables, seed, False)
        pos2 = process_simple(full_positive, variables, seed, True)
        neg2 = process_simple(full_negative, variables, seed, True)
        
        return (pos1, neg1, pos2, neg2)

NODE_CLASS_MAPPINGS["SV-PromptProcessingSimple"] = PromptProcessingSimple
NODE_DISPLAY_NAME_MAPPINGS["SV-PromptProcessingSimple"] = "Simple Processing"

#-------------------------------------------------------------------------------#

class PromptProcessingPromptControl:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True}),
                "steps": ("INT", {"forceInput": True}),
            },
            "optional": {
                "phase": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "variables": ("STRING", {"forceInput": True}),
                "seed": ("INT", {"forceInput": True, "lazy": True}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive", "negative")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Processing"
    
    def check_lazy_status(self, prompt, **kwargs):
        if needs_seed(prompt):
            return ["seed"]
        return []
    
    def run(self, prompt, steps, phase=1, variables="", seed=1):
        prompt = remove_comments(prompt)
        parts = re.split(r"[\n\r]+[\s]*-+[\s]*[\n\r]+", prompt, 1)
        full_positive = parts[0]
        full_negative = parts[1] if len(parts) > 1 else ""
        variables += "\npositive=" + clean_prompt(separate_lora(full_positive)[0])
        
        pos = process_control(full_positive, steps, phase, variables, seed)
        neg = process_control(full_negative, steps, phase, variables, seed)
        
        return (pos, neg)

NODE_CLASS_MAPPINGS["SV-PromptProcessingPromptControl"] = PromptProcessingPromptControl
NODE_DISPLAY_NAME_MAPPINGS["SV-PromptProcessingPromptControl"] = "Control Processing"

#-------------------------------------------------------------------------------#

class PromptProcessingVars:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "variables": ("STRING", {"forceInput": True}),
                "seed": ("INT", {"forceInput": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Processing"
    
    def run(self, prompt, variables="", seed=1):
        prompt = remove_comments(prompt)
        parts = re.split(r"[\n\r]+[\s]*-+[\s]*[\n\r]+", prompt, 1)
        full_positive = parts[0]
        variables += "\npositive=" + clean_prompt(separate_lora(full_positive)[0])
        
        if len(parts) <= 1:
            return (process_vars(parts[0], variables, seed),)
        parts[0] = process_vars(parts[0], variables, seed)
        parts[1] = process_vars(parts[1], variables, seed)
        
        return (f"{parts[0]}\n---\n{parts[1]}",)

NODE_CLASS_MAPPINGS["SV-PromptProcessingVars"] = PromptProcessingVars
NODE_DISPLAY_NAME_MAPPINGS["SV-PromptProcessingVars"] = "Variable Processing"

#-------------------------------------------------------------------------------#

class PromptProcessingWildcards:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True}),
                "wildcards": ("wildcards",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Processing"
    
    def run(self, prompt, wildcards):
        None

#NODE_CLASS_MAPPINGS["SV-PromptProcessingWildcards"] = PromptProcessingWildcards
#NODE_DISPLAY_NAME_MAPPINGS["SV-PromptProcessingWildcards"] = "Wildcard Processing"

#-------------------------------------------------------------------------------#

class LoadWildcards:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dir_path": ("STRING",)
            }
        }
    
    RETURN_TYPES = ("wildcards",)
    RETURN_NAMES = ("wildcards",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/IO"
    
    def run(self, dir_path):
        None

#NODE_CLASS_MAPPINGS["SV-LoadWildcards"] = LoadWildcards
#NODE_DISPLAY_NAME_MAPPINGS["SV-LoadWildcards"] = "Load Wildcards"

#-------------------------------------------------------------------------------#

class ResolutionSelector:
    RATIOS = ["1:1", "5:4", "4:3", "3:2", "16:9", "21:9"]
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 64}),
                "ratio": (ResolutionSelector.RATIOS,),
                "orientation": ("BOOLEAN", {"default": False, "label_on": "portrait", "label_off": "landscape"})
            },
            "optional": {
                "seed": ("*", {"default": -1}),
                "random": ("*", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Input"
    
    def run(self, base, ratio, orientation, seed=-1, random=""):
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

NODE_CLASS_MAPPINGS["SV-ResolutionSelector"] = ResolutionSelector
NODE_DISPLAY_NAME_MAPPINGS["SV-ResolutionSelector"] = "Resolution Selector"

#-------------------------------------------------------------------------------#

class ResolutionSelector2:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 64}),
                "ratio": (ResolutionSelector.RATIOS,),
                "orientation": ("BOOLEAN", {"default": False, "label_on": "portrait", "label_off": "landscape"}),
                "hires": ("FLOAT", {"min": 1, "max": 4, "step": 0.1, "default": 1.5}),
                "batch": ("INT", {"min": 1, "max": 32, "step": 1, "default": 1})
            },
            "optional": {
                "seed": ("*", {"default": -1}),
                "random": ("*", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("RS_OUTPUT",)
    RETURN_NAMES = ("packet",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Input"
    
    def run(self, base, ratio, orientation, hires, batch, seed=-1, random=""):
        result = ResolutionSelector.run(self, base, ratio, orientation, seed, random)
        return ((result[0], result[1], hires, batch),)
        
NODE_CLASS_MAPPINGS["SV-ResolutionSelector2"] = ResolutionSelector2
NODE_DISPLAY_NAME_MAPPINGS["SV-ResolutionSelector2"] = "Resolution Selector 2"

#-------------------------------------------------------------------------------#

class ResolutionSelector2Output:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "packet": ("RS_OUTPUT",)
            }
        }
    
    RETURN_TYPES = ("INT", "INT", "FLOAT", "INT")
    RETURN_NAMES = ("width", "height", "hires ratio", "batch size")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Input"
    
    def run(self, packet):
        if not isinstance(packet, tuple):
            raise TypeError("Invalid packet input type")
        if len(packet) != 4:
            raise ValueError("Invalid packet length")
        return packet

NODE_CLASS_MAPPINGS["SV-ResolutionSelector2Output"] = ResolutionSelector2Output
NODE_DISPLAY_NAME_MAPPINGS["SV-ResolutionSelector2Output"] = "Selector Output"

#-------------------------------------------------------------------------------#

class NormalizeImageSize:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"forceInput": True}),
                "height": ("INT", {"forceInput": True}),
                "size": ("INT", {"min": 64, "max": 4096, "step": 64, "default": 768})
            }
        }
    
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Input"
    
    def run(self, width, height, size):
        ratio = math.sqrt(float(width) / float(height))
        width = math.floor(size * ratio / 64) * 64
        height = math.floor(size / ratio / 64) * 64
        return width, height

NODE_CLASS_MAPPINGS["SV-NormalizeImageSize"] = NormalizeImageSize
NODE_DISPLAY_NAME_MAPPINGS["SV-NormalizeImageSize"] = "Normalize Size"

#-------------------------------------------------------------------------------#

class NormalizeImageSize64:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"forceInput": True}),
                "height": ("INT", {"forceInput": True})
            }
        }
    
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Input"
    
    def run(self, width, height):
        width = math.floor(width / 64) * 64
        height = math.floor(height / 64) * 64
        return width, height

NODE_CLASS_MAPPINGS["SV-NormalizeImageSize64"] = NormalizeImageSize64
NODE_DISPLAY_NAME_MAPPINGS["SV-NormalizeImageSize64"] = "Normalize Size (64)"

#-------------------------------------------------------------------------------#

class BasicParams:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cfg": ("FLOAT", {"min": 0, "max": 20, "step": 0.1, "default": 8.0}),
                "steps": ("INT", {"min": 1, "max": 100, "step": 1, "default": 10}),
                "denoise": ("FLOAT", {"min": 0, "max": 1, "step": 0.01, "default": 1.0}),
                "sampler": (comfy.samplers.SAMPLER_NAMES,)
            }
        }
    
    RETURN_TYPES = ("BP_OUTPUT",)
    RETURN_NAMES = ("packet",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Input"
    
    def run(self, cfg, steps, denoise, sampler):
        if not isinstance(cfg, float) and not isinstance(cfg, int):
            raise TypeError("Invalid cfg input type")
        if not isinstance(steps, int):
            raise TypeError("Invalid steps input type")
        if not isinstance(denoise, float) and not isinstance(denoise, int):
            raise TypeError("Invalid denoise input type")
        if not isinstance(sampler, str):
            raise TypeError("Invalid sampler input type")
        return ((cfg, steps, denoise, sampler, None, None, None, None),)

NODE_CLASS_MAPPINGS["SV-BasicParams"] = BasicParams
NODE_DISPLAY_NAME_MAPPINGS["SV-BasicParams"] = "Params"

#-------------------------------------------------------------------------------#

class BasicParamsPlus:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cfg": ("FLOAT", {"min": 0, "max": 20, "step": 0.1, "default": 8.0}),
                "steps": ("INT", {"min": 1, "max": 100, "step": 1, "default": 10}),
                "denoise": ("FLOAT", {"min": 0, "max": 1, "step": 0.01, "default": 1.0}),
                "sampler": (comfy.samplers.SAMPLER_NAMES,),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES + ["ays"],)
            }
        }
    
    RETURN_TYPES = ("BP_OUTPUT",)
    RETURN_NAMES = ("packet",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Input"
    
    def run(self, cfg, steps, denoise, sampler, scheduler):
        if not isinstance(cfg, float) and not isinstance(cfg, int):
            raise TypeError("Invalid cfg input type")
        if not isinstance(steps, int):
            raise TypeError("Invalid steps input type")
        if not isinstance(denoise, float) and not isinstance(denoise, int):
            raise TypeError("Invalid denoise input type")
        if not isinstance(sampler, str):
            raise TypeError("Invalid sampler input type")
        return ((cfg, steps, denoise, sampler, scheduler, scheduler == "ays", None, None),)

NODE_CLASS_MAPPINGS["SV-BasicParamsPlus"] = BasicParamsPlus
NODE_DISPLAY_NAME_MAPPINGS["SV-BasicParamsPlus"] = "Params Plus"

#-------------------------------------------------------------------------------#

class BasicParamsStartEnd:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cfg": ("FLOAT", {"min": 0, "max": 20, "step": 0.1, "default": 8.0}),
                "steps": ("INT", {"min": 1, "max": 100, "step": 1, "default": 10}),
                "sampler": (comfy.samplers.SAMPLER_NAMES,),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES + ["ays"],),
                "start": ("FLOAT", {"min": 0, "max": 1, "step": 0.01, "default": 0.0}),
                "end": ("FLOAT", {"min": 0, "max": 1, "step": 0.01, "default": 1.0}),
            }
        }
    
    RETURN_TYPES = ("BP_OUTPUT",)
    RETURN_NAMES = ("packet",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Input"
    
    def run(self, cfg, steps, start, end, sampler, scheduler):
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
        return ((cfg, steps, None, sampler, scheduler, scheduler == "ays", start, end),)

NODE_CLASS_MAPPINGS["SV-BasicParamsStartEnd"] = BasicParamsStartEnd
NODE_DISPLAY_NAME_MAPPINGS["SV-BasicParamsStartEnd"] = "Params Start/End"

#-------------------------------------------------------------------------------#

class BasicParamsCustom:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "cfg": ("FLOAT", {"min": 0, "max": 20, "step": 0.1, "default": 8.0}),
                "steps": ("INT", {"min": 1, "max": 100, "step": 1, "default": 10}),
                "sampler": (comfy.samplers.SAMPLER_NAMES,),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES + ["ays"],),
            }
        }
    
    RETURN_TYPES = ("BP_OUTPUT",)
    RETURN_NAMES = ("packet",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Input"
    
    def run(self, cfg=8.0, steps=10, sampler="euler", scheduler="normal"):
        if not isinstance(cfg, float) and not isinstance(cfg, int):
            raise TypeError("Invalid cfg input type")
        if not isinstance(steps, int):
            raise TypeError("Invalid steps input type")
        if not isinstance(sampler, str):
            raise TypeError("Invalid sampler input type")
        if not isinstance(scheduler, str):
            raise TypeError("Invalid scheduler input type")
        return ((cfg, steps, None, sampler, scheduler, scheduler == "ays", None, None),)

NODE_CLASS_MAPPINGS["SV-BasicParamsCustom"] = BasicParamsCustom
NODE_DISPLAY_NAME_MAPPINGS["SV-BasicParamsCustom"] = "Params Custom"

#-------------------------------------------------------------------------------#

class BasicParamsOutput:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "packet": ("BP_OUTPUT",)
            }
        }
    
    RETURN_TYPES = ("FLOAT", "INT", "FLOAT", comfy.samplers.SAMPLER_NAMES, comfy.samplers.SCHEDULER_NAMES, "BOOLEAN", "SAMPLER", "FLOAT", "FLOAT")
    RETURN_NAMES = ("cfg", "steps", "denoise", "sampler", "scheduler", "ays", "SAMPLER", "start", "end")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Input"
    
    def run(self, packet):
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

NODE_CLASS_MAPPINGS["SV-BasicParamsOutput"] = BasicParamsOutput
NODE_DISPLAY_NAME_MAPPINGS["SV-BasicParamsOutput"] = "Params Output"

#-------------------------------------------------------------------------------#

class SamplerNameToSampler:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "name": (comfy.samplers.SAMPLER_NAMES, {"forceInput": True})
            }
        }
    
    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Input"
    
    def run(self, name):
        if not isinstance(name, str):
            raise TypeError("Invalid name input type")
        if name not in comfy.samplers.SAMPLER_NAMES:
            raise ValueError("Invalid name")
        return (comfy.samplers.sampler_object(name),)

NODE_CLASS_MAPPINGS["SV-SamplerNameToSampler"] = SamplerNameToSampler
NODE_DISPLAY_NAME_MAPPINGS["SV-SamplerNameToSampler"] = "Sampler Converter"

#-------------------------------------------------------------------------------#

class StringSeparator:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "separator": ("STRING", {"default": "\\n---\\n"})
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("part1", "part2")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Processing"
    
    def run(self, text, separator="\\n---\\n"):
        if not isinstance(text, str):
            raise TypeError("Invalid text input type")
        if not isinstance(separator, str):
            raise TypeError("Invalid separator input type")
        separator = separator.replace("\\n", "\n").replace("\\t", "\t")
        parts = text.split(separator, 1)
        return parts[0], parts[1] if len(parts) > 1 else ""

NODE_CLASS_MAPPINGS["SV-StringSeparator"] = StringSeparator
NODE_DISPLAY_NAME_MAPPINGS["SV-StringSeparator"] = "String Separator"

#-------------------------------------------------------------------------------#

class LoraSeparator:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True})
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "lora")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Processing"
    
    def run(self, text):
        return separate_lora(text)

NODE_CLASS_MAPPINGS["SV-LoraSeparator"] = LoraSeparator
NODE_DISPLAY_NAME_MAPPINGS["SV-LoraSeparator"] = "Lora Separator"

#-------------------------------------------------------------------------------#

class StringCombine:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "part1": ("STRING", {"forceInput": True}),
                "part2": ("STRING", {"forceInput": True}),
                "separator": ("STRING", {"default": "\\n"})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Processing"
    
    def run(self, part1, part2, separator="\\n"):
        if not isinstance(part1, str) or not isinstance(part2, str):
            raise TypeError("Invalid part input type")
        if not isinstance(separator, str):
            raise TypeError("Invalid separator input type")
        separator = separator.replace("\\n", "\n").replace("\\t", "\t")
        return (part1 + separator + part2,)

NODE_CLASS_MAPPINGS["SV-StringCombine"] = StringCombine
NODE_DISPLAY_NAME_MAPPINGS["SV-StringCombine"] = "String Combine"

#-------------------------------------------------------------------------------#

class LoadTextFile:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "path": ("STRING", {"default": "", "multiline": False})
            }
        }
    
    @classmethod
    def IS_CHANGED(s, path):
        if os.path.exists(path):
            return os.path.getmtime(path)
        return ""
    
    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("content", "success")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/IO"
    
    def run(self, path):
        if not isinstance(path, str):
            raise TypeError("Invalid path input type")
        try:
            with open(path, "r", encoding="utf-8") as file:
                return (file.read(), True)
        except Exception as e:
            print(e)
            return ("", False)

NODE_CLASS_MAPPINGS["SV-LoadTextFile"] = LoadTextFile
NODE_DISPLAY_NAME_MAPPINGS["SV-LoadTextFile"] = "Load Text File"

#-------------------------------------------------------------------------------#

class SaveTextFile:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "path": ("STRING", {"default": "", "multiline": False}),
                "content": ("STRING", {"forceInput": True})
            }
        }
    
    @classmethod
    def IS_CACHED(s, path, content):
        return path + " " + content
    
    OUTPUT_NODE = True
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("success",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/IO"
    
    def run(self, path, content):
        if not isinstance(path, str):
            raise TypeError("Invalid path input type")
        if not isinstance(content, (str, int, float, bool)):
            raise TypeError("Invalid content input type")
        try:
            with open(path, "w") as file:
                file.write(str(content))
            return (True,)
        except:
            return (False,)

NODE_CLASS_MAPPINGS["SV-SaveTextFile"] = SaveTextFile
NODE_DISPLAY_NAME_MAPPINGS["SV-SaveTextFile"] = "Save Text File"

#-------------------------------------------------------------------------------#

class BooleanNot:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("BOOLEAN", {"forceInput": True})
            }
        }
    
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("bool",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Logic"
    
    def run(self, value):
        return (not value,)

NODE_CLASS_MAPPINGS["SV-BooleanNot"] = BooleanNot
NODE_DISPLAY_NAME_MAPPINGS["SV-BooleanNot"] = "Boolean Not"

#-------------------------------------------------------------------------------#

class MathAddInt:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "int": ("INT", {"forceInput": True}),
                "add": ("INT", {"default": 1, "min": -sys.maxsize, "max": sys.maxsize, "step": 1})
            }
        }
    
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Logic"
    
    def run(self, int, add):
        return (int + add,)

NODE_CLASS_MAPPINGS["SV-MathAddInt"] = MathAddInt
NODE_DISPLAY_NAME_MAPPINGS["SV-MathAddInt"] = "Add Int"

#-------------------------------------------------------------------------------#

@VariantSupport()
class MathCompare:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "number": ("*", {"forceInput": True}),
                "operator": ([">", "<", ">=", "<=", "==", "!="],),
                "other": ("FLOAT", {"default": 0, "min": -sys.float_info.max, "max": sys.float_info.max, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("bool",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Logic"
    
    def run(self, number, other, operator):
        if operator == "<":
            return (number < other,)
        if operator == ">":
            return (number > other,)
        if operator == "<=":
            return (number <= other,)
        if operator == ">=":
            return (number >= other,)
        if operator == "==":
            return (number == other,)
        if operator == "!=":
            return (number != other,)
        return (False,)

NODE_CLASS_MAPPINGS["SV-MathCompare"] = MathCompare
NODE_DISPLAY_NAME_MAPPINGS["SV-MathCompare"] = "Simple Compare"

#-------------------------------------------------------------------------------#

@VariantSupport()
class EquationCompare:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "equation": ("STRING", {"multiline": False, "default": ""}),
            },
            "optional": {
                "a": ("*",),
                "b": ("*",),
            }
        }
    
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("bool",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Logic"
    
    def run(self, equation, a=None, b=None):
        equation = re.sub(r"\s+", "", equation)
        equation = re.sub(r"(?<=\d)a(?!nd)", "*" + str(a), equation)
        equation = re.sub(r"a(?!nd)", str(a), equation)
        equation = re.sub(r"(?<=\d)b", "*" + str(b), equation)
        equation = re.sub(r"b", str(b), equation)
        return (evaluateComparison(equation, 0),)

NODE_CLASS_MAPPINGS["SV-EquationCompare"] = EquationCompare
NODE_DISPLAY_NAME_MAPPINGS["SV-EquationCompare"] = "Equation Compare"

#-------------------------------------------------------------------------------#

class SigmaOneStep:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS",)
            }
        }
    
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Sigmas"
    
    def run(self, sigmas):
        lastSigma = sigmas[-1].item()
        return (torch.FloatTensor([lastSigma, 0]).cpu(),)

NODE_CLASS_MAPPINGS["SV-SigmaOneStep"] = SigmaOneStep
NODE_DISPLAY_NAME_MAPPINGS["SV-SigmaOneStep"] = "Sigmas One Step"

#-------------------------------------------------------------------------------#

class SigmaRange:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS",),
                "start": ("INT", {"default": 0, "min": 0, "max": 100}),
                "end": ("INT", {"default": 0, "min": 0, "max": 100}),
            }
        }
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)

    FUNCTION = "run"
    CATEGORY = "SV Nodes/Sigmas"

    def run(self, sigmas, start, end):
        return (sigmas[start:end + 1],)

NODE_CLASS_MAPPINGS["SV-SigmaRange"] = SigmaRange
NODE_DISPLAY_NAME_MAPPINGS["SV-SigmaRange"] = "Sigma Range"

#-------------------------------------------------------------------------------#

class SigmaContinue:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source": ("SIGMAS",),
                "imitate": ("SIGMAS",),
                "steps": ("INT", {"min": 1, "max": 100, "step": 1, "default": 1}),
            }
        }
    
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Sigmas"
    
    def run(self, source, imitate, steps):
        if steps < 1 or len(source) < 1:
            return (torch.FloatTensor([]).cpu(),)
        lastSigma = source[-1].item()
        if lastSigma < 0.0001:
            return (torch.FloatTensor([]).cpu(),)
        return (torch.FloatTensor(calculate_sigma_range(imitate.tolist(), lastSigma, 0, steps)).cpu(),)

NODE_CLASS_MAPPINGS["SV-SigmaContinue"] = SigmaContinue
NODE_DISPLAY_NAME_MAPPINGS["SV-SigmaContinue"] = "Sigma Continue"

#-------------------------------------------------------------------------------#

class SigmaContinueLinear:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source": ("SIGMAS",),
                "steps": ("INT", {"min": 1, "max": 100, "step": 1, "default": 1}),
            }
        }
    
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Sigmas"
    
    def run(self, source, steps):
        if steps < 1 or len(source) < 1:
            return (torch.FloatTensor([]).cpu(),)
        lastSigma = source[-1].item()
        if lastSigma < 0.0001:
            return (torch.FloatTensor([]).cpu(),)
        step = lastSigma / steps
        return (torch.FloatTensor([step * i for i in reversed(range(0, steps + 1))]).cpu(),)

NODE_CLASS_MAPPINGS["SV-SigmaContinueLinear"] = SigmaContinueLinear
NODE_DISPLAY_NAME_MAPPINGS["SV-SigmaContinueLinear"] = "Sigma Linear"

#-------------------------------------------------------------------------------#

class SigmaRemap:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS",),
                "start": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01}),
                "end": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
                "steps": ("INT", {"min": 1, "max": 100, "step": 1, "default": 1}),
            }
        }
    
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Sigmas"
    
    def run(self, sigmas, start, end, steps):
        return (torch.FloatTensor(calculate_sigma_range_percent(sigmas.tolist(), start, end, steps)).cpu(),)

NODE_CLASS_MAPPINGS["SV-SigmaRemap"] = SigmaRemap
NODE_DISPLAY_NAME_MAPPINGS["SV-SigmaRemap"] = "Sigma Remap"

#-------------------------------------------------------------------------------#

class SigmaConcat:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas1": ("SIGMAS",),
                "sigmas2": ("SIGMAS",)
            }
        }
    
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Sigmas"
    
    def run(self, sigmas1, sigmas2):
        list1 = sigmas1.tolist()
        list2 = sigmas2.tolist()
        if len(list1) < 1:
            return (torch.FloatTensor(list2).cpu(),)
        if len(list2) < 1:
            return (torch.FloatTensor(list1).cpu(),)
        if list1[-1] == list2[0]:
            list2 = list2[1:]
        if len(list2) < 1:
            return (torch.FloatTensor(list1).cpu(),)
        return (torch.FloatTensor(list1 + list2).cpu(),)

NODE_CLASS_MAPPINGS["SV-SigmaConcat"] = SigmaConcat
NODE_DISPLAY_NAME_MAPPINGS["SV-SigmaConcat"] = "Sigma Concat"

#-------------------------------------------------------------------------------#

class SigmaEmpty:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {}
        }
    
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Sigmas"
    
    def run(self):
        return (torch.FloatTensor([]).cpu(),)

NODE_CLASS_MAPPINGS["SV-SigmaEmpty"] = SigmaEmpty
NODE_DISPLAY_NAME_MAPPINGS["SV-SigmaEmpty"] = "Sigma Empty"

#-------------------------------------------------------------------------------#

class SigmaAsFloat:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS",)
            }
        }
    
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Sigmas"
    DEPRECATED = True
    
    def run(self, sigmas):
        if len(sigmas) < 1:
            raise ValueError("Invalid sigmas length")
        return (sigmas[0].item(),)

NODE_CLASS_MAPPINGS["SV-SigmaAsFloat"] = SigmaAsFloat
NODE_DISPLAY_NAME_MAPPINGS["SV-SigmaAsFloat"] = "Sigma As Float"

#-------------------------------------------------------------------------------#

class SigmaStartEnd:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS",)
            }
        }
    
    RETURN_TYPES = ("FLOAT", "FLOAT")
    RETURN_NAMES = ("start", "end")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Sigmas"
    
    def run(self, sigmas):
        if len(sigmas) < 1:
            raise ValueError("Invalid sigmas length")
        return (sigmas[0].item(), sigmas[-1].item())

NODE_CLASS_MAPPINGS["SV-SigmaStartEnd"] = SigmaStartEnd
NODE_DISPLAY_NAME_MAPPINGS["SV-SigmaStartEnd"] = "Sigma Start/End"

#-------------------------------------------------------------------------------#

class SigmaLength:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS",)
            }
        }
    
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("length",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Sigmas"
    
    def run(self, sigmas):
        return (len(sigmas),)

NODE_CLASS_MAPPINGS["SV-SigmaLength"] = SigmaLength
NODE_DISPLAY_NAME_MAPPINGS["SV-SigmaLength"] = "Sigma Length"

#-------------------------------------------------------------------------------#

class SigmaStrength:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "sigmas": ("SIGMAS",)
            }
        }
    
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("strength",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Sigmas"
    
    def run(self, model, sigmas):
        sigma = (sigmas[0] - sigmas[-1]) / model.model.latent_format.scale_factor
        return (sigma.item(),)

NODE_CLASS_MAPPINGS["SV-SigmaStrength"] = SigmaStrength
NODE_DISPLAY_NAME_MAPPINGS["SV-SigmaStrength"] = "Sigma Strength"

#-------------------------------------------------------------------------------#

class SigmaReverse:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS",)
            }
        }
    
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Sigmas"
    
    def run(self, sigmas):
        out = sigmas.tolist()
        if out[-1] == 0:
            out[-1] = out[-1] + 0.0001
        out.reverse()
        return (torch.FloatTensor(out).cpu(),)

NODE_CLASS_MAPPINGS["SV-SigmaReverse"] = SigmaReverse
NODE_DISPLAY_NAME_MAPPINGS["SV-SigmaReverse"] = "Sigma Reverse"

#-------------------------------------------------------------------------------#

class NormalizeSamples:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",)
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("normalized",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Sampling"
    
    def run(self, latent):
        samples = latent["samples"]
        out = latent.copy()
        out["samples"] = (samples - samples.mean()) / samples.std()
        return (out,)

NODE_CLASS_MAPPINGS["SV-NormalizeSamples"] = NormalizeSamples
NODE_DISPLAY_NAME_MAPPINGS["SV-NormalizeSamples"] = "Normalize Samples"

#-------------------------------------------------------------------------------#

@VariantSupport()
class ModelName:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("checkpoints"),)
            }
        }
    
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("model name",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Input"
    
    def run(self, model):
        if not isinstance(model, str):
            raise TypeError("Invalid model input type")
        return (model,)

NODE_CLASS_MAPPINGS["SV-ModelName"] = ModelName
NODE_DISPLAY_NAME_MAPPINGS["SV-ModelName"] = "Model Name"

#-------------------------------------------------------------------------------#





# Flow Nodes
#-------------------------------------------------------------------------------#

class PromptPlusModel:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("checkpoints"),),
                "prompt": ("STRING", {"multiline": True})
            }
        }
    
    RETURN_TYPES = ("PPM_OUTPUT",)
    RETURN_NAMES = ("packet",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Input"
    
    def run(self, model, prompt):
        if not isinstance(model, str):
            raise TypeError("Invalid model input type")
        if not isinstance(prompt, str):
            raise TypeError("Invalid prompt input type")
        return ((model, prompt),)

NODE_CLASS_MAPPINGS["SV-PromptPlusModel"] = PromptPlusModel
NODE_DISPLAY_NAME_MAPPINGS["SV-PromptPlusModel"] = "Prompt + Model"

#-------------------------------------------------------------------------------#

@VariantSupport()
class PromptPlusModelOutput:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "packet": ("PPM_OUTPUT",)
            }
        }
    
    RETURN_TYPES = ("*", "STRING")
    RETURN_NAMES = ("model name", "prompt")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Input"
    
    def run(self, packet):
        if not isinstance(packet, tuple):
            raise TypeError("Invalid output input type")
        if len(packet) != 2:
            raise ValueError("Invalid output length")
        return packet

NODE_CLASS_MAPPINGS["SV-PromptPlusModelOutput"] = PromptPlusModelOutput
NODE_DISPLAY_NAME_MAPPINGS["SV-PromptPlusModelOutput"] = "P+M Output"

#-------------------------------------------------------------------------------#

@VariantSupport()
class InputSelect:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "select": ("INT", {"min": 1, "max": 5, "step": 1, "default": 1}),
            },
            "optional": {
                "_1_": ("*", {"lazy": True}),
                "_2_": ("*", {"lazy": True}),
                "_3_": ("*", {"lazy": True}),
                "_4_": ("*", {"lazy": True}),
                "_5_": ("*", {"lazy": True}),
            }
        }
    
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("out",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Flow"
    
    def check_lazy_status(self, select, **kwargs):
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
    
    def run(self, select, _1_=None, _2_=None, _3_=None, _4_=None, _5_=None):
        if select == 1:
            return (_1_,)
        if select == 2:
            return (_2_,)
        if select == 3:
            return (_3_,)
        if select == 4:
            return (_4_,)
        if select == 5:
            return (_5_,)
        return (None,)

NODE_CLASS_MAPPINGS["SV-InputSelect"] = InputSelect
NODE_DISPLAY_NAME_MAPPINGS["SV-InputSelect"] = "Input Select"

#-------------------------------------------------------------------------------#

@VariantSupport()
class InputSelectBoolean:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "select": ("BOOLEAN",),
            },
            "optional": {
                "on": ("*", {"lazy": True}),
                "off": ("*", {"lazy": True}),
            }
        }
    
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("out",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Flow"
    
    def check_lazy_status(self, select, **kwargs):
        if select:
            return ["on"]
        return ["off"]
    
    def run(self, select, on=None, off=None):
        if select:
            return (on,)
        return (off,)

NODE_CLASS_MAPPINGS["SV-InputSelectBoolean"] = InputSelectBoolean
NODE_DISPLAY_NAME_MAPPINGS["SV-InputSelectBoolean"] = "Boolean Select"

#-------------------------------------------------------------------------------#

@VariantSupport()
class InputSelectCompare:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "equation": ("STRING", {"multiline": False, "default": ""})
            },
            "optional": {
                "true": ("*", {"lazy": True}),
                "false": ("*", {"lazy": True}),
                "x": ("*",),
            }
        }
    
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("out",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Flow"
    
    def check_lazy_status(self, x, equation, **kwargs):
        if evaluateComparison(equation, x):
            return ["true"]
        return ["false"]
    
    def run(self, x, equation: str, true=None, false=None):
        if evaluateComparison(equation, x):
            return (true,)
        return (false,)

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

NODE_CLASS_MAPPINGS["SV-InputSelectCompare"] = InputSelectCompare
NODE_DISPLAY_NAME_MAPPINGS["SV-InputSelectCompare"] = "Comparison Select"

#-------------------------------------------------------------------------------#

@VariantSupport()
class FlowBlocker:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("*", {"lazy": True}),
                "block": ("BOOLEAN",),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Flow"
    
    def check_lazy_status(self, block, input=None):
        if not block:
            return ["input"]
        return []

    def run(self, input, block):
        if block:
            return (ExecutionBlocker(None),)
        return (input,)

NODE_CLASS_MAPPINGS["SV-FlowBlocker"] = FlowBlocker
NODE_DISPLAY_NAME_MAPPINGS["SV-FlowBlocker"] = "Blocker"

#-------------------------------------------------------------------------------#

@VariantSupport()
class FlowGate:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("*", {"lazy": True}),
                "open": ("BOOLEAN",),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Flow"
    
    def check_lazy_status(self, open, input=None):
        if open:
            return ["input"]
        return []

    def run(self, input, open):
        if not open:
            return (ExecutionBlocker(None),)
        return (input,)

NODE_CLASS_MAPPINGS["SV-FlowGate"] = FlowGate
NODE_DISPLAY_NAME_MAPPINGS["SV-FlowGate"] = "Gate"

#-------------------------------------------------------------------------------#

@VariantSupport()
class FlowGateMulti:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("*", {"lazy": True}),
                "open 1": ("BOOLEAN",),
                "open 2": ("BOOLEAN",),
                "open 3": ("BOOLEAN",),
                "open 4": ("BOOLEAN",),
                "open 5": ("BOOLEAN",),
            }
        }

    RETURN_TYPES = ("*","*","*","*","*")
    RETURN_NAMES = ("1","2","3","4","5")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Flow"
    
    def check_lazy_status(self, **kwargs):
        if kwargs.get("open 1") or kwargs.get("open 2") or kwargs.get("open 3") or kwargs.get("open 4") or kwargs.get("open 5"):
            return ["input"]
        return []

    def run(self, input, **kwargs):
        block = ExecutionBlocker(None)
        out1 = input if kwargs.get("open 1", False) else block
        out2 = input if kwargs.get("open 2", False) else block
        out3 = input if kwargs.get("open 3", False) else block
        out4 = input if kwargs.get("open 4", False) else block
        out5 = input if kwargs.get("open 5", False) else block
        return out1, out2, out3, out4, out5

NODE_CLASS_MAPPINGS["SV-FlowGateMulti"] = FlowGateMulti
NODE_DISPLAY_NAME_MAPPINGS["SV-FlowGateMulti"] = "Multi Gate"

#-------------------------------------------------------------------------------#

@VariantSupport()
class IfBranch:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("*",),
                "condition": ("BOOLEAN",),
            }
        }
    
    RETURN_TYPES = ("*", "*")
    RETURN_NAMES = ("if", "else")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Flow"
    
    def run(self, input, condition):
        if condition:
            return (input, ExecutionBlocker(None))
        return (ExecutionBlocker(None), input)

NODE_CLASS_MAPPINGS["SV-IfBranch"] = IfBranch
NODE_DISPLAY_NAME_MAPPINGS["SV-IfBranch"] = "If Branch"

#-------------------------------------------------------------------------------#

NUM_FLOW_SOCKETS = 4

@VariantSupport()
class ForLoopOpen:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "remaining": ("INT", {"default": 1, "min": 0, "max": 100000, "step": 1}),
            },
            "optional": {
                "initial_value%d" % i: ("*",) for i in range(1, NUM_FLOW_SOCKETS)
            },
            "hidden": {
                "initial_value0": ("*",)
            }
        }

    RETURN_TYPES = tuple(["FLOW_CONTROL", "INT",] + ["*"] * (NUM_FLOW_SOCKETS-1))
    RETURN_NAMES = tuple(["flow_control", "remaining"] + ["value%d" % i for i in range(1, NUM_FLOW_SOCKETS)])
    FUNCTION = "for_loop_open"

    CATEGORY = "InversionDemo Nodes/Flow"

    def for_loop_open(self, remaining, **kwargs):
        graph = GraphBuilder()
        if "initial_value0" in kwargs:
            remaining = kwargs["initial_value0"]
        while_open = graph.node("SV-WhileLoopOpen", condition=remaining, initial_value0=remaining, **{("initial_value%d" % i): kwargs.get("initial_value%d" % i, None) for i in range(1, NUM_FLOW_SOCKETS)})
        outputs = [kwargs.get("initial_value%d" % i, None) for i in range(1, NUM_FLOW_SOCKETS)]
        return {
            "result": tuple(["stub", remaining] + outputs),
            "expand": graph.finalize(),
        }

NODE_CLASS_MAPPINGS["SV-ForLoopOpen"] = ForLoopOpen
NODE_DISPLAY_NAME_MAPPINGS["SV-ForLoopOpen"] = "For Loop Open"

#-------------------------------------------------------------------------------#

@VariantSupport()
class ForLoopClose:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow_control": ("FLOW_CONTROL", {"rawLink": True}),
            },
            "optional": {
                "initial_value%d" % i: ("*",{"rawLink": True}) for i in range(1, NUM_FLOW_SOCKETS)
            },
        }

    RETURN_TYPES = tuple(["*"] * (NUM_FLOW_SOCKETS-1))
    RETURN_NAMES = tuple(["value%d" % i for i in range(1, NUM_FLOW_SOCKETS)])
    FUNCTION = "for_loop_close"

    CATEGORY = "InversionDemo Nodes/Flow"

    def for_loop_close(self, flow_control, **kwargs):
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

NODE_CLASS_MAPPINGS["SV-ForLoopClose"] = ForLoopClose
NODE_DISPLAY_NAME_MAPPINGS["SV-ForLoopClose"] = "For Loop Close"

#-------------------------------------------------------------------------------#

@VariantSupport()
class WhileLoopOpen:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "condition": ("BOOLEAN", {"default": True}),
            },
            "optional": {
            },
        }
        for i in range(NUM_FLOW_SOCKETS):
            inputs["optional"]["initial_value%d" % i] = ("*",)
        return inputs

    RETURN_TYPES = tuple(["FLOW_CONTROL"] + ["*"] * NUM_FLOW_SOCKETS)
    RETURN_NAMES = tuple(["FLOW_CONTROL"] + ["value%d" % i for i in range(NUM_FLOW_SOCKETS)])
    FUNCTION = "while_loop_open"

    CATEGORY = "InversionDemo Nodes/Flow"

    def while_loop_open(self, condition, **kwargs):
        values = []
        for i in range(NUM_FLOW_SOCKETS):
            values.append(kwargs.get("initial_value%d" % i, None))
        return tuple(["stub"] + values)

NODE_CLASS_MAPPINGS["SV-WhileLoopOpen"] = WhileLoopOpen
NODE_DISPLAY_NAME_MAPPINGS["SV-WhileLoopOpen"] = "While Loop Open"

#-------------------------------------------------------------------------------#

@VariantSupport()
class WhileLoopClose:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "flow_control": ("FLOW_CONTROL", {"rawLink": True}),
                "condition": ("BOOLEAN", {"forceInput": True}),
            },
            "optional": {
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
            }
        }
        for i in range(NUM_FLOW_SOCKETS):
            inputs["optional"]["initial_value%d" % i] = ("*",)
        return inputs

    RETURN_TYPES = tuple(["*"] * NUM_FLOW_SOCKETS)
    RETURN_NAMES = tuple(["value%d" % i for i in range(NUM_FLOW_SOCKETS)])
    FUNCTION = "while_loop_close"

    CATEGORY = "InversionDemo Nodes/Flow"

    def explore_dependencies(self, node_id, dynprompt, upstream):
        node_info = dynprompt.get_node(node_id)
        if "inputs" not in node_info:
            return
        for k, v in node_info["inputs"].items():
            if is_link(v):
                parent_id = v[0]
                if parent_id not in upstream:
                    upstream[parent_id] = []
                    self.explore_dependencies(parent_id, dynprompt, upstream)
                upstream[parent_id].append(node_id)

    def collect_contained(self, node_id, upstream, contained):
        if node_id not in upstream:
            return
        for child_id in upstream[node_id]:
            if child_id not in contained:
                contained[child_id] = True
                self.collect_contained(child_id, upstream, contained)


    def while_loop_close(self, flow_control, condition, dynprompt=None, unique_id=None, **kwargs):
        if not condition:
            # We're done with the loop
            values = []
            for i in range(NUM_FLOW_SOCKETS):
                values.append(kwargs.get("initial_value%d" % i, None))
            return tuple(values)

        # We want to loop
        this_node = dynprompt.get_node(unique_id)
        upstream = {}
        # Get the list of all nodes between the open and close nodes
        self.explore_dependencies(unique_id, dynprompt, upstream)

        contained = {}
        open_node = flow_control[0]
        self.collect_contained(open_node, upstream, contained)
        contained[unique_id] = True
        contained[open_node] = True

        # We'll use the default prefix, but to avoid having node names grow exponentially in size,
        # we'll use "Recurse" for the name of the recursively-generated copy of this node.
        graph = GraphBuilder()
        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.node(original_node["class_type"], "Recurse" if node_id == unique_id else node_id)
            node.set_override_display_id(node_id)
        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.lookup_node("Recurse" if node_id == unique_id else node_id)
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
        my_clone = graph.lookup_node("Recurse" )
        result = map(lambda x: my_clone.out(x), range(NUM_FLOW_SOCKETS))
        return {
            "result": tuple(result),
            "expand": graph.finalize(),
        }

NODE_CLASS_MAPPINGS["SV-WhileLoopClose"] = WhileLoopClose
NODE_DISPLAY_NAME_MAPPINGS["SV-WhileLoopClose"] = "While Loop Close"

#-------------------------------------------------------------------------------#

@VariantSupport()
class IntMathOperation:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 1}),
                "b": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 1}),
                "operation": (["add", "subtract", "multiply", "divide", "modulo", "power"],),
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "int_math_operation"

    CATEGORY = "InversionDemo Nodes/Logic"

    def int_math_operation(self, a, b, operation):
        if operation == "add":
            return (a + b,)
        elif operation == "subtract":
            return (a - b,)
        elif operation == "multiply":
            return (a * b,)
        elif operation == "divide":
            return (a // b,)
        elif operation == "modulo":
            return (a % b,)
        elif operation == "power":
            return (a ** b,)

NODE_CLASS_MAPPINGS["SV-IntMathOperation"] = IntMathOperation
NODE_DISPLAY_NAME_MAPPINGS["SV-IntMathOperation"] = "Int Math Operation"

#-------------------------------------------------------------------------------#

@VariantSupport()
class ToBoolNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("*",),
            },
            "optional": {
                "invert": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "to_bool"

    CATEGORY = "InversionDemo Nodes/Logic"

    def to_bool(self, value, invert = False):
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

        return (result,)

NODE_CLASS_MAPPINGS["SV-ToBoolNode"] = ToBoolNode
NODE_DISPLAY_NAME_MAPPINGS["SV-ToBoolNode"] = "To Bool"

#-------------------------------------------------------------------------------#

@VariantSupport()
class AccumulateNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "to_add": ("*",),
            },
            "optional": {
                "accumulation": ("ACCUMULATION",),
            },
        }

    RETURN_TYPES = ("ACCUMULATION",)
    FUNCTION = "accumulate"

    CATEGORY = "InversionDemo Nodes/Lists"

    def accumulate(self, to_add, accumulation = None):
        if accumulation is None:
            value = [to_add]
        else:
            value = accumulation["accum"] + [to_add]
        return ({"accum": value},)

NODE_CLASS_MAPPINGS["SV-AccumulateNode"] = AccumulateNode
NODE_DISPLAY_NAME_MAPPINGS["SV-AccumulateNode"] = "Accumulate"

#-------------------------------------------------------------------------------#

@VariantSupport()
class AccumulationHeadNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "accumulation": ("ACCUMULATION",),
            },
        }

    RETURN_TYPES = ("ACCUMULATION", "*",)
    FUNCTION = "accumulation_head"

    CATEGORY = "InversionDemo Nodes/Lists"

    def accumulation_head(self, accumulation):
        accum = accumulation["accum"]
        if len(accum) == 0:
            return (accumulation, None)
        else:
            return ({"accum": accum[1:]}, accum[0])

NODE_CLASS_MAPPINGS["SV-AccumulationHeadNode"] = AccumulationHeadNode
NODE_DISPLAY_NAME_MAPPINGS["SV-AccumulationHeadNode"] = "Accumulation Head"

#-------------------------------------------------------------------------------#

@VariantSupport()
class AccumulationTailNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "accumulation": ("ACCUMULATION",),
            },
        }

    RETURN_TYPES = ("ACCUMULATION", "*",)
    FUNCTION = "accumulation_tail"

    CATEGORY = "InversionDemo Nodes/Lists"

    def accumulation_tail(self, accumulation):
        accum = accumulation["accum"]
        if len(accum) == 0:
            return (None, accumulation)
        else:
            return ({"accum": accum[:-1]}, accum[-1])

NODE_CLASS_MAPPINGS["SV-AccumulationTailNode"] = AccumulationTailNode
NODE_DISPLAY_NAME_MAPPINGS["SV-AccumulationTailNode"] = "Accumulation Tail"

#-------------------------------------------------------------------------------#

@VariantSupport()
class AccumulationToListNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "accumulation": ("ACCUMULATION",),
            },
        }

    RETURN_TYPES = ("*",)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = "accumulation_to_list"

    CATEGORY = "InversionDemo Nodes/Lists"

    def accumulation_to_list(self, accumulation):
        return (accumulation["accum"],)

NODE_CLASS_MAPPINGS["SV-AccumulationToListNode"] = AccumulationToListNode
NODE_DISPLAY_NAME_MAPPINGS["SV-AccumulationToListNode"] = "Accumulation To List"

#-------------------------------------------------------------------------------#

@VariantSupport()
class ListToAccumulationNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list": ("*",),
            },
        }

    RETURN_TYPES = ("ACCUMULATION",)
    INPUT_IS_LIST = (True,)

    FUNCTION = "list_to_accumulation"

    CATEGORY = "InversionDemo Nodes/Lists"

    def list_to_accumulation(self, list):
        return ({"accum": list},)

NODE_CLASS_MAPPINGS["SV-ListToAccumulationNode"] = ListToAccumulationNode
NODE_DISPLAY_NAME_MAPPINGS["SV-ListToAccumulationNode"] = "List To Accumulation"

#-------------------------------------------------------------------------------#

@VariantSupport()
class AccumulationGetLengthNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "accumulation": ("ACCUMULATION",),
            },
        }

    RETURN_TYPES = ("INT",)

    FUNCTION = "accumlength"

    CATEGORY = "InversionDemo Nodes/Lists"

    def accumlength(self, accumulation):
        return (len(accumulation['accum']),)

NODE_CLASS_MAPPINGS["SV-AccumulationGetLengthNode"] = AccumulationGetLengthNode
NODE_DISPLAY_NAME_MAPPINGS["SV-AccumulationGetLengthNode"] = "Accumulation Get Length"

#-------------------------------------------------------------------------------#
        
@VariantSupport()
class AccumulationGetItemNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "accumulation": ("ACCUMULATION",),
                "index": ("INT", {"default":0, "step":1})
            },
        }

    RETURN_TYPES = ("*",)

    FUNCTION = "get_item"

    CATEGORY = "InversionDemo Nodes/Lists"

    def get_item(self, accumulation, index):
        return (accumulation['accum'][index],)

NODE_CLASS_MAPPINGS["SV-AccumulationGetItemNode"] = AccumulationGetItemNode
NODE_DISPLAY_NAME_MAPPINGS["SV-AccumulationGetItemNode"] = "Accumulation Get Item"

#-------------------------------------------------------------------------------#
        
@VariantSupport()
class AccumulationSetItemNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "accumulation": ("ACCUMULATION",),
                "index": ("INT", {"default":0, "step":1}),
                "value": ("*",),
            },
        }

    RETURN_TYPES = ("ACCUMULATION",)

    FUNCTION = "set_item"

    CATEGORY = "InversionDemo Nodes/Lists"

    def set_item(self, accumulation, index, value):
        new_accum = accumulation['accum'][:]
        new_accum[index] = value
        return ({"accum": new_accum},)

NODE_CLASS_MAPPINGS["SV-AccumulationSetItemNode"] = AccumulationSetItemNode
NODE_DISPLAY_NAME_MAPPINGS["SV-AccumulationSetItemNode"] = "Accumulation Set Item"

#-------------------------------------------------------------------------------#

@VariantSupport()
class HashObject:
    def __init__(s):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "obj": ("*",)
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("hash",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Flow"
    
    def run(s, obj):
        return (hash_item(obj),)

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

NODE_CLASS_MAPPINGS["SV-HashObject"] = HashObject
NODE_DISPLAY_NAME_MAPPINGS["SV-HashObject"] = "Hash Object"

#-------------------------------------------------------------------------------#

@VariantSupport()
class HashItems:
    def __init__(s):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "_1_": ("*",),
                "_2_": ("*",),
                "_3_": ("*",),
                "_4_": ("*",),
                "_5_": ("*",),
                "_6_": ("*",),
                "_7_": ("*",),
                "_8_": ("*",),
                "_9_": ("*",),
                "_10_": ("*",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("hash",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Flow"
    
    def run(s, _1_=None, _2_=None, _3_=None, _4_=None, _5_=None, _6_=None, _7_=None, _8_=None, _9_=None, _10_=None):
        args = [_1_, _2_, _3_, _4_, _5_, _6_, _7_, _8_, _9_, _10_]
        return (hash_item(args),)

NODE_CLASS_MAPPINGS["SV-HashItems"] = HashItems
NODE_DISPLAY_NAME_MAPPINGS["SV-HashItems"] = "Hash Items"

#-------------------------------------------------------------------------------#

class HashModel:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",)
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("hash",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Flow"
    
    def run(self, model):
        return (hashlib.md5(str(model.model.state_dict()).encode()).hexdigest(),)

NODE_CLASS_MAPPINGS["SV-HashModel"] = HashModel
NODE_DISPLAY_NAME_MAPPINGS["SV-HashModel"] = "Hash Model"

#-------------------------------------------------------------------------------#

@VariantSupport()
class CacheObject:
    hashes = {}
    obj = None
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": ("*", {"lazy": True}),
                "hash": ("STRING", {"forceInput": True}),
                "id": ("STRING",)
            }
        }
    
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("cache",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Flow"
    NOT_IDEMPOTENT = True
    
    @classmethod
    def check_lazy_status(self, hash, id, **kwargs):
        if CacheObject.hashes.get(id) is not None and CacheObject.hashes.get(id) == hash:
            return []
        return ["any"]
    
    def run(self, any, hash, id):
        if self.obj is not None and CacheObject.hashes.get(id) is not None and CacheObject.hashes.get(id) == hash:
            return (self.obj,)
        CacheObject.hashes[id] = hash
        self.obj = any
        return (any,)

NODE_CLASS_MAPPINGS["SV-CacheObject"] = CacheObject
NODE_DISPLAY_NAME_MAPPINGS["SV-CacheObject"] = "Cache Object"

#-------------------------------------------------------------------------------#

@VariantSupport()
class ManualCache:
    hasValue = {}
    obj = None
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": ("*", {"lazy": True}),
                "enable": ("BOOLEAN", {"forceInput": True}),
                "id": ("STRING",),
            }
        }
    
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("cache",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Flow"
    NOT_IDEMPOTENT = True
    
    @classmethod
    def check_lazy_status(self, enable, id, **kwargs):
        if (id is None or id == "") and enable:
            return []
        if enable and ManualCache.hasValue.get(id):
            return []
        return ["any"]
    
    def run(self, any, enable, id):
        if enable and self.obj is not None:
            return (self.obj,)
        if any is None:
            del ManualCache.hasValue[id]
        self.obj = any
        ManualCache.hasValue[id] = True
        return (any,)

NODE_CLASS_MAPPINGS["SV-ManualCache"] = ManualCache
NODE_DISPLAY_NAME_MAPPINGS["SV-ManualCache"] = "Manual Cache"

#-------------------------------------------------------------------------------#

@VariantSupport()
class ClearCustomCaches:
    id = None
    
    @classmethod
    def INPUT_TYPES(s):
        return {}
    
    RETURN_TYPES = ()
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Flow"
    NOT_IDEMPOTENT = True
    OUTPUT_NODE = True
    
    def run(self):
        if self.id is None:
            self.id = round(time.time() * 1000)
            VariableSet.storage = {}
            CacheObject.hashes = {}
            ManualCache.hasValue = {}
        return (None,)

NODE_CLASS_MAPPINGS["SV-ClearCustomCaches"] = ClearCustomCaches
NODE_DISPLAY_NAME_MAPPINGS["SV-ClearCustomCaches"] = "Clear Caches"

#-------------------------------------------------------------------------------#

@VariantSupport()
class FlowNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "_1_": ("*",),
                "_2_": ("*",),
                "_3_": ("*",),
                "_4_": ("*",),
                "_5_": ("*",),
            }
        }
    
    RETURN_TYPES = ("*", "*", "*", "*", "*")
    RETURN_NAMES = ("_1_", "_2_", "_3_", "_4_", "_5_")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Flow"
    
    def run(self, _1_=None, _2_=None, _3_=None, _4_=None, _5_=None):
        return (_1_, _2_, _3_, _4_, _5_)

NODE_CLASS_MAPPINGS["SV-FlowNode"] = FlowNode
NODE_DISPLAY_NAME_MAPPINGS["SV-FlowNode"] = "Flow Node"

#-------------------------------------------------------------------------------#

@VariantSupport()
class FlowPipeInput:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start": ("INT", {"min": 1, "max": 100, "step": 1, "default": 1}),
            },
            "optional": {
                "pipe": ("sv_pipe",),
                "_1_": ("*",),
                "_2_": ("*",),
                "_3_": ("*",),
                "_4_": ("*",),
                "_5_": ("*",),
            }
        }
    
    RETURN_TYPES = ("sv_pipe",)
    RETURN_NAMES = ("pipe",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Pipes"
    
    def run(self, start, pipe=None, **kwargs):
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
        return (pipe,)

NODE_CLASS_MAPPINGS["SV-FlowPipeInput"] = FlowPipeInput
NODE_DISPLAY_NAME_MAPPINGS["SV-FlowPipeInput"] = "Pipe In"

#-------------------------------------------------------------------------------#

@VariantSupport()
class FlowPipeOutput:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("sv_pipe",),
                "index": ("INT", {"min": 1, "max": 100, "step": 1, "default": 1}),
            }
        }
    
    RETURN_TYPES = ("sv_pipe", "*", "*", "*", "*", "*")
    RETURN_NAMES = ("pipe", "_1_", "_2_", "_3_", "_4_", "_5_")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Pipes"
    
    def run(self, pipe, index):
        if not isinstance(pipe, dict):
            raise TypeError("Invalid pipe input type")
        if not isinstance(index, int):
            raise TypeError("Invalid index input type")
        if index < 1:
            raise ValueError("Invalid index value")
        return (pipe, pipe.get(f"_{0 + index}_", None), pipe.get(f"_{1 + index}_", None), pipe.get(f"_{2 + index}_", None), pipe.get(f"_{3 + index}_", None), pipe.get(f"_{4 + index}_", None))

NODE_CLASS_MAPPINGS["SV-FlowPipeOutput"] = FlowPipeOutput
NODE_DISPLAY_NAME_MAPPINGS["SV-FlowPipeOutput"] = "Pipe Out"

#-------------------------------------------------------------------------------#

@VariantSupport()
class FlowPipeInputLarge:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start": ("INT", {"min": 1, "max": 100, "step": 1, "default": 1}),
            },
            "optional": {
                "pipe": ("sv_pipe",),
                "_1_": ("*",),
                "_2_": ("*",),
                "_3_": ("*",),
                "_4_": ("*",),
                "_5_": ("*",),
                "_6_": ("*",),
                "_7_": ("*",),
                "_8_": ("*",),
                "_9_": ("*",),
                "_10_": ("*",)
            }
        }
    
    RETURN_TYPES = ("sv_pipe",)
    RETURN_NAMES = ("pipe",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Pipes"
    
    def run(self, start, pipe=None, **kwargs):
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
        return (pipe,)

NODE_CLASS_MAPPINGS["SV-FlowPipeInputLarge"] = FlowPipeInputLarge
NODE_DISPLAY_NAME_MAPPINGS["SV-FlowPipeInputLarge"] = "Pipe In Large"

#-------------------------------------------------------------------------------#

@VariantSupport()
class FlowPipeOutputLarge:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("sv_pipe",),
                "index": ("INT", {"min": 1, "max": 100, "step": 1, "default": 1}),
            }
        }
    
    RETURN_TYPES = ("sv_pipe", "*", "*", "*", "*", "*", "*", "*", "*", "*", "*")
    RETURN_NAMES = ("pipe", "_1_", "_2_", "_3_", "_4_", "_5_", "_6_", "_7_", "_8_", "_9_", "_10_")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Pipes"
    
    def run(self, pipe, index):
        if not isinstance(pipe, dict):
            raise TypeError("Invalid pipe input type")
        if not isinstance(index, int):
            raise TypeError("Invalid index input type")
        if index < 1:
            raise ValueError("Invalid index value")
        return (pipe, pipe.get(f"_{0 + index}_", None), pipe.get(f"_{1 + index}_", None), pipe.get(f"_{2 + index}_", None), pipe.get(f"_{3 + index}_", None), pipe.get(f"_{4 + index}_", None), pipe.get(f"_{5 + index}_", None), pipe.get(f"_{6 + index}_", None), pipe.get(f"_{7 + index}_", None), pipe.get(f"_{8 + index}_", None), pipe.get(f"_{9 + index}_", None))

NODE_CLASS_MAPPINGS["SV-FlowPipeOutputLarge"] = FlowPipeOutputLarge
NODE_DISPLAY_NAME_MAPPINGS["SV-FlowPipeOutputLarge"] = "Pipe Out Large"

#-------------------------------------------------------------------------------#

@VariantSupport()
class FlowPipeInputIndex:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "index": ("INT", {"min": 1, "max": 100, "step": 1, "default": 1}),
            },
            "optional": {
                "pipe": ("sv_pipe",),
                "value": ("*",),
            }
        }
    
    RETURN_TYPES = ("sv_pipe",)
    RETURN_NAMES = ("pipe",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Pipes"
    
    def run(self, index, pipe=None, value=None):
        if not isinstance(pipe, (dict, type(None))):
            raise TypeError("Invalid pipe input type")
        if not isinstance(index, int):
            raise TypeError("Invalid index input type")
        if index < 1:
            raise ValueError("Invalid index value")
        if value is None:
            return (pipe,)
        pipe = {**pipe} if pipe else {}
        pipe[f"_{index}_"] = value
        return (pipe,)

NODE_CLASS_MAPPINGS["SV-FlowPipeInputIndex"] = FlowPipeInputIndex
NODE_DISPLAY_NAME_MAPPINGS["SV-FlowPipeInputIndex"] = "Pipe In Index"

#-------------------------------------------------------------------------------#

@VariantSupport()
class FlowPipeOutputIndex:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("sv_pipe",),
                "index": ("INT", {"min": 1, "max": 100, "step": 1, "default": 1}),
            }
        }
    
    RETURN_TYPES = ("sv_pipe", "*")
    RETURN_NAMES = ("pipe", "value")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Pipes"
    
    def run(self, pipe, index):
        if not isinstance(pipe, dict):
            raise TypeError("Invalid pipe input type")
        if not isinstance(index, int):
            raise TypeError("Invalid index input type")
        if index < 1:
            raise ValueError("Invalid index value")
        return (pipe, pipe.get(f"_{index}_", None))

NODE_CLASS_MAPPINGS["SV-FlowPipeOutputIndex"] = FlowPipeOutputIndex
NODE_DISPLAY_NAME_MAPPINGS["SV-FlowPipeOutputIndex"] = "Pipe Out Index"

#-------------------------------------------------------------------------------#

@VariantSupport()
class FlowPipeInputKey:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "key": ("STRING", {"multiline": False}),
            },
            "optional": {
                "pipe": ("sv_pipe",),
                "value": ("*",),
            }
        }
    
    RETURN_TYPES = ("sv_pipe",)
    RETURN_NAMES = ("pipe",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Pipes"
    
    def run(self, key, pipe=None, value=None):
        if not isinstance(pipe, (dict, type(None))):
            raise TypeError("Invalid pipe input type")
        if not isinstance(key, str):
            raise TypeError("Invalid key input type")
        if value is None:
            return (pipe,)
        pipe = {**pipe} if pipe else {}
        pipe[key] = value
        return (pipe,)

NODE_CLASS_MAPPINGS["SV-FlowPipeInputKey"] = FlowPipeInputKey
NODE_DISPLAY_NAME_MAPPINGS["SV-FlowPipeInputKey"] = "Pipe In Key"

#-------------------------------------------------------------------------------#

@VariantSupport()
class FlowPipeOutputKey:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("sv_pipe",),
                "key": ("STRING", {"multiline": False}),
            }
        }
    
    RETURN_TYPES = ("sv_pipe", "*")
    RETURN_NAMES = ("pipe", "value")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Pipes"
    
    def run(self, pipe, key):
        if not isinstance(pipe, dict):
            raise TypeError("Invalid pipe input type")
        if not isinstance(key, str):
            raise TypeError("Invalid key input type")
        return (pipe, pipe.get(key, None))

NODE_CLASS_MAPPINGS["SV-FlowPipeOutputKey"] = FlowPipeOutputKey
NODE_DISPLAY_NAME_MAPPINGS["SV-FlowPipeOutputKey"] = "Pipe Out Key"

#-------------------------------------------------------------------------------#

@VariantSupport()
class FlowPipeInputKeyTuple:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "key": ("STRING", {"multiline": False}),
            },
            "optional": {
                "pipe": ("sv_pipe",),
                "_1_": ("*",),
                "_2_": ("*",),
                "_3_": ("*",),
                "_4_": ("*",),
                "_5_": ("*",)
            }
        }
    
    RETURN_TYPES = ("sv_pipe",)
    RETURN_NAMES = ("pipe",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Pipes"
    
    def run(self, key, pipe=None, _1_=None, _2_=None, _3_=None, _4_=None, _5_=None):
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
        return (pipe,)

NODE_CLASS_MAPPINGS["SV-FlowPipeInputKeyTuple"] = FlowPipeInputKeyTuple
NODE_DISPLAY_NAME_MAPPINGS["SV-FlowPipeInputKeyTuple"] = "Pipe In Tuple"

#-------------------------------------------------------------------------------#

@VariantSupport()
class FlowPipeOutputKeyTuple:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("sv_pipe",),
                "key": ("STRING", {"multiline": False}),
            }
        }
    
    RETURN_TYPES = ("sv_pipe", "*", "*", "*", "*", "*")
    RETURN_NAMES = ("pipe", "_1_", "_2_", "_3_", "_4_", "_5_")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Pipes"
    
    def run(self, pipe, key):
        empty = (None, None, None, None, None)
        if key not in pipe or key is None or len(key) == 0:
            return (pipe, *empty)
        if not isinstance(pipe, dict):
            raise TypeError(f"Invalid pipe input type with key '{key}'")
        if not isinstance(key, str):
            raise TypeError(f"Invalid key input type with key '{key}'")
        value = pipe.get(key, None)
        if value is None:
            return (pipe, *empty)
        if not isinstance(value, (tuple, list)):
            raise ValueError(f"Invalid value type with key '{key}'")
        return (pipe, *value)

NODE_CLASS_MAPPINGS["SV-FlowPipeOutputKeyTuple"] = FlowPipeOutputKeyTuple
NODE_DISPLAY_NAME_MAPPINGS["SV-FlowPipeOutputKeyTuple"] = "Pipe Out Tuple"

#-------------------------------------------------------------------------------#

class FlowPipeInputModel:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "key": ("STRING", {"multiline": False, "default": ""}),
            },
            "optional": {
                "pipe": ("sv_pipe",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
            }
        }
    
    RETURN_TYPES = ("sv_pipe",)
    RETURN_NAMES = ("pipe",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Pipes"
    
    def run(self, key, pipe=None, model=None, clip=None, vae=None):
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
        return (pipe,)

NODE_CLASS_MAPPINGS["SV-FlowPipeInputModel"] = FlowPipeInputModel
NODE_DISPLAY_NAME_MAPPINGS["SV-FlowPipeInputModel"] = "Pipe In Model"

#-------------------------------------------------------------------------------#

class FlowPipeOutputModel:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("sv_pipe",),
                "key": ("STRING", {"multiline": False, "default": ""}),
            }
        }
    
    RETURN_TYPES = ("sv_pipe", "MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("pipe", "model", "clip", "vae")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Pipes"
    
    def run(self, pipe, key):
        if not isinstance(pipe, dict):
            raise TypeError("Invalid pipe input type")
        key = f"__model[{key}]__"
        value = pipe.get(key, None)
        if value is None:
            return (pipe, None, None, None)
        if not isinstance(value, (tuple, list)):
            raise ValueError("Invalid value type")
        if len(value) != 3:
            raise ValueError("Invalid value length")
        return (pipe, *value)

NODE_CLASS_MAPPINGS["SV-FlowPipeOutputModel"] = FlowPipeOutputModel
NODE_DISPLAY_NAME_MAPPINGS["SV-FlowPipeOutputModel"] = "Pipe Out Model"

#-------------------------------------------------------------------------------#

class FlowPipeInputParams:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "key": ("STRING", {"multiline": False, "default": ""}),
            },
            "optional": {
                "pipe": ("sv_pipe",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "seed": ("INT", {"forceInput": True}),
                "params": ("BP_OUTPUT",),
                "cfg": ("FLOAT", {"forceInput": True}),
                "steps": ("INT", {"forceInput": True}),
                "denoise": ("FLOAT", {"forceInput": True}),
                "sampler name": (comfy.samplers.SAMPLER_NAMES, {"forceInput": True}),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES, {"forceInput": True}),
                "ays": ("BOOLEAN", {"forceInput": True}),
            }
        }
    
    RETURN_TYPES = ("sv_pipe",)
    RETURN_NAMES = ("pipe",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Pipes"
    
    def run(self, key, pipe=None, positive=None, negative=None, latent=None, seed=None, params=None, cfg=None, steps=None, denoise=None, sampler=None, scheduler=None, ays=None):
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
        return (pipe,)

NODE_CLASS_MAPPINGS["SV-FlowPipeInputParams"] = FlowPipeInputParams
NODE_DISPLAY_NAME_MAPPINGS["SV-FlowPipeInputParams"] = "Pipe In Params"

#-------------------------------------------------------------------------------#

class FlowPipeOutputParams:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("sv_pipe",),
                "key": ("STRING", {"multiline": False, "default": ""}),
            }
        }
    
    RETURN_TYPES = ("sv_pipe", "CONDITIONING", "CONDITIONING", "LATENT", "INT", "FLOAT", "INT", "FLOAT", comfy.samplers.SAMPLER_NAMES, "SAMPLER", comfy.samplers.SCHEDULER_NAMES, "BOOLEAN")
    RETURN_NAMES = ("pipe", "positive", "negative", "latent", "seed", "cfg", "steps", "denoise", "sampler name", "sampler", "scheduler", "ays")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Pipes"
    
    def run(self, pipe, key):
        if not isinstance(pipe, dict):
            raise TypeError("Invalid pipe input type")
        key = f"__params[{key}]__"
        value = pipe.get(key, None)
        if value is None:
            return (pipe, None, None, None, None, None, None, None, None, None, None, None)
        if not isinstance(value, (tuple, list)):
            raise ValueError("Invalid value type")
        if len(value) != 11:
            raise ValueError("Invalid value length")
        return (pipe, *value)

NODE_CLASS_MAPPINGS["SV-FlowPipeOutputParams"] = FlowPipeOutputParams
NODE_DISPLAY_NAME_MAPPINGS["SV-FlowPipeOutputParams"] = "Pipe Out Params"

#-------------------------------------------------------------------------------#

class FlowPipeCombine:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe1": ("sv_pipe",),
                "pipe2": ("sv_pipe",),
            }
        }
    
    RETURN_TYPES = ("sv_pipe",)
    RETURN_NAMES = ("pipe",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Pipes"
    
    def run(self, pipe1, pipe2):
        if not isinstance(pipe1, (dict, type(None))) or not isinstance(pipe2, (dict, type(None))):
            raise TypeError("Invalid pipe input type")
        if pipe1 is None:
            pipe1 = {}
        if pipe2 is None:
            pipe2 = {}
        # remove None values
        pipe1 = {k: v for k, v in pipe1.items() if v is not None}
        pipe2 = {k: v for k, v in pipe2.items() if v is not None}
        return ({**pipe1, **pipe2},)

NODE_CLASS_MAPPINGS["SV-FlowPipeCombine"] = FlowPipeCombine
NODE_DISPLAY_NAME_MAPPINGS["SV-FlowPipeCombine"] = "Pipe Combine"

#-------------------------------------------------------------------------------#

@VariantSupport()
class CheckNone:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": ("*",)
            }
        }
    
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("bool",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Logic"
    
    def run(self, any):
        return (any is None,)

NODE_CLASS_MAPPINGS["SV-CheckNone"] = CheckNone
NODE_DISPLAY_NAME_MAPPINGS["SV-CheckNone"] = "Check None"

#-------------------------------------------------------------------------------#

@VariantSupport()
class CheckNoneNot:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": ("*",)
            }
        }
    
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("bool",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Logic"
    
    def run(self, any):
        return (any is not None,)

NODE_CLASS_MAPPINGS["SV-CheckNoneNot"] = CheckNoneNot
NODE_DISPLAY_NAME_MAPPINGS["SV-CheckNoneNot"] = "Check Not None"

#-------------------------------------------------------------------------------#

@VariantSupport()
class DefaultInt:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": ("*",),
                "default": ("INT", {"min": -sys.maxsize, "max": sys.maxsize, "step": 1, "default": 0, "lazy": True})
            }
        }
    
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Logic"
    
    @classmethod
    def check_lazy_status(self, any, **kwargs):
        if any is None:
            return ["default"]
        return []
    
    def run(self, any, default):
        if any is None or not isinstance(any, int):
            return (default,)
        return (any,)

NODE_CLASS_MAPPINGS["SV-DefaultInt"] = DefaultInt
NODE_DISPLAY_NAME_MAPPINGS["SV-DefaultInt"] = "Default Int"

#-------------------------------------------------------------------------------#

@VariantSupport()
class DefaultFloat:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": ("*",),
                "default": ("FLOAT", {"min": -sys.float_info.max, "max": sys.float_info.max, "step": 0.01, "default": 0.0, "lazy": True})
            }
        }
    
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Logic"
    
    @classmethod
    def check_lazy_status(self, any, **kwargs):
        if any is None:
            return ["default"]
        return []
    
    def run(self, any, default):
        if any is None or not isinstance(any, float):
            return (default,)
        return (any,)

NODE_CLASS_MAPPINGS["SV-DefaultFloat"] = DefaultFloat
NODE_DISPLAY_NAME_MAPPINGS["SV-DefaultFloat"] = "Default Float"

#-------------------------------------------------------------------------------#

@VariantSupport()
class DefaultString:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": ("*",),
                "default": ("STRING", {"multiline": False, "default": "", "lazy": True})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Logic"
    
    @classmethod
    def check_lazy_status(self, any, **kwargs):
        if any is None:
            return ["default"]
        return []
    
    def run(self, any, default):
        if any is None or not isinstance(any, str):
            return (default,)
        return (any,)

NODE_CLASS_MAPPINGS["SV-DefaultString"] = DefaultString
NODE_DISPLAY_NAME_MAPPINGS["SV-DefaultString"] = "Default String"

#-------------------------------------------------------------------------------#

@VariantSupport()
class DefaultBoolean:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": ("*",),
                "default": ("BOOLEAN", {"default": False, "lazy": True})
            }
        }
    
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("bool",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Logic"
    
    @classmethod
    def check_lazy_status(self, any, **kwargs):
        if any is None:
            return ["default"]
        return []
    
    def run(self, any, default):
        if any is None or not isinstance(any, bool):
            return (default,)
        return (any,)

NODE_CLASS_MAPPINGS["SV-DefaultBoolean"] = DefaultBoolean
NODE_DISPLAY_NAME_MAPPINGS["SV-DefaultBoolean"] = "Default Boolean"

#-------------------------------------------------------------------------------#

@VariantSupport()
class DefaultValue:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": ("*",),
                "default": ("*", {"lazy": True}),
            }
        }
    
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("value",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Logic"
    
    @classmethod
    def check_lazy_status(self, any, **kwargs):
        if any is None:
            return ["default"]
        return []
    
    def run(self, any, default):
        if any is None:
            return (default,)
        return (any,)

NODE_CLASS_MAPPINGS["SV-DefaultValue"] = DefaultValue
NODE_DISPLAY_NAME_MAPPINGS["SV-DefaultValue"] = "Default Value"

#-------------------------------------------------------------------------------#

@VariantSupport()
class AnyToAny:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("*",)
            }
        }
    
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes"
    
    def run(self, input):
        return (input,)

NODE_CLASS_MAPPINGS["SV-AnyToAny"] = AnyToAny
NODE_DISPLAY_NAME_MAPPINGS["SV-AnyToAny"] = "Any to Any"

#-------------------------------------------------------------------------------#

@VariantSupport()
class ConsolePrint:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True})
            },
            "optional": {
                "signal": ("*",),
            }
        }
    
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Debug"
    
    def run(self, text, signal=None):
        print(text.replace("_signal_", str(signal)))
        return {}

NODE_CLASS_MAPPINGS["SV-ConsolePrint"] = ConsolePrint
NODE_DISPLAY_NAME_MAPPINGS["SV-ConsolePrint"] = "Console Print"

#-------------------------------------------------------------------------------#

@VariantSupport()
class ConsolePrintMulti:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "signal1": ("*",),
            },
            "optional": {
                "signal2": ("*",),
                "signal3": ("*",),
            }
        }
    
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Debug"
    
    def run(self, text, signal1, signal2=None, signal3=None):
        print(text.replace("_signal1_", str(signal1)).replace("_signal2_", str(signal2)).replace("_signal3_", str(signal3)))
        return {}

NODE_CLASS_MAPPINGS["SV-ConsolePrintMulti"] = ConsolePrintMulti
NODE_DISPLAY_NAME_MAPPINGS["SV-ConsolePrintMulti"] = "Console Print Multi"

#-------------------------------------------------------------------------------#

@VariantSupport()
class ConsolePrintLoop:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
            },
            "optional": {
                "signal1": ("*",),
                "signal2": ("*",),
                "signal3": ("*",),
            }
        }
    
    OUTPUT_NODE = True
    RETURN_TYPES = ("*", "*", "*")
    RETURN_NAMES = ("signal1", "signal2", "signal3")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Debug"
    
    def run(self, text, signal1=None, signal2=None, signal3=None):
        print(text.replace("_signal1_", str(signal1)).replace("_signal2_", str(signal2)).replace("_signal3_", str(signal3)))
        return (signal1, signal2, signal3)

NODE_CLASS_MAPPINGS["SV-ConsolePrintLoop"] = ConsolePrintLoop
NODE_DISPLAY_NAME_MAPPINGS["SV-ConsolePrintLoop"] = "Console Print Loop"

#-------------------------------------------------------------------------------#

@VariantSupport()
class AssertNotNone:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": ("*",)
            }
        }
    
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Debug"
    
    def run(self, any):
        if any is None:
            raise ValueError("AssertNotNone: Value is None")
        return {}

NODE_CLASS_MAPPINGS["SV-AssertNotNone"] = AssertNotNone
NODE_DISPLAY_NAME_MAPPINGS["SV-AssertNotNone"] = "Assert Not None"

#-------------------------------------------------------------------------------#

@VariantSupport()
class TimerStart:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": ("*",)
            }
        }
    
    OUTPUT_NODE = True
    RETURN_TYPES = ("*", "TIMER")
    RETURN_NAMES = ("any", "timestamp")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Debug"
    
    def run(self, any):
        return (any, time.time())

NODE_CLASS_MAPPINGS["SV-TimerStart"] = TimerStart
NODE_DISPLAY_NAME_MAPPINGS["SV-TimerStart"] = "Timer Start"

#-------------------------------------------------------------------------------#

@VariantSupport()
class TimerEnd:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": ("*",),
                "timestamp": ("TIMER",)
            }
        }
    
    OUTPUT_NODE = True
    RETURN_TYPES = ("*", "FLOAT")
    RETURN_NAMES = ("any", "time")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Debug"
    
    def run(self, any, timestamp):
        return (any, time.time() - timestamp)

NODE_CLASS_MAPPINGS["SV-TimerEnd"] = TimerEnd
NODE_DISPLAY_NAME_MAPPINGS["SV-TimerEnd"] = "Timer End"

#-------------------------------------------------------------------------------#

class CurveFromEquation:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "fn": ("STRING", {"multiline": False, "default": "x^2 + 1"})
            }
        }
    
    RETURN_TYPES = ("CURVE",)
    RETURN_NAMES = ("curve",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Logic"
    
    def run(self, fn):
        return (parseCurve(fn),)
        
def parseCurve(curve):
    def f(curve, t):
        curve = re.sub(r"\s+", "", curve)
        curve = collapseSigns(curve)
        while "(" in curve or ")" in curve:
            curve = re.sub(r"\w+\([^()]+\)", lambda x : str(parseCurveFunction(x.group(0), t)), curve)
            curve = re.sub(r"\([^(,)]+\)", lambda x : str(parseCurve(x.group(0)[1:-1])(t)), curve)
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

NODE_CLASS_MAPPINGS["SV-CurveFromEquation"] = CurveFromEquation
NODE_DISPLAY_NAME_MAPPINGS["SV-CurveFromEquation"] = "Curve from Equation"

#-------------------------------------------------------------------------------#

class ApplyCurve:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "curve": ("CURVE",),
                "t": ("FLOAT", {"forceInput": True})
            }
        }
    
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("value",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Logic"
    
    def run(self, curve, t):
        return (curve(t),)

NODE_CLASS_MAPPINGS["SV-ApplyCurve"] = ApplyCurve
NODE_DISPLAY_NAME_MAPPINGS["SV-ApplyCurve"] = "Apply Curve"

#-------------------------------------------------------------------------------#

class ApplyCurveFromStep:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "curve": ("CURVE",),
                "step": ("INT", {"forceInput": True}),
                "steps": ("INT", {"min": 1, "max": 100, "step": 1, "default": 10}),
            }
        }
    
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("value",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Logic"
    
    def run(self, curve, step, steps):
        if step < 1 or step > steps:
            raise ValueError("Invalid step value")
        if steps < 1:
            raise ValueError("Invalid steps value")
        if steps == 1:
            return (curve(1),)
        return (curve((step - 1) / (steps - 1)),)

NODE_CLASS_MAPPINGS["SV-ApplyCurveFromStep"] = ApplyCurveFromStep
NODE_DISPLAY_NAME_MAPPINGS["SV-ApplyCurveFromStep"] = "Apply Curve from Step"

#-------------------------------------------------------------------------------#

@VariantSupport()
class MathOperation:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "op": ("STRING", {"multiline": False, "default": ""})
            },
            "optional": {
                "a": ("*",),
                "b": ("*",),
            }
        }
    
    RETURN_TYPES = ("INT", "FLOAT")
    RETURN_NAMES = ("int", "float")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Logic"
    
    def run(self, op: str, a=None, b=None):
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

NODE_CLASS_MAPPINGS["SV-MathOperation"] = MathOperation
NODE_DISPLAY_NAME_MAPPINGS["SV-MathOperation"] = "Math Operation"

#-------------------------------------------------------------------------------#

class FloatRerouteForSubnodes:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "float": ("FLOAT", {"min": 0, "max": 1, "step": 0.01, "default": 0.0, "forceInput": True})
            }
        }
    
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Flow"
    
    def run(self, float):
        return (float,)

NODE_CLASS_MAPPINGS["SV-FloatRerouteForSubnodes"] = FloatRerouteForSubnodes
NODE_DISPLAY_NAME_MAPPINGS["SV-FloatRerouteForSubnodes"] = "Float Reroute"

#-------------------------------------------------------------------------------#

class ModelReroute:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",)
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Flow"
    
    def run(self, model):
        return (model,)

NODE_CLASS_MAPPINGS["SV-ModelReroute"] = ModelReroute
NODE_DISPLAY_NAME_MAPPINGS["SV-ModelReroute"] = "Model Reroute"

#-------------------------------------------------------------------------------#

class SigmaReroute:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS",)
            }
        }
    
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Flow"
    
    def run(self, sigmas):
        return (sigmas,)

NODE_CLASS_MAPPINGS["SV-SigmaReroute"] = SigmaReroute
NODE_DISPLAY_NAME_MAPPINGS["SV-SigmaReroute"] = "Sigmas Reroute"

#-------------------------------------------------------------------------------#

class ConditioningReroute:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",)
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Flow"
    
    def run(self, conditioning):
        return (conditioning,)

NODE_CLASS_MAPPINGS["SV-ConditioningReroute"] = ConditioningReroute
NODE_DISPLAY_NAME_MAPPINGS["SV-ConditioningReroute"] = "Conditioning Reroute"

#-------------------------------------------------------------------------------#

class ImageReroute:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",)
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Flow"
    
    def run(self, image):
        return (image,)

NODE_CLASS_MAPPINGS["SV-ImageReroute"] = ImageReroute
NODE_DISPLAY_NAME_MAPPINGS["SV-ImageReroute"] = "Image Reroute"

#-------------------------------------------------------------------------------#

@VariantSupport()
class SwapValues:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "_1_": ("*",),
                "_2_": ("*",),
                "swap": ("BOOLEAN", {"default": True})
            }
        }
    
    RETURN_TYPES = ("*", "*")
    RETURN_NAMES = ("_2_", "_1_")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Logic"
    
    def run(self, _1_, _2_, swap):
        if swap:
            return (_2_, _1_)
        return (_1_, _2_)

NODE_CLASS_MAPPINGS["SV-SwapValues"] = SwapValues
NODE_DISPLAY_NAME_MAPPINGS["SV-SwapValues"] = "Swap"

#-------------------------------------------------------------------------------#

@VariantSupport()
class VariableSet:
    storage = {}
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "key": ("STRING",)
            },
            "optional": {
                "signal": ("signal",),
                "value": ("*",),
                "set": ("BOOLEAN", {"defaultInput": True, "default": True})
            }
        }
    
    OUTPUT_NODE = True
    RETURN_TYPES = ("signal", "STRING")
    RETURN_NAMES = ("signal", "value")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Logic"
    
    def run(self, value, key, set):
        if set == False:
            return (None, None)
        if value is None:
            del VariableSet.storage[key]
            return (None, None)
        VariableSet.storage[key] = value
        return (None, value)
    
NODE_CLASS_MAPPINGS["SV-VariableSet"] = VariableSet
NODE_DISPLAY_NAME_MAPPINGS["SV-VariableSet"] = "Var Set"

#-------------------------------------------------------------------------------#

@VariantSupport()
class VariableGet:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "key": ("STRING",)
            },
            "optional": {
                "signal": ("signal",)
            }
        }
    
    RETURN_TYPES = ("signal", "*")
    RETURN_NAMES = ("signal", "value")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Logic"
    
    def run(self, key, signal):
        if key not in VariableSet.storage:
            return (None, None)
        return (None, VariableSet.storage[key])

NODE_CLASS_MAPPINGS["SV-VariableGet"] = VariableGet
NODE_DISPLAY_NAME_MAPPINGS["SV-VariableGet"] = "Var Get"

#-------------------------------------------------------------------------------#

class VariableClear:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "signal": ("signal",)
            }
        }
    
    RETURN_TYPES = ("signal",)
    RETURN_NAMES = ("signal",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Logic"
    
    def run(self, signal):
        VariableSet.storage = {}
        return (None,)

NODE_CLASS_MAPPINGS["SV-VariableClear"] = VariableClear
NODE_DISPLAY_NAME_MAPPINGS["SV-VariableClear"] = "Var Clear"
#-------------------------------------------------------------------------------#

class EmptyValue:
    def __init__(self):
        pass

@VariantSupport()
class ValueRepeater:
    savedValue = EmptyValue()
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("*", {"lazy": True}),
                "repeat": ("BOOLEAN", {"defaultInput": True})
            }
        }
    
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("value",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Logic"
    
    def check_lazy_status(self, value, repeat):
        if repeat and type(self.savedValue) is not EmptyValue:
            return []
        return ["value"]
    
    def run(self, value, repeat):
        if repeat and type(self.savedValue) is not EmptyValue:
            return (self.savedValue,)
        self.savedValue = value
        return (value,)

NODE_CLASS_MAPPINGS["SV-ValueRepeater"] = ValueRepeater
NODE_DISPLAY_NAME_MAPPINGS["SV-ValueRepeater"] = "Value Repeater"

#-------------------------------------------------------------------------------#

class UnitFloat:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "float": ("FLOAT", {"min": 0, "max": 1, "step": 0.01, "default": 0.0})
            }
        }
    
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Input"
    
    def run(self, float):
        return (float,)

NODE_CLASS_MAPPINGS["SV-UnitFloat"] = UnitFloat
NODE_DISPLAY_NAME_MAPPINGS["SV-UnitFloat"] = "Unit Float"

#-------------------------------------------------------------------------------#

class MetadataJson:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "prompt": ("STRING", {"forceInput": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/IO"
    
    def run(self, prompt):
        return (json.dumps({"extra": {
            "prompt": prompt,
        }}),)

NODE_CLASS_MAPPINGS["SV-MetadataJson"] = MetadataJson
NODE_DISPLAY_NAME_MAPPINGS["SV-MetadataJson"] = "Metadata Json"

#-------------------------------------------------------------------------------#

class PadImage:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "padding": ("INT", {"default": 32, "min": 0, "max": 9999, "step": 1}),
                "color": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "run"
    CATEGORY = "SV Nodes/image"

    def run(self, image, padding, color):
        d1, d2, d3, d4 = image.size()

        new_image = torch.ones(
            (d1, d2 + padding * 2, d3 + padding * 2, d4),
            dtype=torch.float32,
        ) * color

        new_image[:, padding:padding + d2, padding:padding + d3, :] = image

        return (new_image,)

NODE_CLASS_MAPPINGS["SV-PadImage"] = PadImage
NODE_DISPLAY_NAME_MAPPINGS["SV-PadImage"] = "Pad Image"

#-------------------------------------------------------------------------------#
# Experiments

# By Thyri on discord
class CompressConds:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
            },
            "optional": {
                "negative": ("CONDITIONING",),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Experiments"
    
    
    def run(self, positive, negative=None):
        pos = self.compress(positive)
        neg = self.compress(negative) if negative is not None else None
        return pos, neg
        
    def compress(self, conditioning):
        out = copy.deepcopy(conditioning)
        for o in out:
            o[0] = self.reduc(o[0])
        return out
    
    def reduc(self, o):
        u,s,v = torch.svd(o)
        return (u[0,:75,:75]@s[0,:75].diag()@v[0,:,:75].T)[None]

NODE_CLASS_MAPPINGS["SV-CompressConds"] = CompressConds
NODE_DISPLAY_NAME_MAPPINGS["SV-CompressConds"] = "Compress Conds"
