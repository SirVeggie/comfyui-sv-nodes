import os
import time
import comfy.samplers
import folder_paths
import node_helpers
import hashlib
import math
import random as _random
import json
import re
import torch
import sys

class AnyType(str):
    def __eq__(self, __value: object) -> bool:
        return True
    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")

#-------------------------------------------------------------------------------#

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

#-------------------------------------------------------------------------------#

class PromptProcessing:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "forceInput": True})
            },
            "optional": {
                "variables": ("STRING", {"multiline": True, "forceInput": True, "default": ""}),
                "seed": ("INT", {"forceInput": True, "default": 1})
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
                "variables": ("STRING", {"forceInput": True, "default": ""}),
                "seed": ("INT", {"forceInput": True, "default": 1}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "lora")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Processing"
    
    def run(self, text, step, progress, variables="", seed=1):
        text = remove_comments(text)
        return LoraSeparator.run(self, process_advanced(text, variables, seed, step, progress))

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
                "seed": ("INT", {"forceInput": True}),
            }
        }
    
    RETURN_TYPES = ("sv_prompt", "STRING")
    RETURN_NAMES = ("prompt", "lora")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Processing"
    
    def run(self, prompt, steps, phase=1, variables="", seed=1):
        prompt = remove_comments(prompt)
        prompt, lora = LoraSeparator.run(self, prompt)
        parts = re.split(r"[\n\r]+[\s]*-+[\s]*[\n\r]+", prompt, 1)
        full_positive = parts[0]
        full_negative = parts[1] if len(parts) > 1 else ""
        
        result = []
        
        for i in range(1, steps + 1):
            pos = process_advanced(full_positive, variables, seed, i, i / steps + phase - 1)
            neg = process_advanced(full_negative, variables, seed, i, i / steps + phase - 1)
            result.append((pos, neg))
        
        return result, lora

#-------------------------------------------------------------------------------#

class PromptProcessingEncode:
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
        cache = {}
        pconds = []
        nconds = []
        
        for i in range(1, steps + 1):
            start = (i - 1) / steps
            end = i / steps
            pos, neg = prompt[i - 1]
            
            if pos not in cache:
                cache[pos] = encode(clip, pos)
            pcond = node_helpers.conditioning_set_values(cache[pos], {"start_percent": start, "end_percent": end})
            pconds += pcond
            if neg not in cache:
                cache[neg] = encode(clip, neg)
            ncond = node_helpers.conditioning_set_values(cache[neg], {"start_percent": start, "end_percent": end})
            nconds += ncond
        
        return pconds, nconds

def encode(clip, text):
    tokens = clip.tokenize(text)
    output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
    cond = output.pop("cond")
    return [[cond, output]]

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
        return ((cfg, steps, denoise, sampler, "normal", False),)

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
        return ((cfg, steps, denoise, sampler, scheduler, scheduler == "ays"),)

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
        return ((cfg, steps, 1.0, sampler, scheduler, scheduler == "ays"),)

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
    
    RETURN_TYPES = ("FLOAT", "INT", "FLOAT", comfy.samplers.SAMPLER_NAMES, comfy.samplers.SCHEDULER_NAMES, "BOOLEAN")
    RETURN_NAMES = ("cfg", "steps", "denoise", "sampler", "scheduler", "ays")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Input"
    
    def run(self, packet):
        if not isinstance(packet, tuple):
            raise TypeError("Invalid packet input type")
        if len(packet) != 6:
            raise ValueError("Invalid packet length")
        cfg = packet[0] or 8.0
        steps = packet[1] or 10
        denoise = packet[2] or 1.0
        sampler = packet[3] or comfy.samplers.SAMPLER_NAMES[0]
        scheduler = comfy.samplers.SCHEDULER_NAMES[0] if packet[4] in [None, "ays"] else packet[4]
        ays = packet[5] or False
        return cfg, steps, denoise, sampler, scheduler, ays

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
        prompt = re.sub(r"<l\w+:[^>]+>", "", text, 0, re.IGNORECASE)
        text = remove_comments(text)
        lora = "".join(re.findall(r"<l\w+:[^>]+>", text, re.IGNORECASE))
        return (prompt, lora)

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

#-------------------------------------------------------------------------------#

class InputSelect:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "select": ("INT", {"min": 1, "max": 5, "step": 1}),
            },
            "optional": {
                "_1_": (any_type,),
                "_2_": (any_type,),
                "_3_": (any_type,),
                "_4_": (any_type,),
                "_5_": (any_type,),
            }
        }
    
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("out",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Flow"
    
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

#-------------------------------------------------------------------------------#

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
                "on": (any_type,),
                "off": (any_type,)
            }
        }
    
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("out",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Flow"
    
    def run(self, select, on=None, off=None):
        if select:
            return (on,)
        return (off,)

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

#-------------------------------------------------------------------------------#

class MathCompare:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "number": ("INT,FLOAT", {"forceInput": True}),
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
    
    def run(self, sigmas):
        if len(sigmas) < 1:
            raise ValueError("Invalid sigmas length")
        return (sigmas[0].item(),)

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

#-------------------------------------------------------------------------------#

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
    
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("model name",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Input"
    
    def run(self, model):
        if not isinstance(model, str):
            raise TypeError("Invalid model input type")
        return (model,)

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

#-------------------------------------------------------------------------------#

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
    
    RETURN_TYPES = (any_type, "STRING")
    RETURN_NAMES = ("model name", "prompt")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Input"
    
    def run(self, packet):
        if not isinstance(packet, tuple):
            raise TypeError("Invalid output input type")
        if len(packet) != 2:
            raise ValueError("Invalid output length")
        return packet

#-------------------------------------------------------------------------------#

class CacheShield:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (any_type,)
            }
        }
    
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("any",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Flow"
    
    def run(self, any):
        return (any,)
    
    @classmethod
    def IS_CACHED(s, any):
        try:
            return hash_item(any)
        except:
            return ""

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

#-------------------------------------------------------------------------------#

class CacheShieldProxy:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "check": (any_type,)
            },
            "optional": {
                "_1_": (any_type,),
                "_2_": (any_type,),
                "_3_": (any_type,),
                "_4_": (any_type,),
                "_5_": (any_type,)
            }
        }
    
    RETURN_TYPES = (any_type, any_type, any_type, any_type, any_type)
    RETURN_NAMES = ("_1_", "_2_", "_3_", "_4_", "_5_")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Flow"
    
    def run(self, check, _1_=None, _2_=None, _3_=None, _4_=None, _5_=None):
        return (_1_, _2_, _3_, _4_, _5_)
    
    @classmethod
    def IS_CACHED(s, check, **kwargs):
        return CacheShield.IS_CACHED(s, check)

#-------------------------------------------------------------------------------#

class FlowManualCache:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (any_type,),
                "enabled": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("any",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Flow"
    
    def run(self, any, enabled):
        return (any,)
    
    @classmethod
    def IS_CONTROLLED(s, any, enabled):
        if enabled:
            return "cached"
        return None

#-------------------------------------------------------------------------------#

class FlowBlockSignal:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("bsignal",)
    RETURN_NAMES = ("signal",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Flow"
    
    def run(self, enabled):
        return (None,)
    
    @classmethod
    def IS_CONTROLLED(s, enabled):
        if enabled:
            return "blocked"
        return None

#-------------------------------------------------------------------------------#

class FlowBlock:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "signal": ("bsignal",),
                "any": (any_type,),
            }
        }
    
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("any",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Flow"
    
    def run(self, signal, any):
        return (any,)

#-------------------------------------------------------------------------------#

class FlowBlockSimple:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (any_type,),
                "enabled": ("BOOLEAN", {"default": False, "label_on": "block", "label_off": "allow"}),
            }
        }
    
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("any",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Flow"
    
    def run(self, any, enabled):
        return (any,)
    
    @classmethod
    def IS_CONTROLLED(s, any, enabled):
        if enabled:
            return "blocked"
        return None

#-------------------------------------------------------------------------------#
# Strongly referencing rgthree's any switch

def is_none(value):
    if value is not None:
        if isinstance(value, dict) and 'model' in value and 'clip' in value:
            return is_context_empty(value)
    return value is None
def is_context_empty(ctx):
    return not ctx or all(v is None for v in ctx.values())

class FlowContinue:
    CONTINUE = True
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "_1_": (any_type,),
                "_2_": (any_type,),
                "_3_": (any_type,),
                "_4_": (any_type,),
                "_5_": (any_type,),
            }
        }
    
    RETURN_TYPES = (any_type, "INT")
    RETURN_NAMES = ("any", "index")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Flow"
    
    def run(self, **kwargs):
        for key, value in kwargs.items():
            if key.startswith("_") and key.endswith("_") and not is_none(value):
                return (value, parse_index(key))
        return (None, 0)

#-------------------------------------------------------------------------------#

class FlowContinueSimple:
    CONTINUE = True
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (any_type,),
            }
        }
    
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("any",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Flow"
    
    def run(self, any):
        return (any,)

#-------------------------------------------------------------------------------#

class FlowNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "_1_": (any_type,),
                "_2_": (any_type,),
                "_3_": (any_type,),
                "_4_": (any_type,),
                "_5_": (any_type,),
            }
        }
    
    RETURN_TYPES = (any_type, any_type, any_type, any_type, any_type)
    RETURN_NAMES = ("_1_", "_2_", "_3_", "_4_", "_5_")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Flow"
    
    def run(self, _1_=None, _2_=None, _3_=None, _4_=None, _5_=None):
        return (_1_, _2_, _3_, _4_, _5_)

#-------------------------------------------------------------------------------#

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
                "_1_": (any_type,),
                "_2_": (any_type,),
                "_3_": (any_type,),
                "_4_": (any_type,),
                "_5_": (any_type,),
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

#-------------------------------------------------------------------------------#

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
                "_1_": (any_type,),
                "_2_": (any_type,),
                "_3_": (any_type,),
                "_4_": (any_type,),
                "_5_": (any_type,),
                "_6_": (any_type,),
                "_7_": (any_type,),
                "_8_": (any_type,),
                "_9_": (any_type,),
                "_10_": (any_type,)
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

#-------------------------------------------------------------------------------#

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
                "value": (any_type,),
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

#-------------------------------------------------------------------------------#

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
                "value": (any_type,),
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

#-------------------------------------------------------------------------------#

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
                "_1_": (any_type,),
                "_2_": (any_type,),
                "_3_": (any_type,),
                "_4_": (any_type,),
                "_5_": (any_type,)
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

#-------------------------------------------------------------------------------#

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
    
    RETURN_TYPES = ("sv_pipe", any_type, any_type, any_type, any_type, any_type)
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

#-------------------------------------------------------------------------------#

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
    
    RETURN_TYPES = ("sv_pipe", any_type, any_type, any_type, any_type, any_type, any_type, any_type, any_type, any_type, any_type)
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

#-------------------------------------------------------------------------------#

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
    
    RETURN_TYPES = ("sv_pipe", any_type)
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

#-------------------------------------------------------------------------------#

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
    
    RETURN_TYPES = ("sv_pipe", any_type)
    RETURN_NAMES = ("pipe", "value")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Pipes"
    
    def run(self, pipe, key):
        if not isinstance(pipe, dict):
            raise TypeError("Invalid pipe input type")
        if not isinstance(key, str):
            raise TypeError("Invalid key input type")
        return (pipe, pipe.get(key, None))

#-------------------------------------------------------------------------------#

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
    
    RETURN_TYPES = ("sv_pipe", any_type, any_type, any_type, any_type, any_type)
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

#-------------------------------------------------------------------------------#

class CheckNone:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (any_type,)
            }
        }
    
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("bool",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Logic"
    
    def run(self, any):
        return (any is None,)

#-------------------------------------------------------------------------------#

class CheckNoneNot:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (any_type,)
            }
        }
    
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("bool",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Logic"
    
    def run(self, any):
        return (any is not None,)

#-------------------------------------------------------------------------------#

class AnyToAny:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": (any_type,)
            }
        }
    
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("output",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes"
    
    def run(self, input):
        return (input,)

#-------------------------------------------------------------------------------#

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
                "signal": (any_type,),
            }
        }
    
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Debug"
    
    def run(self, text, signal=None):
        print(text.replace("_signal_", str(signal)))
        return {}

#-------------------------------------------------------------------------------#

class ConsolePrintMulti:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "signal1": (any_type,),
            },
            "optional": {
                "signal2": (any_type,),
                "signal3": (any_type,),
            }
        }
    
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Debug"
    
    def run(self, text, signal1, signal2=None, signal3=None):
        print(text.replace("_signal1_", str(signal1)).replace("_signal2_", str(signal2)).replace("_signal3_", str(signal3)))
        return {}

#-------------------------------------------------------------------------------#

class AssertNotNone:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (any_type,)
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

#-------------------------------------------------------------------------------#

class TimerStart:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (any_type,)
            }
        }
    
    OUTPUT_NODE = True
    RETURN_TYPES = (any_type, "TIMER")
    RETURN_NAMES = ("any", "timestamp")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Debug"
    
    def run(self, any):
        return (any, time.time())

#-------------------------------------------------------------------------------#

class TimerEnd:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (any_type,),
                "timestamp": ("TIMER",)
            }
        }
    
    OUTPUT_NODE = True
    RETURN_TYPES = (any_type, "FLOAT")
    RETURN_NAMES = ("any", "time")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Debug"
    
    def run(self, any, timestamp):
        return (any, time.time() - timestamp)

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
    CATEGORY = "SV Nodes/Logic"
    
    def run(self, float):
        return (float,)

#-------------------------------------------------------------------------------#

class SwapValues:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "_1_": (any_type,),
                "_2_": (any_type,),
                "swap": ("BOOLEAN", {"default": True})
            }
        }
    
    RETURN_TYPES = (any_type, any_type)
    RETURN_NAMES = ("_2_", "_1_")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Logic"
    
    def run(self, _1_, _2_, swap):
        if swap:
            return (_2_, _1_)
        return (_1_, _2_)

#-------------------------------------------------------------------------------#

NODE_CLASS_MAPPINGS = {
    "SV-SimpleText": SimpleText,
    "SV-PromptProcessing": PromptProcessing,
    "SV-PromptProcessingRecursive": PromptProcessingRecursive,
    "SV-PromptProcessingAdvanced": PromptProcessingAdvanced,
    "SV-PromptProcessingEncode": PromptProcessingEncode,
    "SV-ResolutionSelector": ResolutionSelector,
    "SV-ResolutionSelector2": ResolutionSelector2,
    "SV-ResolutionSelector2Output": ResolutionSelector2Output,
    "SV-NormalizeImageSize": NormalizeImageSize,
    "SV-BasicParams": BasicParams,
    "SV-BasicParamsPlus": BasicParamsPlus,
    "SV-BasicParamsCustom": BasicParamsCustom,
    "SV-BasicParamsOutput": BasicParamsOutput,
    "SV-SamplerNameToSampler": SamplerNameToSampler,
    "SV-StringSeparator": StringSeparator,
    "SV-LoraSeparator": LoraSeparator,
    "SV-StringCombine": StringCombine,
    "SV-InputSelect": InputSelect,
    "SV-InputSelectBoolean": InputSelectBoolean,
    "SV-LoadTextFile": LoadTextFile,
    "SV-SaveTextFile": SaveTextFile,
    "SV-BooleanNot": BooleanNot,
    "SV-MathAddInt": MathAddInt,
    "SV-MathCompare": MathCompare,
    "SV-SigmaOneStep": SigmaOneStep,
    "SV-SigmaRange": SigmaRange,
    "SV-SigmaContinue": SigmaContinue,
    "SV-SigmaContinueLinear": SigmaContinueLinear,
    "SV-SigmaRemap": SigmaRemap,
    "SV-SigmaConcat": SigmaConcat,
    "SV-SigmaAsFloat": SigmaAsFloat,
    "SV-SigmaLength": SigmaLength,
    "SV-ModelName": ModelName,
    "SV-PromptPlusModel": PromptPlusModel,
    "SV-PromptPlusModelOutput": PromptPlusModelOutput,
    "SV-CacheShield": CacheShield,
    "SV-CacheShieldProxy": CacheShieldProxy,
    "SV-HashModel": HashModel,
    "SV-FlowManualCache": FlowManualCache,
    "SV-FlowBlockSignal": FlowBlockSignal,
    "SV-FlowBlock": FlowBlock,
    "SV-FlowBlockSimple": FlowBlockSimple,
    "SV-FlowContinue": FlowContinue,
    "SV-FlowContinueSimple": FlowContinueSimple,
    "SV-FlowNode": FlowNode,
    "SV-CheckNone": CheckNone,
    "SV-CheckNoneNot": CheckNoneNot,
    "SV-AnyToAny": AnyToAny,
    "SV-ConsolePrint": ConsolePrint,
    "SV-ConsolePrintMulti": ConsolePrintMulti,
    "SV-AssertNotNone": AssertNotNone,
    "SV-TimerStart": TimerStart,
    "SV-TimerEnd": TimerEnd,
    "SV-FlowPipeInput": FlowPipeInput,
    "SV-FlowPipeInputLarge": FlowPipeInputLarge,
    "SV-FlowPipeInputIndex": FlowPipeInputIndex,
    "SV-FlowPipeInputKey": FlowPipeInputKey,
    "SV-FlowPipeInputKeyTuple": FlowPipeInputKeyTuple,
    "SV-FlowPipeCombine": FlowPipeCombine,
    "SV-FlowPipeOutput": FlowPipeOutput,
    "SV-FlowPipeOutputLarge": FlowPipeOutputLarge,
    "SV-FlowPipeOutputIndex": FlowPipeOutputIndex,
    "SV-FlowPipeOutputKey": FlowPipeOutputKey,
    "SV-FlowPipeOutputKeyTuple": FlowPipeOutputKeyTuple,
    "SV-FloatRerouteForSubnodes": FloatRerouteForSubnodes,
    "SV-SwapValues": SwapValues
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SV-SimpleText": "Simple Text",
    "SV-PromptProcessing": "Prompt Processing",
    "SV-PromptProcessingRecursive": "Recursive Processing",
    "SV-PromptProcessingAdvanced": "Advanced Processing",
    "SV-PromptProcessingEncode": "Encode Prompt",
    "SV-ResolutionSelector": "Resolution Selector",
    "SV-ResolutionSelector2": "Resolution Selector 2",
    "SV-ResolutionSelector2Output": "Selector Output",
    "SV-NormalizeImageSize": "Normalize Image Size",
    "SV-BasicParams": "Params",
    "SV-BasicParamsPlus": "Params Plus",
    "SV-BasicParamsCustom": "Params Custom",
    "SV-BasicParamsOutput": "Params Output",
    "SV-SamplerNameToSampler": "Sampler Converter",
    "SV-StringSeparator": "String Separator",
    "SV-LoraSeparator": "Lora Separator",
    "SV-StringCombine": "String Combine",
    "SV-InputSelect": "Input Select",
    "SV-InputSelectBoolean": "Boolean Select",
    "SV-LoadTextFile": "Load Text File",
    "SV-SaveTextFile": "Save Text File",
    "SV-BooleanNot": "Boolean Not",
    "SV-MathAddInt": "Add Int",
    "SV-MathCompare": "Compare",
    "SV-SigmaOneStep": "Sigmas One Step",
    "SV-SigmaRange": "Sigma Range",
    "SV-SigmaContinue": "Sigma Continue",
    "SV-SigmaContinueLinear": "Sigma Linear",
    "SV-SigmaRemap": "Sigma Remap",
    "SV-SigmaConcat": "Sigma Concat",
    "SV-SigmaAsFloat": "Sigma As Float",
    "SV-SigmaLength": "Sigma Length",
    "SV-ModelName": "Model Name",
    "SV-PromptPlusModel": "Prompt + Model",
    "SV-PromptPlusModelOutput": "P+M Output",
    "SV-CacheShield": "Cache Shield",
    "SV-CacheShieldProxy": "Cache Proxy",
    "SV-HashModel": "Hash Model",
    "SV-FlowManualCache": "Manual Cache",
    "SV-FlowBlockSignal": "Block Signal",
    "SV-FlowBlock": "Flow Block",
    "SV-FlowBlockSimple": "Simple Block",
    "SV-FlowContinue": "Flow Continue",
    "SV-FlowContinueSimple": "Simple Continue",
    "SV-FlowNode": "Flow Node",
    "SV-CheckNone": "Check None",
    "SV-CheckNoneNot": "Check Not None",
    "SV-AnyToAny": "Any to Any",
    "SV-ConsolePrint": "Console Print",
    "SV-ConsolePrintMulti": "Console Print Multi",
    "SV-AssertNotNone": "Assert Not None",
    "SV-TimerStart": "Timer Start",
    "SV-TimerEnd": "Timer End",
    "SV-FlowPipeInput": "Pipe In",
    "SV-FlowPipeInputLarge": "Pipe In Large",
    "SV-FlowPipeInputIndex": "Pipe In Index",
    "SV-FlowPipeInputKey": "Pipe In Key",
    "SV-FlowPipeInputKeyTuple": "Pipe In Tuple",
    "SV-FlowPipeCombine": "Pipe Combine",
    "SV-FlowPipeOutput": "Pipe Out",
    "SV-FlowPipeOutputLarge": "Pipe Out Large",
    "SV-FlowPipeOutputIndex": "Pipe Out Index",
    "SV-FlowPipeOutputKey": "Pipe Out Key",
    "SV-FlowPipeOutputKeyTuple": "Pipe Out Tuple",
    "SV-FloatRerouteForSubnodes": "Float",
    "SV-SwapValues": "Swap"
}

#-------------------------------------------------------------------------------#

def approx_index(reference: list[float], value: float):
    if value > reference[0]:
        raise ValueError("Value is greater than the maximum value in the reference list")
    if value == 0:
        return len(reference) - 1
    for i in range(len(reference)):
        if value > reference[i]:
            return i - 1 + (value - reference[i-1]) / (reference[i] - reference[i-1])

def calculate_sigma_range(reference: list[float], start: float, end: float, steps: int):
    start_percentage = approx_index(reference, start) / (len(reference) - 1)
    end_percentage = approx_index(reference, end) / (len(reference) - 1)
    return calculate_sigma_range_percent(reference, start_percentage, end_percentage, steps)

def calculate_sigma_range_percent(reference: list[float], start: float, end: float, steps: int):
    sigmas = []
    dist = (end - start) / steps
    for i in range(steps + 1):
        approx = (start + dist * i) * (len(reference) - 1)
        delta = approx - int(approx)
        lower_index = math.floor(approx)
        upper_index = math.ceil(approx)
        sigmas.append(reference[lower_index] + delta * (reference[upper_index] - reference[lower_index]))
    return sigmas

#-------------------------------------------------------------------------------#

processing_depth = 5
var_char = "$"
char_pair = ("{", "}")
adv_char_pair = ("[", "]")
bracket_pairs = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")]

#-------------------------------------------------------------------------------#

def parse_index(input: str):
    # parse index from input, eg. _1_
    if input.startswith("_") and input.endswith("_"):
        try:
            return int(input[1:-1])
        except:
            raise ValueError(f"Invalid index input: {input}")

def input_add(input: str, value: int):
    # add value to input index, eg. _1_ + 1 = _2_
    index = parse_index(input)
    return f"_{index + value}_"

#-------------------------------------------------------------------------------#

def parse_vars(variables: str):
    vars: dict[str, str] = {}
    lines = variables.split("\n")
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            continue
        if line[0] == "#":
            continue
        if line.startswith("//"):
            continue
        
        parts = line.split("=", 1)
        if len(parts) != 2:
            log_error(f"Invalid variable definition: {line}")
            continue
        
        name = parts[0].strip()
        text = parts[1].strip()
        vars[name] = text
    return vars
def build_var(name: str):
    if " " in name:
        return f"{var_char}({name})"
    return f"{var_char}{name}"
def clean_prompt(prompt: str):
    prompt = re.sub(r"\s*[\n\r,][,\s]*", ", ", prompt)
    prompt = re.sub(r"\s+", " ", prompt)
    return re.sub(r"[,\s]+$", "", prompt)
def remove_comments(prompt: str):
    # remove comments from prompt, accepts // and # comments
    lines = prompt.split("\n")
    lines = [line for line in lines if not line.strip().startswith("//") and not line.strip().startswith("#")]
    return "\n".join(lines)
def process(prompt, output: int, variables: str, seed: int):
    prompt = remove_comments(prompt)
    prompt = clean_prompt(prompt)
    
    vars = parse_vars(variables)
    names: list[str] = vars.keys()
    names = sorted(names, key=len, reverse=True)
    
    depth = 0
    previous_prompt = prompt
    while depth < processing_depth:
        prompt = decode(prompt, output, seed)
        
        for name in names:
            if name not in prompt:
                continue
            
            text = vars[name]
            if " " not in name:
                prompt = prompt.replace(f"{var_char}{name}", text)
            prompt = prompt.replace(f"{var_char}({name})", text)
        
        if prompt == previous_prompt:
            break
        previous_prompt = prompt
        depth += 1
    
    return clean_prompt(prompt)

def log_error(message):
    return
def error_context(text, i):
    return text[max(0, i-10):min(len(text), i+10)]
def is_opening(text, i):
    list = [char_pair, adv_char_pair] + bracket_pairs
    list = [item[0] for item in list]
    return text[i] in list and (i == 0 or text[i-1] != '\\')
def is_closing(text, i):
    list = [char_pair, adv_char_pair] + bracket_pairs
    list = [item[1] for item in list]
    return text[i] in list and (i == 0 or text[i-1] != '\\')
def get_pair(bracket: str, opening: bool):
    list = [char_pair, adv_char_pair] + bracket_pairs
    for pair in list:
        if opening and pair[0] == bracket:
            return pair[1]
        if not opening and pair[1] == bracket:
            return pair[0]
    return None
def decode(text: str, output: int, seed: int):
    depth = 0
    start = -1
    end = -1
    mode = "random"
    count = 0
    splits = []
    rand = _random.Random(seed)
    
    if len(text) == 0:
        return text
    
    i = -1
    while i + 1 < len(text):
        i += 1
        
        if is_opening(text, i):
            if depth == 0 and text[i] != char_pair[0]:
                continue
            if depth == 0:
                start = i
            depth += 1
        elif is_closing(text, i):
            if depth > 0:
                depth -= 1
            if depth == 0 and text[i] == char_pair[1] and start != -1:
                end = i
        elif text[i] == '|' and depth == 1:
            splits.append(i)
        elif text[i] == ':' and depth == 1:
            splits.append(i)
            mode = "hr"
        
        if end != -1:
            if mode == "hr" and len(splits) > 2:
                log_error("Warning: multiple splits in hr mode")
                return text
            
            if mode == "hr":
                part1 = text[start+1:splits[0]]
                part2 = text[splits[0]+1:end if len(splits) == 1 else splits[1]]
                part3 = "'2" if len(splits) == 1 else text[splits[1]+1:end]
                if part2 == "'1":
                    part2 = part1
                if part3 == "'1":
                    part3 = part1
                if part3 == "'2":
                    part3 = part2
                part = [part1, part2, part3][output]
                text = text[:start] + part + text[end+1:]
                
            elif mode == "random":
                parts = []
                if len(splits) == 0:
                    parts.append(text[start+1:end])
                else:
                    for k in range(len(splits)):
                        if k == 0:
                            parts.append(text[start+1:splits[k]])
                        else:
                            parts.append(text[splits[k-1]+1:splits[k]])
                    parts.append(text[splits[-1]+1:end])
                
                count += 1
                part = rand.choice(parts)
                text = text[:start] + part + text[end+1:]
            
            else:
                start += 1
            
            i = start - 1
            start = -1
            end = -1
            splits = []
            mode = "random"
    
    return text

#-------------------------------------------------------------------------------#
# Advanced Prompt Processing

def process_advanced(prompt, variables: str, seed: int, step: int, progress: float):
    prompt = remove_comments(prompt)
    prompt = clean_prompt(prompt)
    
    if len(prompt) == 0:
        return prompt
    
    vars = parse_vars(variables)
    names: list[str] = vars.keys()
    names = sorted(names, key=len, reverse=True)
    
    depth = 0
    previous_prompt = prompt
    while depth < processing_depth:
        prompt = decode_advanced(prompt, seed, step, progress)
        
        for name in names:
            if name not in prompt:
                continue
            
            text = vars[name]
            if " " not in name:
                prompt = prompt.replace(f"{var_char}{name}", text)
            prompt = prompt.replace(f"{var_char}({name})", text)
        
        if prompt == previous_prompt:
            break
        previous_prompt = prompt
        depth += 1
    
    return clean_prompt(prompt)

def decode_advanced(text: str, seed: int, step: int, progress: float):
    depth = 0
    start = -1
    end = -1
    mode = ""
    splits = []
    pipes = 0
    colons = 0
    rand = _random.Random(seed)
    
    if len(text) == 0:
        return text
    
    brackets = []
    i = -1
    while i + 1 < len(text):
        i += 1
        
        closing = is_closing(text, i) and not is_opening(text, i)
        if closing and len(brackets) == 0:
            raise ValueError(f"Invalid bracket closing: {text[i]} at {error_context(text, i)}")
        if closing and brackets[-1][1] != get_pair(text[i], False):
            raise ValueError(f"Invalid bracket closing: {text[i]} at {error_context(text, i)}")
        closing = is_closing(text, i) and len(brackets) and brackets[-1][1] == get_pair(text[i], False)
        opening = not closing and is_opening(text, i)
        
        if opening:
            brackets.append((i, text[i]))
            if depth == 0 and text[i] not in [char_pair[0], adv_char_pair[0]]:
                continue
            if depth == 0:
                start = i
                mode = "curly" if text[i] == char_pair[0] else "square"
            depth += 1
        elif closing:
            prev = brackets.pop()
            if prev[1] != get_pair(text[i], False):
                raise ValueError(f"Invalid bracket closing: {text[i]} at {error_context(text, i)}")
            if depth == 1 and text[i] in [char_pair[1], adv_char_pair[1]] and start != -1:
                end = i
            if depth <= 0 and text[i] in [char_pair[1], adv_char_pair[1]]:
                raise ValueError(f"Invalid bracket closing: {text[i]} at {error_context(text, i)}")
            if depth > 0:
                depth -= 1
        elif text[i] == '|' and depth == 1:
            splits.append((i, '|'))
            pipes += 1
        elif text[i] == ':' and depth == 1:
            splits.append((i, ':'))
            colons += 1
        
        if end != -1:
            if mode == "curly" and pipes + colons == 0:
                text = text[:start] + text[start+1:end] + text[end+1:]
            elif mode == "curly" and colons > 0 and pipes > 0:
                raise ValueError(f"Invalid curly bracket content at {text[start:end+1]}")
            elif mode == "curly" and colons > 0:
                # part1 = text[start+1:splits[0][0]]
                # text = text[:start] + part1 + text[end+1:]
                parts = []
                for k in range(len(splits)):
                    if k == 0:
                        parts.append(text[start+1:splits[k][0]])
                    else:
                        parts.append(text[splits[k-1][0]+1:splits[k][0]])
                parts.append(text[splits[-1][0]+1:end])
                
                index = min(int(progress), len(parts) - 1)
                part = parts[index]
                text = text[:start] + part + text[end+1:]
            elif mode == "curly" and pipes > 0:
                parts = []
                for k in range(len(splits)):
                    if k == 0:
                        parts.append(text[start+1:splits[k][0]])
                    else:
                        parts.append(text[splits[k-1][0]+1:splits[k][0]])
                parts.append(text[splits[-1][0]+1:end])
                
                part = rand.choice(parts)
                text = text[:start] + part + text[end+1:]
            
            elif mode == "square" and pipes + colons == 0:
                text = text[:start] + text[start+1:end] + text[end+1:]
            elif mode == "square" and pipes > 0 and colons > 0:
                raise ValueError(f"Invalid square bracket content at {text[start:end+1]}")
            elif mode == "square" and pipes > 0:
                parts = []
                for k in range(len(splits)):
                    if k == 0:
                        parts.append(text[start+1:splits[k][0]])
                    else:
                        parts.append(text[splits[k-1][0]+1:splits[k][0]])
                parts.append(text[splits[-1][0]+1:end])
                norm = (step - 1) % len(parts)
                part = parts[norm]
                text = text[:start] + part + text[end+1:]
            elif mode == "square" and colons > 0:
                if colons % 2 != 0:
                    raise ValueError(f"Invalid square bracket content at {text[start:end+1]}")
                parts = []
                weights = []
                if colons == 2:
                    parts.append(text[start+1:splits[0][0]])
                    parts.append(text[splits[0][0]+1:splits[1][0]])
                    weight = text[splits[1][0]+1:end]
                    weights = [weight, "end"]
                else:
                    for k in range(int(len(splits) / 2)):
                        index = k * 2
                        if index == 0:
                            parts.append(text[start+1:splits[index][0]])
                        else:
                            parts.append(text[splits[index-1][0]+1:splits[index][0]])
                        weights.append(text[splits[index][0]+1:splits[index+1][0]])
                    parts.append(text[splits[-1][0]+1:end])
                    for k in range(len(parts)):
                        if re.match(r"'\d+", parts[k]) is not None:
                            index = int(parts[k][1:]) - 1
                            if index < 0 or index >= len(parts) or index == k:
                                raise ValueError(f"Invalid square bracket pointer {parts[k]} at {text[start:end+1]}")
                            parts[k] = parts[index]
                    weights.append("end")
                
                part = ""
                for k in range(len(weights)):
                    if weights[k] == "end":
                        part = parts[k]
                        break
                    is_step = str(weights[k]).startswith("s")
                    weight = int(weights[k][1:]) if is_step else float(weights[k])
                    if is_step and step <= weight:
                        part = parts[k]
                        break
                    if not is_step and progress <= weight:
                        part = parts[k]
                        break
                    
                text = text[:start] + part + text[end+1:]
                
            else:
                raise ValueError("Unexpected bracket evaluation")
            
            i = start - 1
            start = -1
            end = -1
            splits = []
            mode = ""
            pipes = 0
            colons = 0
    
    return text