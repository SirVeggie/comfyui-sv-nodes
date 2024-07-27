import os
import comfy.samplers
import folder_paths
import hashlib
import math
import random as _random
import json
import re
import torch

class AnyType(str):
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
        if not isinstance(text, str) or not isinstance(variables, str):
            raise TypeError("Invalid text input type")
        if not isinstance(seed, int):
            raise TypeError("Invalid seed input type")
        return process(text, 0, variables, seed), process(text, 1, variables, seed), process(text, 2, variables, seed)
    
    @classmethod
    def IS_CACHED(s, text, variables, seed):
        return f"{text} {variables} {seed}"

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
        return ((cfg, steps, denoise, sampler, "normal"),)

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
                "scheduler": (comfy.samplers.SCHEDULER_NAMES,)
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
        return ((cfg, steps, denoise, sampler, scheduler),)

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
    
    RETURN_TYPES = ("FLOAT", "INT", "FLOAT", comfy.samplers.SAMPLER_NAMES, comfy.samplers.SCHEDULER_NAMES)
    RETURN_NAMES = ("cfg", "steps", "denoise", "sampler", "scheduler")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Input"
    
    def run(self, packet):
        if not isinstance(packet, tuple):
            raise TypeError("Invalid packet input type")
        if len(packet) != 5:
            raise ValueError("Invalid packet length")
        return packet[0], packet[1], packet[2], packet[3], packet[4]

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
        if not isinstance(text, str):
            raise TypeError("Invalid text input type")
        prompt = re.sub(r"<l\w+:[^>]+>", "", text, 0, re.IGNORECASE)
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
    RETURN_NAMES = ("value",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Logic"
    
    def run(self, value):
        return (not value,)

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
        if steps < 1:
            return (torch.FloatTensor([]).cpu(),)
        lastSigma = source[-1].item()
        if lastSigma < 0.0001:
            raise ValueError("Invalid source sigma")
        imitateRaw = imitate.tolist()
        length = len(imitateRaw)
        start = 0
        while lastSigma < imitateRaw[start]:
            start += 1
        result = [lastSigma]
        for i in range(1, steps + 1):
            progress = i / steps
            sigma_i = round((length - 1 - start) * progress + start)
            result.append(imitateRaw[sigma_i])
        return (torch.FloatTensor(result).cpu(),)

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
            if isinstance(any, (tuple, list)):
                is_changed = False
                for item in any:
                    is_changed = is_changed or CacheShield.IS_CHANGED(s, item)
                return is_changed
            elif isinstance(any, (str, int, float, bool, type(None))):
                return any
            elif isinstance(any, (list, tuple, dict)):
                return hashlib.md5(json.dumbs(any, sort_keys=True).encode()).hexdigest()
            else:
                if hasattr(any, "__dict__"):
                    return hashlib.md5(json.dumps(any.__dict__, sort_keys=True).encode()).hexdigest()
                else:
                    return hashlib.md5(repr(any).encode()).hexdigest()
        except:
            return ""

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

class FlowSelect:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "select": ("INT", {"min": 1, "max": 5, "step": 1}),
            },
            "optional": {
                "_1_": (any_type, {"default": None}),
                "_2_": (any_type, {"default": None}),
                "_3_": (any_type, {"default": None}),
                "_4_": (any_type, {"default": None}),
                "_5_": (any_type, {"default": None}),
            }
        }
    
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("out",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Flow"
    
    def run(self, select, _1_=None, _2_=None, _3_=None, _4_=None, _5_=None):
        raise NotImplementedError("This node is not working yet")
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

class FlowContinue:
    CONTINUE = True
    
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
    
    RETURN_TYPES = (any_type, "INT")
    RETURN_NAMES = ("any", "index")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes/Flow"
    
    def run(self, _1_=None, _2_=None, _3_=None, _4_=None, _5_=None):
        for i, any in enumerate((_1_, _2_, _3_, _4_, _5_)):
            if any is not None:
                return (any, i+1)
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
        if not isinstance(pipe, dict):
            raise TypeError("Invalid pipe input type")
        if not isinstance(key, str):
            raise TypeError("Invalid key input type")
        value = pipe.get(key, None)
        if not isinstance(value, (tuple, list)):
            raise ValueError("Invalid value type")
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
    CATEGORY = "SV Nodes/Output"
    
    def run(self, text, signal=None):
        print(text.replace("_signal_", str(signal)))
        return {}

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
    "SV-ResolutionSelector": ResolutionSelector,
    "SV-ResolutionSelector2": ResolutionSelector2,
    "SV-ResolutionSelector2Output": ResolutionSelector2Output,
    "SV-BasicParams": BasicParams,
    "SV-BasicParamsPlus": BasicParamsPlus,
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
    "SV-SigmaOneStep": SigmaOneStep,
    "SV-SigmaRange": SigmaRange,
    "SV-SigmaContinue": SigmaContinue,
    "SV-SigmaAsFloat": SigmaAsFloat,
    "SV-SigmaLength": SigmaLength,
    "SV-ModelName": ModelName,
    "SV-PromptPlusModel": PromptPlusModel,
    "SV-PromptPlusModelOutput": PromptPlusModelOutput,
    "SV-CacheShield": CacheShield,
    "SV-FlowManualCache": FlowManualCache,
    "SV-FlowBlockSignal": FlowBlockSignal,
    "SV-FlowBlock": FlowBlock,
    "SV-FlowBlockSimple": FlowBlockSimple,
    # "SV-FlowSelect": FlowSelect,
    "SV-FlowContinue": FlowContinue,
    "SV-FlowContinueSimple": FlowContinueSimple,
    "SV-FlowNode": FlowNode,
    "SV-CheckNone": CheckNone,
    "SV-CheckNoneNot": CheckNoneNot,
    "SV-AnyToAny": AnyToAny,
    "SV-ConsolePrint": ConsolePrint,
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
    "SV-ResolutionSelector": "Resolution Selector",
    "SV-ResolutionSelector2": "Resolution Selector 2",
    "SV-ResolutionSelector2Output": "Selector Output",
    "SV-BasicParams": "Params",
    "SV-BasicParamsPlus": "Params Plus",
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
    "SV-SigmaOneStep": "Sigmas One Step",
    "SV-SigmaRange": "Sigma Range",
    "SV-SigmaContinue": "Sigma Continue",
    "SV-SigmaAsFloat": "Sigma As Float",
    "SV-SigmaLength": "Sigma Length",
    "SV-ModelName": "Model Name",
    "SV-PromptPlusModel": "Prompt + Model",
    "SV-PromptPlusModelOutput": "P+M Output",
    "SV-CacheShield": "Cache Shield",
    "SV-FlowManualCache": "Manual Cache",
    "SV-FlowBlockSignal": "Block Signal",
    "SV-FlowBlock": "Flow Block",
    "SV-FlowBlockSimple": "Simple Block",
    # "SV-FlowSelect": "Flow Select",
    "SV-FlowContinue": "Flow Continue",
    "SV-FlowContinueSimple": "Simple Continue",
    "SV-FlowNode": "Flow Node",
    "SV-CheckNone": "Check None",
    "SV-CheckNoneNot": "Check Not None",
    "SV-AnyToAny": "Any to Any",
    "SV-ConsolePrint": "Console Print",
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

processing_depth = 5
var_char = "$"
char_open = "{"
char_close = "}"

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
    return re.sub(r"\s+", " ", prompt)
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
def is_opening(text, i):
    list = [char_open, '{', '(', '[', '<']
    return text[i] in list and (i == 0 or text[i-1] != '\\')
def is_closing(text, i):
    list = [char_close, '}', ')', ']', '>']
    return text[i] in list and (i == 0 or text[i-1] != '\\')
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
            if depth == 0 and text[i] != char_open:
                continue
            if depth == 0:
                start = i
            depth += 1
        elif is_closing(text, i):
            if depth > 0:
                depth -= 1
            if depth == 0 and text[i] == char_close and start != -1:
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