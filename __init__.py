import os
import comfy.samplers
import folder_paths
import hashlib
import math
import random as _random
import json
import re

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")

#-------------------------------------------------------------------------------#

class PromptProcessing:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": (any_type, {"multiline": False})
            },
            "optional": {
                "variables": (any_type, {"multiline": True, "default": ""}),
                "seed": (any_type, {"default": 1})
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("1st pass", "2nd pass", "3rd pass")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes"
    
    def run(self, text, variables="", seed=1):
        if not isinstance(text, str) or not isinstance(variables, str):
            raise TypeError("Invalid text input type")
        if not isinstance(seed, int):
            raise TypeError("Invalid seed input type")
        return process(text, 0, variables, seed), process(text, 1, variables, seed), process(text, 2, variables, seed)
    
    @classmethod
    def IS_CHANGED(s, text, variables, seed):
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
                "orientation": ("BOOLEAN", {"default": False, "label_on": "landscape", "label_off": "portrait"})
            },
            "optional": {
                "seed": ("*", {"default": 1}),
                "random": ("*", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes"
    
    def run(self, base, ratio, orientation, seed=1, random=""):
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
        elif random:
            random = random.replace(" ", "").split(",")
            rand = _random.Random(seed)
            ratio = rand.choice(random).split(":")
        ratio = math.sqrt(float(ratio[0]) / float(ratio[1]))
        
        width = math.floor(base * ratio / 64) * 64 
        height = math.floor(base / ratio / 64) * 64
        
        if not orientation:
            width, height = height, width
        return width, height

#-------------------------------------------------------------------------------#

class ResolutionSelector2:
    RATIOS = ["1:1", "5:4", "4:3", "3:2", "16:9", "21:9"]
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 64}),
                "ratio": (ResolutionSelector2.RATIOS,),
                "orientation": ("BOOLEAN", {"default": False, "label_on": "landscape", "label_off": "portrait"}),
                "hires": ("FLOAT", {"min": 1, "max": 4, "step": 0.1, "default": 1.5}),
                "batch": ("INT", {"min": 1, "max": 32, "step": 1, "default": 1})
            },
            "optional": {
                "seed": ("*", {"default": 1}),
                "random": ("*", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("RS_OUTPUT",)
    RETURN_NAMES = ("packet",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes"
    
    def run(self, base, ratio, orientation, hires, batch, seed=1, random=""):
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
    CATEGORY = "SV Nodes"
    
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
                "sampler": (comfy.samplers.SAMPLER_NAMES, )
            }
        }
    
    RETURN_TYPES = ("BP_OUTPUT",)
    RETURN_NAMES = ("packet",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes"
    
    def run(self, cfg, steps, denoise, sampler):
        if not isinstance(cfg, float) and not isinstance(cfg, int):
            raise TypeError("Invalid cfg input type")
        if not isinstance(steps, int):
            raise TypeError("Invalid steps input type")
        if not isinstance(denoise, float) and not isinstance(denoise, int):
            raise TypeError("Invalid denoise input type")
        if not isinstance(sampler, str):
            raise TypeError("Invalid sampler input type")
        return ((cfg, steps, denoise, sampler),)

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
    
    RETURN_TYPES = ("FLOAT", "INT", "FLOAT", "SAMPLER")
    RETURN_NAMES = ("cfg", "steps", "denoise", "sampler")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes"
    
    def run(self, packet):
        if not isinstance(packet, tuple):
            raise TypeError("Invalid packet input type")
        if len(packet) != 4:
            raise ValueError("Invalid packet length")
        sampler = comfy.samplers.sampler_object(packet[3])
        return packet[0], packet[1], packet[2], sampler
    

#-------------------------------------------------------------------------------#

class StringSeparator:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": (any_type,),
                "separator": ("STRING", {"default": "\\n---\\n"})
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("part1", "part2")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes"
    
    def run(self, text, separator="\\n---\\n"):
        if not isinstance(text, str):
            raise TypeError("Invalid text input type")
        if not isinstance(separator, str):
            raise TypeError("Invalid separator input type")
        separator = separator.replace("\\n", "\n").replace("\\t", "\t")
        parts = text.split(separator, 1)
        return parts[0], parts[1] if len(parts) > 1 else ""

#-------------------------------------------------------------------------------#

class StringCombine:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "part1": (any_type,),
                "part2": (any_type,),
                "separator": ("STRING", {"default": "\\n"})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes"
    
    def run(self, part1, part2, separator="\\n"):
        if not isinstance(part1, str) or not isinstance(part2, str):
            raise TypeError("Invalid part input type")
        if not isinstance(separator, str):
            raise TypeError("Invalid separator input type")
        separator = separator.replace("\\n", "\n").replace("\\t", "\t")
        return (part1 + separator + part2,)

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
            return float("NaN")
        return ""
    
    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("content", "success")
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes"
    
    def run(self, path):
        if not isinstance(path, str):
            raise TypeError("Invalid path input type")
        try:
            with open(path, "r") as file:
                return (file.read(), True)
        except:
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
                "content": (any_type,)
            }
        }
    
    @classmethod
    def IS_CHANGED(s, path, content):
        
        m = hashlib.sha256()
        m.update(content.encode())
        return m.hexdigest()
    
    OUTPUT_NODE = True
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("success",)
    
    FUNCTION = "run"
    CATEGORY = "SV Nodes"
    
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
    CATEGORY = "SV Nodes"
    
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
    CATEGORY = "SV Nodes"
    
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
    CATEGORY = "SV Nodes"
    
    def run(self, any):
        return (any,)
    
    @classmethod
    def IS_CHANGED(s, any):
        try:
            if isinstance(any, (str, int, float, bool, type(None))):
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
    CATEGORY = "SV Nodes"
    
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
    CATEGORY = "SV Nodes"
    
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
    CATEGORY = "SV Nodes"
    
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
    CATEGORY = "SV Nodes"
    
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
    CATEGORY = "SV Nodes"
    
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
    CATEGORY = "SV Nodes"
    
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
    CATEGORY = "SV Nodes"
    
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
    CATEGORY = "SV Nodes"
    
    def run(self, _1_=None, _2_=None, _3_=None, _4_=None, _5_=None):
        return (_1_, _2_, _3_, _4_, _5_)

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
    CATEGORY = "SV Nodes"
    
    def run(self, any):
        return (any is None,)

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

NODE_CLASS_MAPPINGS = {
    "SV-PromptProcessing": PromptProcessing,
    "SV-ResolutionSelector": ResolutionSelector,
    "SV-ResolutionSelector2": ResolutionSelector2,
    "SV-ResolutionSelector2Output": ResolutionSelector2Output,
    "SV-BasicParams": BasicParams,
    "SV-BasicParamsOutput": BasicParamsOutput,
    "SV-StringSeparator": StringSeparator,
    "SV-StringCombine": StringCombine,
    "SV-LoadTextFile": LoadTextFile,
    "SV-SaveTextFile": SaveTextFile,
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
    "SV-AnyToAny": AnyToAny
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SV-PromptProcessing": "Prompt Processing",
    "SV-ResolutionSelector": "Resolution Selector",
    "SV-ResolutionSelector2": "Resolution Selector 2",
    "SV-ResolutionSelector2Output": "Selector Output",
    "SV-BasicParams": "Params",
    "SV-BasicParamsOutput": "Params Output",
    "SV-StringSeparator": "String Separator",
    "SV-StringCombine": "String Combine",
    "SV-LoadTextFile": "Load Text File",
    "SV-SaveTextFile": "Save Text File",
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
    "SV-AnyToAny": "Any to Any"
}

#-------------------------------------------------------------------------------#

processing_depth = 5
var_char = "$"
char_open = "{"
char_close = "}"

#-------------------------------------------------------------------------------#

def parse_vars(variables: str):
    vars: dict[str, str] = {}
    lines = variables.split("\n")
    for line in lines:
        line = line.strip()
        if len(line) == 0:
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
def process(prompt, output: int, variables: str, seed: int):
    prompt = re.sub(r"[\s,]*[\n\r]+[\s,]*", ", ", prompt)
    prompt = re.sub(r"\s+", " ", prompt)
    
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
    
    prompt = re.sub(r"[\s\n\r]*,[,\s\n\r]*", ", ", prompt)
    prompt = re.sub(r"\s+", " ", prompt)
    return prompt

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