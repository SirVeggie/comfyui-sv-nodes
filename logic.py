import math
import re
import random as _random

#-------------------------------------------------------------------------------#
# Helper functions

def default(value, *args):
    if value is not None:
        return value
    for arg in args:
        if arg is not None:
            return arg
    return None

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
    if steps == 0:
        return []
    sigmas = []
    dist = (end - start) / steps
    for i in range(steps + 1):
        approx = (start + dist * i) * (len(reference) - 1)
        delta = approx - int(approx)
        lower_index = math.floor(approx)
        upper_index = math.ceil(approx)
        sigmas.append(reference[lower_index] + delta * (reference[upper_index] - reference[lower_index]))
    if sigmas[0] == sigmas[-1]:
        return [sigmas[0]]
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
# Introspection

def needs_seed(prompt: str):
    old_prompt = ""
    while prompt != old_prompt:
        if re.search(r"{[^\[\]{}()<>]*\|[^\[\]{}()<>]*}", prompt):
            return True
        old_prompt = prompt
        prompt = re.sub(r"[\([{<][^\[\]{}()<>]*[)\]}>]", "", prompt)
    return False

#-------------------------------------------------------------------------------#

def separate_lora(text: str):
    prompt = re.sub(r"<l\w+:[^>]+>", "", text, 0, re.IGNORECASE)
    text = remove_comments(text)
    lora = "".join(re.findall(r"<l\w+:[^>]+>", text, re.IGNORECASE))
    return (prompt, lora)

def unescape_prompt(text: str):
    return re.sub(r"\\([<>\[\]|:])", r"\1", text)

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
def finalize_prompt(prompt: str):
    return re.sub(r"\\:", ":", prompt)
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
    
    return finalize_prompt(clean_prompt(prompt))

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
        elif text[i] == ':' and text[i-1] != '\\' and depth == 1:
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

def separate_lora_advanced(prompts: list[str]):
    lora = separate_lora(prompts[0][0])[1]
    for i in range(len(prompts)):
        prompt = separate_lora(prompts[i][0])[0]
        prompt = clean_prompt(prompt)
        prompts[i] = (prompt, prompts[i][1])
    return prompts, lora

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
    
    return finalize_prompt(clean_prompt(prompt))

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
        elif text[i] == ':' and text[i-1] != '\\' and depth == 1:
            splits.append((i, ':'))
            colons += 1
        
        if end != -1:
            if mode == "curly" and pipes + colons == 0:
                text = text[:start] + text[start+1:end] + text[end+1:]
            elif mode == "curly" and colons > 0 and pipes > 0:
                raise ValueError(f"Invalid curly bracket content at {text[start:end+1]}")
            elif mode == "curly" and colons > 0:
                parts = []
                for k in range(len(splits)):
                    if k == 0:
                        parts.append(text[start+1:splits[k][0]])
                    else:
                        parts.append(text[splits[k-1][0]+1:splits[k][0]])
                parts.append(text[splits[-1][0]+1:end])
                for k in range(len(parts)):
                    if re.match(r"'\d+$", parts[k]) is not None:
                        index = int(parts[k][1:]) - 1
                        if index < 0 or index >= len(parts) or index == k:
                            raise ValueError(f"Invalid curly bracket pointer {parts[k]} at {text[start:end+1]}")
                        parts[k] = parts[index]
                
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
                for k in range(len(parts)):
                    if re.match(r"'\d+$", parts[k]) is not None:
                        index = int(parts[k][1:]) - 1
                        if index < 0 or index >= len(parts) or index == k:
                            raise ValueError(f"Invalid square bracket pointer {parts[k]} at {text[start:end+1]}")
                        parts[k] = parts[index]
                
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
                if colons == 2 and not text[splits[0][0]+1:splits[1][0]].replace(".", "", 1).isdigit():
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
                    weights.append("end")
                
                for k in range(len(parts)):
                    if re.match(r"'\d+$", parts[k]) is not None:
                        index = int(parts[k][1:]) - 1
                        if index < 0 or index >= len(parts) or index == k:
                            raise ValueError(f"Invalid square bracket pointer {parts[k]} at {text[start:end+1]}")
                        parts[k] = parts[index]
                
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
                    if not is_step and progress < weight:
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
    
    if (depth != 0):
        raise ValueError("Invalid syntax: mismatched brackets")
    
    return text

#-------------------------------------------------------------------------------#
# Simple Prompt Processing

def process_simple(prompt, variables: str, seed: int, hires: bool):
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
        prompt = decode_simple(prompt, seed, hires)
        
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
    
    return finalize_prompt(clean_prompt(prompt))

def decode_simple(text: str, seed: int, hires: bool):
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
            if depth == 0 and text[i] not in [char_pair[0]]:
                continue
            if depth == 0:
                start = i
                mode = "curly" if text[i] == char_pair[0] else "square"
            depth += 1
        elif closing:
            prev = brackets.pop()
            if prev[1] != get_pair(text[i], False):
                raise ValueError(f"Invalid bracket closing: {text[i]} at {error_context(text, i)}")
            if depth == 1 and text[i] in [char_pair[1]] and start != -1:
                end = i
            if depth <= 0 and text[i] in [char_pair[1]]:
                raise ValueError(f"Invalid bracket closing: {text[i]} at {error_context(text, i)}")
            if depth > 0:
                depth -= 1
        elif text[i] == '|' and depth == 1:
            splits.append((i, '|'))
            pipes += 1
        elif text[i] == ':' and text[i-1] != '\\' and depth == 1:
            splits.append((i, ':'))
            colons += 1
        
        if end != -1:
            if mode == "curly" and pipes + colons == 0:
                text = text[:start] + text[start+1:end] + text[end+1:]
            elif mode == "curly" and colons > 0 and pipes > 0:
                raise ValueError(f"Invalid curly bracket content at {text[start:end+1]}")
            elif mode == "curly" and colons > 0:
                parts = []
                for k in range(len(splits)):
                    if k == 0:
                        parts.append(text[start+1:splits[k][0]])
                    else:
                        parts.append(text[splits[k-1][0]+1:splits[k][0]])
                parts.append(text[splits[-1][0]+1:end])
                for k in range(len(parts)):
                    if re.match(r"'\d+$", parts[k]) is not None:
                        index = int(parts[k][1:]) - 1
                        if index < 0 or index >= len(parts) or index == k:
                            raise ValueError(f"Invalid curly bracket pointer {parts[k]} at {text[start:end+1]}")
                        parts[k] = parts[index]
                
                index = min(1 if hires else 0, len(parts) - 1)
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
                for k in range(len(parts)):
                    if re.match(r"'\d+$", parts[k]) is not None:
                        index = int(parts[k][1:]) - 1
                        if index < 0 or index >= len(parts) or index == k:
                            raise ValueError(f"Invalid square bracket pointer {parts[k]} at {text[start:end+1]}")
                        parts[k] = parts[index]
                
                part = rand.choice(parts)
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
    
    if (depth != 0):
        raise ValueError("Invalid syntax: mismatched brackets")
    
    return text

#-------------------------------------------------------------------------------#
# Prompt Control Prompt Processing

def process_control(prompt: str, steps: int, phase: int, variables: str, seed: int):
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
        prompt = decode_control(prompt, steps, phase, seed)
        
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
    
    return finalize_prompt(clean_prompt(prompt))

def decode_control(text: str, steps: int, phase: int, seed: int):
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
                raise ValueError(f"Invalid bracket pair when closing: {text[i]} at {error_context(text, i)}")
            if depth == 1 and text[i] in [char_pair[1], adv_char_pair[1]] and start != -1:
                end = i
            # if depth <= 0 and text[i] in [char_pair[1], adv_char_pair[1]]:
            #     raise ValueError(f"Invalid bracket closing: {text[i]} at {error_context(text, i)}")
            if depth > 0:
                depth -= 1
        elif text[i] == '|' and depth == 1:
            splits.append((i, '|'))
            pipes += 1
        elif text[i] == ':' and text[i-1] != '\\' and depth == 1:
            splits.append((i, ':'))
            colons += 1
        
        if end == -1:
            continue
        
        if mode == "curly" and pipes + colons == 0:
            text = text[:start] + text[start+1:end] + text[end+1:]
        elif mode == "curly" and colons > 0 and pipes > 0:
            raise ValueError(f"Invalid curly bracket content at {text[start:end+1]}")
        elif mode == "curly" and colons > 0:
            parts = []
            for k in range(len(splits)):
                if k == 0:
                    parts.append(text[start+1:splits[k][0]])
                else:
                    parts.append(text[splits[k-1][0]+1:splits[k][0]])
            parts.append(text[splits[-1][0]+1:end])
            for k in range(len(parts)):
                if re.match(r"'\d+$", parts[k]) is not None:
                    index = int(parts[k][1:]) - 1
                    if index < 0 or index >= len(parts) or index == k:
                        raise ValueError(f"Invalid curly bracket pointer {parts[k]} at {text[start:end+1]}")
                    parts[k] = parts[index]
            
            index = min(phase - 1, len(parts) - 1)
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
            for k in range(len(parts)):
                if re.match(r"'\d+$", parts[k]) is not None:
                    index = int(parts[k][1:]) - 1
                    if index < 0 or index >= len(parts) or index == k:
                        raise ValueError(f"Invalid square bracket pointer {parts[k]} at {text[start:end+1]}")
                    parts[k] = parts[index]
            
            part = rand.choice(parts)
            text = text[:start] + part + text[end+1:]
        
        elif mode == "square" and text[start+1:end].startswith("SEQ"):
            brackets.append((start, "["))
            start += 1
        elif mode == "square" and pipes + colons == 0:
            text = text[:start] + text[start+1:end] + text[end+1:]
        elif mode == "square" and pipes > 0 and colons > 0:
            brackets.append((start, "["))
            start += 1
        elif mode == "square" and pipes > 0:
            text = text[:end] + ":" + str(1.0 / steps) + text[end:]
            brackets.append((start, "["))
            start += 1
        elif mode == "square" and colons == 1 and not text[start+1:splits[0][0]].replace(".", "").isdigit() and not text[splits[0][0]+1:end].replace(".", "").isdigit():
            left = text[start+1:splits[0][0]]
            right = text[splits[0][0]+1:end]
            part = f"{left}:0.333:({left}, {right}:0.5):0.666:{right}";
            text = text[:start+1] + part + text[end:]
        elif mode == "square" and colons == 1 and text[start+1:splits[0][0]].replace(".", "").isdigit() and text[splits[0][0]+1:end].replace(".", "").isdigit():
            left = text[start+1:splits[0][0]]
            right = text[splits[0][0]+1:end]
            left = float(left)
            right = float(right)
            part = "SEQ"
            delta = right - left
            for k in range(steps - 1):
                part += ":" + str(left + delta * (k / (steps - 1)))[:5] + ":" + str((k + 1) / steps)
            part += ":" + str(right) + ":1.0"
            text = text[:start+1] + part + text[end:]
            brackets.append((start, "["))
            start += 1
        elif mode == "square" and colons == 1:
            reason = f"syntax [number:number] and [prompt:prompt] can't mix between numbers and prompts"
            raise ValueError(f"Invalid square bracket content at {text[start:end+1]}: {reason}")
        elif mode == "square" and colons > 0:
            if colons % 2 != 0:
                raise ValueError(f"Invalid square bracket content at {text[start:end+1]}: invalid number of colons ({colons})")
            
            # build part and timing lists
            parts = []
            weights = []
            if colons == 2 and not text[splits[0][0]+1:splits[1][0]].replace(".", "", 1).isdigit():
                parts.append(text[start+1:splits[0][0]])
                parts.append(text[splits[0][0]+1:splits[1][0]])
                weight = text[splits[1][0]+1:end]
                weights = [float(weight), 9999]
            else:
                for k in range(int(len(splits) / 2)):
                    index = k * 2
                    if index == 0:
                        parts.append(text[start+1:splits[index][0]])
                    else:
                        parts.append(text[splits[index-1][0]+1:splits[index][0]])
                    weights.append(float(text[splits[index][0]+1:splits[index+1][0]]))
                parts.append(text[splits[-1][0]+1:end])
                weights.append(9999)
            
            # expand pointers ('1, '2, etc)
            for k in range(len(parts)):
                if re.match(r"'\d+$", parts[k]) is not None:
                    index = int(parts[k][1:]) - 1
                    if index < 0 or index >= len(parts) or index == k:
                        raise ValueError(f"Invalid square bracket pointer {parts[k]} at {text[start:end+1]}")
                    parts[k] = parts[index]
            
            # build SEQ
            part = "SEQ"
            partCount = 0
            latestPart = ""
            for k in range(len(weights)):
                if weights[k] > phase - 1 and (k <= 0 or weights[k-1] < phase):
                    partCount += 1
                    latestPart = parts[k]
                    part += ":" + parts[k] + ":"
                    part += (str(min(1.0, weights[k] - (phase - 1))) if weights[k] != 9999 else "1.0")
            
            # remove SEQ if only one part
            if partCount == 1:
                part = latestPart
            text = text[:start+1] + part + text[end:]
            brackets.append((start, "["))
            start += 1
            
        else:
            raise ValueError("Unexpected bracket evaluation")
        
        i = start - 1
        start = -1
        end = -1
        splits = []
        mode = ""
        pipes = 0
        colons = 0
    
    if (depth != 0):
        raise ValueError("Invalid syntax: mismatched brackets")
    
    return text

#-------------------------------------------------------------------------------#
# Variable Only Prompt Processing

def process_vars(prompt, variables: str):
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
    
    return finalize_prompt(clean_prompt(prompt))