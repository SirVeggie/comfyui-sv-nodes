#!/usr/bin/env python3
"""Migrate comfyui-sv-nodes V1 __init__.py to ComfyUI V3 schema."""

from __future__ import annotations

import ast
import importlib.util
import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "__init__.py"
BACKUP = ROOT / "__init__v1_backup.py"
BORROWED = ROOT / "borrowed.py"
OUTPUT = ROOT / "__init__.py"

HEADER = '''import os
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

'''

FOOTER = '''

#-------------------------------------------------------------------------------#
# Extension

class SVNodesExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return NODE_LIST

async def comfy_entrypoint() -> SVNodesExtension:
    return SVNodesExtension()
'''


def _install_comfy_mocks():
    import types
    import sys

    if "comfy" in sys.modules:
        return

    comfy = types.ModuleType("comfy")
    samplers = types.ModuleType("comfy.samplers")
    samplers.SAMPLER_NAMES = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral"]
    samplers.SCHEDULER_NAMES = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]
    comfy.samplers = samplers
    sys.modules["comfy"] = comfy
    sys.modules["comfy.samplers"] = samplers

    folder_paths = types.ModuleType("folder_paths")

    def _fake_list(_kind):
        return ["model_a.safetensors", "model_b.safetensors"]

    folder_paths.get_filename_list = _fake_list
    sys.modules["folder_paths"] = folder_paths

    node_helpers = types.ModuleType("node_helpers")
    node_helpers.conditioning_set_values = lambda c, v: c
    sys.modules["node_helpers"] = node_helpers

    graph_utils = types.ModuleType("comfy_execution.graph_utils")
    graph_utils.GraphBuilder = type("GraphBuilder", (), {"node": lambda *a, **k: None, "finalize": lambda s: {}})
    graph_utils.is_link = lambda v: isinstance(v, list)
    sys.modules["comfy_execution"] = types.ModuleType("comfy_execution")
    sys.modules["comfy_execution.graph_utils"] = graph_utils

    graph_mod = types.ModuleType("comfy_execution.graph")
    graph_mod.ExecutionBlocker = type("ExecutionBlocker", (), {"__init__": lambda s, v: None})
    sys.modules["comfy_execution.graph"] = graph_mod


def load_v1_module():
    _install_comfy_mocks()
    import sys
    sys.path.insert(0, str(ROOT))
    source = BACKUP.read_text(encoding="utf-8")
    source = source.replace("from .logic import", "from logic import")
    source = source.replace("from .borrowed import", "from borrowed import")
    tmp = ROOT / "_migrate_v1_load.py"
    tmp.write_text(source, encoding="utf-8")
    borrowed_mod = None
    try:
        spec = importlib.util.spec_from_file_location("sv_nodes_v1", tmp)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if BORROWED.exists():
            bspec = importlib.util.spec_from_file_location("borrowed", BORROWED)
            borrowed_mod = importlib.util.module_from_spec(bspec)
            bspec.loader.exec_module(borrowed_mod)
        mod._borrowed_mod = borrowed_mod
        return mod
    finally:
        if tmp.exists():
            tmp.unlink()


def parse_mappings(*sources: str) -> tuple[dict[str, str], dict[str, str]]:
    node_ids: dict[str, str] = {}
    display_names: dict[str, str] = {}
    for source in sources:
        for m in re.finditer(
            r'^(?:NODE|BORROWED)_CLASS_MAPPINGS\["([^"]+)"\]\s*=\s*(\w+)',
            source,
            re.MULTILINE,
        ):
            node_ids[m.group(2)] = m.group(1)
        for m in re.finditer(
            r'^(?:NODE|BORROWED)_DISPLAY_NAME_MAPPINGS\["([^"]+)"\]\s*=\s*"([^"]*)"',
            source,
            re.MULTILINE,
        ):
            display_names[m.group(1)] = m.group(2)
    return node_ids, display_names


def split_classes(source: str) -> list[tuple[str, str, str | None]]:
    """Return list of (class_name, class_source, node_id or None)."""
    node_ids, _ = parse_mappings(source)
    id_by_class = node_ids

    sections = re.split(r'\n(?=#-{10,}#)', source)
    classes: list[tuple[str, str, str | None]] = []
    for section in sections:
        class_matches = list(re.finditer(r'^class\s+(\w+)\s*[:(]', section, re.MULTILINE))
        for i, m in enumerate(class_matches):
            name = m.group(1)
            if name in ("SmartType",):
                continue
            start = m.start()
            end = class_matches[i + 1].start() if i + 1 < len(class_matches) else len(section)
            map_m = re.search(r'\n(?:NODE|BORROWED)_(?:CLASS|DISPLAY)_MAPPINGS\[', section[start:end])
            if map_m:
                end = start + map_m.start()
            chunk = section[start:end].strip()
            node_id = id_by_class.get(name)
            if node_id and i > 0:
                helpers: list[str] = []
                for j in range(i - 1, -1, -1):
                    helper_name = class_matches[j].group(1)
                    if id_by_class.get(helper_name):
                        break
                    h_start = class_matches[j].start()
                    h_end = class_matches[j + 1].start()
                    helpers.insert(0, section[h_start:h_end].strip())
                if helpers:
                    chunk = "\n\n".join(helpers) + "\n\n" + chunk
            classes.append((name, chunk, node_id))
    return classes


def get_class_attr(source: str, attr: str) -> str | None:
    m = re.search(rf'^\s*{attr}\s*=\s*(.+)$', source, re.MULTILINE)
    return m.group(1).strip() if m else None


def has_decorator(source: str, dec: str) -> bool:
    return f"@{dec}" in source


def get_method_body(source: str, method_names: list[str]) -> tuple[str, str] | None:
    for name in method_names:
        m = re.search(
            rf'^(\s*)def\s+{re.escape(name)}\s*\((.*?)\)\s*:(.*?)(?=\n\1(?:def |@classmethod)|\n#-{10,}#|\nclass |\Z)',
            source,
            re.MULTILINE | re.DOTALL,
        )
        if m:
            return name, m.group(0)
    return None


def format_options(options: dict) -> str:
    if not options:
        return ""
    parts = []
    for k, v in options.items():
        nk = {"forceInput": "force_input", "rawLink": "raw_link", "defaultInput": "force_input"}.get(k, k)
        if isinstance(v, str):
            parts.append(f'{nk}={v!r}')
        elif isinstance(v, bool):
            parts.append(f"{nk}={v}")
        elif isinstance(v, (int, float)):
            parts.append(f"{nk}={v}")
        else:
            parts.append(f"{nk}={v!r}")
    return ", " + ", ".join(parts)


MATCHTYPE_INPUT_OPTS = {"lazy", "raw_link", "rawLink", "advanced", "tooltip"}
COMBO_EXCLUDED_OPTS = {"forceInput", "force_input", "defaultInput"}


def format_combo_options(options: dict) -> str:
    return format_options({k: v for k, v in options.items() if k not in COMBO_EXCLUDED_OPTS})


def input_to_code(name: str, spec: tuple, optional: bool, use_match: bool, template: str) -> str:
    type_part = spec[0]
    opts = spec[1] if len(spec) > 1 else {}

    if type_part == "*":
        opts = {k: v for k, v in opts.items() if k in MATCHTYPE_INPUT_OPTS}
        extra = format_options(opts)
        if optional:
            extra += (", optional=True" if extra else ", optional=True")
        if opts.get("lazy") and "lazy=True" not in extra:
            extra += ", lazy=True"
        return f'io.MatchType.Input({name!r}, template={template}{extra})'

    extra = format_options(opts)
    if optional:
        extra += (", optional=True" if extra else ", optional=True")
    if opts.get("lazy"):
        extra += (", lazy=True" if "lazy=True" not in extra else "")

    if isinstance(type_part, list):
        options_repr = repr(type_part)
        extra = format_combo_options(opts)
        if optional:
            extra += (", optional=True" if extra else ", optional=True")
        if opts.get("lazy") and "lazy=True" not in extra:
            extra += ", lazy=True"
        return f"io.Combo.Input({name!r}, options={options_repr}{extra})"

    if isinstance(type_part, str):
        type_map = {
            "STRING": "io.String", "INT": "io.Int", "FLOAT": "io.Float", "BOOLEAN": "io.Boolean",
            "IMAGE": "io.Image", "MASK": "io.Mask", "LATENT": "io.Latent",
            "CONDITIONING": "io.Conditioning", "MODEL": "io.Model", "VAE": "io.Vae", "CLIP": "io.Clip",
            "wildcards": "Wildcards", "sv_prompt": "SvPrompt", "cond_list": "CondList",
            "RS_OUTPUT": "RsOutput", "PPM_OUTPUT": "PpmOutput", "BP_OUTPUT": "BpOutput",
            "SIGMAS": "Sigmas", "SAMPLER": "SvSampler", "FLOW_CONTROL": "FlowControl",
            "sv_pipe": "SvPipe", "ACCUMULATION": "Accumulation", "signal": "Signal",
            "TIMER": "Timer", "CURVE": "Curve",
        }
        io_type = type_map.get(type_part)
        if io_type:
            if io_type in ("Wildcards", "SvPrompt", "CondList", "RsOutput", "PpmOutput", "BpOutput",
                           "Sigmas", "SvSampler", "FlowControl", "SvPipe", "Accumulation",
                           "Signal", "Timer", "Curve"):
                return f"{io_type}.Input({name!r}{extra})"
            return f"{io_type}.Input({name!r}{extra})"

    # dynamic expression (folder_paths, comfy.samplers, class attrs)
    if isinstance(type_part, str):
        expr = type_part
    else:
        expr = repr(type_part)
    combo_extra = format_combo_options(opts)
    if optional:
        combo_extra += (", optional=True" if combo_extra else ", optional=True")
    if opts.get("lazy") and "lazy=True" not in combo_extra:
        combo_extra += ", lazy=True"
    return f"io.Combo.Input({name!r}, options={expr}{combo_extra})"


def output_to_code(type_part, name: str | None, template: str, use_match: bool) -> str:
    extra = f", display_name={name!r}" if name else ""
    if type_part == "*":
        return f"io.MatchType.Output(template={template}{extra})"
    type_map = {
        "STRING": "io.String", "INT": "io.Int", "FLOAT": "io.Float", "BOOLEAN": "io.Boolean",
        "IMAGE": "io.Image", "MASK": "io.Mask", "LATENT": "io.Latent",
        "CONDITIONING": "io.Conditioning", "MODEL": "io.Model", "VAE": "io.Vae", "CLIP": "io.Clip",
        "wildcards": "Wildcards", "sv_prompt": "SvPrompt", "cond_list": "CondList",
        "RS_OUTPUT": "RsOutput", "PPM_OUTPUT": "PpmOutput", "BP_OUTPUT": "BpOutput",
        "SIGMAS": "Sigmas", "SAMPLER": "SvSampler", "FLOW_CONTROL": "FlowControl",
        "sv_pipe": "SvPipe", "ACCUMULATION": "Accumulation", "signal": "Signal",
        "TIMER": "Timer", "CURVE": "Curve",
    }
    if isinstance(type_part, str) and type_part in type_map:
        t = type_map[type_part]
        return f"{t}.Output({extra.lstrip(', ')})" if extra else f"{t}.Output()"
    if isinstance(type_part, str):
        return f"io.Custom({type_part!r}).Output({extra.lstrip(', ')})" if extra else f"io.Custom({type_part!r}).Output()"
    return f"io.String.Output({extra.lstrip(', ')})" if extra else "io.String.Output()"


def eval_input_types(cls, mod):
    if not hasattr(cls, "INPUT_TYPES"):
        return {}
    try:
        return cls.INPUT_TYPES()
    except Exception as e:
        print(f"  WARN: could not eval INPUT_TYPES for {cls.__name__}: {e}")
        return {}


def has_star_types(inputs_dict, return_types=()) -> bool:
    for section in ("required", "optional", "hidden"):
        for _name, spec in inputs_dict.get(section, {}).items():
            if isinstance(spec, tuple) and spec[0] == "*":
                return True
            if isinstance(spec, str) and spec == "*":
                return True
    return "*" in return_types


def generate_define_schema(cls, mod, node_id: str, display_name: str, class_source: str) -> str:
    inputs_dict = eval_input_types(cls, mod)
    return_types = getattr(cls, "RETURN_TYPES", ())
    use_match = has_decorator(class_source, "VariantSupport") or has_star_types(inputs_dict, return_types)
    template = "cls._match_template" if use_match else "None"
    input_lines = []
    for section in ("required", "optional"):
        for name, spec in inputs_dict.get(section, {}).items():
            if not isinstance(spec, tuple):
                continue
            input_lines.append(
                "            " + input_to_code(name, spec, section == "optional", use_match, template) + ","
            )

    hidden = inputs_dict.get("hidden", {})
    hidden_lines = []
    hidden_map = {
        "UNIQUE_ID": "io.Hidden.unique_id",
        "PROMPT": "io.Hidden.prompt",
        "EXTRA_PNGINFO": "io.Hidden.extra_pnginfo",
        "DYNPROMPT": "io.Hidden.dynprompt",
    }
    for key, val in hidden.items():
        if isinstance(val, str) and val in hidden_map:
            hidden_lines.append(f"            {hidden_map[val]},")
        elif key == "initial_value0" or (isinstance(val, tuple) and val[0] == "*"):
            # special hidden wildcard - still use hidden list with dynprompt if present
            pass

    return_types = getattr(cls, "RETURN_TYPES", ())
    return_names = getattr(cls, "RETURN_NAMES", ())
    output_lines = []
    for i, rt in enumerate(return_types):
        rn = return_names[i] if i < len(return_names) else None
        output_lines.append("            " + output_to_code(rt, rn, template, use_match) + ",")

    category = getattr(cls, "CATEGORY", "SV Nodes")
    schema_flags = []
    if getattr(cls, "OUTPUT_NODE", False):
        schema_flags.append("is_output_node=True")
    if getattr(cls, "DEPRECATED", False):
        schema_flags.append("is_deprecated=True")
    if getattr(cls, "NOT_IDEMPOTENT", False):
        schema_flags.append("not_idempotent=True")
    if "expand" in class_source or "GraphBuilder" in class_source and "finalize" in class_source:
        if any(x in class_source for x in ("for_loop_open", "for_loop_close", "while_loop_close")):
            schema_flags.append("enable_expand=True")

    flags_str = ""
    match_decl = ""
    if use_match:
        match_decl = f"    _match_template = io.MatchType.Template({cls.__name__!r})\n\n"

    hidden_block = ""
    if hidden_lines:
        hidden_block = ",\n            hidden=[\n" + "\n".join(hidden_lines) + "\n            ],"

    if schema_flags:
        joiner = ",\n            "
        flags_str = joiner + ",\n            ".join(schema_flags)

    return match_decl + f'''    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id={node_id!r},
            display_name={display_name!r},
            category={category!r},
            inputs=[
{chr(10).join(input_lines) if input_lines else "            "}
            ],
            outputs=[
{chr(10).join(output_lines) if output_lines else "            "}
            ]{hidden_block}{flags_str}
        )
'''


def transform_method(method_src: str, old_name: str) -> str:
    src = method_src
    # instance -> classmethod execute
    src = re.sub(rf'def\s+{re.escape(old_name)}\s*\(\s*self\s*\)\s*:', "def execute(cls):", src)
    src = re.sub(rf'def\s+{re.escape(old_name)}\s*\(\s*self\s*,', "def execute(cls,", src)
    src = re.sub(rf'def\s+{re.escape(old_name)}\s*\(\s*s\s*,', "def execute(cls,", src)
    src = re.sub(r'^(\s*)def\s+', r'\1@classmethod\n\1def ', src, count=1)

    # check_lazy_status self -> cls
    src = re.sub(r'def check_lazy_status\(self,', "def check_lazy_status(cls,", src)
    src = re.sub(r'def check_lazy_status\(cls,', "@classmethod\n    def check_lazy_status(cls,", src)
    src = src.replace("def check_lazy_status(cls,", "@classmethod\n    def check_lazy_status(cls,")
    src = re.sub(r'(@classmethod\n\s*)@classmethod\n', r'\1', src)

    # IS_CHANGED / IS_CACHED -> fingerprint_inputs
    src = re.sub(r'def IS_CHANGED\(', "def fingerprint_inputs(", src)
    src = re.sub(r'def IS_CACHED\(', "def fingerprint_inputs(", src)
    src = re.sub(r'def IS_CHANGED\(s,', "def fingerprint_inputs(cls,", src)
    src = re.sub(r'def IS_CACHED\(s,', "def fingerprint_inputs(cls,", src)

    # VALIDATE_INPUTS -> validate_inputs
    src = src.replace("VALIDATE_INPUTS", "validate_inputs")

    # hidden input access is handled per-node (e.g. WhileLoopClose uses cls.hidden)

    # return transformations
    src = re.sub(
        r'return\s+\{\s*"result"\s*:\s*([^,]+),\s*"expand"\s*:\s*([^}]+)\}',
        r'return io.NodeOutput(\1, expand=\2)',
        src,
    )
    src = re.sub(r'return\s+\{\s*\}', "return io.NodeOutput()", src)
    src = re.sub(r'return\s+\(\s*\)', "return io.NodeOutput()", src)

    # return tuple -> NodeOutput (simple cases)
    def ret_repl(m):
        inner = m.group(1).strip()
        if not inner:
            return "return io.NodeOutput()"
        return f"return io.NodeOutput({inner})"

    src = re.sub(r'return\s+\(([^()]*(?:\([^()]*\)[^()]*)*)\)\s*$', ret_repl, src, flags=re.MULTILINE)

    # bare tuple return without parens in while_loop_close
    src = re.sub(
        r'(\n\s+)return tuple\(values\)',
        r'\1return io.NodeOutput(*values)',
        src,
    )
    src = re.sub(
        r'(\n\s+)return tuple\(\[([^\]]+)\]\)',
        r'\1return io.NodeOutput(\2)',
        src,
    )

    # add return type hint on execute
    src = re.sub(
        r'(def execute\(cls,[^)]*\)):',
        r'\1 -> io.NodeOutput:',
        src,
    )

    return src


def transform_class(name: str, source: str, node_id: str | None, mod, display_names: dict) -> str:
    if not node_id:
        return ""

    cls = getattr(mod, name, None)
    if cls is None and getattr(mod, "_borrowed_mod", None) is not None:
        cls = getattr(mod._borrowed_mod, name, None)
    if cls is None:
        print(f"  SKIP: class {name} not in module")
        return ""

    display_name = display_names.get(node_id, node_id)

    # strip old v1 boilerplate
    body = source
    body = re.sub(r'@VariantSupport\(\)\s*\n', '', body)
    body = re.sub(r'class\s+' + re.escape(name) + r'\s*:', f"class {name}(io.ComfyNode):", body)
    body = re.sub(
        rf'(class {re.escape(name)}\(io\.ComfyNode\):)\s*\n\s*def __init__\(self\)[^:]*:\s*\n\s*pass\s*\n',
        r'\1\n',
        body,
    )
    body = re.sub(r'^\s*def __init__\(self\)[^:]*:\s*\n\s*pass\s*\n', '', body, flags=re.MULTILINE)

    # remove v1 class attributes
    for attr in (
        "RETURN_TYPES", "RETURN_NAMES", "FUNCTION", "CATEGORY",
        "OUTPUT_NODE", "DEPRECATED", "NOT_IDEMPOTENT", "EXPERIMENTAL",
    ):
        body = re.sub(rf'^\s*{attr}\s*=.*\n', '', body, flags=re.MULTILINE)

    # remove INPUT_TYPES method
    body = re.sub(
        r'@classmethod\s*\n\s*def INPUT_TYPES\([^)]*\):.*?(?=\n    (?:@classmethod|def |\w+\s*=))',
        '',
        body,
        flags=re.DOTALL,
    )

    # remove IS_CHANGED / IS_CACHED at classmethod level (will be re-added from transform)
    # transform execute method
    func_name = getattr(cls, "FUNCTION", "run")
    method = get_method_body(body, [func_name, "run", "patch", "for_loop_open", "for_loop_close",
                                     "while_loop_open", "while_loop_close", "int_math_operation",
                                     "to_bool", "accumulate", "accumulation_head", "accumulation_tail",
                                     "accumulation_to_list", "list_to_accumulation", "accumlength",
                                     "get_item", "set_item"])
    if method:
        old_name, method_src = method
        new_method = transform_method(method_src, old_name)
        body = body.replace(method_src, new_method)

    # fix cross-class calls ResolutionSelector.run(self -> ResolutionSelector.execute
    body = body.replace(f"{name}.run(self,", f"{name}.execute(")
    body = re.sub(r'(\w+)\.run\(self,', r'\1.execute(', body)

    # instance state -> class-level keyed by unique_id for cache nodes
    if name in ("CacheObject", "ManualCache"):
        body = body.replace("self.obj", f"{name}._objs.get(cls.hidden.unique_id)")
        body = re.sub(
            rf'class {name}\(io\.ComfyNode\):',
            f"class {name}(io.ComfyNode):\n    _objs: dict = {{}}",
            body,
        )
        body = body.replace(
            f"{name}._objs.get(cls.hidden.unique_id) is not None",
            f"{name}._objs.get(cls.hidden.unique_id) is not None",
        )
        # assignments
        body = re.sub(
            rf'{name}\._objs\.get\(cls\.hidden\.unique_id\)\s*=\s*',
            f"{name}._objs[cls.hidden.unique_id] = ",
            body,
        )
        body = body.replace(
            f"if enable and {name}._objs.get(cls.hidden.unique_id) is not None:",
            f"if enable and cls.hidden.unique_id in {name}._objs:",
        )
        body = body.replace(
            f"return ({name}._objs.get(cls.hidden.unique_id),)",
            f"return io.NodeOutput({name}._objs[cls.hidden.unique_id])",
        )

    if name == "ClearCustomCaches":
        body = body.replace("if self.id is None:", "if ClearCustomCaches._id is None:")
        body = body.replace("self.id = round", "ClearCustomCaches._id = round")
        body = re.sub(
            r'class ClearCustomCaches\(io\.ComfyNode\):',
            "class ClearCustomCaches(io.ComfyNode):\n    _id = None",
            body,
        )

    if name == "ValueRepeater":
        body = re.sub(r'^\s*savedValue\s*=.*\n', '', body, flags=re.MULTILINE)
        body = body.replace("type(self.savedValue)", "type(ValueRepeater._saved.get(cls.hidden.unique_id, EmptyValue()))")
        body = body.replace("self.savedValue", "ValueRepeater._saved.get(cls.hidden.unique_id, EmptyValue())")
        body = re.sub(
            r'class ValueRepeater\(io\.ComfyNode\):',
            "class ValueRepeater(io.ComfyNode):\n    _saved: dict = {}",
            body,
        )
        body = body.replace(
            "ValueRepeater._saved.get(cls.hidden.unique_id, EmptyValue()) = value",
            "ValueRepeater._saved[cls.hidden.unique_id] = value",
        )

    # WhileLoopClose: use cls.hidden for dynprompt/unique_id
    if name == "WhileLoopClose":
        body = body.replace(
            "def while_loop_close(self, flow_control, condition, dynprompt=None, unique_id=None, **kwargs):",
            "def execute(cls, flow_control, condition, **kwargs) -> io.NodeOutput:",
        )
        body = body.replace(
            "def execute(cls, flow_control, condition, dynprompt=None, unique_id=None, **kwargs):",
            "def execute(cls, flow_control, condition, **kwargs) -> io.NodeOutput:",
        )
        body = body.replace(
            "def execute(cls, flow_control, condition, =None, unique_id=None, **kwargs):",
            "def execute(cls, flow_control, condition, **kwargs) -> io.NodeOutput:",
        )
        body = body.replace("dynprompt.get_node(unique_id)", "cls.hidden.dynprompt.get_node(cls.hidden.unique_id)")
        body = body.replace("original_node = dynprompt.get_node(node_id)", "original_node = cls.hidden.dynprompt.get_node(node_id)")
        body = body.replace("self.explore_dependencies(unique_id, dynprompt", "cls.explore_dependencies(cls.hidden.unique_id, cls.hidden.dynprompt")
        body = body.replace("self.collect_contained(", "cls.collect_contained(")
        body = body.replace("contained[unique_id]", "contained[cls.hidden.unique_id]")
        body = body.replace('graph.lookup_node("Recurse" )', 'graph.lookup_node("Recurse")')
        body = body.replace("my_clone.out(x)", "my_clone.out(x)")
        # convert helper methods to classmethods
        body = body.replace("def explore_dependencies(self,", "@classmethod\n    def explore_dependencies(cls,")
        body = body.replace("def collect_contained(self,", "@classmethod\n    def collect_contained(cls,")
        body = body.replace("self.explore_dependencies(", "cls.explore_dependencies(")
        body = body.replace("self.collect_contained(", "cls.collect_contained(")

    # PromptProcessingEncode: instance cache helpers -> classmethods
    if name == "PromptProcessingEncode":
        body = body.replace("def cacheHas(self,", "@classmethod\n    def cacheHas(cls,")
        body = body.replace("def cacheSet(self,", "@classmethod\n    def cacheSet(cls,")
        body = body.replace("def cacheGet(self,", "@classmethod\n    def cacheGet(cls,")
        body = body.replace("def cacheClean(self,", "@classmethod\n    def cacheClean(cls,")
        body = body.replace("self.cacheHas(", "cls.cacheHas(")
        body = body.replace("self.cacheSet(", "cls.cacheSet(")
        body = body.replace("self.cacheGet(", "cls.cacheGet(")
        body = body.replace("self.cacheClean()", "cls.cacheClean()")
        body = body.replace("return pconds, nconds", "return io.NodeOutput(pconds, nconds)")

    # CompressConds
    if name == "CompressConds":
        body = body.replace("def compress(self,", "@classmethod\n    def compress(cls,")
        body = body.replace("def reduc(self,", "@classmethod\n    def reduc(cls,")
        body = body.replace("self.compress(", "cls.compress(")
        body = body.replace("self.reduc(", "cls.reduc(")
        body = body.replace("return pos, neg", "return io.NodeOutput(pos, neg)")

    # remove mapping lines from class block
    body = re.sub(r'\nNODE_CLASS_MAPPINGS\[.*\nNODE_DISPLAY_NAME_MAPPINGS\[.*', '', body)
    body = re.sub(r'\nNODE_CLASS_MAPPINGS\[.*', '', body)
    body = re.sub(r'\nNODE_DISPLAY_NAME_MAPPINGS\[.*', '', body)

    schema = generate_define_schema(cls, mod, node_id, display_name, source)

    class_header = f"class {name}(io.ComfyNode):"
    if class_header in body:
        body = body.replace(class_header, class_header + "\n" + schema, 1)

    return f"\n#-------------------------------------------------------------------------------#\n\n{body.strip()}\n"


def extract_helpers(source: str) -> str:
    """Keep module-level helper functions (encode, parseCurve, evaluateComparison, etc.)."""
    helpers = []
    for func_name in ("encode", "parseCurve", "parseCurveFunction", "collapseSigns", "parseCurvePart",
                      "evaluateComparison"):
        m = re.search(rf'^(def {func_name}\(.*)', source, re.MULTILINE | re.DOTALL)
        if not m:
            continue
        start = m.start()
        # find end at next def/class at column 0 or NODE_
        rest = source[start:]
        end_m = re.search(r'\n(?=def |class |NODE_CLASS_MAPPINGS)', rest[1:])
        chunk = rest[: end_m.start() + 1] if end_m else rest
        if func_name == "encode":
            helpers.append(chunk.strip())
    return "\n\n".join(helpers)


def main():
    # Always migrate from the fullest V1 snapshot available, not git HEAD.
    if not BACKUP.exists() or "NODE_CLASS_MAPPINGS" not in BACKUP.read_text(encoding="utf-8"):
        text = SOURCE.read_text(encoding="utf-8")
        if "NODE_CLASS_MAPPINGS" not in text:
            raise SystemExit(f"No V1 source found in {SOURCE} or {BACKUP}")
        BACKUP.write_text(text, encoding="utf-8")
        print(f"Backed up workspace V1 to {BACKUP}")
    else:
        print(f"Using V1 snapshot {BACKUP}")

    text = BACKUP.read_text(encoding="utf-8")
    borrowed_text = BORROWED.read_text(encoding="utf-8") if BORROWED.exists() else ""

    node_ids, display_names = parse_mappings(text, borrowed_text)
    classes = split_classes(text) + split_classes(borrowed_text)

    print("Loading V1 module for INPUT_TYPES introspection...")
    mod = load_v1_module()

    out = [HEADER]
    node_list: list[str] = []

    helpers_added = set()
    for class_name, class_source, node_id in classes:
        if not node_id:
            continue
        print(f"Migrating {class_name} ({node_id})...")
        converted = transform_class(class_name, class_source, node_id, mod, display_names)
        if converted:
            out.append(converted)
            node_list.append(class_name)
            if class_name == "PromptProcessingEncode" and "encode" not in helpers_added:
                h = extract_helpers(text)
                if h:
                    out.append(f"\n#-------------------------------------------------------------------------------#\n\n{h}\n")
                    helpers_added.add("encode")
            if class_name == "CurveFromEquation" and "parseCurve" not in helpers_added:
                # helpers are inside class_source for CurveFromEquation
                helpers_added.add("parseCurve")

    if not node_list:
        raise SystemExit("Migration produced 0 nodes; refusing to overwrite output.")

    out.append(f"\nNODE_LIST = [\n    " + ",\n    ".join(node_list) + ",\n]\n")
    out.append(FOOTER)

    result = "".join(out)
    for wrong, right in (("io.VAE", "io.Vae"), ("io.CLIP", "io.Clip"), ("io.CLip", "io.Clip")):
        result = result.replace(wrong, right)

    OUTPUT.write_text(result, encoding="utf-8")
    print(f"Wrote migrated {OUTPUT} ({len(node_list)} nodes)")


if __name__ == "__main__":
    main()
