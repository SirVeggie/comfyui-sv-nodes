#!/usr/bin/env python3
"""Post-process migrated V3 __init__.py to fix systematic conversion issues."""

import re
from pathlib import Path

PATH = Path(__file__).resolve().parent / "__init__.py"

EXEC_NAMES = (
    "run", "patch", "get_imagesize", "for_loop_open", "for_loop_close",
    "while_loop_open", "while_loop_close", "int_math_operation",
    "to_bool", "accumulate", "accumulation_head", "accumulation_tail",
    "accumulation_to_list", "list_to_accumulation", "accumlength",
    "get_item", "set_item",
)


def convert_exec_methods(text: str) -> str:
    for name in EXEC_NAMES:
        text = re.sub(
            rf'^(\s*)def {name}\(self\)\s*:',
            r'\1@classmethod\n\1def execute(cls):',
            text,
            flags=re.MULTILINE,
        )
        text = re.sub(
            rf'^(\s*)def {name}\(self(?:,|\))',
            r'\1@classmethod\n\1def execute(cls,',
            text,
            flags=re.MULTILINE,
        )
        text = re.sub(
            rf'^(\s*)def {name}\(s(?:,|\))',
            r'\1@classmethod\n\1def execute(cls,',
            text,
            flags=re.MULTILINE,
        )
    text = text.replace('def execute(cls,:', 'def execute(cls)')
    return text


def add_execute_return_hint(text: str) -> str:
    return re.sub(
        r'def execute\(cls,([^)]*)\):',
        r'def execute(cls,\1) -> io.NodeOutput:',
        text,
    )


def fix_returns(text: str) -> str:
    text = re.sub(
        r'return\s+\{\s*"result"\s*:\s*([^,]+),\s*"expand"\s*:\s*([^}]+)\}',
        r'return io.NodeOutput(\1, expand=\2)',
        text,
    )
    text = re.sub(r'return\s+\{\s*\}', 'return io.NodeOutput()', text)

    def repl(m):
        indent, inner = m.group(1), m.group(2).strip()
        if inner.startswith("io.NodeOutput"):
            return m.group(0)
        if not inner:
            return f"{indent}return io.NodeOutput()"
        return f"{indent}return io.NodeOutput({inner})"

    text = re.sub(r'^(\s*)return \(([^;\n]+)\)\s*$', repl, text, flags=re.MULTILINE)
    text = text.replace('return tuple(values)', 'return io.NodeOutput(*values)')
    return text


def fix_fingerprint(text: str) -> str:
    text = re.sub(r'^\s*def IS_CACHED\(', '    @classmethod\n    def fingerprint_inputs(', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*def IS_CHANGED\(', '    @classmethod\n    def fingerprint_inputs(', text, flags=re.MULTILINE)
    text = text.replace('def fingerprint_inputs(s,', 'def fingerprint_inputs(cls,')
    text = text.replace('def fingerprint_inputs(self,', 'def fingerprint_inputs(cls,')
    return text


def fix_check_lazy(text: str) -> str:
    text = re.sub(r'(\n    )def check_lazy_status\(self,', r'\1@classmethod\n\1def check_lazy_status(cls,', text)
    return text


def fix_resolution_selector_call(text: str) -> str:
    return text.replace(
        'ResolutionSelector.run(self, base, ratio, orientation, seed, random)',
        'ResolutionSelector.execute(base, ratio, orientation, seed=seed, random=random)',
    )


def fix_while_loop_close(text: str) -> str:
    if 'class WhileLoopClose' not in text:
        return text
    text = text.replace(
        'def while_loop_close(self, flow_control, condition, dynprompt=None, unique_id=None, **kwargs):',
        '@classmethod\n    def execute(cls, flow_control, condition, **kwargs) -> io.NodeOutput:',
    )
    text = text.replace(
        'def execute(cls, flow_control, condition, dynprompt=None, unique_id=None, **kwargs):',
        'def execute(cls, flow_control, condition, **kwargs) -> io.NodeOutput:',
    )
    text = text.replace(
        'def execute(cls, flow_control, condition, =None, unique_id=None, **kwargs):',
        'def execute(cls, flow_control, condition, **kwargs) -> io.NodeOutput:',
    )
    text = text.replace('dynprompt.get_node(unique_id)', 'cls.hidden.dynprompt.get_node(cls.hidden.unique_id)')
    text = text.replace(
        'original_node = dynprompt.get_node(node_id)',
        'original_node = cls.hidden.dynprompt.get_node(node_id)',
    )
    text = text.replace(
        'self.explore_dependencies(unique_id, dynprompt',
        'cls.explore_dependencies(cls.hidden.unique_id, cls.hidden.dynprompt',
    )
    text = text.replace('self.collect_contained(', 'cls.collect_contained(')
    text = text.replace('contained[unique_id]', 'contained[cls.hidden.unique_id]')
    text = re.sub(r'def explore_dependencies\(self,', '@classmethod\n    def explore_dependencies(cls,', text)
    text = re.sub(r'def collect_contained\(self,', '@classmethod\n    def collect_contained(cls,', text)
    return text


def fix_prompt_encode(text: str) -> str:
    if 'class PromptProcessingEncode' not in text:
        return text
    for name in ('cacheHas', 'cacheSet', 'cacheGet', 'cacheClean'):
        text = re.sub(rf'def {name}\(self,', f'@classmethod\n    def {name}(cls,', text)
        text = text.replace(f'self.{name}(', f'cls.{name}(')
    text = text.replace('return pconds, nconds', 'return io.NodeOutput(pconds, nconds)')
    return text


def fix_compress_conds(text: str) -> str:
    if 'class CompressConds' not in text:
        return text
    text = re.sub(r'def compress\(self,', '@classmethod\n    def compress(cls,', text)
    text = re.sub(r'def reduc\(self,', '@classmethod\n    def reduc(cls,', text)
    text = text.replace('self.compress(', 'cls.compress(')
    text = text.replace('self.reduc(', 'cls.reduc(')
    text = text.replace('return pos, neg', 'return io.NodeOutput(pos, neg)')
    return text


def fix_cache_nodes(text: str) -> str:
    for name in ('CacheObject', 'ManualCache'):
        if f'class {name}' not in text:
            continue
        if f'{name}._objs' not in text:
            text = text.replace(
                f'class {name}(io.ComfyNode):',
                f'class {name}(io.ComfyNode):\n    _objs: dict = {{}}',
                1,
            )
        text = text.replace('self.obj', f'{name}._objs.get(cls.hidden.unique_id)')
        text = text.replace(
            f'{name}._objs.get(cls.hidden.unique_id) = ',
            f'{name}._objs[cls.hidden.unique_id] = ',
        )
    if 'class ClearCustomCaches' in text:
        text = text.replace('if self.id is None:', 'if ClearCustomCaches._id is None:')
        text = text.replace('self.id = round', 'ClearCustomCaches._id = round')
        if 'ClearCustomCaches._id' in text and '_id = None' not in text.split('class ClearCustomCaches')[1][:120]:
            text = text.replace(
                'class ClearCustomCaches(io.ComfyNode):',
                'class ClearCustomCaches(io.ComfyNode):\n    _id = None',
                1,
            )
    return text


def fix_io_type_names(text: str) -> str:
    for wrong, right in (("io.VAE", "io.Vae"), ("io.CLIP", "io.Clip")):
        text = text.replace(wrong, right)
    return text


def fix_v1_mapping_lines(text: str) -> str:
    text = re.sub(r'^NODE_CLASS_MAPPINGS\[.*\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'^NODE_DISPLAY_NAME_MAPPINGS\[.*\n', '', text, flags=re.MULTILINE)
    return text


def fix_combo_input_kwargs(text: str) -> str:
    text = re.sub(
        r'(io\.Combo\.Input\([^)]*),\s*force_input=(?:True|False)',
        r'\1',
        text,
    )
    return text


def fix_matchtype_input_kwargs(text: str) -> str:
    text = re.sub(
        r'(io\.MatchType\.Input\([^)]*),\s*default=(?:-?\d+(?:\.\d+)?|\'[^\']*\'|"[^"]*")',
        r'\1',
        text,
    )
    text = re.sub(
        r'(io\.MatchType\.Input\([^)]*),\s*force_input=(?:True|False)',
        r'\1',
        text,
    )
    return text


def fix_double_classmethod(text: str) -> str:
    return re.sub(r'(@classmethod\n\s*){2,}', '@classmethod\n    ', text)


def fix_empty_classmethod_lines(text: str) -> str:
    return re.sub(r'\n    @classmethod\n    \n    def ', '\n    @classmethod\n    def ', text)


def dedupe_blank_sections(text: str) -> str:
    return re.sub(r'(#-{10,}#\n\n){2,}', r'\1', text)


def cleanup_orphan_classmethods(text: str) -> str:
    text = re.sub(r'\n    @classmethod\n\n    def ', '\n    @classmethod\n    def ', text)
    text = re.sub(r'\n    @classmethod\n\n    def ', '\n    @classmethod\n    def ', text)
    text = re.sub(r'\n    @classmethod\n\n\n    def ', '\n    @classmethod\n    def ', text)
    return text


def fix_while_loop_unique_id(text: str) -> str:
    return text.replace('node_id == unique_id', 'node_id == cls.hidden.unique_id')


def dedupe_encode(text: str) -> str:
    parts = text.split('def encode(clip, text):')
    if len(parts) <= 2:
        return text
    first = parts[0]
    last = parts[-1]
    return first + 'def encode(clip, text):' + last


def fix_schema_flags(text: str) -> str:
    text = re.sub(
        r'(\])\n(\s+)(hidden|is_output_node|is_deprecated|not_idempotent|enable_expand)=',
        r'\1,\n\2\3=',
        text,
    )
    text = re.sub(r'\],,', '],', text)
    return text


def fix_value_repeater(text: str) -> str:
    if 'class ValueRepeater' not in text:
        return text
    text = text.replace(
        'class EmptyValue:\nclass ValueRepeater',
        'class EmptyValue:\n    pass\n\nclass ValueRepeater',
    )
    text = re.sub(
        r'(class ValueRepeater\(io\.ComfyNode\):.*?)(\n    _objs: dict = \{\})',
        r'\1',
        text,
        count=1,
        flags=re.DOTALL,
    )
    text = re.sub(r'^\s*savedValue\s*=.*\n', '', text, flags=re.MULTILINE)
    return text


def fix_borrowed_artifacts(text: str) -> str:
    text = re.sub(
        r'^BORROWED_(?:CLASS|DISPLAY)_MAPPINGS\[.*$',
        '',
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(r'\n    @classmethod\n\n    def execute', '\n    @classmethod\n    def execute', text)
    return text


def main():
    text = PATH.read_text(encoding='utf-8')
    text = convert_exec_methods(text)
    text = add_execute_return_hint(text)
    text = fix_returns(text)
    text = fix_fingerprint(text)
    text = fix_check_lazy(text)
    text = fix_resolution_selector_call(text)
    text = fix_while_loop_close(text)
    text = fix_while_loop_unique_id(text)
    text = fix_prompt_encode(text)
    text = fix_compress_conds(text)
    text = fix_cache_nodes(text)
    text = fix_value_repeater(text)
    text = fix_borrowed_artifacts(text)
    text = fix_v1_mapping_lines(text)
    text = fix_schema_flags(text)
    text = fix_io_type_names(text)
    text = fix_combo_input_kwargs(text)
    text = fix_matchtype_input_kwargs(text)
    text = fix_double_classmethod(text)
    text = fix_empty_classmethod_lines(text)
    text = cleanup_orphan_classmethods(text)
    text = dedupe_encode(text)
    text = dedupe_blank_sections(text)
    PATH.write_text(text, encoding='utf-8')
    print(f"Fixed {PATH}")


if __name__ == '__main__':
    main()
