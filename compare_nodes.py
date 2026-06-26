import re
from pathlib import Path

ROOT = Path("S:/Library/Projects/comfyui-sv-nodes-v3")

def node_ids_from_v1(text: str) -> set[str]:
    active = set(re.findall(r'^NODE_CLASS_MAPPINGS\["([^"]+)"\]', text, re.M))
    active |= set(re.findall(r'^BORROWED_CLASS_MAPPINGS\["([^"]+)"\]', text, re.M))
    return active

backup = (ROOT / "__init__v1_backup.py").read_text(encoding="utf-8")
borrowed = (ROOT / "borrowed.py").read_text(encoding="utf-8")
migrated = (ROOT / "__init__.py").read_text(encoding="utf-8")

v1_all = node_ids_from_v1(backup) | node_ids_from_v1(borrowed)
v3 = set(re.findall(r"node_id='([^']+)'", migrated))

print("origin v1 nodes:", len(v1_all))
print("migrated:", len(v3))
print("missing:", sorted(v1_all - v3))
print("extra:", sorted(v3 - v1_all))
