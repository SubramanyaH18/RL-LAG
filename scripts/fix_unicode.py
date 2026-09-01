"""Fix all print()-level cp1252 crashes by patching the few offending lines.
Only touches lines that write non-ASCII to stdout at runtime; docstrings/comments are safe.
"""
import re

FIXES = {
    # (file, old_fragment, new_fragment)
    # policies.py
    'policies.py': [
        (
            'print(f"[checkpoint] resumed from step {step} \u2190 {path}")',
            'print(f"[checkpoint] resumed from step {step} <- {path}")',
        ),
    ],
    # train_ppo.py  (already fixed the arrow in banner; one more)
    'train_ppo.py': [
        (
            'print(f"[train_ppo] Checkpoint dir overridden \u2192 {args.checkpoint_dir}")',
            'print(f"[train_ppo] Checkpoint dir overridden -> {args.checkpoint_dir}")',
        ),
    ],
    # llm_client.py
    'llm_client.py': [
        (
            'print(f"[ollama] client (re)created \u2192 {host}")',
            'print(f"[ollama] client (re)created -> {host}")',
        ),
    ],
    # eval.py  (summary separator lines printed to console)
    'eval.py': [
        (
            'print("\\n\u2500\u2500\u2500 Summary \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")',
            'print("\\n--- Summary ------------------------------------")',
        ),
        (
            'print("\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")',
            'print("------------------------------------------")',
        ),
    ],
}

import sys
total = 0
for fname, patches in FIXES.items():
    try:
        with open(fname, encoding='utf-8') as f:
            text = f.read()
        original = text
        for old, new in patches:
            if old in text:
                text = text.replace(old, new)
                total += 1
                sys.stdout.buffer.write(f"  FIXED {fname}: ...{new[:60]}\n".encode('utf-8'))
            else:
                sys.stdout.buffer.write(f"  SKIP  {fname}: pattern not found: {old[:60]}\n".encode('utf-8'))
        if text != original:
            with open(fname, 'w', encoding='utf-8') as f:
                f.write(text)
    except FileNotFoundError:
        sys.stdout.buffer.write(f"  MISSING: {fname}\n".encode('utf-8'))

sys.stdout.buffer.write(f"\nTotal fixes applied: {total}\n".encode('utf-8'))
