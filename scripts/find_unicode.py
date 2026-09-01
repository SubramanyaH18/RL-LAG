"""Scan all .py files for non-ASCII chars that would crash cp1252 Windows console.
Writes results to scripts/unicode_hits.txt in UTF-8.
"""
import os
import sys

PY_FILES = [
    'policies.py', 'reward.py', 'rollout.py', 'train_ppo.py',
    'retrieval.py', 'eval.py', 'llm_client.py', 'solver.py',
    'decomposition.py', 'graph_builder.py', 'pipeline.py', 'app.py',
]

out = open('scripts/unicode_hits.txt', 'w', encoding='utf-8')

for fname in PY_FILES:
    if not os.path.exists(fname):
        continue
    with open(fname, encoding='utf-8') as f:
        lines = f.readlines()
    hits = []
    for i, line in enumerate(lines, 1):
        for ch in line:
            if ord(ch) > 127:
                try:
                    ch.encode('cp1252')
                except UnicodeEncodeError:
                    hits.append((i, repr(ch), line.rstrip()))
                    break
    if hits:
        out.write(f"\n{fname}:\n")
        for lineno, char, content in hits:
            out.write(f"  L{lineno}: {char}  ->  {content[:100]}\n")
    else:
        out.write(f"{fname}: OK\n")

out.close()
print("Done - results written to scripts/unicode_hits.txt")
