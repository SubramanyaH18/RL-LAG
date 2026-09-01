"""Print per-question eval breakdown from results.json."""
import json
from pathlib import Path
from collections import defaultdict

data = json.loads(Path('results.json').read_text(encoding='utf-8'))
summary = data['summary']
per_q = data['per_question']

print("=== Summary ===")
for cond, m in summary.items():
    print(f"  {cond:<24}  EM={m['em']:.4f}  F1={m['f1']:.4f}  (n={m['n']})")

# Per question-type breakdown
print("\n=== By question type ===")
type_stats = defaultdict(lambda: defaultdict(lambda: {'em': 0.0, 'f1': 0.0, 'n': 0}))
for q in per_q:
    qtype = q.get('type', 'unknown')
    for cond in summary:
        if cond in q:
            type_stats[qtype][cond]['em'] += q[cond]['em']
            type_stats[qtype][cond]['f1'] += q[cond]['f1']
            type_stats[qtype][cond]['n'] += 1

for qtype, conds in sorted(type_stats.items()):
    print(f"\n  Type: {qtype}")
    for cond, m in conds.items():
        n = m['n']
        print(f"    {cond:<24}  EM={m['em']/n:.4f}  F1={m['f1']/n:.4f}  (n={n})")

# Best individual question performances
print("\n=== PPO best questions (top 5 by F1) ===")
ppo_results = [(q['question'][:70], q.get('ppo_trained_policy', {}).get('f1', 0),
                q.get('ppo_trained_policy', {}).get('prediction', '')[:50],
                q.get('gold', '')) for q in per_q]
ppo_results.sort(key=lambda x: x[1], reverse=True)
for q, f1, pred, gold in ppo_results[:5]:
    print(f"  F1={f1:.2f}  Q: {q}")
    print(f"         Gold: {gold}  Pred: {pred}")

print("\n=== Metadata ===")
for k, v in data.get('metadata', {}).items():
    print(f"  {k}: {v}")
