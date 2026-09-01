"""
Show how many unique questions the PPO training drew from,
and compute accuracy on the training question pool using results.json.
"""
import json
from pathlib import Path

# Training draws from hotpot_questions.jsonl (full pool = 17,388)
# Each of 200 steps samples one question at random from the pool.
# With 200 draws from 17,388 questions, ~199 are unique (collision prob ~1.1%)

pool_path  = Path('corpus/hotpot_questions.jsonl')
train_path = Path('corpus/hotpot_train.jsonl')
results_path = Path('results.json')

pool  = [json.loads(l) for l in pool_path.read_text(encoding='utf-8').splitlines() if l.strip()]
train = [json.loads(l) for l in train_path.read_text(encoding='utf-8').splitlines() if l.strip()]

print(f"=== Training Question Pool ===")
print(f"  Full question pool (hotpot_questions.jsonl) : {len(pool):,} questions")
print(f"  Training subset   (hotpot_train.jsonl)      : {len(train):,} questions")
print(f"  PPO steps run                               : 200")
print(f"  Questions sampled per step                  : 1 (random, with replacement)")
print(f"  Expected unique questions sampled           : ~{200 - int(200**2 / (2*len(pool)))} of {len(pool):,}")

# Count types in pool
from collections import Counter
pool_types  = Counter(q.get('type','?') for q in pool)
train_types = Counter(q.get('type','?') for q in train)
pool_levels = Counter(q.get('level','?') for q in pool)

print(f"\n=== Pool composition ===")
for t, c in sorted(pool_types.items()):
    print(f"  Type '{t}': {c:,} ({100*c/len(pool):.1f}%)")
print()
for lv, c in sorted(pool_levels.items()):
    print(f"  Level '{lv}': {c:,} ({100*c/len(pool):.1f}%)")

# Eval results cover first 100 questions of hotpot_questions.jsonl
# Check how many of those 100 overlap with hotpot_train.jsonl
if results_path.exists():
    results = json.loads(results_path.read_text(encoding='utf-8'))
    eval_qs = {q['question'] for q in results['per_question']}
    train_qs = {q['question'] for q in train}
    overlap = eval_qs & train_qs
    print(f"=== Eval vs Training subset overlap ===")
    print(f"  Eval questions (first 100 of pool)  : {len(eval_qs)}")
    print(f"  Training subset questions           : {len(train_qs)}")
    print(f"  Overlap (same questions in both)    : {len(overlap)}")

    # Accuracy on overlapping questions (train subset questions that appeared in eval)
    if overlap:
        print(f"\n=== PPO accuracy on questions also in training subset ===")
        overlap_results = [q for q in results['per_question'] if q['question'] in overlap]
        for cond in ['zero_shot_baseline','random_init_policy','rule_based_pipeline','ppo_trained_policy']:
            ems = [q[cond]['em'] for q in overlap_results if cond in q]
            f1s = [q[cond]['f1'] for q in overlap_results if cond in q]
            if ems:
                print(f"  {cond:<24} EM={sum(ems)/len(ems):.4f}  F1={sum(f1s)/len(f1s):.4f}  (n={len(ems)})")
    else:
        print("  No overlap — eval and training subsets are disjoint.")

print(f"\n=== Summary of training correctness from checkpoint ===")
import torch
ckpts = sorted(Path('checkpoints').glob('checkpoint_step_*.pt'))
d = torch.load(ckpts[-1], map_location='cpu', weights_only=False)
hist = d['reward_history']
cors = [h.get('components',{}).get('correctness',0.0) for h in hist]
nonzero = [c for c in cors if c > 0]
print(f"  Total training rollouts   : {len(cors)}")
print(f"  Rollouts correctness > 0  : {len(nonzero)} ({100*len(nonzero)/len(cors):.1f}%)")
print(f"  Rollouts correctness = 1  : {sum(1 for c in cors if c == 1.0)} (exact match during training)")
print(f"  Mean correctness          : {sum(cors)/len(cors):.4f}")
print(f"  Max correctness           : {max(cors):.4f}")
