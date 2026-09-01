"""
Analyze PPO training questions and their correctness scores.
Shows which questions were sampled, their gold answers, predicted answers,
and the correctness (EM/F1) component from the reward at each step.
"""
import json, torch
from pathlib import Path
from collections import Counter

# Load latest checkpoint
ckpt_dir = Path('checkpoints')
ckpts = sorted(ckpt_dir.glob('checkpoint_step_*.pt'))
latest = ckpts[-1]
data = torch.load(latest, map_location='cpu', weights_only=False)
hist = data.get('reward_history', [])

print(f"=== PPO Training Summary ===")
print(f"Latest checkpoint : {latest.name}")
print(f"Total steps logged: {len(hist)}")

# Extract per-step info
questions_seen = []
for h in hist:
    questions_seen.append({
        'step':        h.get('step', '?'),
        'question':    h.get('question', ''),
        'gold':        h.get('gold_answer', ''),
        'prediction':  h.get('final_answer', ''),
        'reward':      h.get('reward', 0.0),
        'correctness': h.get('components', {}).get('correctness', 0.0),
        'em':          1.0 if h.get('components', {}).get('correctness', 0.0) == 1.0 else 0.0,
    })

# Check what fields are actually stored
sample_keys = list(hist[0].keys()) if hist else []
print(f"Keys in reward_history entries: {sample_keys}\n")

# If question/gold not stored in history, count unique steps
total_steps = len(hist)
rewards     = [h['reward'] for h in hist]
cors        = [h.get('components', {}).get('correctness', 0.0) for h in hist]
nonzero_cor = [c for c in cors if c > 0.0]

print(f"=== Correctness (EM+F1 blend) across {total_steps} training steps ===")
print(f"  Steps with correctness > 0  : {len(nonzero_cor)} / {total_steps} ({100*len(nonzero_cor)/total_steps:.1f}%)")
print(f"  Mean correctness (all steps): {sum(cors)/len(cors):.4f}")
print(f"  Mean correctness (>0 only)  : {sum(nonzero_cor)/len(nonzero_cor):.4f}" if nonzero_cor else "  No non-zero correctness scores")
print(f"  Max correctness in training : {max(cors):.4f}")

# Reward breakdown
print(f"\n=== Reward breakdown across {total_steps} steps ===")
print(f"  Mean total reward  : {sum(rewards)/len(rewards):.4f}")
print(f"  Best reward        : {max(rewards):.4f}  (step {hist[rewards.index(max(rewards))].get('step','?')})")
print(f"  Min reward         : {min(rewards):.4f}")

# Component averages
comp_keys = ['correctness','retrieval_presence','token_efficiency','logical_consistency','grounding']
print(f"\n=== Mean component scores across all steps ===")
for k in comp_keys:
    vals = [h.get('components', {}).get(k, 0.0) for h in hist]
    print(f"  {k:<24} : {sum(vals)/len(vals):.4f}")

# Reward over time (10-step buckets)
print(f"\n=== Reward trend (20-step buckets) ===")
bucket = 20
for i in range(0, len(hist), bucket):
    chunk = rewards[i:i+bucket]
    cor_chunk = cors[i:i+bucket]
    print(f"  steps {i:>3}-{i+len(chunk)-1:<3} : reward={sum(chunk)/len(chunk):.4f}  correctness={sum(cor_chunk)/len(cor_chunk):.4f}")
