"""Print checkpoint inventory and reward summary."""
import torch
from pathlib import Path

ckpt_dir = Path('checkpoints')
ckpts = sorted(ckpt_dir.glob('checkpoint_step_*.pt'))
print(f'=== Checkpoint inventory ({len(ckpts)} files) ===')
for c in ckpts:
    data = torch.load(c, map_location='cpu', weights_only=False)
    hist = data.get('reward_history', [])
    last_reward = hist[-1]['reward'] if hist else 'N/A'
    print(f'  {c.name}  step={data["step"]}  entries={len(hist)}  last_reward={last_reward}')

if not ckpts:
    print('No checkpoints found.')
    exit()

latest = ckpts[-1]
data = torch.load(latest, map_location='cpu', weights_only=False)
hist = data.get('reward_history', [])
rewards = [h['reward'] for h in hist]

if rewards:
    best_idx = rewards.index(max(rewards))
    print(f'\n=== Reward summary across {len(rewards)} logged steps ===')
    print(f'  Mean : {sum(rewards)/len(rewards):.4f}')
    print(f'  Best : {max(rewards):.4f}  (step {hist[best_idx]["step"]})')
    print(f'  Last : {rewards[-1]:.4f}')
    print(f'  Min  : {min(rewards):.4f}')

print(f'\n=== Next resume command ===')
print(f'  python train_ppo.py --steps 100 --checkpoint-every 50')
print(f'  (will auto-resume from step {data["step"]})')
