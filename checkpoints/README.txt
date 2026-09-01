PPO checkpoint directory
========================

Checkpoint files (checkpoint_step_NNNN.pt) are excluded from git via .gitignore
because they can be large (>100 MB for all three policy networks).

On Colab/Kaggle, override the checkpoint path with:

    python train_ppo.py --checkpoint-dir /content/drive/MyDrive/rl-lag/checkpoints

To resume from the latest checkpoint (default behaviour):

    python train_ppo.py --steps 200

To start fresh (ignore any existing checkpoint):

    python train_ppo.py --steps 200 --no-resume

CPU smoke-test (no Ollama needed):

    python train_ppo.py --steps 5 --checkpoint-every 5 --no-resume --dry-run
