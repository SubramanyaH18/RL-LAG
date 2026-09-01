# Clean-corpus swap — what changed vs. the original RL-LAG_train.zip

This package is `RL_-_LAG_train.zip` with the cleaned HotpotQA corpus
(`rl_lag_clean_corpus_dropin.zip`) already applied, using the project's own
`apply_clean_corpus.py`. Nothing in the training/inference code had to
change — `retrieval.py` rebuilds the FAISS index from whatever is in
`corpus/` on first run, so no hardcoded paths or shapes needed touching.

## Files replaced
- `corpus/hotpot_corpus.jsonl` — 87,889 → 87,590 passages (contaminated /
  answer-leaking documents removed)
- `corpus/hotpot_questions.jsonl` — 90,447 → 17,388 questions (de-duped /
  leaking questions removed)
- `corpus/hotpot_train.jsonl`, `corpus/hotpot_val.jsonl`, `corpus/knowledge.txt`
  — byte-identical to the originals (the drop-in's own notes say these were
  already clean, confirmed by diff)

## Files removed (stale / would silently mislead you on the new corpus)
- `vector_db/knowledge.faiss`, `vector_db/metadata.json` — built from the
  old corpus; `retrieval.py`'s `_load_or_build()` regenerates both
  automatically the first time you run `app.py` or `train_ppo.py`
- `checkpoints/checkpoint_step_*.pt` — PPO policy weights learned against
  retrieval behavior from the old (contaminated, different-sized) corpus;
  per the drop-in script's own warning these don't transfer, kept
  `checkpoints/README.txt` only
- `demo_cache.json` — cached LLM responses keyed to the old corpus's
  retrieved context; would just be dead weight
- `results.json` — stale eval numbers (EM/F1) from a run against the old
  corpus
- `.git/`, `__pycache__/`, and the transient `backup_*/` folder the apply
  script creates (useful when you run it locally, not needed in a
  redistributed zip)

## Next steps on your machine
```bash
pip install -r requirements.txt
ollama pull qwen2.5:3b-instruct   # swapped from llama3.2:3b — see below
python train_ppo.py --steps 200 --no-resume
# or: streamlit run app.py
```
First run re-embeds ~87.6k passages into `vector_db/` — slow on CPU, fine
on Colab/GPU (see `colab_train.ipynb`).
