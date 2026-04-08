## Live Eval Dataset

This dataset is for practical evaluation, not training.

Purpose:

- store GT-tagged real-time run captures,
- replay the exact same sequences on multiple models,
- compare models on fixed live-like inputs.

Workflow:

1. Run `gpt_run.py` or `nsu_run_fusion_v6_overlap_hold.py`
2. Set GT using `N`, `B`, `G`
3. Perform the target sign and let the script save the capture
4. Build the dataset:

```powershell
& C:\SignProject\myenv\Scripts\python.exe C:\SignProject\build_live_eval_dataset.py --copy
```

5. Evaluate a model on the saved live-like dataset:

```powershell
& C:\SignProject\myenv\Scripts\python.exe C:\SignProject\evaluate_live_eval_dataset.py --model-dir "C:\SignProject\experiments\glove_fusion\2026-04-07_stable_hybrid_18w_clean_v1"
```

Recommended use:

- do not mix this dataset into training,
- use it as a fixed practical benchmark,
- compare vision-only and fusion on exactly the same captures.
