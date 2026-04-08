# Full Context Handoff (2026-04-06)

This file is meant to be the single handoff document for a new chat.
It combines:
- older project history,
- the main conclusions already recorded in earlier handoff files,
- the current merged `C:\SignProject` workspace state,
- and the additional work/debugging done in this chat on 2026-04-06.

If a new chat starts from this file, it should be able to continue without losing project context.

## 1. Project Goal

The goal is not simply:
- "fusion is more accurate than vision."

The better research framing is:
- vision-only sign recognition has limitations when signs are visually similar, partially occluded, or when landmark detection is unstable,
- smart-glove sensors can provide complementary cues,
- and the real question is whether multimodal fusion improves recognition performance and stability, especially for confusing similar-sign groups.

Recommended thesis-style framing:
- vision-based sign recognition is limited by hand occlusion, landmark dropout, and visual overlap among similar signs,
- smart-glove sensors provide finger bend and motion cues but are also affected by drift and wear/calibration variation,
- therefore vision and sensors may complement each other,
- and the study aims to verify whether MediaPipe-based visual features fused with glove sensors improve both recognition performance and practical robustness, especially in similar-sign groups.

Important hypotheses discussed:
- vision-only will confuse similar signs more often,
- sensor-only is limited by itself but can help as an auxiliary cue,
- vision+sensor fusion can reduce intra-group confusion,
- a role-separated, reliability-aware fusion structure is more stable than dumping all sensor channels in directly,
- this last hypothesis is exactly what justifies the current `stable hybrid` candidate.

## 2. Overall Historical Progression

The model development path up to now is:

1. Start with naive fusion:
- vision branch + sensor branch, both fed into an LSTM-based fusion structure.
- This did not give stable gains.

2. Improve raw input quality:
- add hold/freeze logic and zero handling for hand landmark dropout,
- reduce catastrophic failure when landmarks disappear.

3. Add detection-state flags:
- `left_detected`, `right_detected`, `left_held`, `right_held`, `overlap_flag`, `ambiguous_flag`
- to explicitly tell the model when vision is unreliable.

4. Realize that latest fusion was still not consistently better than glove vision-only:
- this led to deeper sensor usefulness analysis.

5. Sensor analysis conclusion:
- IMU sequence information is useful,
- flex raw full sequence can be harmful/noisy,
- flex should not be fed as a raw sequence branch in the same way.

6. New structured direction:
- remove/reduce problematic flex raw usage,
- convert flex into `flex posture` summary features,
- keep vision central,
- use IMU/flags as correction/support information.

7. Friend's gate line:
- friend used a vision-main, sensor-correction gate design,
- this produced strong single-run peaks.

8. Final current candidate:
- combine the strong points of both lines:
  - our role separation (`IMU useful`, `flex raw harmful`, `flex posture summary`),
  - friend's vision-guided gate idea,
- leading to the current `stable hybrid` model.

This is the right way to explain the evolution:
- naive fusion was unstable,
- analysis showed why,
- two improvement directions emerged,
- and `stable hybrid` is the merged result of those lessons.

## 3. Final Comparison Frame Required by Professor

The final comparison frame the professor wants is:

- bare hand + vision
- glove + vision
- glove + fusion

The practical questions are:
- how good is vision when bare-handed,
- how much does glove wearing hurt vision-only performance,
- how much does sensor fusion recover or improve that loss.

Important caution:
- current results are still mostly 8-word pilot / structure-search results,
- so `stable hybrid` must **not** be described as the final confirmed thesis model,
- only as the **current best final candidate**.

Final confirmation still requires:
- full-word final training,
- full-word final run validation,
- and the professor's 3-axis comparison on the final vocabulary.

## 4. Current Workspace State (`C:\SignProject`)

This folder is **not** the old notebook folder as-is anymore.

Current status:
- the folder was originally a long-used notebook local workspace,
- it contained many legacy experiments, results, temporary artifacts, and mixed states,
- then the `lsm-version` branch from GitHub `nowonjong/nsu` (desktop-side latest core code) was brought into:
  - `C:\SignProject\_lsm_version_snapshot`
- and selectively merged into the current workspace.

Meaning:
- `C:\SignProject` should now be treated as the latest practical merged working folder,
- even if `git status` is not perfectly clean.

Important user-confirmed workspace facts:
- core code files are aligned to the latest `lsm-version` versions:
  - `C:\SignProject\nsu_train.py`
  - `C:\SignProject\nsu_run_fusion_v6_overlap_hold.py`
  - `C:\SignProject\nsu_collect.py`
  - `C:\SignProject\gpt_train.py`
  - `C:\SignProject\run_fusion_with_arduino.py`
  - `C:\SignProject\run_fusion_with_arduino.bat`
- old legacy artifacts were moved to:
  - `C:\SignProject\_archive_notebook_legacy_2026-04-05`
- files that differed before merge were backed up to:
  - `C:\SignProject\_merge_backup_pre_lsm_2026-04-05`

## 5. Main Candidate / Baselines / Comparison Models

### Current main candidate

- `fusion_train_output_hold_v4_flags_gate_flexposture_stable`
- Single-run metrics:
  - Test Accuracy `0.9464`
  - Macro F1 `0.9446`

### Important comparison models

- Vision-only latest baseline:
  - `vision_results_from_fusion_hold_v4_flags`
  - `0.9286 / 0.9280`

- Input-structured fusion v2:
  - `fusion_train_output_hold_v4_flags_imu_flags_flexposture_v2`
  - `0.9286 / 0.9260`
  - this is our cleaned structured fusion without the final gate refinement

- Friend raw gate fusion:
  - `gksdydtjr/fusion_train_output_hold_v4_flags_gate_v2`
  - `0.9643 / 0.9625`
  - strongest single-run peak among major compared models

- Friend rewritten pipeline references:
  - `gksdydtjr/vision_only_v1`
  - `gksdydtjr/fusion_v2`

### Repeated-seed conclusion

Canonical repeated-seed result:
- `stable hybrid` (10 seeds, 42-51)
  - mean Acc `0.9225`
  - std Acc `0.0362`
  - mean Macro F1 `0.9170`
  - std Macro F1 `0.0396`
  - min Acc `0.8500`
  - max Acc `0.9821`

Friend alpha 0.3 gate summary:
- mean Acc `0.9022`
- std Acc `0.0494`
- mean Macro F1 `0.8925`
- std Macro F1 `0.0581`

Main interpretation:
- friend gate had stronger peak runs,
- but `stable hybrid` had better average performance and lower spread,
- so `stable hybrid` became the preferred **current best final candidate**.

## 6. Why `stable hybrid` Is the Current Candidate

`stable hybrid` should be described as:
- vision-main,
- IMU/flags correction,
- flex posture auxiliary summary,
- selected because of better mean and stability, not only peak score.

Structure summary:
- vision branch is the main branch,
- `imu_flags` branch provides gate-based correction to vision,
- `flex raw` full sequence is **not** directly used as a raw time-series branch,
- instead, `flex posture v2` (44-dim summary) is used as a separate auxiliary branch.

One-line philosophy:
- vision is the primary source,
- IMU/flags correct or support vision,
- flex is summarized instead of trusted as raw sequence.

## 7. Why It Is Still Not Final

This must be stated clearly:

- `stable hybrid` is **not** the final confirmed project model.
- It is the **current most promising final candidate**.

Why not final yet:
- many comparisons were still 8-word pilot / mid-stage structure-search experiments,
- final full-word training is not done,
- final full-word run validation is not done,
- the final thesis framing requires the 3-axis comparison:
  - bare + vision
  - glove + vision
  - glove + fusion
- final conclusion must come after full-word evaluation.

## 8. Existing Key Result Files

Main summary files:
- `C:\SignProject\results_summary\results_overview_2026-04-04.md`
- `C:\SignProject\results_summary\results_table_2026-04-04.csv`

Older main handoff/history:
- `C:\SignProject\handoff_2026-04-04_full_session_gate_stability.md`
- `C:\SignProject\notebook_sync_handoff_2026-04-05.md`

Important note:
- some older Korean markdown files appear with mojibake/encoding damage in terminal output,
- but their content was already interpreted and summarized into the current working understanding.

## 9. Work Done in This Chat on 2026-04-06

This section is critical because a new chat must know exactly what was added/changed today.

### 9.1 3-axis repeated-seed comparison preparation

The user wanted to compare:
- bare-hand vision-only
- glove vision-only
- glove fusion

We aligned repeated-seed comparisons as follows.

#### Bare-hand vision-only (`vision_only_zero_v1`)

Repeated 10 seeds:
- created outputs:
  - `cmp_seed_bare_vision42` to `cmp_seed_bare_vision51`
- summary:
  - mean Acc `0.9038`
  - std Acc `0.0202`
  - mean Macro F1 `0.9006`
  - std Macro F1 `0.0235`

Worst bare-hand seed used later for presentation:
- `cmp_seed_bare_vision48`
  - Accuracy `0.8625`
  - Macro F1 `0.8502`

#### Glove vision-only

Repeated 10 seeds on the glove dataset using only vision features:
- outputs:
  - `cmp_seed_glove_vision42` to `cmp_seed_glove_vision51`
- summary:
  - mean Acc `0.7863`
  - std Acc `0.0807`
  - mean Macro F1 `0.7606`
  - std Macro F1 `0.1003`

Important clarification:
- this was not "reading stable hybrid output and using only vision,"
- it was retraining on the same glove dataset using vision-only input.

#### Glove fusion

User reminded correctly:
- glove fusion 10-seed result already existed,
- so that should remain the canonical source,
- not be re-done unnecessarily.

Canonical glove fusion summary:
- stable hybrid 10-seed:
  - mean Acc `0.9225`
  - std Acc `0.0362`
  - mean Macro F1 `0.9170`
  - std Macro F1 `0.0396`

#### Important split caution

Bare-hand and glove repeated comparisons are useful directionally, but:
- bare-hand `vision_only_zero_v1` had split `160 / 80 / 80`,
- glove hold-v4-flags experiments usually used `248 / 16 / 56`,
- so bare-vs-glove comparison is **not** a perfectly identical final matched benchmark.

It is still useful as pilot evidence, but should be stated cautiously.

### 9.2 Files added for repeated-seed work

#### `gpt_train.py`

Patched so the seed can be set from environment:
- `NSU_RANDOM_SEED`

This allows multi-seed automation without manual editing.

#### `run_multiseed_comparison.py`

Added helper script for repeated-seed experiments.

Purpose:
- run multiple seeds,
- collect per-run metrics,
- write summaries.

Summary files written:
- `C:\SignProject\results_summary\multiseed_per_run.csv`
- `C:\SignProject\results_summary\multiseed_group_summary.csv`
- `C:\SignProject\results_summary\multiseed_group_summary.json`

### 9.3 Slide/visual generation work

Several temporary presentation PNG/MD files were created during iteration.
User later asked to delete the temporary versions and keep only the final small set.

Current remaining slide-related files in `results_summary` after cleanup:
- `slide1_bare_vision_worst_seed48_en_2026-04-06.png`
- `slide2_stable_hybrid_mean_en_2026-04-06.png`
- `slide_metrics_final_en_2026-04-06.md`

User also decided confusion-matrix use should be minimal:
- bare-hand vision confusion matrix
- stable hybrid confusion matrix

### 9.4 Collect vs run alignment investigation

We checked whether current `nsu_collect.py` is the same line as the one that generated the stable hybrid dataset.

Result:
- current `nsu_collect.py` hash matches `_lsm_version_snapshot/nsu_collect.py`,
- it differs from `_merge_backup_pre_lsm_2026-04-05/nsu_collect.py`,
- dataset metadata in `dataset_fusion_hold_v4_flags` matches current collect behavior.

Practical conclusion:
- current `nsu_collect.py` is effectively the same collect-code family that produced the stable-hybrid dataset.

### 9.5 Important collect vs run differences found

Before patching, we found run and collect were not fully aligned.

Main mismatches found:
- collect used handedness score as a side hint with threshold `0.55`, but did not hard-drop low-score candidates,
- run hard-dropped candidates below `0.70`,
- collect aligned sensor packets to frame host time,
- run had been taking only the latest packet,
- collect's `held` flag behavior effectively reflected overlap-freeze semantics more than missing-freeze,
- run had been reflecting missing-freeze in the held flag.

### 9.6 `nsu_run_fusion_v6_overlap_hold.py` patches applied

Run was patched to be closer to collect and to the stable hybrid training assumptions.

Changes made:
- default model directory changed to:
  - `fusion_cmp_seed45_hybrid_stable`
- handedness threshold changed to `0.55`
- candidate filtering changed so low-score handedness no longer hard-drops hands; it now behaves more like collect and uses side hints
- added frame-time-aligned sensor packet retrieval:
  - `get_aligned_packet(frame_host_time_ms)`
- changed runtime to use aligned sensor packet instead of simply latest packet
- aligned `held` flag semantics more closely to collect, using overlap-freeze behavior for model flags
- fixed a bug after sensor API changes where `show_wait_sensor_screen()` still unpacked 3 values instead of 4 from `get_latest()`

Current practical default run model:
- `fusion_cmp_seed45_hybrid_stable`

### 9.7 Arduino work done

User asked whether Arduino code was running.

Actions completed:
- Arduino CLI index update
- install `arduino:avr` core
- compile sensor sender sketch
- upload to `COM9`

Result:
- Arduino sensor-send code is uploaded and available on `COM9`.

### 9.8 Real-time run debugging added

Because user found real-time performance disappointing, we added debug capture saving to the run script.

New behavior:
- every time a 60-frame sequence is classified,
- the exact sequence the model saw is saved,
- along with prediction/confidence/margin and zero-frame metadata.

Added in:
- `C:\SignProject\nsu_run_fusion_v6_overlap_hold.py`

Saved to:
- `C:\SignProject\run_debug_captures`

Each capture produces:
- `*.npy` : the 60x158 raw runtime sequence
- `*.json`: metadata including:
  - predicted label,
  - confidence,
  - margin,
  - motion,
  - hand_count,
  - zero-frame counts,
  - sensor timing delta.

This was added specifically so future chats can compare real-time inputs directly against training data instead of guessing.

## 10. Real-Time Performance Discussion in This Chat

This is important because the new chat must not misstate what has or has not been proven.

### 10.1 User-reported real-time issue

User reported:
- when doing `gamsahamnida`, the result often came out as `sada`,
- `billida` also felt bad / was not appearing reliably in actual use,
- despite offline 9x% numbers.

### 10.2 What was first suspected

At first, possible causes discussed included:
- no `none` class in labels,
- lack of reject logic,
- domain gap between curated collect data and real-time input,
- timing/window mismatch.

Important correction:
- user rightly pointed out that `none` absence was not itself a contradiction, because for offline word-level confusion analysis, having no `none` class was actually useful and intentional.
- The distinction that must be preserved:
  - no `none` is fine for offline forced word classification and confusion analysis,
  - reject/none-like behavior can still be useful for a real-time demo system.

### 10.3 What was actually verified

We then checked the evidence more carefully.

Verified facts:
- label order is not broken:
  - `labels.json` and runtime loading are consistent
- stable hybrid seed45 is not a model that inherently fails to learn `gamsahamnida` or `billida`
  - offline test for seed45:
    - `gamsahamnida` recall `1.0000`
    - `billida` recall `1.0000`
- across multiple stable hybrid seeds:
  - `billida` remains generally strong,
  - `gamsahamnida` can vary a little by seed, but it is not a hopeless class.

So:
- this is not explained by a simple class-order bug,
- and not explained by "the model never learned these classes."

### 10.4 What the saved real-time captures showed

Saved real-time debug captures currently include:
- `gamsahamnida` 1 capture
- `sada` 2 captures
- `billida` several captures
- `banggeum` 1 capture

Key observations from the saved captures:

#### `gamsahamnida`

Saved file:
- `run_debug_captures\20260406_190712_921296_gamsahamnida.json`

Observed:
- it was predicted as `gamsahamnida`, not `sada`
- but confidence was only `0.541`
- margin was only `0.093`
- zero-frame counts were `0`

Interpretation:
- the runtime `gamsahamnida` example that was captured was correctly classified,
- but it is near a boundary and not very stable.

#### `billida`

Multiple captured `billida` samples were all predicted as `billida`.
Some had:
- right-hand missing frames,
- e.g. 4 to 10 right-missing frames in some captures,
- while still being classified as `billida`.

Motion comparison:
- runtime `billida` motion was lower than training average,
- and runtime had more right-hand missing frames than typical training samples.

#### Training-vs-run differences that were numerically confirmed

For the captured examples we compared against training examples:

- `gamsahamnida`
  - runtime motion ~ `0.8305`
  - training average motion ~ `1.1`

- `billida`
  - runtime motion roughly `0.6959 ~ 0.7832`
  - training average motion ~ `0.9151`
  - runtime had more right-hand missing frames than training average

This means:
- even when the runtime sample looks visually "the same" to a human,
- the actual numeric sequence the model sees can be smaller-motion and noisier than training samples.

### 10.5 Most careful current interpretation

The strongest evidence-based interpretation at the end of this chat is:

- there is **not yet proof** of a structural model-input mismatch,
- there is **not yet proof** of a class mapping bug,
- there **is** evidence that real-time inputs can differ from training inputs in motion size and landmark stability,
- and at least one captured `gamsahamnida` example was low-margin / unstable even when correct.

So the honest conclusion is:
- the current bottleneck is more likely in real-time input quality / timing / sequence characteristics than in a simple broken model wiring bug,
- but more failed real-time captures are still needed before making stronger claims.

## 11. Important Clarification About Runtime Input Structure

This point must be preserved exactly, because the user explicitly challenged sloppy explanations here.

Question asked:
- does current run actually match the stable-hybrid runtime structure,
- meaning vision + IMU-based gated branch + flex posture auxiliary, rather than something structurally different?

Answer:
- yes, current run does match the stable hybrid structure.

More precisely:
- runtime initially builds per-frame `158`-dim raw stream:
  - `126` vision
  - `26` raw sensor
  - `6` flags
- but the model does **not** consume all 26 raw sensor channels directly,
- instead, runtime reads `config.json` and uses:
  - `18` selected raw sensor indices (IMU-centered selection),
  - `6` flags,
  - plus flex posture extracted from the raw flex channels into a separate `44`-dim auxiliary representation.

So current run is aligned with stable hybrid logic:
- vision main branch
- IMU/flags gate branch
- flex posture auxiliary branch

The user correctly pointed out that saying "run just uses raw sensor directly in the final same way" is too sloppy.
The precise statement is:
- run stores a `158`-dim raw sequence,
- then selects/configures the IMU-centered sensor branch and flex posture branch according to the model config.

## 12. Can Saved Runtime Captures Be Reused for Training?

This was also discussed carefully.

Correct interpretation:
- saved runtime captures are **not** the final 150-dim model input,
- they are the raw `158`-dim sequence stream.

That means:
- they cannot be described as "already-final model input features,"
- but they **can** be reused as raw dataset samples for retraining,
- because the train pipeline also starts from the same raw sequence representation and then:
  - selects sensor indices,
  - standardizes modalities,
  - extracts flex posture,
  - and feeds the structured model inputs.

Important caution:
- runtime captures should not be auto-labeled by the model prediction,
- they must be human-confirmed and manually assigned to the correct word class,
- otherwise mislabeled noisy data can pollute the dataset.

## 13. Current Recommendation Going Forward

After all discussions in this chat, the realistic view is:

- it is probably **not** worth inventing a radically new train architecture right now,
- because the current architecture already works well offline,
- and the stronger current suspicion is a real-time input / timing / distribution issue.

So the most practical path is:

1. keep `stable hybrid` as the main current candidate,
2. keep current runtime structure aligned to the stable hybrid config,
3. collect failed and borderline real-time captures using `run_debug_captures`,
4. manually label the useful ones,
5. add them as supplemental real-world samples,
6. retrain/fine-tune the same stable-hybrid pipeline,
7. re-test real-time performance.

In other words:
- not "GG,"
- but the next bottleneck is more likely
  - data / runtime distribution,
  - not architecture invention.

## 14. Important Files to Know Right Now

Core code:
- `C:\SignProject\nsu_train.py`
- `C:\SignProject\nsu_run_fusion_v6_overlap_hold.py`
- `C:\SignProject\nsu_collect.py`
- `C:\SignProject\gpt_train.py`
- `C:\SignProject\run_multiseed_comparison.py`

Main model/result folders:
- `C:\SignProject\fusion_train_output_hold_v4_flags_gate_flexposture_stable`
- `C:\SignProject\fusion_cmp_seed45_hybrid_stable`
- `C:\SignProject\vision_results_from_fusion_hold_v4_flags`
- `C:\SignProject\fusion_train_output_hold_v4_flags_imu_flags_flexposture_v2`
- `C:\SignProject\gksdydtjr\fusion_train_output_hold_v4_flags_gate_v2`

Repeated-seed summaries:
- `C:\SignProject\results_summary\multiseed_per_run.csv`
- `C:\SignProject\results_summary\multiseed_group_summary.csv`
- `C:\SignProject\results_summary\multiseed_group_summary.json`

Runtime debug capture folder:
- `C:\SignProject\run_debug_captures`

Legacy/archive references:
- `C:\SignProject\_archive_notebook_legacy_2026-04-05`
- `C:\SignProject\_merge_backup_pre_lsm_2026-04-05`
- `C:\SignProject\_lsm_version_snapshot`

## 15. What a New Chat Should Assume

If a new chat starts from this file, it should assume:

- `C:\SignProject` is the merged current working folder.
- `stable hybrid` is the current best final candidate, not final confirmed model.
- The thesis framing should center on similar-sign confusion and practical robustness.
- The professor's final comparison frame is:
  - bare hand + vision
  - glove + vision
  - glove + fusion
- The offline architecture search has largely narrowed to stable hybrid.
- Real-time issues are currently being investigated using saved runtime captures.
- Current runtime default model is `fusion_cmp_seed45_hybrid_stable`.
- Arduino sensor sender has been uploaded to `COM9`.
- The current best next move is to keep gathering and analyzing real-time captures, then use confirmed useful samples for retraining/fine-tuning.

## 16. One-Paragraph Executive Summary

This project evolved from unstable naive vision+sensor fusion into a role-separated multimodal design where vision stays central, IMU/flags act as corrective support, and flex is reduced to posture-summary features. That evolution led to the current `stable hybrid` candidate, which did not always have the best single run but showed the best repeated-seed mean and stability. `C:\SignProject` is now a merged latest working folder reflecting the `lsm-version` core code and curated results. In this chat, repeated-seed bare/glove baselines were added, run-vs-collect behavior was aligned, Arduino was prepared on `COM9`, runtime debug capture saving was added, and the first careful real-time analysis suggested that the current bottleneck is more likely runtime input/timing/distribution mismatch than a simple class-order or model-wiring bug. The immediate next step is to keep using `run_debug_captures` to gather real real-time failure cases and use human-labeled useful samples for stable-hybrid retraining.
