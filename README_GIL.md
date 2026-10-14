# Gil project review snapshot

Read **Peer_Review.md first**. October 14 comparison found a stronger application-only candidate; the current 64/32 model is preserved as the reviewed reference, not declared the strongest finalized team choice. Candidate adoption and a matching new error review remain a separate next iteration.

## Files to use

- gil_project2.ipynb: executed reference model, gender excluded, age retained, weight decay 0.0001, seed 42, cutoff 0.121.
- Error_Analysis.ipynb: all 95 reference-model eyeball errors, with draft notes and correct-case comparisons.
- error_analysis/application_no_gender_weight_decay_20261013T064129748622Z/eyeball_error_review.csv: current reference review, 67 FP and 28 FN. Team verification is pending.
- Blackbox_Error_Hypotheses.ipynb: nine recorded checks for that reference model.
- Final_Split_Comparison.ipynb: corrected three-protocol evaluation of the reference.
- Decision_Audit.ipynb: October 13 finite candidate search; Peer_Review_Comparison.ipynb: October 14 expanded comparison that updates its recommendation.
- output/documents/: conceptual review and error-hypothesis explanation. The October 14 Peer_Review.md is the latest recommendation and supersedes the older document's keep-model conclusion.

## Run locally or in Colab

Install requirements_gil.txt. Put application_train.csv in the repository working directory. Keep dev_split_manifest.json, active_error_analysis_run.json and the included model_runs folders. All necessary saved inputs for the reference error analysis are included; retraining is optional. Run notebooks with this repository as the working directory. For Colab, clone/download the repository, change directory into it and upload the raw data there. Do not upload only a notebook and omit its supporting files.

The nonrandom comparison also needs previous_application.csv and bureau.csv. Those raw Kaggle tables are intentionally not committed. New main-model training exports a new run; run Error_Analysis next so the output pointer and review match it. Do not mix a new model with the old 95-error CSV. The historical gender comparison is specific to the reference model and should be updated if a different model is adopted.

To reproduce the peer comparison, run `python peer_review/run_comparison.py` from the root. It uses the frozen reviewed nn1 and Gil source snapshots, application_train only, fixed train/blackbox rows and three seeds. This reruns six fits without selecting on eyeball/test outcomes. The saved baseline probabilities and protocol are included.

Earlier 10-case starter CSV and peer notebooks remain historical/independent work. No colleague's modeling notebook was changed by this review. Report historical dev/test exposure and unresolved fairness acceptance; none of these models is certified for deployment.
