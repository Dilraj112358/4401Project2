# Final Project 2 package — October 14, 2026

Start with these files:

- [Final model notebook](Final_Project_Model.ipynb)
- [Final error analysis notebook](Final_Error_Analysis.ipynb)
- [Final error review CSV](Final_Error_Analysis.csv): 81 errors, 53 false positives and 28 false negatives.
- [Model explanation and team contributions](output/documents/Project_2_Final_Model_and_Team_Contributions.docx)

The selected model adapts Dilraj's application-only recipe to Gil's fixed evaluation protocol: gender excluded, training-only preprocessing, a 128/64 neural network with a linear skip, and a blackbox-F1-selected cutoff of 0.167. Stanley's experiments informed the architecture review. This is one fitted model, not an ensemble of the three people's predictions. The document distinguishes attributed work from shared notebooks whose authorship is not established.

Three controlled seeds improved mean blackbox AUC over the previous baseline. Seed 42 was fixed before inspecting the new eyeball/test outputs. Main stratified test AUC is 0.7567, F1 0.3065, accuracy 85.51%, and recall 39.66%. Random and previous-application membership splits are also evaluated. New-client transfer is weaker than the previous model; fairness disparities and recall limitations remain. Historical dev/test exposure prevents calling these results a pristine holdout or deployment approval.

## Run

Clone the entire repository, install `requirements_final.txt`, put Kaggle `application_train.csv` in the repository root, and run the notebooks from that root. Raw Kaggle tables are not included. Default evaluation loads the frozen checkpoint and fitted preprocessing in `final_project/selected`. `RETRAIN=True` writes a separate retrained run; it does not replace the selected model or error CSV. `RERUN_SPLITS=True` retrains the alternatives and also needs `previous_application.csv`. That table determines the nonrandom partition; it is not a predictor source.

The standalone training implementation reproduced the selected blackbox probabilities exactly in the tested environment. Supporting code, metrics, predictions, fairness tables, hypothesis checks and provenance are under `final_project`. Original peer notebooks and older Gil outputs remain historical references; use the files linked above together for the current model.

## Before submission

The code and error inventory have been checked. The 81 factual case notes are assistant drafts: the team should verify them and fill `reviewer`, `team_review_status`, and `team_comment`. The notebook preserves those edits for the same model and case IDs. No labels were changed merely because the model disagreed. Nine prior hypotheses were rechecked using fixed definitions; these are exploratory associations on repeatedly used blackbox data, not causal findings.
