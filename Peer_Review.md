# Peer review and finalization recommendation

October 14, 2026. Reviewed the complete main-branch tree at commit e2a1de71249f46df8a83a30632ef258ba52477be in Dilraj112358/4401Project2. Main was the only remote branch in this checkout. Attribution follows the user's identification of Stanley's notebook and Dilraj's nn1 files; authorship of other uploaded files is not inferred from filenames.

## Recommendation

Do not describe Gil's original model as the strongest finalized team model yet. A controlled application-only adaptation of Dilraj's fuller recipe improves AUC and F1 in all three seeds on Gil's exact train/blackbox rows, with gender excluded. It meets the previously recorded practical AUC switching rule. Recommend advancing this candidate to a final model/error-review iteration, while explicitly reviewing its recall and fairness tradeoffs. This is practical repeatability across initialization seeds, not independent confirmation or proof of optimality.

The existing 64/32 model and its 95-error analysis remain a consistent, reproducible reference. They are preserved together. No candidate eyeball or test predictions were inspected in this comparison, no active model pointer changed, and the 95 rows must not be relabeled as the candidate's errors. A replacement requires a new matching review export, hypothesis checks, three-split evaluation and model documentation. The current branch is a review snapshot, not a declaration that either model is deployment-ready or team-approved.

## Controlled comparison

Two challengers, each across seeds 42/43/44, used 184,506 training and 61,002 blackbox applicants. Both use only application_train predictors and exclude gender. Baseline predictions come from the completed October 13 audit on those same rows and seeds. The protocol was saved before fitting. The simple challenger substitutes the peer preprocessing bundle: quantile transformation, categorical missing/rare handling, missing indicators and constant/duplicate-column removal. It is not an isolated test of the quantile transform. The full challenger adds peer application feature engineering and the regularized_all CreditNet recipe: 128/64 units, linear skip connection, SiLU, batch normalization, dropout, input noise, AdamW, scheduler and best-checkpoint restoration. No ensemble or extra CSV is needed.

| Recipe | Mean blackbox AUC | Mean F1 | Recall | FPR | AUC wins |
|---|---:|---:|---:|---:|---:|
| baseline | 0.747075 | 0.296473 | 44.07% | 13.46% | 0/3 |
| quantile_current | 0.745818 | 0.291678 | 41.16% | 12.39% | 0/3 |
| peer_recipe | 0.754961 | 0.302354 | 41.16% | 11.52% | 3/3 |

At each model's F1-selected threshold, the fuller recipe reduces both false alarms and recall. At the baseline's overall FPR cap, mean candidate recall is 44.81%, versus 44.07%; mean gender FPR gap is 6.55 versus 5.67 percentage points, and age gap is 23.34 versus 22.31. Thus improved ranking is not a uniform fairness improvement. These threshold matches use blackbox labels and are descriptive.

For fixed seed42, candidate AUC is 0.754731 versus 0.746842, F1 0.302551 versus 0.295141, and its F1 cutoff is 0.167 versus 0.121. The 0.167 cutoff belongs only to the candidate checkpoint. Do not copy it onto the old model. Seed42 remains the fixed reference; selecting seed44 because it has the highest observed AUC would introduce another choice.

The October 13 audit remains accurate for the simpler candidate recipes it tested. It did not test this richer combined recipe. The new result updates the recommendation rather than proving the earlier finite search was optimal.

## What each repository document contributes

| File | Findings and implications |
|---|---|
| Stanley_Project_2_NN_Model.ipynb | Reported test AUC 0.7417 baseline and 0.7484 improved. Improved model uses dropout, batch normalization and weighting. Preprocessing is fitted before carving out validation, so validation influences its transform. best_state is saved but never restored before testing. total_loss is not reset each epoch in the improved loop, making its loss log invalid. The prose confuses 128 units with 128 layers. These issues should be fixed before treating its result as a clean best-checkpoint comparison. |
| Project_2_Baseline_and_Improved_Model.ipynb | Train-only preprocessing and restored checkpoints; 0.7470 to 0.7511 test AUC. Several changes are bundled, so improvement cannot be attributed solely to engineered features. At cutoff 0.5 the improved model has zero recall and F1 despite 91.93% accuracy. Keep Gil's validation-selected threshold workflow. |
| nn1.ipynb | Application-only ensemble reports test AUC 0.76075 and validation AUC 0.75893. Uses 64/16/20 partitions, gender inputs, engineered features, quantile preprocessing and stronger regularization. Useful ideas, but its absolute score is not a controlled comparison with Gil's 60/20/20 plus eyeball protocol. The controlled adaptation above is the relevant new evidence. |
| nn1_Improvement.ipynb | Reports test AUC 0.77126 and validation 0.77154 using previous applications, bureau, installments and a five-model ensemble. The raw_baseline in this notebook already includes history, so it is not an application-only baseline. At 0.5, difficulty recall is 2.64%; threshold/fairness checks still matter. Its saved bureau coverage is 6.997%, versus 85.685% in Additional_Data under similar eligibility rules: verify source CSV version/completeness and rerun before relying on feature attribution. A stale paragraph incorrectly says history is absent. |
| Project_2_Home_Credit_History_Features.ipynb | Application plus history test AUC 0.7555 versus 0.7466; best validation 0.7540 versus 0.7392. Supports predictive value of history, consistent with Gil's past experiments. One split/seed does not establish that the gain cannot be random. Adding these data remains a deliberate scope choice; do not claim they were useless. |
| Additional_Data.ipynb | Preparation only, not a model win. Good practices include per-applicant aggregation, time eligibility, currency handling, grouping payment parts, missingness/presence distinction and one-to-one joins. No need to add these tables to reproduce the new application-only gain. |
| Nonrandom_Split_Comparison.ipynb | Loan-size stress test, correctly qualified as unproven deployment simulation. Credit cards add about 0.00096 AUC to bureau/installments, but F1 and recall fall at selected cutoffs, with no better matched-FPR detection in that run. No basis to restore the rejected loan split or add credit cards merely for accuracy. Keep the previous-application deployment scenario, with its own stated assumptions. |
| Project_2_DATA_4401.ipynb | Early baseline with a two-way 80/20 split and a partial saved training log; no completed performance evidence establishing superiority. |
| gil_project2.ipynb and Error_Analysis.ipynb | Repository copies are stale relative to local work: older gender-inclusive baseline and a ten-error review with threshold0.139. Replace together with the current reference, its run pointer, manifest and saved predictions; do not mix generations. |
| 4401 Project 2 Error Analysis.xlsx | Sheet1 has generic headers in A1:P1 and no populated observations beneath them. Formatting extends beyond the data. It provides no completed case findings to incorporate. |
| error_analysis/starter_review_seed42.csv | Ten-case historical starter, not the current95-error review. Keep labeled historical. |
| README.md and .gitignore | Minimal repository setup. Added README_GIL.md identifies the current reference, review snapshot, dependencies, data requirements and paths. |

## Finalization steps supported by this review

1. Advance the fuller application-only candidate for the next model iteration; retain the old model as the comparison baseline. Decide the acceptable recall/fairness tradeoff explicitly rather than calling AUC improvement a fairness improvement.
2. If selected, freeze its checkpoint and threshold using blackbox data, then generate a fresh error inventory on the same500 eyeball IDs. Review all its errors; do not force it to have95 rows or reuse old case conclusions unchanged.
3. Recheck the recorded error hypotheses and group metrics, and update the three-split evaluation for the new configuration. State historical exposure and exploratory selection limitations.
4. Only then label the replacement model and matching error spreadsheet finalized for the course. Team verification of manually drafted notes remains necessary.

## Evidence and reproduction

Peer source links are pinned to the reviewed commit: https://github.com/Dilraj112358/4401Project2/tree/e2a1de71249f46df8a83a30632ef258ba52477be . Full source was inspected locally; uploaded raw datasets were not available in GitHub and peers' full history experiments were not rerun. The coverage discrepancy is unresolved, not proof of a particular bug.

Saved new evidence: peer_review/20261014/Protocol.md, results.json, summary.json and probability/checkpoint files in the local workspace. Peer_Review_Comparison.ipynb displays the saved evidence without training. The branch includes peer_review/run_comparison.py and the reviewed nn1 source snapshot for reproduction from the repository root, using application_train.csv supplied separately. Fitting again writes comparison outputs; it does not update the active model or its error review. Historical main-model source is preserved alongside the comparison so future edits cannot silently change the reference.
