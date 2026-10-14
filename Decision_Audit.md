# Project 2 decision audit

October 13, 2026. Keep the current application-only neural network with gender excluded, age retained, 64 / 32 hidden units, weight decay 0.0001, and a 0.121 cutoff. The audit supports this as a defensible classroom model within the chosen scope. It does not establish a global optimum, acceptable lending fairness, or readiness for deployment.

## What was tested

Eight configurations across seeds 42, 43 and 44 produced 24 fits. All used the existing training and blackbox rows, train-only preprocessing, Adam learning rate 0.001, batch size 256 and 15 epochs; the best checkpoint was selected by blackbox AUC. Candidate selection did not evaluate eyeball or test predictions. The baseline seed 42 reproduced the active model's predictions to numerical precision.

Before running the candidates, Protocol.md recorded a practical switching rule: mean paired AUC gain of at least 0.001, with improvement in all three seeds. F1 decreases above 0.005 or matched-FPR recall decreases above one percentage point require tradeoff review. This is a practical rule for this audit, not a significance test or a rule retroactively applied to earlier choices. Three seeds measure initialization sensitivity on one partition, not three independent population samples.

| Candidate | Mean blackbox AUC | Change | Seeds improving | Mean F1 |
|---|---:|---:|---:|---:|
| Current 64 / 32 | 0.747075 | +0.000000 | 0/3 | 0.296473 |
| Remove age | 0.745060 | -0.002015 | 0/3 | 0.293401 |
| Remove weight decay | 0.745463 | -0.001612 | 0/3 | 0.293393 |
| One layer of 32 | 0.746389 | -0.000686 | 1/3 | 0.293260 |
| 128 / 64 | 0.747544 | +0.000469 | 2/3 | 0.295957 |
| 64 / 32 / 16 | 0.747516 | +0.000441 | 2/3 | 0.295967 |
| Credit and annuity to income | 0.747635 | +0.000560 | 2/3 | 0.295311 |
| External score interactions | 0.747689 | +0.000614 | 2/3 | 0.295779 |

None met the switching rule. Larger networks and explicit ratio or score-interaction features offered small average AUC gains but lost in one seed and had lower mean F1. The current model is not the highest mean AUC in this table; its retention follows the recorded minimum-gain and consistency rule.

## Decisions and evidence

| Decision | Audit conclusion | Reason and limitation |
|---|---|---|
| Application-only predictors | Keep as a scope choice | Additional tables previously improved AUC, especially repayment history. They were excluded for the requested simplicity, not because they contain no signal. AUC optimization is conditional on this scope. |
| Exclude gender | Keep | Direct gender use is avoided. Earlier comparisons showed a narrower gender FPR gap. Exclusion does not remove proxies or certify fairness. |
| Retain age | Keep for this classroom model with an explicit tradeoff | Removing age from the current gender-excluded model reduced AUC in every seed by 0.002015 on average. At matched overall FPR, the age FPR gap widened from 22.30 to 25.69 percentage points. This empirical comparison is not a legal or ethical justification for deployment. |
| Train-only preprocessing | Keep | IDs are unique; ID, target and gender are excluded; declared feature lists cover every input without duplication. Imputation, scaling and missing indicators use training data. Employment sentinel 365243 is cleaned. These checks do not prove every raw record is correct. |
| 64 / 32 architecture | Keep | Smaller, wider and deeper alternatives did not meet the recorded switching rule. No claim of optimal architecture over all possible networks. |
| Weight decay 0.0001 | Keep | Removing it reduced AUC in all three seeds and worsened the matched-FPR age gap. Other strengths were not exhaustively searched. |
| Unweighted loss | Keep provisionally | Earlier oversampling and class-weight experiments did not justify replacement. They used an older configuration, so they are historical evidence, not new proof for the gender-excluded model. No business cost matrix supports a particular weight. |
| Select checkpoint by blackbox AUC | Keep | Ranking is the optimizing metric; current seed 42 selects epoch 10. Train AUC 0.7595 versus blackbox 0.7468 suggests a modest generalization gap, not evidence that more depth is needed. Without an achievable-error benchmark, high bias cannot be diagnosed conclusively. |
| 12.1 percent F1 cutoff | Keep | Exact cutoff search gives 0.1212775 and increases F1 by only 0.000064. Five-fold transferred cutoff F1 is 0.29441 versus current 0.29514. F1 is a project proxy, not a known profit-optimal operating rule. |
| Probability calibration | Record a useful follow-up | Cross-fitted calibration improves Brier score 0.068866 to 0.068639 and log loss 0.252386 to 0.249903. Raw mean score 6.60 percent is below observed 8.07 percent. No final calibrator is adopted; ranking within each fold is unchanged. Probability estimates should not be called calibrated. |
| Stratified main split | Keep for source-population evaluation | It preserves difficulty prevalence and supports the fixed error-review workflow. It is not more demographically fair, and is not chosen because it has the largest AUC. |
| Previous-application nonrandom split | Keep as a deployment scenario | Train and dev have prior application records; test has none. This tests transfer to clients with no recorded prior Home Credit application. Missing records do not prove no credit history. If actual deployment targets new clients, future tuning should also use a separate representative new-client dev set. |
| Bureau availability | Keep as an audit label | Eligible bureau presence subdivides the no-prior-application test group. It does not add predictive bureau features or imply three descending levels of credit information. |
| Fixed eyeball and blackbox roles | Keep with exposure disclosure | All 500 eye cases stay out of new candidate fits and tuning. Review all 95 current errors and compare correct cases. Historical full-dev use cannot be undone by renaming the sets. |
| Error hypotheses | Keep as descriptive evidence | Blackbox checks distinguish observed associations from proposed causes. Young/low EXT3 false positives and older/high EXT2 false negatives are strong patterns. Hypotheses of general low-score or young-group overprediction were not supported. Multiple exploratory comparisons do not prove causation. |
| Fairness evaluation | Keep measures; unresolved acceptance decision | Report FPR, recall, PPV, NPV and prediction rates with group counts and uncertainty. Gender and age gaps remain. No client-approved cost tradeoff or acceptable disparity tolerance exists; do not invent a post hoc pass mark. |

## Corrected three-split evaluation

Final_Split_Comparison.ipynb previously used the earlier gender-inclusive configuration. It now uses the selected gender-excluded configuration, with train-only preprocessing and independent dev-based epoch and threshold selection in each protocol. Random and stratified partitions now have equal sizes: 184,506 train, 61,002 blackbox and 61,503 test. The fixed 500 eyeball applicants are excluded from all alternative partitions. IDs are disjoint within each protocol; the alternative partitions cover the same remaining 307,011 applicants.

| Protocol | Test applicants | Threshold | Test AUC | Test F1 | Test recall | Test FPR |
|---|---:|---:|---:|---:|---:|---:|
| Stratified | 61,503 | 0.121 | 0.74948 | 0.30063 | 43.26% | 12.69% |
| Random | 61,503 | 0.128 | 0.75027 | 0.29398 | 45.91% | 14.70% |
| No prior application | 16,421 | 0.144 | 0.76854 | 0.26858 | 35.48% | 8.15% |

These are different evaluation populations, not a contest for the best split. The nonrandom test has 5.96 percent difficulty prevalence versus about 8.07 percent for the stratified test. Its higher AUC does not make it the best main split. Random catches more positives at the cost of more false alarms; stratified has higher F1 here. The current configuration was fixed before these new test evaluations.

Within the no-prior-application test, 13,956 applicants have an eligible bureau record and 2,465 do not. With versus without bureau: AUC 0.7664 versus 0.7794, recall 35.09 versus 37.22 percent, and FPR 8.35 versus 7.00 percent. There is no uniform evidence that the no-bureau group performs worse. These are descriptive subgroup results at the frozen 0.144 cutoff, not evidence that removing bureau information improves a person's prediction.

## Error analysis and what remains unfinished

The main model and error predictions stay unchanged: 95 errors among 500 fixed eyeball applicants, comprising 67 false positives and 28 false negatives. There are 91 matched correct-case comparisons and four unmatched errors. The spreadsheet contains assistant-drafted case notes that the team must actually review; generated notes are not a substitute for the lecture's manual inspection.

Nine recorded hypotheses were checked against the larger blackbox. Findings include a modest score-adjusted loan-to-income association, concentrated false negatives among older applicants with high EXT_SOURCE_2, concentrated false positives among younger applicants with low EXT_SOURCE_3, score-conflict associations and a high false-negative rate among positive cases with the employment sentinel. Missing scores have mixed effects. Only 22.38 percent of blackbox errors are within 0.02 of the threshold, so threshold proximity does not explain most errors. Refer to Blackbox_Error_Hypotheses.ipynb and the error-hypothesis Word report for denominators, uncertainty, counterexamples and limitations.

The new ratio and interaction experiments connect error analysis to an attempted remedy; they did not yield a consistent practical upgrade. A failed remedy still provides evidence when its hypothesis, comparison and outcome are recorded. Remaining gaps are human review of the 95 notes, explicit business/fairness acceptance criteria before any deployment, and a fresh independent evaluation if stronger generalization claims are needed. Repeated blackbox use and earlier test exposure mean the reported uncertainty is descriptive and conditional; no pristine final holdout should be claimed. Calibration cross-fitting also remains conditional on a checkpoint previously selected using the whole blackbox.

## Reproducibility and sources

Assignment and lectures reviewed: DATA4401-Project02.pdf, DATA4401-Lec09-Yearning03.pdf, DATA4401-Lec10-Fairness.pdf and DATA4401-Lec11-Yearning04.pdf. The assignment requires the three deployment protocols, an NN, metrics, fairness discussion and iterative error analysis; it does not require every attempted improvement to succeed.

- Main run: model_runs/application_no_gender_weight_decay_20261013T064129748622Z
- Audit run: decision_audit/20261013T181905Z; recorded protocol, source hashes, 24 checkpoints, probabilities, learning curves and summaries.
- Corrected split run: final_split_runs/20261013T182318505371Z
- Main review: error_analysis/application_no_gender_weight_decay_20261013T064129748622Z/eyeball_error_review.csv
- Audit training script: diagnostics/decision_audit.py; calibration: diagnostics/audit_calibration_threshold.py.
- Decision_Audit.ipynb displays the saved audit without retraining or replacing the active model. Re-running the training script creates a new audit run and updates its pointer.
