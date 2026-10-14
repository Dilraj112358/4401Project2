# Final error analysis: model without gender

## Model and review population

Gender is excluded from predictive inputs and retained for auditing. Age remains. Application-only 64–32 network, Adam weight decay 0.0001, seed 42. Selected epoch: 10. Threshold: 12.1%, selected by blackbox F1. This cutoff is a model score, not a demonstrated calibrated probability.

The same 500-person eyeball pool now contains **95 errors: 67 false positives and 28 false negatives**. There are 393 true negatives and 12 true positives. Export all current errors; the old 89-row count is superseded. All rows contain factual observations, pattern hypotheses and an analyst case note. Human verification remains pending, not falsely marked complete.

## What gender removal changed

The comparison uses the same 61,002 blackbox applicants. Each model has its own blackbox F1 threshold. One training seed was rerun; these differences are descriptive, not established statistical significance.

| Metric | Previous model | Without gender |
|---|---:|---:|
| Threshold | 17.20% | 12.10% |
| AUC | 0.7494 | 0.7468 |
| Accuracy | 84.19% | 83.33% |
| Precision | 23.12% | 22.41% |
| Recall | 41.22% | 43.23% |
| F1 | 0.2963 | 0.2951 |
| False-positive rate | 12.04% | 13.15% |
| Gender FPR gap (percentage points) | 7.99 | 5.82 |
| Age FPR gap (percentage points) | 22.41 | 22.17 |

At matched overall FPR of approximately 12.04%, recall changes from 41.22% to 40.57%; the gender FPR gap changes from 7.98 to 5.58 points and the age gap from 22.40 to 20.43. These matched-FPR thresholds are diagnostic only; the submitted CSV uses 12.1%. This separates some operating-point effects from ranking changes but does not establish individual causality.

Gender exclusion is the agreed project constraint; the model was not chosen because it improved test or eyeball performance. AUC is slightly lower. Gender disparity remains despite removing the direct input. Age and proxy features remain, and a narrower gap alone does not mean every group benefits. No post hoc numerical tolerance is presented as a legal or fairness certification.

| Blackbox group | Previous FPR | New FPR | Previous recall | New recall |
|---|---:|---:|---:|---:|
| F | 9.4% | 11.2% | 35.1% | 39.2% |
| M | 17.4% | 17.0% | 49.4% | 48.6% |
| Under 30 | 24.3% | 25.0% | 58.1% | 58.1% |
| 30 to under 45 | 14.4% | 15.2% | 45.5% | 46.5% |
| 45 to under 60 | 7.6% | 9.4% | 29.2% | 33.6% |
| 60 and older | 1.9% | 2.8% | 9.3% | 14.8% |

At the chosen F1 thresholds, women's false-positive rate increases while men's decreases slightly; the narrower gender gap is not an across-the-board reduction in false alarms. The oldest group's difficulty recall remains much lower than the youngest group's. Counts, metric-specific denominators, NPV, precision, favorable prediction rates and descriptive Wilson intervals are in the fairness exports. The single XNA record cannot support a group fairness conclusion. Intervals do not account for repeated model selection.

The previous and new eyeball predictions yield 80 persistent errors, 9 previous errors now correct, and 15 previously correct records now wrong. 396 are correct in both. A revised model shifts errors rather than universally fixing them.

## Error patterns and priorities

| Eyeball group | Errors / eligible applicants | Error rate |
|---|---:|---:|
| Older high-score actual difficulty | 11 / 12 | 91.7% |
| Younger low-score actual no difficulty | 7 / 13 | 53.8% |
| Any external score missing | 57 / 324 | 17.6% |
| All external scores present | 38 / 176 | 21.6% |
| Within 0.02 of threshold | 20 / 43 | 46.5% |

1. **Missed difficulty among older applicants with relatively high external scores remains a concern.** The 11/12 pattern is unchanged in aggregate from the previous review. It is a small, exploratory group, not proof that age or an external score caused the mistakes. Prioritize the potential harm of missed difficulty and the age recall disparity.
2. **Younger applicants with low external scores still receive false alarms.** The 7/13 pattern is also unchanged in aggregate. The fairness table quantifies this concern on the larger blackbox set. It does not imply all younger applicants are overpredicted for the same reason.
3. **Missing external information is not a sufficient explanation.** Compare rates against all eligible records: the missing-score group's error rate is lower than the complete-score group's here. This does not prove missingness helps; the groups differ in other ways and outcome mix.
4. **Most errors are outside the near-threshold band.** Only 20 of 95 errors are within 0.02 of the cutoff. Moving the cutoff changes false positives and false negatives together; it does not improve ranking or establish a root cause.
5. **Use correct cases as context.** 91 errors have a correct-case match on actual outcome and EXT_SOURCE_2 within 0.05. Matching uses replacement, so controls are not independent and can repeat. Other characteristics differ. The remaining 4 cases have no eligible match and are not assigned fabricated comparisons.

## Representative case interpretations

### Applicant 449715

False negative: actual TARGET 1, score 0.7%, cutoff 12.1%; age 55.5. EXT_SOURCE_1=missing, EXT_SOURCE_2=0.684, EXT_SOURCE_3=0.674.

Score is 11.42 percentage points from the cutoff; this case is outside the predefined 2-point near-threshold band. Difficulty occurred despite both EXT_SOURCE_2 and EXT_SOURCE_3 being at least 0.50; relatively high scores are not guarantees. Credit amount / reported income is 6.70; this descriptive ratio alone does not establish affordability or the cause of the error. Gender is shown for audit only and did not enter this model. Age remains an input; this row does not establish its causal effect.

Correct case 361395: EXT_SOURCE_2 0.651 versus 0.684; EXT_SOURCE_3 0.369 versus 0.674; age 42.1 versus 55.5. Other inputs differ; this is not a causal comparison.

### Applicant 259646

False negative: actual TARGET 1, score 1.5%, cutoff 12.1%; age 62.1. EXT_SOURCE_1=missing, EXT_SOURCE_2=0.664, EXT_SOURCE_3=missing.

Score is 10.58 percentage points from the cutoff; this case is outside the predefined 2-point near-threshold band. Credit amount / reported income is 12.43; this descriptive ratio alone does not establish affordability or the cause of the error. Gender is shown for audit only and did not enter this model. Age remains an input; this row does not establish its causal effect.

Correct case 361395: EXT_SOURCE_2 0.651 versus 0.664; EXT_SOURCE_3 0.369 versus missing; age 42.1 versus 62.1. Other inputs differ; this is not a causal comparison.

### Applicant 305291

False positive: actual TARGET 0, score 36.3%, cutoff 12.1%; age 26.2. EXT_SOURCE_1=0.323, EXT_SOURCE_2=0.094, EXT_SOURCE_3=0.161.

Score is 24.17 percentage points from the cutoff; this case is outside the predefined 2-point near-threshold band. Both EXT_SOURCE_2 and EXT_SOURCE_3 are below 0.25, yet no difficulty was recorded; low scores are not sufficient to establish difficulty. Credit amount / reported income is 1.78; this descriptive ratio alone does not establish affordability or the cause of the error. Gender is shown for audit only and did not enter this model. Age remains an input; this row does not establish its causal effect.

Correct case 425677: EXT_SOURCE_2 0.096 versus 0.094; EXT_SOURCE_3 0.658 versus 0.161; age 31.4 versus 26.2. Other inputs differ; this is not a causal comparison.

### Applicant 400464

False positive: actual TARGET 0, score 12.3%, cutoff 12.1%; age 53.3. EXT_SOURCE_1=missing, EXT_SOURCE_2=0.545, EXT_SOURCE_3=0.264.

Score is 0.18 percentage points from the cutoff; classification is sensitive to a small threshold change, whose other errors must also be counted. Credit amount / reported income is 4.49; this descriptive ratio alone does not establish affordability or the cause of the error. Gender is shown for audit only and did not enter this model. Age remains an input; this row does not establish its causal effect.

Correct case 254373: EXT_SOURCE_2 0.543 versus 0.545; EXT_SOURCE_3 0.590 versus 0.264; age 60.9 versus 53.3. Other inputs differ; this is not a causal comparison.

### Applicant 228526

False negative: actual TARGET 1, score 2.6%, cutoff 12.1%; age 62.3. EXT_SOURCE_1=0.698, EXT_SOURCE_2=0.038, EXT_SOURCE_3=0.683.

Score is 9.46 percentage points from the cutoff; this case is outside the predefined 2-point near-threshold band. External scores point in different directions under the exploratory high/low boundaries; neither score alone determines the outcome. Raw employment duration contains the known placeholder; training already replaced it with missingness and an indicator, so this is not an uncorrected preprocessing defect. Credit amount / reported income is 3.36; this descriptive ratio alone does not establish affordability or the cause of the error. Gender is shown for audit only and did not enter this model. Age remains an input; this row does not establish its causal effect.

No correct case with the same actual outcome within 0.05 on EXT_SOURCE_2 in this eyeball pool.

## Bias and variance

Selected-model training AUC is 0.7595, versus blackbox 0.7468, a gap of 0.0126. The gap alone is not a bias–variance decomposition or proof of severe overfitting. We lack an attainable-error benchmark for quantifying avoidable bias. Historical depth/width, regularization and feature experiments remain supporting evidence; they were not all rerun for the gender-excluded model.

## Exploratory final test assessment

Test AUC 0.7495, accuracy 83.75%, precision 23.03%, recall 43.26%, F1 0.3006. Test outcomes did not select this checkpoint or cutoff. Previous test exposure means this is exploratory, not an untouched final holdout. No test-case records enter manual error review.

## Friday submission and remaining work

Submit **eyeball_error_review.csv** after verifying the drafted case observations. The lecture does not require exactly 89 errors. Use `reviewer`, `team_review_status`, and `team_comment` for real human review. Do not mark the assistant's draft as a teammate's completed review.

Supporting files: `eyeball_predictions.csv` contains all 500 eyeball cases; `correct_case_comparisons.csv` supplies matched correct cases; `eyeball_pattern_summary.csv` supplies denominators; `fairness_metrics.csv` and `fairness_disparities.csv` describe blackbox groups; `gender_removal_comparison.csv` and its group companion document the latest intervention. These supporting files need not all be separate submissions; they support the report and presentation. The raw all_eyeball_errors.csv is an audit export, not a second review to complete.

No verified label corruption was identified, so no labels or difficult cases were removed. Preserve the current model during the review. A future intervention should follow a specific hypothesis, use blackbox evaluation, and start a new model-specific review if adopted. Older split comparisons and the conceptual Word document describe the prior model until separately updated; do not present their numbers as this run's results.

Provenance: application_no_gender_weight_decay_20261013T064129748622Z. Prior comparison: application_weight_decay_20261013T045633838008Z. Current predictors exclude CODE_GENDER; raw audit attributes preserve it. Historical full-dev exposure and repeated blackbox use limit confirmatory claims.
