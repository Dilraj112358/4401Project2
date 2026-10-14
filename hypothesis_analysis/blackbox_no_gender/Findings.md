# Blackbox tests of error-analysis hypotheses

Protocol was recorded before analysis in Protocol.md. Frozen model: application_no_gender_weight_decay_20261013T064129748622Z. Only 61,002 blackbox applicants were used. TARGET means repayment difficulty. No model, threshold, or error-review CSV was changed.

## H1: Overprediction for EXT_SOURCE_2 below 0.11

N=2,537; observed difficulty 23.02%; mean prediction 18.26%. Predicted minus observed: -4.76 percentage points, approximate 95% interval [-6.34, -3.18] percentage points. Result: negative gap, with the descriptive interval below zero. The predeclared illustrative practical margin is +2 percentage points; a point estimate below this is not evidence of large overprediction. False positives alone cannot establish miscalibration because most members of even a risky group may have no difficulty.

**H1 is not supported: the observed direction is underprediction, not overprediction.**

## H2: High credit-to-income ratio and false positives at similar external scores

Excluded 0 applicants with invalid/missing ratio inputs. Primary score-band width 0.20: standardized FPR is 12.03% for ratio <=4 and 13.63% for ratio >4. Adjusted difference: 1.61 points, approximate 95% interval [1.16, 2.05] percentage points. Result: positive gap, with the descriptive interval above zero. Included 54,720 of 56,077 eligible actual negatives (97.6%) across 105 common-support strata. Retention is 97.3% of low-ratio negatives and 98.0% of high-ratio negatives.

At narrower 0.10 score bands, difference=0.60 points, interval [0.16, 1.04] percentage points, with 81.6% negative coverage. These are standardized associations within retained strata, not causal effects. Both groups use the same pooled-stratum weights. Other variables and within-band score differences can confound the association. Missing scores form their own bands; their unobserved values are not held constant.

**H2 has modest descriptive support after score adjustment.** Before adjustment, FPR is 13.62% for ratio <=4 and 12.39% for ratio >4: the direction reverses. This is why raw counts and unadjusted rates are insufficient. Narrower bands reduce both the difference and overlap coverage, so the sensitivity change is not attributable solely to finer score control. This does not establish that high loan ratios are erroneous signals or should be removed.

## H3: Younger false alarms and calibration

Under 30: N=8,900, observed difficulty 11.31%, mean prediction 9.62%. Calibration residual -1.70 points, interval [-2.33, -1.07] percentage points. Under-30 minus age-30+ calibration residual is -0.26 points, interval [-0.93, 0.41] percentage points.

The age table and fixed-bin reliability table separate average calibration from FPR. Aggregate agreement does not guarantee calibration in every probability bin. A high false-positive rate may persist even with calibrated probabilities and different risk distributions. This diagnostic does not decompose the age FPR gap into base rates, calibration, and discrimination, so it cannot prove that only one explains the disparity.

**H3 is not supported as an overprediction explanation for younger false alarms.** All four age groups have negative average calibration residuals. The younger-minus-older interval includes zero. Under-30 bins with at least 100 applicants also all have mean predictions below observed rates; small bins are too uncertain for strong conclusions. This does not establish that base rates alone explain the FPR gap.

## Limits and next decisions

Overall mean prediction 6.60%, observed difficulty 8.07%. Hypotheses were motivated by eyeball errors and evaluated using blackbox aggregates. The blackbox set already selected checkpoint and threshold; intervals condition on the fitted model and do not correct for model selection, repeated dev use, multiple hypotheses or possible applicant dependence. Treat the results as exploratory evidence, not independent confirmation or causal proof. No individual blackbox cases were displayed or added to manual review. No test predictions were loaded.

H1/H3 concern probability calibration, not accuracy or a decision cutoff. H2 concerns false alarms among actual negatives; context tables also show prevalence and calibration among all applicants. Any subsequent model change should be evaluated as a separate planned experiment, not selected to remove these particular errors.
