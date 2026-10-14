# This code is embedded in both notebooks so no extra module upload is required.
def fairness_tables(predictions, attributes, cutoff):
    """Audit TARGET=1 difficulty; undefined rates stay NaN, never zero."""
    data = predictions.merge(attributes[['SK_ID_CURR', 'CODE_GENDER', 'DAYS_BIRTH']],
                             on='SK_ID_CURR', validate='one_to_one')
    data['age_group'] = pd.cut(-data.DAYS_BIRTH / 365.25, [0, 30, 45, 60, np.inf],
        right=False, labels=['Under 30', '30 to under 45', '45 to under 60', '60 and older'])
    data['label'] = data.predicted_probability.ge(cutoff)
    rows = []
    for attribute in ['CODE_GENDER', 'age_group']:
        for group, part in data.groupby(attribute, observed=True, dropna=False):
            actual = part.actual_TARGET.eq(1); label = part.label
            tp=int((actual & label).sum()); fn=int((actual & ~label).sum())
            fp=int((~actual & label).sum()); tn=int((~actual & ~label).sum())
            counts = {'false_positive_rate': (fp, fp+tn), 'recall': (tp, tp+fn),
                      'precision': (tp, tp+fp), 'npv': (tn, tn+fn),
                      'specificity': (tn, tn+fp), 'false_negative_rate': (fn, fn+tp),
                      'favorable_prediction_rate': (tn+fn, len(part)),
                      'difficulty_prevalence': (tp+fn, len(part))}
            row=dict(attribute=attribute, group=str(group), n=len(part),
                     positives=tp+fn, negatives=tn+fp, tp=tp, fn=fn, fp=fp, tn=tn)
            for metric, (success, total) in counts.items():
                rate=success/total if total else np.nan
                row[metric]=rate; row[metric+'_denominator']=total
                # Descriptive 95% Wilson interval; does not account for repeated dev selection.
                if total:
                    z=1.959963984540054; den=1+z*z/total
                    center=(rate+z*z/(2*total))/den
                    half=z*np.sqrt(rate*(1-rate)/total+z*z/(4*total*total))/den
                    row[metric+'_lower']=max(0.,center-half); row[metric+'_upper']=min(1.,center+half)
                else:
                    row[metric+'_lower']=np.nan; row[metric+'_upper']=np.nan
            row['small_group_warning']=min(tp+fn,tn+fp,tp+fp,tn+fn)<30
            rows.append(row)
    groups=pd.DataFrame(rows)
    gaps=[]
    for attribute in groups.attribute.unique():
        part=groups[groups.attribute.eq(attribute)]
        for metric in ['false_positive_rate','recall','precision','npv','favorable_prediction_rate','specificity']:
            eligible=part[part[metric+'_denominator'].ge(30)&part[metric].notna()]
            if len(eligible)>=2:
                lo=eligible.loc[eligible[metric].idxmin()]; hi=eligible.loc[eligible[metric].idxmax()]
                gaps.append(dict(attribute=attribute,metric=metric,eligible_groups=len(eligible),
                    excluded_groups='; '.join(part.loc[~part.index.isin(eligible.index),'group']),
                    lowest_group=lo['group'],highest_group=hi['group'],
                    minimum=lo[metric],maximum=hi[metric],gap_pp=100*(hi[metric]-lo[metric])))
    return groups, pd.DataFrame(gaps)
