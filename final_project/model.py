"""Final application-only neural network; adapted from Dilraj's nn1 and Gil's audit.
Gender excluded. All fitted transformations use training rows only.
"""
import copy,gc,hashlib,json,math,random,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib
from torch import nn
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import MissingIndicator,SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,confusion_matrix,average_precision_score,brier_score_loss,log_loss
RANDOM_STATE=42
DEVICE=torch.device('cpu')
MAX_EPOCHS=40
PATIENCE=7
BATCH_SIZE=2048
CONFIG=dict(name='final_application',feature_set='application_only',hidden=[128,64],dropout=.35,input_dropout=.08,weight_decay=.03)
def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

def safe_divide(numerator, denominator):
    return numerator.div(denominator.replace(0, np.nan))

def engineer_features(raw, engineered=True):
    x = raw.drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore').copy()
    employment_sentinel = x['DAYS_EMPLOYED'].eq(365243)
    x['DAYS_EMPLOYED'] = x['DAYS_EMPLOYED'].mask(employment_sentinel)
    x = x.drop(columns=['CODE_GENDER'], errors='ignore')
    if not engineered:
        return x.replace([np.inf, -np.inf], np.nan)
    extra = {}
    extra['EMPLOYMENT_UNKNOWN'] = employment_sentinel.astype(float)
    extra['APPLICATION_MISSING_COUNT'] = x.isna().sum(axis=1)
    extra['AGE_YEARS'] = -x['DAYS_BIRTH'] / 365.25
    extra['EMPLOYMENT_YEARS'] = -x['DAYS_EMPLOYED'] / 365.25
    ratios = {'CREDIT_TO_INCOME': ('AMT_CREDIT', 'AMT_INCOME_TOTAL'), 'ANNUITY_TO_INCOME': ('AMT_ANNUITY', 'AMT_INCOME_TOTAL'), 'CREDIT_TO_ANNUITY': ('AMT_CREDIT', 'AMT_ANNUITY'), 'ANNUITY_TO_CREDIT': ('AMT_ANNUITY', 'AMT_CREDIT'), 'GOODS_TO_CREDIT': ('AMT_GOODS_PRICE', 'AMT_CREDIT'), 'GOODS_TO_INCOME': ('AMT_GOODS_PRICE', 'AMT_INCOME_TOTAL'), 'INCOME_PER_PERSON': ('AMT_INCOME_TOTAL', 'CNT_FAM_MEMBERS'), 'CREDIT_PER_PERSON': ('AMT_CREDIT', 'CNT_FAM_MEMBERS'), 'ANNUITY_PER_PERSON': ('AMT_ANNUITY', 'CNT_FAM_MEMBERS'), 'CHILDREN_FRACTION': ('CNT_CHILDREN', 'CNT_FAM_MEMBERS'), 'EMPLOYMENT_TO_AGE': ('DAYS_EMPLOYED', 'DAYS_BIRTH'), 'REGISTRATION_TO_AGE': ('DAYS_REGISTRATION', 'DAYS_BIRTH'), 'ID_CHANGE_TO_AGE': ('DAYS_ID_PUBLISH', 'DAYS_BIRTH'), 'PHONE_CHANGE_TO_AGE': ('DAYS_LAST_PHONE_CHANGE', 'DAYS_BIRTH'), 'SOCIAL_DEFAULT_RATE_30': ('DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_30_CNT_SOCIAL_CIRCLE'), 'SOCIAL_DEFAULT_RATE_60': ('DEF_60_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE')}
    for name, (numerator, denominator) in ratios.items():
        extra[name] = safe_divide(x[numerator], x[denominator])
    extra['CREDIT_MINUS_GOODS'] = x['AMT_CREDIT'] - x['AMT_GOODS_PRICE']
    extra['CREDIT_MINUS_GOODS_FRACTION'] = safe_divide(extra['CREDIT_MINUS_GOODS'], x['AMT_CREDIT'])
    extra['INCOME_MINUS_ANNUITY'] = x['AMT_INCOME_TOTAL'] - x['AMT_ANNUITY']
    extra['ADULT_COUNT'] = (x['CNT_FAM_MEMBERS'] - x['CNT_CHILDREN']).clip(lower=1)
    extra['INCOME_PER_ADULT'] = safe_divide(x['AMT_INCOME_TOTAL'], extra['ADULT_COUNT'])
    extra['CAR_AGE_TO_AGE'] = safe_divide(x['OWN_CAR_AGE'], extra['AGE_YEARS'])
    extra['EMPLOYMENT_START_AGE'] = extra['AGE_YEARS'] - extra['EMPLOYMENT_YEARS']
    ext = x[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']]
    extra['EXT_MEAN'] = ext.mean(axis=1)
    extra['EXT_STD'] = ext.std(axis=1)
    extra['EXT_MIN'] = ext.min(axis=1)
    extra['EXT_MAX'] = ext.max(axis=1)
    extra['EXT_RANGE'] = extra['EXT_MAX'] - extra['EXT_MIN']
    extra['EXT_MISSING_COUNT'] = ext.isna().sum(axis=1)
    extra['EXT_PRODUCT'] = ext.prod(axis=1, min_count=3)
    for a, b in [(1, 2), (1, 3), (2, 3)]:
        extra[f'EXT_{a}_TIMES_{b}'] = x[f'EXT_SOURCE_{a}'] * x[f'EXT_SOURCE_{b}']
    extra['EXT_MEAN_TIMES_AGE'] = extra['EXT_MEAN'] * extra['AGE_YEARS']
    extra['EXT_MEAN_TO_CREDIT_INCOME'] = safe_divide(extra['EXT_MEAN'], extra['CREDIT_TO_INCOME'])
    extra['EXT_MEAN_TIMES_ANNUITY_CREDIT'] = extra['EXT_MEAN'] * extra['ANNUITY_TO_CREDIT']
    documents = [c for c in x if c.startswith('FLAG_DOCUMENT_')]
    bureau = [c for c in x if c.startswith('AMT_REQ_CREDIT_BUREAU_')]
    housing = [c for c in x if c.endswith(('_AVG', '_MODE', '_MEDI')) and pd.api.types.is_numeric_dtype(x[c])]
    contacts = ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL']
    addresses = [c for c in x if '_NOT_' in c]
    extra['DOCUMENT_COUNT'] = x[documents].sum(axis=1)
    extra['CONTACT_COUNT'] = x[contacts].sum(axis=1)
    extra['ADDRESS_MISMATCH_COUNT'] = x[addresses].sum(axis=1)
    extra['HOUSING_MISSING_COUNT'] = x[housing].isna().sum(axis=1)
    extra['BUREAU_ACTIVITY_INDEX'] = x[bureau].sum(axis=1, min_count=1)
    extra['BUREAU_MISSING_COUNT'] = x[bureau].isna().sum(axis=1)
    extra['APPLICATION_HOUR_SIN'] = np.sin(2 * np.pi * x['HOUR_APPR_PROCESS_START'] / 24)
    extra['APPLICATION_HOUR_COS'] = np.cos(2 * np.pi * x['HOUR_APPR_PROCESS_START'] / 24)
    return pd.concat([x, pd.DataFrame(extra, index=x.index)], axis=1).replace([np.inf, -np.inf], np.nan)

def fit_preprocessing(train_frame, validation_frame):
    numeric = train_frame.select_dtypes(include=['number', 'bool']).columns.tolist()
    categorical = train_frame.select_dtypes(include=['object', 'category']).columns.tolist()
    continuous = [c for c in numeric if train_frame[c].nunique() > 2]
    binary = [c for c in numeric if c not in continuous]
    continuous_pipe = Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)), ('quantile', QuantileTransformer(n_quantiles=512, output_distribution='normal', subsample=100000, random_state=RANDOM_STATE))])
    category_pipe = Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='Missing')), ('encode', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=50, sparse_output=False, dtype=np.float32))])
    prep = ColumnTransformer([('continuous', continuous_pipe, continuous), ('binary', SimpleImputer(strategy='most_frequent', keep_empty_features=True), binary), ('missing', MissingIndicator(features='all', error_on_new=False), numeric), ('categorical', category_pipe, categorical)], sparse_threshold=0, n_jobs=1)
    a = np.clip(prep.fit_transform(train_frame), -4, 4).astype(np.float32)
    b = np.clip(prep.transform(validation_frame), -4, 4).astype(np.float32)
    names = prep.get_feature_names_out()
    variance = VarianceThreshold(threshold=0.0)
    a = variance.fit_transform(a)
    b = variance.transform(b)
    names = names[variance.get_support()]
    import hashlib
    groups, unique = ({}, [])
    for j in range(a.shape[1]):
        key = hashlib.sha256(a[:, j].tobytes()).digest()
        prior = groups.setdefault(key, [])
        if not any((np.array_equal(a[:, j], a[:, k]) for k in prior)):
            unique.append(j)
            prior.append(j)
    unique = np.asarray(unique)
    bundle = {'preprocessor': prep, 'variance': variance, 'unique': unique, 'feature_names': names[unique]}
    a, b = (np.ascontiguousarray(a[:, unique]), np.ascontiguousarray(b[:, unique]))
    assert np.isfinite(a).all() and np.isfinite(b).all()
    return (bundle, a, b)

class CreditNet(nn.Module):

    def __init__(self, input_dim, hidden=(128, 64), dropout=0.2, input_dropout=0.03):
        super().__init__()
        self.input_dropout = nn.Dropout(input_dropout)
        layers, width = ([], input_dim)
        for next_width in hidden:
            layers.extend([nn.Linear(width, next_width), nn.BatchNorm1d(next_width), nn.SiLU(), nn.Dropout(dropout)])
            width = next_width
        layers.append(nn.Linear(width, 1))
        self.deep = nn.Sequential(*layers)
        self.linear = nn.Linear(input_dim, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return (self.deep(self.input_dropout(x)) + self.linear(x)).squeeze(1)

def make_model(config, input_dim):
    return CreditNet(input_dim, tuple(config['hidden']), config['dropout'], config.get('input_dropout', 0.03)).to(DEVICE)

@torch.no_grad()
def predict_network(model, x, batch_size=8192):
    model.eval()
    values = []
    for start in range(0, len(x), batch_size):
        inputs = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32, device=DEVICE)
        values.append(torch.sigmoid(model(inputs)).cpu().numpy())
    return np.concatenate(values)

def train_trial(config, x_train, x_validation, seed=42):
    seed_everything(seed)
    start = time.perf_counter()
    model = make_model(config, x_train.shape[1])
    prior = float(y_train.mean())
    nn.init.constant_(model.linear.bias, math.log(prior / (1 - prior)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('lr', 0.001), weight_decay=config.get('weight_decay', 0.01))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=2e-05)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config.get('pos_weight', 1.0), device=DEVICE))
    xt = torch.as_tensor(np.ascontiguousarray(x_train), dtype=torch.float32, device=DEVICE)
    yt = torch.as_tensor(y_train, dtype=torch.float32, device=DEVICE)
    best_auc, best_epoch, best_state, stale = (-np.inf, 0, None, 0)
    history = []
    for epoch in range(1, config.get('max_epochs', MAX_EPOCHS) + 1):
        model.train()
        order = torch.randperm(len(xt), device=DEVICE)
        loss_sum = 0.0
        for indices in order.split(BATCH_SIZE):
            if len(indices) < 2:
                continue
            xb, yb = (xt[indices], yt[indices])
            noise = config.get('noise', 0.02)
            if noise:
                xb = xb + noise * torch.randn_like(xb)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
        pred = predict_network(model, x_validation)
        auc = roc_auc_score(y_val, pred)
        history.append({'epoch': epoch, 'train_loss': loss_sum / len(xt), 'val_auc': auc, 'lr': optimizer.param_groups[0]['lr']})
        scheduler.step(auc)
        material_improvement = auc > best_auc + 1e-05
        if auc > best_auc:
            best_auc, best_epoch = (auc, epoch)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        stale = 0 if material_improvement else stale + 1
        if epoch == 1 or epoch % 5 == 0:
            print(f"{config['name']} seed={seed} epoch={epoch:02d}: val AUC={auc:.6f}, best={best_auc:.6f}", flush=True)
        if stale >= PATIENCE:
            break
    model.load_state_dict(best_state)
    val_pred = predict_network(model, x_validation)
    elapsed = time.perf_counter() - start
    checkpoint_name = f"{config['name']}_seed{seed}.pt"
    torch.save({'state_dict': best_state, 'config': config, 'seed': seed, 'input_dim': x_train.shape[1], 'best_epoch': best_epoch}, ARTIFACT_DIR / checkpoint_name)
    record = {'name': config['name'], 'feature_set': config['feature_set'], 'seed': seed, 'val_auc': float(roc_auc_score(y_val, val_pred)), 'best_epoch': best_epoch, 'epochs_run': epoch, 'seconds': round(elapsed, 1), 'checkpoint': checkpoint_name, 'config': config.copy(), 'history': history, 'val_pred': val_pred}
    print(f"Finished {config['name']}: AUC={record['val_auc']:.6f}, epoch={best_epoch}, {elapsed:.1f}s", flush=True)
    del model, xt, yt, optimizer
    gc.collect()
    return record

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

def load_data(path='application_train.csv'):
    df=pd.read_csv(path)
    assert df.SK_ID_CURR.is_unique and set(df.TARGET.unique())=={0,1}
    return df

def main_partitions(df):
    train,test=train_test_split(df.index,test_size=.20,random_state=42,stratify=df.TARGET)
    train,dev=train_test_split(train,test_size=.25,random_state=42,stratify=df.loc[train,'TARGET'])
    eye_ids=set(json.loads(Path('final_project/selected/dev_split_manifest.json').read_text())['eyeball_ids'])
    eye=dev[df.loc[dev,'SK_ID_CURR'].isin(eye_ids).to_numpy()]
    black=dev[~df.loc[dev,'SK_ID_CURR'].isin(eye_ids).to_numpy()]
    parts={'train':train,'blackbox':black,'eyeball':eye,'test':test}
    assert [len(x) for x in parts.values()]==[184506,61002,500,61503]
    assert set(df.loc[eye,'SK_ID_CURR'])==eye_ids
    assert sum(map(len,parts.values()))==len(set().union(*map(set,parts.values())))==len(df)
    return parts

def transform(bundle,raw):
    frame=engineer_features(raw)
    assert 'CODE_GENDER' not in frame and 'TARGET' not in frame and 'SK_ID_CURR' not in frame
    a=np.clip(bundle['preprocessor'].transform(frame),-4,4).astype(np.float32)
    a=bundle['variance'].transform(a)[:,bundle['unique']]
    assert np.isfinite(a).all()
    return np.ascontiguousarray(a)

def select_threshold(y,p):
    thresholds=np.arange(10,501)/1000;order=np.argsort(p);pp=p[order];yy=np.asarray(y)[order]
    tp=np.r_[np.cumsum(yy[::-1])[::-1],0];at=np.searchsorted(pp,thresholds)
    score=2*tp[at]/(len(pp)-at+yy.sum())
    return float(thresholds[np.argmax(score)])

def metrics(y,p,threshold):
    tn,fp,fn,tp=confusion_matrix(y,p>=threshold,labels=[0,1]).ravel()
    return dict(n=len(y),prevalence=float(np.mean(y)),auc=roc_auc_score(y,p),average_precision=average_precision_score(y,p),accuracy=(tp+tn)/len(y),precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn),f1=2*tp/(2*tp+fp+fn),false_positive_rate=fp/(fp+tn),brier=brier_score_loss(y,p),log_loss=log_loss(y,p),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))

def fit_recipe(df,train,blackbox,folder,seed=42):
    global y_train,y_val,ARTIFACT_DIR
    ARTIFACT_DIR=Path(folder);ARTIFACT_DIR.mkdir(parents=True,exist_ok=True)
    y_train=df.loc[train,'TARGET'].to_numpy(dtype=np.float32);y_val=df.loc[blackbox,'TARGET'].to_numpy()
    bundle,a,b=fit_preprocessing(engineer_features(df.loc[train]),engineer_features(df.loc[blackbox]))
    record=train_trial(CONFIG,a,b,seed=seed);prob=record.pop('val_pred')
    joblib.dump(bundle,ARTIFACT_DIR/'preprocessor.joblib')
    (ARTIFACT_DIR/'training_record.json').write_text(json.dumps(record,indent=2))
    return bundle,ARTIFACT_DIR/record['checkpoint'],select_threshold(y_val,prob)

def evaluate_frozen(df,parts,folder,checkpoint_path,threshold):
    folder=Path(folder);bundle=joblib.load(folder/'preprocessor.joblib')
    ck=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    model=make_model(ck['config'],ck['input_dim']);model.load_state_dict(ck['state_dict'])
    rows=[]
    for partition,idx in parts.items():
        p=predict_network(model,transform(bundle,df.loc[idx]));y=df.loc[idx,'TARGET'].to_numpy()
        pred=pd.DataFrame({'SK_ID_CURR':df.loc[idx,'SK_ID_CURR'].to_numpy(),'actual_TARGET':y,'predicted_probability':p,'predicted_label':(p>=threshold).astype(int)})
        if partition!='train':
            pred.to_json(folder/(partition+'_predictions.json'),orient='records',indent=2)
            groups,gaps=fairness_tables(pred,df,threshold)
            groups.to_json(folder/(partition+'_fairness.json'),orient='records',indent=2)
            gaps.to_json(folder/(partition+'_fairness_gaps.json'),orient='records',indent=2)
        rows.append(dict(partition=partition,threshold=threshold,selected_epoch=ck['best_epoch'],**metrics(y,p,threshold)))
    result=pd.DataFrame(rows);result.to_json(folder/'metrics.json',orient='records',indent=2)
    meta=dict(model_id='final_application_peer_seed42',threshold=threshold,seed=42,selected_epoch=ck['best_epoch'],config=ck['config'],input_dim=ck['input_dim'],excluded_predictors=['CODE_GENDER'],age_retained=True,selection_partition='blackbox',historical_exposure=True)
    (folder/'model_info.json').write_text(json.dumps(meta,indent=2))
    return result

def run_main(retrain=False):
    torch.set_num_threads(1);df=load_data();parts=main_partitions(df);folder=Path('final_project/selected')
    if retrain:
        # Separate outputs preserve the shipped checkpoint and its matching error review.
        target=Path('final_project/retrained');_,ck,t=fit_recipe(df,parts['train'],parts['blackbox'],target)
        return evaluate_frozen(df,parts,target,ck,t)
    result=evaluate_frozen(df,parts,folder,folder/'checkpoint.pt',.167)
    expected=np.load('peer_review/20261014/peer_recipe_seed42_probabilities.npy') if Path('peer_review/20261014/peer_recipe_seed42_probabilities.npy').exists() else None
    if expected is not None:
        observed=pd.read_json(folder/'blackbox_predictions.json').predicted_probability.to_numpy()
        assert np.allclose(observed,expected,atol=1e-7)
    return result

def run_alternative_splits():
    torch.set_num_threads(1);df=load_data();parts=main_partitions(df)
    eligible=df.index[~df.index.isin(parts['eyeball'])]
    tr,te=train_test_split(eligible,test_size=61503,random_state=42)
    tr,va=train_test_split(tr,test_size=61002,random_state=42)
    previous=pd.read_csv('previous_application.csv',usecols=['SK_ID_CURR','SK_ID_PREV','DAYS_DECISION'])
    assert previous.SK_ID_PREV.is_unique
    known_ids=set(previous.loc[previous.DAYS_DECISION.lt(0),'SK_ID_CURR'])
    has=df.loc[eligible,'SK_ID_CURR'].isin(known_ids).to_numpy();known=eligible[has];new=eligible[~has]
    ptr,pva=train_test_split(known,test_size=.20,random_state=42,stratify=df.loc[known,'TARGET'])
    all_results=[pd.read_json('final_project/selected/metrics.json').assign(protocol='stratified')]
    for name,partition in [('random',{'train':tr,'blackbox':va,'test':te}),('no_previous_application',{'train':ptr,'blackbox':pva,'test':new})]:
        assert sum(map(len,partition.values()))==len(set().union(*map(set,partition.values())))==len(eligible)
        assert not set().union(*map(set,partition.values()))&set(parts['eyeball'])
        folder=Path('final_project/splits')/name
        _,ck,t=fit_recipe(df,partition['train'],partition['blackbox'],folder)
        # Freeze this protocol's dev-selected settings before any test inference.
        (folder/'selection.json').write_text(json.dumps({'threshold':t,'test_used_for_selection':False,'split_ids':{k:df.loc[v,'SK_ID_CURR'].tolist() for k,v in partition.items()}},indent=2))
        all_results.append(evaluate_frozen(df,partition,folder,ck,t).assign(protocol=name))
    comparison=pd.concat(all_results,ignore_index=True);comparison.to_json('final_project/split_comparison.json',orient='records',indent=2)
    return comparison
