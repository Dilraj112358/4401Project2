import ast,json,random,copy,math,time,gc,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch,joblib
from torch import nn
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import MissingIndicator,SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,QuantileTransformer
from sklearn.metrics import roc_auc_score,confusion_matrix,roc_curve
ROOT=Path.cwd();OUT=ROOT/'peer_review/20261014';OUT.mkdir(parents=True,exist_ok=True)
(OUT/'Protocol.md').write_text('''# Peer comparison protocol recorded before fitting
Compare two application-only, gender-excluded challengers with Gil's saved three-seed baseline on identical train/blackbox IDs: (1) peer quantile preprocessing with current 64/32 training; (2) peer engineered application features, preprocessing and regularized_all CreditNet training. Seeds42/43/44. No additional CSV predictors; no eyeball or test evaluation for candidate selection. Baseline audit rule: mean AUC gain>=.001 and wins in all three seeds; inspect F1 drops>.005, matched-FPR recall drops>1pp and group disparities before recommending replacement. This compares complete recipes, not causal attribution to any one component. Conditional on previously used blackbox; peer discovery is exploratory. Preserve current model/error outputs during comparison.
''')
torch.set_num_threads(1)
notebook=json.load(open('peer_review/reference_gil_project2.ipynb'));s={}
for i in [2,4,6,8,9,10]:exec(''.join(notebook['cells'][i]['source']),s)
raw=s['df'];Xtr=s['X_train'];Xva=s['X_val'];yt=s['y_train'];yv=s['y_val'];ids=s['applicant_ids'].loc[Xva.index].to_numpy()
y_train=yt.to_numpy(dtype=np.float32);y_val=yv.to_numpy();RANDOM_STATE=42;DEVICE=torch.device('cpu');MAX_EPOCHS=40;PATIENCE=7;BATCH_SIZE=2048;ARTIFACT_DIR=OUT
peer=json.load(open('peer_review/source_nn1.ipynb'))
for idx in [6,8,10]:
 source=''.join(peer['cells'][idx]['source'])
 if idx==6:source=source.replace("x['CODE_GENDER'] = x['CODE_GENDER'].replace('XNA', np.nan)","x = x.drop(columns=['CODE_GENDER'], errors='ignore')")
 tree=ast.parse(source);tree.body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef,ast.Import,ast.ImportFrom))];exec(compile(tree,'peer_definitions','exec'),globals())
def seed_everything(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
exec(Path('diagnostics/fairness_metrics_source.py').read_text(),globals())
def measures(p,t):
 labels=p>=t;tn,fp,fn,tp=confusion_matrix(yv,labels,labels=[0,1]).ravel();f=pd.DataFrame({'SK_ID_CURR':ids,'actual_TARGET':y_val,'predicted_probability':p});g,gaps=fairness_tables(f,raw,t);gap=gaps[gaps.metric.eq('false_positive_rate')].set_index('attribute').gap_pp
 return dict(auc=roc_auc_score(yv,p),f1=2*tp/(2*tp+fp+fn),recall=tp/(tp+fn),precision=tp/(tp+fp),fpr=fp/(fp+tn),gender_gap_pp=gap.CODE_GENDER,age_gap_pp=gap.age_group)
def cutoff(p):
 thresholds=np.arange(10,501)/1000;order=np.argsort(p);pp=p[order];yy=y_val[order];tp=np.r_[np.cumsum(yy[::-1])[::-1],0];at=np.searchsorted(pp,thresholds);f=2*tp[at]/(len(pp)-at+yy.sum());return float(thresholds[np.argmax(f)])
baseline=json.load(open('decision_audit/20261013T181905Z/results.json'))
print('Baseline record keys',baseline[0].keys(),flush=True)
results=[]
for variant in ['quantile_current','peer_recipe']:
 tr=raw.loc[Xtr.index].drop(columns=['TARGET','SK_ID_CURR','CODE_GENDER']);va=raw.loc[Xva.index].drop(columns=['TARGET','SK_ID_CURR','CODE_GENDER'])
 tr=engineer_features(tr,variant=='peer_recipe');va=engineer_features(va,variant=='peer_recipe')
 assert 'CODE_GENDER' not in tr
 bundle,a,b=fit_preprocessing(tr,va);joblib.dump(bundle,OUT/(variant+'_preprocessor.joblib'))
 for seed in [42,43,44]:
  if variant=='peer_recipe':
   config=dict(name=variant,feature_set='application_only',hidden=[128,64],dropout=.35,input_dropout=.08,weight_decay=.03)
   record=train_trial(config,a,b,seed);p=record.pop('val_pred');epoch=record['best_epoch']
  else:
   seed_everything(seed);model=nn.Sequential(nn.Linear(a.shape[1],64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1));opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001);loss=nn.BCEWithLogitsLoss();best=-1
   from torch.utils.data import DataLoader,TensorDataset
   loader=DataLoader(TensorDataset(torch.from_numpy(a),torch.tensor(y_train)),batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(seed));v=torch.from_numpy(b)
   for e in range(1,16):
    model.train()
    for xx,yy in loader:opt.zero_grad();l=loss(model(xx).squeeze(1),yy);l.backward();opt.step()
    model.eval()
    with torch.no_grad():pp=torch.sigmoid(model(v).squeeze(1)).numpy()
    auc=roc_auc_score(y_val,pp)
    if auc>best:best=auc;p=pp.copy();epoch=e;state=copy.deepcopy(model.state_dict())
   torch.save(state,OUT/f'{variant}_seed{seed}.pt')
  np.save(OUT/f'{variant}_seed{seed}_probabilities.npy',p)
  basep=np.load(f'decision_audit/20261013T181905Z/current/probabilities_{seed}.npy');bm=measures(basep,cutoff(basep));t=cutoff(p);m=measures(p,t)
  fpr,tpr,thresholds=roc_curve(y_val,p);ok=np.where(fpr<=bm['fpr'])[0];mt=float(thresholds[ok[-1]])
  row=dict(variant=variant,seed=seed,selected_epoch=epoch,threshold=t,metrics=m,auc_gain=m['auc']-bm['auc'],baseline=bm,matched_threshold=mt,matched=measures(p,mt));results.append(row);(OUT/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(row),flush=True)
