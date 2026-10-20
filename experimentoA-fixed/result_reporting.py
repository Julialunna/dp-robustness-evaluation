"""Persistência/plots da grade; não altera o treinamento."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config

METRICS = {
 'accuracy':['final_global_accuracy','acc_noise_008','acc_noise_018','acc_noise_026'],
 'balanced_accuracy':['final_global_balanced_accuracy','balanced_acc_noise_008','balanced_acc_noise_018','balanced_acc_noise_026'],
 'f1_macro':['final_global_f1_macro','f1_noise_008','f1_noise_018','f1_noise_026'],
}
NOISE = [0, .08, .18, .26]
GROUP = ['dataset','extractor','mode','partitioner','dirichlet_alpha','epsilon','partition_label','model_label']

def atomic_csv(df,filename):
 p=Path(filename);p.parent.mkdir(parents=True, exist_ok=True)
 temp=p.with_suffix(p.suffix+'.tmp');df.to_csv(temp,index=False);temp.replace(p)

def summary_table(df):
 valid=df.loc[df.status.eq('ok')].copy()
 if valid.empty:return pd.DataFrame(columns=GROUP+['n_runs','seeds'])
 cols=[v for v in [*sum(METRICS.values(),[]), 'retention_acc_008','retention_acc_018','retention_acc_026','retention_f1_008','retention_f1_018','retention_f1_026','final_validation_accuracy','best_validation_accuracy'] if v in valid]
 rows=[]
 for _,g in valid.groupby(GROUP,dropna=False):
  r={k:(None if pd.isna(g.iloc[0][k]) else g.iloc[0][k]) for k in GROUP}
  r['n_runs']=len(g);r['seeds']=','.join(str(v) for v in sorted(g.seed.astype(int)))
  for col in cols:
   v=pd.to_numeric(g[col],errors='coerce').dropna()
   r[col+'_mean']=v.mean() if len(v) else np.nan
   r[col+'_std']=v.std(ddof=1) if len(v)>1 else np.nan
   r[col+'_n']=len(v)
  rows.append(r)
 return pd.DataFrame(rows)

def make_plots(df,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 valid=df[df.status.eq('ok')]
 written=[]
 for label,g in valid.groupby('partition_label'):
  for typ in ['accuracy','balanced_accuracy','f1_macro','retention','drop']:
   columns=METRICS.get(typ,METRICS['accuracy'])
   clean=pd.to_numeric(g.final_global_accuracy,errors='coerce')
   vals=[]
   for i,col in enumerate(columns):
    v=pd.to_numeric(g[col],errors='coerce') if col in g else pd.Series(np.nan,index=g.index)
    if typ=='retention':v=v.div(clean).mul(100) if i else pd.Series(100.,index=g.index)
    elif typ=='drop':v=(clean-v).mul(100)
    else:v=v.mul(100)
    vals.append(v)
   means=[v.mean() for v in vals]
   if any(np.isnan(x) for x in means):
    print(f'[plots] ausência de colunas no {label}/{typ}; gráfico ignorado',flush=True);continue
   sd=[v.std(ddof=1) if v.count()>1 else 0 for v in vals]
   fig,ax=plt.subplots(figsize=(8,5.2),dpi=175)
   ax.errorbar(NOISE,means,yerr=sd,marker='o',capsize=4,linewidth=2,label=f'Experimento A (n={len(g)})')
   ax.set_xticks(NOISE)
   ax.set_xlabel('Desvio-padrão do ruído gaussiano (σ)')
   yl={'accuracy':'Acurácia (%)','balanced_accuracy':'Acurácia balanceada (%)','f1_macro':'F1-macro (%)', 'retention':'Retenção de acurácia (%)','drop':'Queda absoluta (p.p.)'}[typ]
   ax.set_ylabel(yl);ax.set_title(f'PathMNIST + DINOv2-S — {label}\n{yl} | β={config.CVAE_BETA} sem DP')
   ax.grid(alpha=.25);ax.legend();fig.tight_layout()
   name=f'{typ}_{label.lower().replace(" ","_").replace("=","").replace("α", "alpha")}.png'
   fig.savefig(out/name);plt.close(fig);written.append(name)
 return written

def publish(df,base,total):
 base=Path(base);root=Path('/output')
 atomic_csv(df,base/'grid_results.csv');atomic_csv(df,root/'grid_results.csv')
 agg=summary_table(df)
 atomic_csv(agg,base/'grid_summary.csv');atomic_csv(agg,root/'grid_summary.csv')
 ok=df[df.status.eq('ok')];err=df[df.status.eq('erro')]
 metrics={'total_runs':int(total),'completed_runs':int(len(ok)+len(err)),'successful_runs':int(len(ok)),
          'failed_runs':int(len(err)), 'pending_runs':int(total-len(ok)-len(err)), 'beta':float(config.CVAE_BETA)}
 if len(ok):
  vals=pd.to_numeric(ok.final_global_accuracy,errors='coerce').dropna()
  if len(vals):metrics['mean_accuracy']=float(vals.mean())
  if len(vals)>1:metrics['std_accuracy']=float(vals.std(ddof=1))
 p=root/'metrics.json';p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix('.json.tmp');tmp.write_text(json.dumps(metrics,indent=2),encoding='utf-8');tmp.replace(p)
 return agg
