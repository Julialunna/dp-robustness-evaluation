"""Persistência/plots da grade; não altera o treinamento."""
import os
from pathlib import Path
import numpy as np
import pandas as pd

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

def make_plots(df,out):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
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

COLUMNS = [
    "descricao_experimento",
    "acuracia_limpos", "acuracia_corrompidos",
    "f1_macro_limpos", "f1_macro_corrompidos",
    "acuracia_balanceada_limpos", "acuracia_balanceada_corrompidos",
]


def compact_results(df):
    """Uma linha por execução concluída e nível de corrupção; métricas em [0, 1]."""
    rows = []
    for _, run in df.loc[df["status"].eq("ok")].iterrows():
        mode = run["mode"]
        cvae = "sem CVAE" if mode == "baseline" else "com CVAE"
        dp = "com DP" if mode in {"cvae_dp", "cvae_decoder_dp"} else "sem DP"
        description = (
            f"{cvae}; {dp}; modo={mode}; dataset={run['dataset']}; "
            f"{run['partition_label']}; seed={run['seed']}"
        )
        if dp == "com DP":
            description += f"; epsilon={run.get('epsilon')}"
        for suffix, sigma in [("008", 0.08), ("018", 0.18), ("026", 0.26)]:
            rows.append({
                "descricao_experimento": f"{description}; ruído gaussiano σ={sigma}",
                "acuracia_limpos": run.get("final_global_accuracy"),
                "acuracia_corrompidos": run.get(f"acc_noise_{suffix}"),
                "f1_macro_limpos": run.get("final_global_f1_macro"),
                "f1_macro_corrompidos": run.get(f"f1_noise_{suffix}"),
                "acuracia_balanceada_limpos": run.get("final_global_balanced_accuracy"),
                "acuracia_balanceada_corrompidos": run.get(f"balanced_acc_noise_{suffix}"),
            })
    return pd.DataFrame(rows, columns=COLUMNS)


def publish(df, base, total):
    """Publica somente a descrição e as métricas solicitadas na grade."""
    base = Path(base)
    root = Path(os.environ.get("FL_OUTPUT_ROOT", "output")).resolve()
    compact = compact_results(df)
    atomic_csv(compact, base / "grid_results.csv")
    atomic_csv(compact, root / "grid_results.csv")
    return compact
