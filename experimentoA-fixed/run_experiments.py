"""Experimento A: 3 partições x NUM_RUNS seeds, com resultados incrementais.
Comando: python run_experiments.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path
import pandas as pd
import config
from result_reporting import publish, make_plots

SEEDS=list(config.GRID_SEEDS[:config.NUM_RUNS])
PARTITIONS=[('iid',None,'IID'),('dirichlet',1.0,'Não-IID alpha 1.0'),('dirichlet',0.5,'Não-IID alpha 0.5')]
MODE='cvae_decoder_no_dp'
ROOT=Path('/output')
BASE=ROOT/'experiments'/f'experiment_A_{config.DATASET}_beta{config.CVAE_BETA}_sigma008_018_026'

def get_specs():
 result=[]
 for partitioner,alpha,label in PARTITIONS:
  for seed in SEEDS:
   key=f'{partitioner}_alpha{alpha if alpha is not None else "iid"}_seed{seed}'
   spec={'seed':int(seed),'dataset':config.DATASET,'extractor':config.EXTRACTOR_NAME,
         'mode':MODE,'partitioner':partitioner,'dirichlet_alpha':alpha,'epsilon':None,
         'partition_label':label,'model_label':'A: decoder separado sem DP (beta=0.1)',
         'beta':float(config.CVAE_BETA),'sigma_1':.08,'sigma_2':.18,'sigma_3':.26,
         '_dir':BASE/key}
   result.append(spec)
 return result

def load(spec):
 f=spec['_dir']/'summary.json'
 if not f.exists():return None
 try:
  d=json.loads(f.read_text())
  for key in ['seed','dataset','mode','partitioner','dirichlet_alpha','epsilon']:
   if d.get(key)!=spec[key]:return None
  if d.get('status') not in ('ok','erro'):return None
  return d
 except (OSError,ValueError):return None

def dataframe(specs):
 rows=[]
 for s in specs:
  d=load(s)
  row={k:v for k,v in s.items() if k!='_dir'}
  if d:row.update(d)
  else:row['status']='pendente'
  rows.append(row)
 return pd.DataFrame(rows)

def write_state(specs):
 df=dataframe(specs)
 publish(df,BASE,len(specs))
 make_plots(df,BASE/'plots')
 return df

def main():
 if not SEEDS or len(set(SEEDS))!=len(SEEDS) or len(SEEDS)!=config.NUM_RUNS:
  raise ValueError('NUM_RUNS e GRID_SEEDS inconsistentes')
 assert [round(config.EVAL_GAUSSIAN_NOISE_STD1,2),round(config.EVAL_GAUSSIAN_NOISE_STD2,2),round(config.EVAL_GAUSSIAN_NOISE_STD3,2)]==[.08,.18,.26]
 BASE.mkdir(parents=True,exist_ok=True)
 specs=get_specs()
 print(f'EXPERIMENTO A: {len(specs)} runs ({len(SEEDS)} seeds x 3 partições): {SEEDS}',flush=True)
 write_state(specs)
 for n,s in enumerate(specs,1):
  prev=load(s)
  if prev and prev['status']=='ok':
   print(f'[{n}/{len(specs)}] {s["partition_label"]} seed={s["seed"]} já concluído',flush=True)
   continue
  folder=s['_dir'];folder.mkdir(parents=True,exist_ok=True)
  cmd=[sys.executable,'main.py','--dataset',config.DATASET,'--partitioner',s['partitioner'],
       '--dirichlet-alpha',str(s['dirichlet_alpha'] or 1.),'--mode',MODE,'--epsilon','0',
       '--seed',str(s['seed']),'--num-rounds',str(config.NUM_SERVER_ROUNDS),'--output-dir',str(folder)]
  print(f'[{n}/{len(specs)}] {s["partition_label"]} seed={s["seed"]}',flush=True)
  with (folder/'stdout.log').open('w') as output:
   finished=subprocess.run(cmd,stdout=output,stderr=subprocess.STDOUT,check=False)
  if not (folder/'summary.json').exists():
   failure={k:s[k] for k in ['seed','dataset','mode','partitioner','dirichlet_alpha','epsilon']}
   failure.update(status='erro', error=f'main.py terminou com {finished.returncode}; veja stdout.log')
   (folder/'summary.json').write_text(json.dumps(failure,indent=2))
  write_state(specs)
 df=write_state(specs)
 print(f'FIM: {sum(df.status.eq("ok"))}/{len(specs)} sucessos, {sum(df.status.eq("erro"))} falhas.',flush=True)
 print(f'CSV completos: {ROOT}/grid_results.csv e {ROOT}/grid_summary.csv; gráficos em {BASE}/plots',flush=True)

if __name__=='__main__':main()
