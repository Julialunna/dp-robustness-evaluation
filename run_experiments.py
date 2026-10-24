"""Baseline ou Experimento B: 3 partições x NUM_RUNS seeds.
Comando: python run_experiments.py --mode baseline
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import pandas as pd
import config

SEEDS=list(config.GRID_SEEDS[:config.NUM_RUNS])
PARTITIONS=[('iid',None,'IID'),('dirichlet',1.0,'Não-IID alpha 1.0'),('dirichlet',0.5,'Não-IID alpha 0.5')]
MODE='cvae_decoder_no_dp'
ROOT=Path(os.environ.get('FL_OUTPUT_ROOT', 'output')).resolve()
BASE=ROOT/'experiments'/f'experiment_B_{config.DATASET}_beta{config.CVAE_BETA}_sigma008_018_026'

METRICS = [
 'final_global_accuracy', 'acc_noise_008', 'acc_noise_018', 'acc_noise_026',
 'final_global_balanced_accuracy', 'balanced_acc_noise_008',
 'balanced_acc_noise_018', 'balanced_acc_noise_026',
 'final_global_f1_macro', 'f1_noise_008', 'f1_noise_018', 'f1_noise_026',
]
SUMMARY_COLUMNS = ['dataset', 'mode', 'partitioner', 'dirichlet_alpha'] + [
 f'{metric}_{stat}' for metric in METRICS for stat in ('mean', 'std')
]


def experiment_dir(mode):
 if mode == 'baseline':
  return ROOT/'experiments'/f'baseline_{config.DATASET}_sigma008_018_026'
 return BASE


def parse_args(argv=None):
 parser = argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--mode', choices=['baseline', MODE], default=MODE)
 return parser.parse_args(argv)


def aggregate_baseline(df):
 """Agrega apenas sucessos; métricas ausentes permanecem ausentes."""
 rows = []
 for partitioner, alpha, _ in PARTITIONS:
  mask = df['status'].eq('ok') & df['partitioner'].eq(partitioner)
  mask &= df['dirichlet_alpha'].isna() if alpha is None else df['dirichlet_alpha'].eq(alpha)
  valid = df.loc[mask]
  row = dict(dataset=config.DATASET, mode='baseline',
             partitioner=partitioner, dirichlet_alpha=alpha)
  for metric in METRICS:
   values = pd.to_numeric(valid.get(metric, pd.Series(dtype=float)), errors='coerce')
   row[f'{metric}_mean'] = values.mean()
   row[f'{metric}_std'] = values.std(ddof=1) if values.count() >= 2 else float('nan')
  rows.append(row)
 return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def get_specs(mode=MODE):
 result=[]
 for partitioner,alpha,label in PARTITIONS:
  for seed in SEEDS:
   key=f'{partitioner}_alpha{alpha if alpha is not None else "iid"}_seed{seed}'
   spec={'seed':int(seed),'dataset':config.DATASET,'extractor':config.EXTRACTOR_NAME,
         'mode':mode,'partitioner':partitioner,'dirichlet_alpha':alpha,'epsilon':None,
         'partition_label':label,'model_label':('Baseline DINOv2 + FedAvg' if mode == 'baseline' else 'B: decoder separado sem DP (beta=0.01)'),
         'beta':float(config.CVAE_BETA),'sigma_1':.08,'sigma_2':.18,'sigma_3':.26,
         'server_rounds':config.NUM_SERVER_ROUNDS, '_dir':experiment_dir(mode)/key}
   result.append(spec)
 return result

def load(spec):
 f=spec['_dir']/'summary.json'
 if not f.exists():return None
 try:
  d=json.loads(f.read_text())
  if not isinstance(d, dict):return None
  for key in ['seed','dataset','mode','partitioner','dirichlet_alpha','epsilon']:
   if key not in d or d[key]!=spec[key]:return None
  if 'server_rounds' in d and d['server_rounds'] != spec['server_rounds']:return None
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
 base = specs[0]['_dir'].parent
 if specs[0]['mode'] == 'baseline':
  temporary = base/'grid_summary.csv.tmp'
  aggregate_baseline(df).to_csv(temporary, index=False)
  temporary.replace(base/'grid_summary.csv')
 else:
  from result_reporting import publish, make_plots

  publish(df,base,len(specs))
  make_plots(df,base/'plots')
 return df

def main():
 args = parse_args()
 base = experiment_dir(args.mode)
 if not SEEDS or len(set(SEEDS))!=len(SEEDS) or len(SEEDS)!=config.NUM_RUNS:
  raise ValueError('NUM_RUNS e GRID_SEEDS inconsistentes')
 assert [round(config.EVAL_GAUSSIAN_NOISE_STD1,2),round(config.EVAL_GAUSSIAN_NOISE_STD2,2),round(config.EVAL_GAUSSIAN_NOISE_STD3,2)]==[.08,.18,.26]
 if args.mode == 'baseline':
  expected = {'DATASET':'pathmnist', 'EXTRACTOR_NAME':'dinov2_s',
              'NUM_PARTITIONS':10, 'NUM_SERVER_ROUNDS':20,
              'LOCAL_EPOCHS':1, 'BATCH_SIZE':256, 'NUM_RUNS':10}
  for name, value in expected.items():
   if getattr(config, name) != value:
    raise ValueError(f'Baseline requer config.{name}={value!r}')
 base.mkdir(parents=True,exist_ok=True)
 specs=get_specs(args.mode)
 print(f'{args.mode}: {len(specs)} runs ({len(SEEDS)} seeds x 3 partições): {SEEDS}',flush=True)
 write_state(specs)
 for n,s in enumerate(specs,1):
  prev=load(s)
  if prev and prev['status']=='ok':
   print(f'[{n}/{len(specs)}] {s["partition_label"]} seed={s["seed"]} já concluído',flush=True)
   continue
  folder=s['_dir'];folder.mkdir(parents=True,exist_ok=True)
  cmd=[sys.executable,'main.py','--dataset',config.DATASET,'--partitioner',s['partitioner'],
       '--dirichlet-alpha',str(s['dirichlet_alpha'] or 1.),'--mode',s['mode'],'--epsilon','0',
       '--seed',str(s['seed']),'--num-rounds',str(config.NUM_SERVER_ROUNDS),'--output-dir',str(folder)]
  print(f'[{n}/{len(specs)}] {s["partition_label"]} seed={s["seed"]}',flush=True)
  with (folder/'stdout.log').open('a') as output:
   finished=subprocess.run(cmd,stdout=output,stderr=subprocess.STDOUT,check=False)
  current = load(s)
  if current is None or (finished.returncode != 0 and current['status'] == 'ok'):
   failure={k:s[k] for k in ['seed','dataset','mode','partitioner','dirichlet_alpha','epsilon','server_rounds']}
   failure.update(status='erro', error=f'main.py terminou com {finished.returncode}; veja stdout.log')
   (folder/'summary.json').write_text(json.dumps(failure,indent=2))
  write_state(specs)
 df=write_state(specs)
 print(f'FIM: {sum(df.status.eq("ok"))}/{len(specs)} sucessos, {sum(df.status.eq("erro"))} falhas.',flush=True)
 if args.mode == 'baseline':
  print(f'CSV agregado: {base}/grid_summary.csv', flush=True)
 else:
  print(f'CSV simplificado: {ROOT}/grid_results.csv; gráficos em {base}/plots',flush=True)

if __name__=='__main__':main()
