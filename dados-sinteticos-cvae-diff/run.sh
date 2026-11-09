#!/usr/bin/env bash
# =============================================================================
# run.sh: estágios do gerador de dados sintéticos (chamado pelo flautim.yaml; roda igual na máquina local).
#
#   bash run.sh check                 # 1-2 min: python, torch, GPU, arquivos do dataset, DINOv2 carrega
#   bash run.sh smoke                 # ~10 min: bloodmnist, 1000 linhas por split, 2 épocas; saída em ../output/smoke
#   bash run.sh treinar [flags...]    # treina os dois estágios (padrão: pathmnist)
#   bash run.sh gerar [flags...]      # gera N_SINT sintéticos com os modelos treinados
#   bash run.sh tudo [flags...]       # treinar e depois gerar
#
# Variáveis (com padrão):
#   SINT_DATA=dataset        pasta com <flag>_64.npz e metadata/ (no Flautim, o dataset montado)
#   SINT_OUTPUT=../output    saída (no Flautim, o volume /output)
#   AUTO_DOWNLOAD=1          se o .npz faltar, baixa do Zenodo (md5 conferido) para $SINT_OUTPUT/dados
#   BACKBONE=dinov2_vits14   o do dp_FL (torch.hub, precisa de internet); dinov2_vitb14 lê os pesos de
#                            $SINT_DATA/metadata/weights (sem internet)
#   N_SINT=10000             quantos sintéticos o estágio gerar produz
#   MODELOS=                 pasta com <flag>/cvae.pt e <flag>/difusao.pt para o gerar; vazio: usa
#                            $SINT_OUTPUT/modelos se existir, senão ./modelos (modelos copiados para o workspace)
#   TORCH_HOME=./cache/torch cache do torch.hub (o .flautimignore não o envia)
#   EXTRA_TREINO=""          argumentos extras para treinar.py, ex.: "--epocas 40 --cond-classe 1"
#   EXTRA_GERAR=""           argumentos extras para gerar.py, ex.: "--balanceado --passos 100"
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"

export SINT_DATA="${SINT_DATA:-dataset}"
export SINT_OUTPUT="${SINT_OUTPUT:-../output}"
export AUTO_DOWNLOAD="${AUTO_DOWNLOAD:-1}"
export BACKBONE="${BACKBONE:-dinov2_vits14}"
export N_SINT="${N_SINT:-10000}"
export TORCH_HOME="${TORCH_HOME:-$PWD/cache/torch}"
EXTRA_TREINO="${EXTRA_TREINO:-}"
EXTRA_GERAR="${EXTRA_GERAR:-}"
PY="${PY:-python3}"
mkdir -p "$SINT_OUTPUT" "$TORCH_HOME"
OUT="$(cd "$SINT_OUTPUT" && pwd)"
mkdir -p "$OUT/logs"
FALHAS=()

filtra () { grep --line-buffered -v -i -E "warning|xformers"; }

check () {
  "$PY" - <<'PYEOF' 2>&1 | filtra
import os, platform, sys, torch, numpy as np
print(f"  python  {platform.python_version()}  torch {torch.__version__}  cuda {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  GPU     {p.name}  {p.total_memory / 2**30:.1f} GiB")
d = os.environ["SINT_DATA"]
npz = sorted(f for f in os.listdir(d) if f.endswith(".npz")) if os.path.isdir(d) else []
print(f"  dados   {d}: {', '.join(npz) or 'nenhum .npz (o treino vai baixar do Zenodo)'}")
sys.path.insert(0, ".")
import extrator
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = extrator.construir(os.environ["BACKBONE"], d, dev)
print(f"  backbone {os.environ['BACKBONE']}: embedding de {extrator.extrair(m, np.zeros((2, 64, 64, 3), np.uint8), dev).shape[1]} números")
PYEOF
  [[ ${PIPESTATUS[0]} -eq 0 ]] || FALHAS+=("check")
}

treinar () {  # treinar <flag> [args...]
  local flag="$1"; shift
  # shellcheck disable=SC2086
  "$PY" -u treinar.py --flag "$flag" "$@" $EXTRA_TREINO 2>&1 | filtra | tee "$OUT/logs/${flag}${TAG:-}_treino.log"
  [[ ${PIPESTATUS[0]} -eq 0 ]] || { FALHAS+=("${flag}:treino"); return 1; }
}

gerar () {  # gerar <flag> <pasta de modelos> [args...]
  local flag="$1" modelos="$2"; shift 2
  # shellcheck disable=SC2086
  "$PY" -u gerar.py --modelos "$modelos/$flag" --flag "$flag" "$@" $EXTRA_GERAR 2>&1 | filtra \
      | tee "$OUT/logs/${flag}${TAG:-}_gerar.log"
  [[ ${PIPESTATUS[0]} -eq 0 ]] || { FALHAS+=("${flag}:gerar"); return 1; }
}

pasta_modelos () {  # onde estão os modelos de <flag> para o estágio gerar
  local flag="$1"
  if [[ -n "${MODELOS:-}" ]]; then echo "$MODELOS"
  elif [[ -f "$OUT/modelos/$flag/difusao.pt" ]]; then echo "$OUT/modelos"
  else echo "$PWD/modelos"; fi
}

ESTAGIO="${1:-tudo}"; shift || true
T0=$(date +%s)
case "$ESTAGIO" in
  check)   check ;;
  smoke)   TAG=_smoke treinar bloodmnist --limite 1000 --epocas 2 --epocas-cvae 3 --passos 20 --saida "$OUT/smoke" \
           && TAG=_smoke gerar bloodmnist "$OUT/smoke/modelos" --n 200 --passos 20 --saida "$OUT/smoke" ;;
  treinar) for f in ${@:-pathmnist}; do treinar "$f" --saida "$OUT"; done ;;
  gerar)   for f in ${@:-pathmnist}; do gerar "$f" "$(pasta_modelos "$f")" --saida "$OUT"; done ;;
  tudo)    for f in ${@:-pathmnist}; do treinar "$f" --saida "$OUT" && gerar "$f" "$OUT/modelos" --saida "$OUT"; done ;;
  *) echo "estágio desconhecido: $ESTAGIO (check | smoke | treinar [flags] | gerar [flags] | tudo [flags])"; exit 2 ;;
esac
echo "tempo total: $(( ($(date +%s) - T0) / 60 )) min; saída em $OUT"
if (( ${#FALHAS[@]} )); then echo "FALHARAM: ${FALHAS[*]} (ver $OUT/logs/)"; exit 1; fi
