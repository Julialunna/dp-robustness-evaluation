# Gerador de dados sintéticos (MedMNIST): CVAE de embeddings + difusão

Este código treina um gerador de imagens sintéticas e gera um conjunto sintético rotulado. Ele **só treina e gera**:
não tem privacidade diferencial, aprendizado federado nem avaliação dos sintéticos. É um ponto de partida para vocês
explorarem.

## A ideia

O gerador tem dois estágios:

```
                       estágio 1 (cvae.py)                      estágio 2 (difusao.py)
classe y ──► z ~ N(0, I) ──► CVAE ──► embedding DINOv2 (384 números) ──► difusão ──► imagem 64x64
```

- **Estágio 1: CVAE.** Aprende a distribuição dos embeddings DINOv2 de cada classe. É o mesmo CVAE do dp_FL
  (`dp_cvae.py`), com uma mudança: os embeddings são padronizados antes do treino (z-score por dimensão).
- **Estágio 2: difusão.** Uma U-Net aprende a desenhar uma imagem a partir de um embedding: p(imagem | embedding).
  O embedding de entrada pode vir de uma imagem real ou do CVAE.
  - **Por que difusão:** um decodificador determinístico (embedding → imagem, treinado com erro de pixel) devolve a
    média das imagens possíveis, e a imagem sai borrada. A difusão sorteia uma imagem compatível com o embedding, e o
    detalhe fino sai nítido.
- **Para gerar:** escolha a classe, sorteie um embedding com o CVAE e desenhe a imagem com a difusão.

Os embeddings vêm do DINOv2 ViT-S/14 congelado, com o pré-processamento do dp_FL (`extractor.py`): a imagem é
ampliada para 224 e a saída é o token CLS.

## Arquivos

| arquivo | o que faz |
|---|---|
| `treinar.py` | lê o dataset, extrai os embeddings (com cache), treina o CVAE e a difusão, salva os modelos |
| `gerar.py` | carrega os modelos e gera N imagens sintéticas com rótulos |
| `cvae.py` | estágio 1: o CVAE |
| `difusao.py` | estágio 2: a U-Net, o treino e a amostragem |
| `extrator.py` | embeddings DINOv2 |
| `dados.py` | lê o `<flag>_64.npz` do MedMNIST; se faltar, baixa do Zenodo com md5 conferido |
| `run.sh` | os estágios (`check`, `smoke`, `treinar`, `gerar`, `tudo`); o Flautim chama este script |
| `flautim.yaml`, `.flautimignore`, `requirements.txt` | os jobs do Flautim |
| `metadata/` | nomes das classes e URLs dos 8 MedMNIST (o código roda mesmo sem o dataset montado) |
| `modelos/` | vazia; coloque aqui modelos já treinados para só gerar (ver abaixo) |

## Como rodar no Flautim

Abra a pasta na extensão do Flautim e lance os jobs nesta ordem:

| job | o que faz | tempo |
|---|---|---|
| `sint-check` | confere Python, torch, GPU, os arquivos do dataset e se o DINOv2 carrega | 1–2 min |
| `sint-smoke` | treino e geração em miniatura (BloodMNIST, 1000 imagens, 2 épocas); a saída vai para `smoke/` | ~10 min (estimativa) |
| `sint-path` | PathMNIST completo: embeddings, CVAE, difusão e 10 000 sintéticos | ~3 h (estimativa) |
| `sint-gerar-path` | só gera, com modelos copiados para `modelos/pathmnist/` | ~30 min |

As imagens do `sint-smoke` saem como ruído, o que é esperado com 2 épocas. Ele só confere que tudo roda até o fim.

Para outro dataset, troque `pathmnist` no comando do job. As opções são `bloodmnist`, `dermamnist`, `octmnist`,
`tissuemnist`, `organamnist`, `organcmnist` e `organsmnist`.

Para mudar hiperparâmetros sem editar o código, descomente `EXTRA_TREINO` ou `EXTRA_GERAR` no `flautim.yaml`.

## Como rodar na sua máquina

```bash
pip install -r requirements.txt
# teste rápido, roda em CPU (~1 min):
python treinar.py --flag bloodmnist --data /caminho/do/dataset --saida ../output \
       --limite 300 --epocas 1 --epocas-cvae 2 --largura 32 --mult 1,1,1,1 --blocos 1 --passos 4
python gerar.py --modelos ../output/modelos/bloodmnist --n 100 --passos 4 --saida ../output
# a sério (precisa de GPU):
python treinar.py --flag pathmnist --data /caminho/do/dataset --saida ../output
python gerar.py --modelos ../output/modelos/pathmnist --saida ../output
```

- `--data` é a pasta com `pathmnist_64.npz` etc. Se o arquivo não estiver lá, o código baixa do Zenodo para
  `<saida>/dados/`.
- O DINOv2 ViT-S/14 vem do `torch.hub`, por isso a primeira execução precisa de internet. Sem internet, use
  `--backbone dinov2_vitb14`: ele lê os pesos de `<data>/metadata/weights/` e gera embeddings de 768 números.

## O que sai

```
<saida>/
  modelos/<flag>/
    cvae.pt, difusao.pt          os dois modelos
    historico_cvae.csv           perda por época, no treino e na validação
    historico_difusao.csv
    amostras.png                 8 imagens geradas por classe no fim do treino: olhe esta figura primeiro
    config.json                  argumentos, tempos, versões
  sinteticos/
    sinteticos_<flag>.npz        o conjunto sintético
    amostras_<flag>.png
  emb/                           cache dos embeddings (reaproveitado se rodar de novo)
  logs/
```

A difusão é salva no fim de cada época. Se o job cair no meio, o `difusao.pt` tem a última época completa.

**Como ler os sintéticos:**

```python
import numpy as np
z = np.load("sinteticos_pathmnist.npz")
X = z["images"]       # (N, 64, 64, 3) uint8, como o train_images do MedMNIST (sem o canal nos datasets em cinza)
y = z["labels"]       # (N,) int64
E = z["embeddings"]   # (N, 384) float32: o embedding DINOv2 que o CVAE sorteou para cada imagem
nomes = z["classes"]  # nomes das classes
```

Por padrão as classes saem na proporção do treino. Com `--balanceado`, saem com o mesmo número por classe.

## Parâmetros principais

| parâmetro | padrão | efeito |
|---|---|---|
| `--epocas-cvae` | 30 | épocas do CVAE (rápido: ~1 s por época) |
| `--latente`, `--oculta`, `--beta` | 32, 256, 1,0 | tamanho do CVAE e peso do termo KL |
| `--epocas` | 20 | épocas da difusão; é aqui que vai o tempo |
| `--largura`, `--mult`, `--blocos` | 64, `1,2,3,4`, 2 | tamanho da U-Net (23 M parâmetros no padrão) |
| `--cond-classe` | 0 | 1 condiciona a difusão também na classe, além do embedding |
| `--passos` | 50 | passos de amostragem: menos é mais rápido, com imagens piores |
| `--guia` | 2,0 | o quanto a imagem obedece ao embedding (1 = sem guia; valores maiores reforçam o embedding e reduzem a variedade) |
| `--n` (gerar) | 10 000 | quantos sintéticos gerar |
| `--limite` | 0 | usa só esse número de imagens por split; serve para testes |

## Tempo e memória

Os tempos medidos vêm de um job anterior com a mesma difusão, rodado no perfil `medio` do Flautim (MIG 1g.18gb):

- **Embeddings do PathMNIST:** ~2,5 min, só na primeira vez (depois vêm do cache).
- **Difusão:** ~197 s por época em 45 mil imagens, ou seja, ~6,5 min por época nas 90 mil do treino. As 20 épocas
  padrão levam ~2,2 h.
- **Geração:** ~6 imagens por segundo, ou seja, ~27 min para 10 000.
- **CPU:** só para testes. Com o tamanho padrão, gerar 36 imagens levou ~5 min numa CPU de 16 núcleos.
- **Memória:** num teste em CPU, `--mult 1,2` estourou a memória. Com essa opção, a atenção do meio da U-Net fica
  em 32×32. Para testes pequenos, use `--mult 1,1,1,1 --blocos 1`.

## O que este código não faz

- **Não avalia os sintéticos.** A avaliação usada no projeto está em `../cvae_embeddings/gerador_imagens.py`:
  - um classificador fixo nas imagens geradas;
  - TSTR (treinar em sintéticos, testar em reais);
  - Fréchet e precision/recall/density/coverage;
  - memorização.
- **Sem DP e sem divisão privada/pública.** Os dois estágios treinam no treino inteiro.
- **64×64.** `--size 128` funciona com o `.npz` 128 do Zenodo, mas a difusão fica ~4× mais cara.
- **Embeddings diferentes dos do dp_FL.** O dp_FL extrai os embeddings das imagens 224 do Hugging Face. Aqui eles
  vêm das imagens 64 ampliadas para 224, então os valores não são os mesmos.

## Referência: o que esperar

Rode o código antes de ler esta seção.

A mesma arquitetura já rodou uma vez no PathMNIST (job `cvae-ger-path`, 2026-11-04). Naquela execução, o CVAE treinou
numa metade do treino e a difusão na outra, com 40 épocas. Uma ResNet-18 treinada em imagens reais avaliou o
resultado:

- **Reconstruções** (difusão a partir do embedding de uma imagem real da validação): acurácia balanceada **0,93**.
- **Sintéticos:** **0,87**. As classes mais difíceis foram debris e estroma.
- **TSTR, ResNet treinada em 10 000 sintéticos:**
  - **0,75** no teste real, contra 0,68 com 10 000 reais;
  - **0,82** na validação, contra 0,86 com 10 000 reais.
- **Ressalvas:** foi uma semente só, e a base real estava instável. Na mesma origem dos dados, os sintéticos treinam
  quase tão bem quanto o real. Não está estabelecido que treinem melhor.

Os modelos daquele job leem direto no `gerar.py`:

```bash
python gerar.py --cvae caminho/cvae_privado.pt --difusao caminho/difusao.pt --flag pathmnist
```
