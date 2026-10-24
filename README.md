# Libras - reconhecimento de alfabeto (estatico) e palavras (dinamico)

Projeto de TCC: reconhecimento de sinais de Libras a partir de landmarks de
maos, rosto (e, no alfabeto, so maos) extraidos com MediaPipe, classificados
por uma CNN (alfabeto) e uma LSTM (palavras).

## Estrutura

```
libras/                                            modulo comum (NAO duplicar funcoes daqui em outro lugar)
  landmarks.py                                      detectores MediaPipe (mao, rosto, pose), normalizacao da mao, amostragem de frames, leitura de zip/CSV
  features.py                                       representacao dos sinais: face normalizada, trajetoria do punho, velocidade, montagem do vetor por quadro
  experiments.py                                    sementes, dobras por sinalizador, metricas e gravacao em resultados/<experimento>/
  training_utils.py                                 filtro --classes, sinalizador, contagem de amostras

extract_landmarks_imagem.py                          extrai landmarks do alfabeto (imagens, dataset archive/train e archive/test)
extract_landmarks_video_vlibras.py                   extrai landmarks so-maos de videos (V-Librasil)
extract_landmarks_video_hands_face_minds_libras.py   extrai landmarks maos+rosto de videos (MINDS-Libras)

train_cnn.py                                         treina o modelo do alfabeto (CNN 1D)
train_lstm_hands_face.py                             treina o modelo de palavras (LSTM); com --no_face e so maos
train_lstm.py                                        atalho para train_lstm_hands_face.py --no_face (LSTM so-maos)
train_baseline.py                                    baseline (Random Forest / k-NN) sobre os mesmos vetores achatados
resumo_resultados.py                                 junta resultados/*/metricas.json numa tabela (media +- desvio)
matriz.py                                             gera a matriz de confusao de um modelo ja treinado (le a representacao do _config.json do modelo)

app.py                                                servidor Flask (demo ao vivo pela webcam)
templates/index.html                                  pagina da demo (modo estatico e dinamico)
```

## 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

As versoes em `requirements.txt` sao um ponto de partida - se o `pip install`
der conflito no seu ambiente, ajuste a versao do pacote que falhar e teste de
novo (nao tenho como validar contra o seu Python/SO exatos daqui).

## 2. Baixar os modelos do MediaPipe

```bash
wget -O hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

wget -O face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

Ambos ficam na raiz do projeto (mesma pasta do `app.py`).

Para a **trajetoria do punho** (opcional, so em `--trajectory`) tambem e preciso o
modelo de pose, usado apenas para achar os ombros:

```bash
wget -O pose_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
```

## 3. Datasets

### Alfabeto (imagens)

Pasta `archive/`, com a estrutura original do dataset preservada:

```
archive/
  train/<letra>/*.jpg
  test/<letra>/*.jpg
```

<!-- TODO: preencher com a fonte exata do dataset do alfabeto (nome, link,
licenca) e confirmar se ele fica versionado no repositorio ou precisa ser
baixado separadamente. -->

### MINDS-Libras (palavras, video)

<!-- TODO: preencher com o link/fonte oficial do MINDS-Libras e a licenca de uso. -->

Extraido com `extract_landmarks_video_hands_face_minds_libras.py` (maos +
rosto), usando o nome do arquivo pra identificar classe e sinalizador (ex.:
`01AcontecerSinalizador02-5.mp4` -> classe `Acontecer`, sinalizador `02`).

### V-Librasil (palavras, video)

<!-- TODO: preencher com o link/fonte oficial do V-Librasil, a licenca, e
como o annotations.csv e estruturado (colunas video_name/class). -->

Extraido com `extract_landmarks_video_vlibras.py` (so maos), via
`--annotations_csv annotations.csv` ou `--dataset_dir` com uma subpasta por
classe.

## 4. Extrair landmarks

Alfabeto:

```bash
python extract_landmarks_imagem.py --dataset_dir ./archive --output landmarks.npz
```

MINDS-Libras:

```bash
python extract_landmarks_video_hands_face_minds_libras.py \
  --dataset_dir ./minds-libras \
  --filename_regex '^\d+([A-Za-zÀ-ÿ]+)Sinalizador\d+-\d+\.mp4$' \
  --output landmarks_video_minds.npz \
  --failures_log falhas_minds.csv \
  --quiet
```

MINDS-Libras com trajetoria do punho (precisa de `pose_landmarker.task`). O `.npz` sai com
tudo que as representacoes abaixo precisam, entao e uma extracao so:

```bash
python extract_landmarks_video_hands_face_minds_libras.py \
  --dataset_dir ./minds-libras \
  --filename_regex '^\d+([A-Za-zÀ-ÿ]+)Sinalizador\d+-\d+\.mp4$' \
  --trajectory \
  --output landmarks_video_minds_traj.npz \
  --failures_log falhas_minds_traj.csv \
  --quiet
```

Ao final ele avisa quantos videos ficaram sem nenhum quadro com ombros detectados.
O extrator de V-Librasil (so maos) ainda nao tem a opcao de trajetoria.

V-Librasil:

```bash
python extract_landmarks_video_vlibras.py \
  --dataset_dir ./v-librasil \
  --annotations_csv annotations.csv \
  --output landmarks_video_vlibrasil.npz
```

## 5. Treinar e avaliar (experimentos)

Cada treino **repete** (dobras x sementes) e grava tudo em `resultados/<experimento>/`:

| Arquivo | Conteudo |
|---|---|
| `metricas.json` | resumo (media, desvio entre dobras, desvio entre sementes), cada rodada, metricas por classe |
| `matriz_confusao.csv` / `.png` | contagens somadas sobre todas as rodadas / versao normalizada por classe real (PNG so com ate 60 classes) |
| `confusoes_principais.csv` | os pares de sinais mais confundidos (base da analise de erros) |
| `argumentos.json` | todos os argumentos usados, versoes de Python/NumPy/scikit-learn/TensorFlow |
| `contagem_amostras.csv` | amostras por classe e por divisao (por dobra) |

Duas fontes de variacao, separadas de proposito: **entre dobras** (muda quem esta no teste: outros
sinalizadores) e **entre sementes** (muda so o sorteio do modelo; a divisao dos dados e fixa).
A acuracia e o F1 macro sao reportados como media +- desvio; a "media" e a media das dobras, cada
dobra ja promediada sobre as sementes. O F1 macro considera so as classes que aparecem no teste da
rodada.

Opcoes comuns (as de representacao estao na tabela mais abaixo):

| Opcao | Efeito |
|---|---|
| `--experiment NOME` | nome da pasta em `resultados/` (obrigatorio, ou use `--output`, de onde o nome e tirado) |
| `--folds K` | `0` = uma divisao unica por sinalizador (`--test_size`); `K >= 2` = validacao cruzada por sinalizador com K dobras; `K` = numero de sinalizadores = deixa um de fora por rodada (atalho: `--leave_one_signer_out`) |
| `--seeds 0 1 2` | sementes do TensorFlow (padrao: 3). Um treino por semente em cada dobra |
| `--deterministic` | mesma semente, mesmo resultado exato (mais lento). Sem ela, a semente ja reduz muito a variacao |
| `--output modelo.keras` | opcional: salva o modelo da **primeira** rodada (+ `_labels.json`, `_config.json`). Os experimentos nao precisam salvar modelos |
| `--classes "A,B"` ou `@arquivo.txt` | treina so um subconjunto do vocabulario |

O numero de treinos e `dobras x sementes` (cada um leva da ordem de 1 minuto no MINDS-Libras).
No PowerShell, escreva cada comando em uma linha so.

### Os experimentos (a) a (d)

(a) **CNN no alfabeto**, com a divisao `train/`/`test/` original do dataset (o `.npz` precisa ter sido
gerado com o `extract_landmarks_imagem.py` atual), e o baseline correspondente:

```bash
python train_cnn.py --data landmarks.npz --experiment alfabeto_cnn --output modelo_alfabeto.keras
python train_baseline.py --data landmarks.npz --model rf  --experiment alfabeto_rf
python train_baseline.py --data landmarks.npz --model knn --experiment alfabeto_knn
```

(b) **LSTM so maos**, (c) **LSTM maos + rosto** (com e sem rosto), e (d) **baseline** sobre os mesmos
vetores achatados, no MINDS-Libras, divisao por sinalizador. Troque `--folds 4` por um valor entre 2 e o
numero de sinalizadores (o log mostra `Sinalizadores encontrados (N)`):

```bash
python train_lstm_hands_face.py --data landmarks_video_minds_traj.npz --no_face --folds 4 --experiment minds_lstm_maos
python train_lstm_hands_face.py --data landmarks_video_minds_traj.npz --folds 4 --experiment minds_lstm_maos_rosto
python train_baseline.py --data landmarks_video_minds_traj.npz --no_face --model rf --folds 4 --experiment minds_rf_maos
python train_baseline.py --data landmarks_video_minds_traj.npz --model rf --folds 4 --experiment minds_rf_maos_rosto
```

As representacoes da issue 2 (face normalizada, trajetoria, velocidade) entram na mesma comparacao:

```bash
python train_lstm_hands_face.py --data landmarks_video_minds_traj.npz --face_norm --folds 4 --experiment minds_lstm_facenorm
python train_lstm_hands_face.py --data landmarks_video_minds_traj.npz --face_norm --trajectory --folds 4 --experiment minds_lstm_traj
python train_lstm_hands_face.py --data landmarks_video_minds_traj.npz --face_norm --trajectory --velocity --folds 4 --experiment minds_lstm_traj_vel
```

Todas leem o mesmo `.npz` e usam a mesma divisao por sinalizador, entao os resultados sao
comparaveis entre si. Para montar a tabela de todos de uma vez (grava `resultados/resumo.md` e
`resumo.csv`):

```bash
python resumo_resultados.py
```

O baseline treina em **todo** o treino da divisao (nao precisa de validacao), ou seja, com um pouco mais
de dados que os modelos neurais, que separam uma parte para o early stopping. Isso favorece o
baseline: se a CNN/LSTM ainda ganhar, o ganho nao e artefato.

### Representacoes do sinal (LSTM maos + rosto)

| Opcao | O que entra em cada quadro |
|---|---|
| (nenhuma) | forma das duas maos + rosto cru (coordenadas do recorte). E a representacao "atual" |
| `--no_face` | so a forma das maos |
| `--face_norm` | rosto relativo ao centro dos olhos e dividido pela distancia entre eles (sem posicao absoluta nem escala) |
| `--trajectory` | + posicao de cada punho em relacao ao ponto medio dos ombros, em larguras de ombro (onde a mao esta: perto do rosto, no peito...) |
| `--velocity` | + variacao dessa posicao entre quadros consecutivos |

`--trajectory` e `--velocity` exigem `.npz` extraido com `--trajectory`; `--face_norm` exige `.npz`
extraido com a versao atual do extrator. Um `.npz` antigo continua funcionando sem essas opcoes.

Quando se passa `--output`, o treino grava ao lado do modelo um `<modelo>_config.json` com a
representacao usada. O `app.py` e o `matriz.py` leem esse arquivo para montar exatamente as mesmas
features; um modelo sem ele e tratado como representacao "atual".

### V-Librasil (so maos)

Poucos sinalizadores (3): deixa um de fora por rodada. Com menos de 3 sinalizadores no treino nao
sobra validacao, entao o treino usa `--epochs` fixas, sem early stopping (vale baixar `--epochs`):

```bash
python train_lstm.py --data landmarks_video_vlibrasil.npz --leave_one_signer_out --signer_regex "SEU_PADRAO_AQUI" --epochs 40 --experiment vlibrasil_lstm_maos
```

<!-- TODO: confirmar o padrao de nome de arquivo do V-Librasil pra preencher
--signer_regex corretamente (o padrao default, Sinalizador(\d+), e do
MINDS-Libras). -->

## 6. Matriz de confusao de um modelo salvo

O resultado "oficial" de cada experimento (todas as rodadas somadas) ja esta em
`resultados/<experimento>/matriz_confusao.png`. O `matriz.py` serve para olhar **um** modelo salvo,
no teste da divisao unica (`--folds 0`) do treino:

```bash
python matriz.py --data landmarks_video_minds.npz --model modelo_palavras_minds.keras
```

Precisa dos mesmos `--test_size`/`--signer_regex`, se voce mudou o padrao; a representacao vem do
`_config.json` do modelo.

## 7. Rodar a demo

```bash
python app.py
```

Abre em `http://localhost:5000` (ou a porta exposta pelo Codespace). A pagina
tem um seletor entre modo "Alfabeto (estatico)" e "Palavras (dinamico)".

## Observacoes

- As funcoes de deteccao, normalizacao e leitura de zip/CSV moram em
  `libras/` e sao compartilhadas por todos os extratores - no reimplemente
  localmente num script novo, importe de la.
- `extract_landmarks_video_hands_face_minds_libras.py` e
  `extract_landmarks_video_vlibras.py` ainda tem cada um sua propria
  `find_class_dirs` de video (`find_class_dirs_video` em `libras/landmarks.py`,
  compartilhada entre os dois) - e diferente do `find_class_dirs` do alfabeto
  (que rastreia o split `train`/`test`), por isso nao foi unificada com ela.