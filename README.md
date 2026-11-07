# Libras - reconhecimento de alfabeto (estatico), palavras (dinamico) e frases com LLM

Projeto de TCC: reconhecimento de sinais de Libras a partir de landmarks de
maos e rosto (no alfabeto, so maos) extraidos com MediaPipe, classificados
por uma CNN (alfabeto) e uma LSTM (palavras). As palavras reconhecidas sao
organizadas em frases em portugues por agentes de LLM (Groq como principal e
Gemini como reserva).

## Ambiente de desenvolvimento

Todo o projeto - extracao de landmarks, treinamento, avaliacao, demo Flask e
frases com LLM - foi desenvolvido no **Visual Studio Codespaces (VS Code)**.

## Estrutura

```
libras/                                            modulo comum
  landmarks.py                                      detectores MediaPipe, normalizacao, amostragem de frames, leitura de zip/CSV
  features.py                                       normalizacao do rosto e trajetoria do punho
  training_utils.py                                 filtro --classes, sinalizador, contagem de amostras

extract_landmarks_imagem.py                          extrai landmarks do alfabeto (imagens, dataset archive/train e archive/test)
extract_landmarks_minds_libras.py                    extrai landmarks maos+rosto+trajetoria dos videos do MINDS-Libras (recorta e interpola)
extract_landmarks_video_hands_face_minds_libras.py   versao anterior do extrator de maos+rosto do MINDS-Libras
extract_landmarks_vlibras.py                         extrai landmarks dos videos do V-Librasil (Nucleo Comum)

train_cnn.py                                         treina o modelo do alfabeto (CNN 1D)
train_lstm.py                                        treina o modelo de palavras (LSTM)
train_baseline.py                                    baseline (Random Forest / k-NN) para comparar com a CNN e a LSTM
matriz.py                                            gera a matriz de confusao do modelo unificado

frases.py                                            monta frases a partir das palavras reconhecidas (spaCy + agentes LLM)
avaliar_frases.py                                    avalia as frases geradas (ex.: sacreBLEU) contra glosas_referencia.csv
glosas_referencia.csv                                glosas/frases de referencia para a avaliacao

landmarks.npz                                        landmarks do alfabeto
landmarks_unificado.npz                              landmarks de palavras: MINDS-Libras + V-Librasil unidos
modelo_alfabeto.keras                                modelo do alfabeto treinado
modelo_libras_unificado.keras (+ _config/_labels)    modelo de palavras treinado (27 classes)
resultados/                                          resultados dos experimentos (ex.: baseline em JSON)

app.py                                               servidor Flask (demo ao vivo pela webcam)
templates/index.html                                 pagina da demo (modo estatico e dinamico)
requirements.txt                                     dependencias
```

## 1. Instalar dependencias

```bash
pip install uv
uv venv --python 3.11 .venv
source .venv/bin/activate
python --version   # deve mostrar 3.11.x
uv pip install -r requirements.txt
uv pip uninstall opencv-python opencv-contrib-python
uv pip install opencv-python-headless==4.10.0.84
```

Como usamos o `.venv`, o `pip` pode nao existir dentro dele. Para instalar o
Groq (agente principal do LLM) pelo proprio ambiente:

```bash
python -m ensurepip --upgrade
python -m pip install groq
```

Modelo de portugues do spaCy:

```bash
python -m spacy download pt_core_news_sm
```

As versoes em `requirements.txt` sao um ponto de partida - se o `pip install`
der conflito no seu ambiente, ajuste a versao do pacote que falhar e teste de
novo.

## 2. Baixar os modelos do MediaPipe

```bash
wget -O hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

wget -O face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

wget -O pose_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
```

Os tres ficam na raiz do projeto (mesma pasta do `app.py`).

## 3. Configurar o LLM (agentes Groq e Gemini)

A parte de LLM funciona com **agentes**: o **Groq** e o agente principal e o
**Gemini** e o segundo (reserva), usado se o principal falhar.

No terminal do projeto faça:

```bash
export GROQ_API_KEY=sua_chave_groq
export GROQ_MODELO=openai/gpt-oss-20b

export GEMINI_API_KEY=sua_chave_gemini
export GEMINI_MODELO=gemini-3.8-flash
export GEMINI_MODELO_RESERVA=
```

`GEMINI_MODELO_RESERVA` é opcional (pode ficar vazio).

## 4. Datasets

### Alfabeto (imagens)

Pasta `archive/`, com a estrutura original do dataset preservada:

```
archive/
  train/<letra>/*.jpg
  test/<letra>/*.jpg
```

### Palavras (video): MINDS-Libras + V-Librasil

Usamos o **MINDS-Libras** e o **V-Librasil** (Nucleo Comum), apenas com
palavras especificas, e adicionamos mais videos ao dataset.

- **MINDS-Libras**: 20 classes (Acontecer, Aluno, Amarelo, America, Aproveitar,
  Bala, Banco, Banheiro, Barulho, Cinco, Conhecer, Espelho, Esquina, Filho,
  Maca, Medo, Ruim, Sapo, Vacina, Vontade) - 800 amostras.
- **V-Librasil (Nucleo Comum)**: 7 classes (Compreender, Desculpa, Não, Onde,
  Por favor, Sim, Surdo) - 138 amostras.
- **Unificado** (`landmarks_unificado.npz`): as duas fontes juntas, 938
  amostras e 27 classes. A chave `origem` indica de qual dataset veio cada
  amostra.

Vocabulario atual do modelo unificado:

```json
[
  "Acontecer",
  "Aluno",
  "Amarelo",
  "America",
  "Aproveitar",
  "Bala",
  "Banco",
  "Banheiro",
  "Barulho",
  "Cinco",
  "Compreender",
  "Conhecer",
  "Desculpa",
  "Espelho",
  "Esquina",
  "Filho",
  "Maca",
  "Medo",
  "Não",
  "Onde",
  "Por favor",
  "Ruim",
  "Sapo",
  "Sim",
  "Surdo",
  "Vacina",
  "Vontade"
]
```

O nome do arquivo identifica classe e sinalizador (MINDS-Libras), ex.:
`01AcontecerSinalizador02-5.mp4` -> classe `Acontecer`, sinalizador `02`.

Cada amostra de palavra tem 30 frames e 406 features por frame (126 das maos,
276 do rosto normalizado e 4 da trajetoria do punho).

## 5. Extrair landmarks

Alfabeto:

```bash
python extract_landmarks_imagem.py --dataset_dir ./archive --output landmarks.npz
```

MINDS-Libras (maos + rosto + trajetoria; recorta o trecho ativo do sinal e
interpola os frames sem deteccao):

```bash
python extract_landmarks_minds_libras.py \
  --dataset_dir ./minds-libras \
  --output landmarks_minds_libras_trimmed_interpolated.npz \
  --failures_log falhas_minds.csv \
  --quiet
```

V-Librasil (Nucleo Comum):

```bash
python extract_landmarks_vlibras.py \
  --dataset_dir ./v-librasil \
  --output landmarks_nucleo_comum.npz
```

## 6. Treinar

Alfabeto (usa a divisao `train/`/`test/` original do dataset, se o `.npz`
tiver essa informacao):

```bash
python train_cnn.py --data landmarks.npz --output modelo_alfabeto.keras \
  --counts_output resultados_alfabeto_contagem.csv
```

Palavras (LSTM), com rosto normalizado e trajetoria:

```bash
python train_lstm.py --data landmarks_unificado.npz \
  --face_norm --trajectory \
  --output modelo_libras_unificado.keras
```

O treino salva ao lado do modelo o `_labels.json` (classes) e o `_config.json`
(representacao usada).

Baseline (Random Forest ou k-NN) com a mesma representacao e a mesma divisao
do modelo neural:

```bash
python train_baseline.py --data landmarks_unificado.npz \
  --config modelo_libras_unificado_config.json
```

Para treinar so um subconjunto do vocabulario, use `--classes`:
`--classes "Sim,Não,Onde"` ou `--classes @lista.txt` (uma classe por linha).

## 7. Avaliar

Matriz de confusao do modelo unificado (reproduz a divisao de teste do
treino e mostra acuracia geral, por origem e por classe):

```bash
python matriz.py --model modelo_libras_unificado.keras \
  --data landmarks_unificado.npz --normalize
```

Frases (LLM) contra as glosas de referencia:

```bash
python avaliar_frases.py
```

## 8. Rodar a demo

```bash
python app.py
```

Abre em `http://localhost:5000` (ou a porta exposta pelo Codespace). A pagina
tem um seletor entre modo "Alfabeto (estatico)" e "Palavras (dinamico)".

