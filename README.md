# Libras - reconhecimento de alfabeto (estatico), palavras (dinamico) e frases com LLM

Projeto de TCC: reconhecimento de sinais de Libras a partir de landmarks de
maos e rosto (no alfabeto, so maos) extraidos com MediaPipe, classificados
por uma CNN (alfabeto) e uma LSTM (palavras). As palavras reconhecidas sao
organizadas em frases em portugues por agentes de LLM (Groq como principal e
Gemini como reserva).

## Onde cada parte foi feita

- **Alfabeto manual (estatico)**: extracao de landmarks e treinamento feitos
  no proprio **Visual Studio Codespaces (VS Code)**.
- **Palavras (dinamico, videos)**: extracao de landmarks e treinamento feitos
  no **Google Colab**.
- **Visual Studio Codespaces (VS Code)**: todo o resto do TCC (demo Flask,
  avaliacao, frases/LLM, matriz de confusao, documentacao).

## Estrutura

```
libras/                                            modulo comum
  landmarks.py                                      detectores MediaPipe, normalizacao, amostragem de frames, leitura de zip/CSV
  training_utils.py                                 filtro --classes, sinalizador, contagem de amostras

extract_landmarks_imagem.py                          extrai landmarks do alfabeto (imagens, dataset archive/train e archive/test)
extract_landmarks_video_hands_face_minds_libras.py   extrai landmarks maos+rosto de videos (MINDS-Libras)

train_cnn.py                                         treina o modelo do alfabeto (CNN 1D)
train_lstm.py                                        treina o modelo de palavras (LSTM)
train_baseline.py                                    modelo base para comparacao
matriz.py                                            gera a matriz de confusao de um modelo ja treinado
resumo_resultados.py                                 resume os resultados dos treinos/avaliacoes

frases.py                                            monta frases a partir das palavras reconhecidas (spaCy + agentes LLM)
avaliar_frases.py                                    avalia as frases geradas (ex.: sacreBLEU) contra glosas_referencia.csv
glosas_referencia.csv                                glosas/frases de referencia para a avaliacao
diagnostico_deteccao.py                              diagnostico da deteccao de maos/rosto

modelo_alfabeto.keras                                modelo do alfabeto treinado
modelo_libras_unificado.keras (+ _config/_labels)    modelo unificado de palavras treinado

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
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task
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

Usamos o **MINDS-Libras** e o **V-Librasil**, apenas com palavras
especificas, e adicionamos mais videos ao dataset. Vocabulario atual:

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

## 5. Extrair landmarks

### Alfabeto (feito no Codespace)

```bash
python extract_landmarks_imagem.py --dataset_dir ./archive --output landmarks.npz
```

### Palavras, maos + rosto (feito no Colab)

```bash
python extract_landmarks_video_hands_face_minds_libras.py \
  --dataset_dir ./minds-libras \
  --filename_regex '^\d+([A-Za-zÀ-ÿ]+)Sinalizador\d+-\d+\.mp4$' \
  --output landmarks_video_minds.npz \
  --failures_log falhas_minds.csv \
  --quiet
```

## 6. Treinar

### Alfabeto (feito no Codespace)

Usa a divisao `train/`/`test/` original do dataset, se o `.npz` tiver essa
informacao:

```bash
python train_cnn.py --data landmarks.npz --output modelo_alfabeto.keras \
  --counts_output resultados_alfabeto_contagem.csv
```

### Palavras, LSTM (feito no Colab)

```bash
!python train_unificado.py --minds landmarks_minds_libras_trimmed_interpolated.npz --nucleo landmarks_nucleo_comum.npz
```

Para treinar so um subconjunto do vocabulario, use `--classes`:
`--classes "Sim,Não,Onde"` ou `--classes @lista.txt` (uma classe por linha).

## 7. Avaliar

Matriz de confusao:

```bash
python matriz.py --data landmarks_video_minds.npz --model modelo_libras_unificado.keras
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

## Observacoes

- As funcoes de deteccao, normalizacao e leitura de zip/CSV estão em
  `libras/` e sao compartilhadas por todos os extratores - nao reimplemente
  localmente num script novo, importe de la.