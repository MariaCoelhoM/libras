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
  training_utils.py                                 filtro --classes, sinalizador (MINDS e V-Librasil), divisao por sinalizador, contagem
  experiments.py                                    divisoes, metricas e gravacao de resultados em resultados/<experimento>/

extract_landmarks_imagem.py                          extrai landmarks do alfabeto (imagens, dataset archive/train e archive/test)
extract_landmarks_minds_libras.py                    extrai landmarks maos+rosto+trajetoria dos videos (MINDS-Libras e V-Librasil, via --filename_regex); recorta e interpola

train_cnn.py                                         treina o modelo do alfabeto (CNN 1D)
train_lstm.py                                        une MINDS + V-Librasil e treina o modelo de palavras (LSTM), dividindo por sinalizador
train_baseline.py                                    baseline (Random Forest / k-NN), no mesmo teste da LSTM (--split)
matriz.py                                            matriz de confusao do modelo unificado, no teste salvo pelo treino
resumo_resultados.py                                 junta resultados/<experimento>/metricas.json numa tabela (resumo.md e resumo.csv)
tabela_resultados.py                                 tabela por modelo e por origem (MINDS x Nucleo Comum), com os baselines (tabela_modelos.md e .csv)
diagnostico_deteccao.py                              diagnostico da deteccao de maos e rosto nos videos

frases.py                                            monta frases a partir das palavras reconhecidas (spaCy + agentes LLM)
avaliar_frases.py                                    avalia as frases geradas (BLEU e chrF com IC95%)
glosas_desenvolvimento.csv                           frases usadas para escrever as regras (NAO usar para medir)
glosas_teste.csv                                     frases de teste, separadas das de desenvolvimento

landmarks.npz                                        landmarks do alfabeto
landmarks_unificado.npz                              landmarks de palavras: MINDS-Libras + V-Librasil unidos
modelo_alfabeto.keras                                modelo do alfabeto treinado
modelo_libras_unificado_treino_val.keras (+ _config/_labels/_split/_relatorio)  modelo de palavras treinado com treino+validacao (27 classes) e a divisao usada
resultados/                                          resultados dos experimentos (LSTM, baselines, BLEU/chrF)

app.py                                               servidor Flask (demo ao vivo pela webcam)
templates/index.html                                 pagina da demo (modo estatico e dinamico)
requirements.txt                                     dependencias
```

## 1. Instalar dependencias

```bash
pip install uv
uv venv --python 3.14 .venv
source .venv/bin/activate
python --version   # deve mostrar 3.14.x
uv pip install -r requirements.txt
uv pip uninstall opencv-python opencv-contrib-python
uv pip install opencv-python-headless==4.13.0.92
```

Como o `.venv` e criado pelo `uv`, o `pip` pode nao existir dentro dele, e o
`spacy download` precisa dele. Por isso, antes de baixar o modelo de portugues do
spaCy (o comando instala a versao compativel com o `spacy` fixado em
`requirements.txt`):

```bash
python -m ensurepip --upgrade
python -m spacy download pt_core_news_sm
```

O Groq e o Gemini nao pedem instalacao a parte: o `groq` ja esta no
`requirements.txt` e o Gemini e chamado por HTTP (`urllib`, em `frases.py`).

Todas as versoes de `requirements.txt` estao fixadas e sao as do ambiente que
treinou o modelo final e gerou `resultados/lstm_treino_val/` (Python 3.14.2). O
`tensorflow==2.22.0rc0` e uma versao candidata (pre-release). Os experimentos
`lstm_unificado`, `lstm_unificado_semPesos` e `modelo_alfabeto` foram rodados antes,
em Python 3.11.17 com numpy 2.4.6 e tensorflow 2.21.0 (registrado em cada
`argumentos.json`, campo `versoes`).

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

Defina cada chave entre aspas simples e sem Enter dentro delas (uma quebra de
linha na chave faz a API recusar o cabecalho). O `frases.py` remove espacos e
quebras de linha das pontas e mascara a chave nas mensagens de erro. Nunca
versione chaves; se usar um arquivo `.env` (lido pelo `avaliar_frases.py`),
ele precisa estar no `.gitignore`.

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

Os arquivos de entrada do treino (`landmarks_minds_libras_trimmed_interpolated.npz`
e `landmarks_nucleo_comum.npz`) nao estao no repositorio: sao gerados na secao 5.
O `landmarks_unificado.npz`, que o treino produz a partir deles, esta versionado.

O nome do arquivo identifica classe e sinalizador:

- MINDS-Libras: `01AcontecerSinalizador02-5.mp4` -> classe `Acontecer`, sinalizador `02`.
- V-Librasil: `Não_Articulador20.mp4` -> classe `Não`, articulador (sinalizador) `20`.

O treino divide por sinalizador (nenhuma pessoa em treino e teste ao mesmo tempo),
separadamente em cada base (uma base com menos de 3 sinalizadores nao e dividida),
e usa o prefixo da base para nao confundir o
sinalizador `01` do MINDS com o articulador `01` do V-Librasil.

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

V-Librasil (Nucleo Comum), com o MESMO extrator e o mesmo recorte do rosto do
MINDS-Libras (so muda o regex que le a classe do nome do arquivo; o grupo 1 e a
classe). O resultado deve ter 138 amostras:

```bash
python extract_landmarks_minds_libras.py \
  --dataset_dir ./v-librasil \
  --filename_regex "^(Compreender|Desculpa|Não|Onde|Por favor|Sim|Surdo)_Articulador\d+\.mp4$" \
  --output landmarks_nucleo_comum.npz \
  --failures_log falhas_nucleo.csv \
  --quiet
```

## 6. Treinar

Alfabeto (usa a divisao `train/`/`test/` original do dataset, se o `.npz`
tiver essa informacao):

```bash
python train_cnn.py --data landmarks.npz --output modelo_alfabeto.keras \
  --counts_output resultados_alfabeto_contagem.csv
```

Palavras (LSTM): une os dois `.npz` (gera `landmarks_unificado.npz`), divide por
sinalizador e treina com rosto normalizado e trajetoria. O treino tem duas etapas.

Etapa 1, com validacao e EarlyStopping (serve para definir o numero de epocas; o
modelo fica em `/tmp` e nao e versionado):

```bash
python train_lstm.py \
  --minds landmarks_minds_libras_trimmed_interpolated.npz \
  --nucleo landmarks_nucleo_comum.npz \
  --output /tmp/lstm_unificado.keras
```

Nessa etapa o treino parou na epoca 19 de 40 (melhor epoca 9). Os resultados ficam
em `resultados/lstm_unificado/`.

Etapa 2, modelo final: treino com treino+validacao (`--val_size 0`, sem
EarlyStopping) e as mesmas 19 epocas, no mesmo teste:

```bash
python train_lstm.py \
  --minds landmarks_minds_libras_trimmed_interpolated.npz \
  --nucleo landmarks_nucleo_comum.npz \
  --val_size 0 --epochs 19 \
  --experiment lstm_treino_val \
  --output modelo_libras_unificado_treino_val.keras
```

O treino salva ao lado do modelo o `_labels.json` (classes), o `_config.json`
(representacao usada), o `_relatorio.txt` e o `_split.npz` (indices de treino,
validacao e teste). Os resultados versionaveis ficam em `resultados/<experimento>/`
(`metricas.json`, `matriz_confusao.csv/png`, `argumentos.json`, `relatorio.txt`...).
`--test_size` e a fracao de SINALIZADORES no teste (por base); `--seed` muda o
sorteio dos sinalizadores. O teste depende so de `--test_size` e `--seed`, entao as
duas etapas avaliam as mesmas 235 amostras (MINDS 200, Nucleo Comum 35).

Variante sem pesos de classe (a padrao usa `--class_weight balanced`):

```bash
python train_lstm.py --minds landmarks_minds_libras_trimmed_interpolated.npz \
  --nucleo landmarks_nucleo_comum.npz --class_weight none \
  --experiment lstm_unificado_semPesos --output /tmp/lstm_sem_pesos.keras
```

Com `--output /tmp/...`, o modelo e seus arquivos (`_labels.json`, `_config.json`,
`_split.npz`) ficam em `/tmp`; os resultados versionaveis vao para
`resultados/lstm_unificado_semPesos/`. A divisao depende so de `--seed` e
`--test_size`, entao o teste e o mesmo do treino com pesos.

O modelo versionado (`modelo_libras_unificado_treino_val.keras`, usado pela demo) e o
da etapa 2, treinado com pesos (padrao). `resultados/lstm_treino_val/` guarda os
resultados dele (LSTM, baselines e matriz). `resultados/lstm_unificado/` guarda a
etapa 1 e `resultados/lstm_unificado_semPesos/` guarda o experimento de comparacao. A matriz
de confusao do modelo sem pesos (o `.keras` fica em `/tmp`) e gerada assim:

```bash
python matriz.py --model /tmp/lstm_sem_pesos.keras --data landmarks_unificado.npz \
  --normalize --output resultados/lstm_unificado_semPesos/matriz_confusao_teste.png
```

Baseline (Random Forest ou k-NN) com a mesma representacao e EXATAMENTE o mesmo
teste da LSTM (nao use `--split` junto com `--classes`):

```bash
python train_baseline.py --data landmarks_unificado.npz \
  --config modelo_libras_unificado_treino_val_config.json \
  --split modelo_libras_unificado_treino_val_split.npz --model rf \
  --results_dir resultados/lstm_treino_val
python train_baseline.py --data landmarks_unificado.npz \
  --config modelo_libras_unificado_treino_val_config.json \
  --split modelo_libras_unificado_treino_val_split.npz --model knn \
  --results_dir resultados/lstm_treino_val
```

Para treinar so um subconjunto do vocabulario, use `--classes`:
`--classes "Sim,Não,Onde"` ou `--classes @lista.txt` (uma classe por linha).

## 7. Avaliar

Matriz de confusao do modelo unificado (le a divisao de teste salva pelo treino em
`_split.npz` e mostra acuracia geral, por origem e por classe; a acuracia deve bater
com a do log do treino):

```bash
python matriz.py --model modelo_libras_unificado_treino_val.keras \
  --data landmarks_unificado.npz --normalize \
  --output resultados/lstm_treino_val/matriz_confusao_teste.png
```

Tabela por modelo e por origem (LSTM, Random Forest, k-NN, no mesmo teste), para
colar no texto:

```bash
python tabela_resultados.py
```

Salva `resultados/tabela_modelos.md` e `resultados/tabela_modelos.csv`.

Frases contra as referencias (BLEU e chrF com IC95%). As glosas do CSV sao separadas
por `|` (ex.: `POR FAVOR|ONDE|BANHEIRO`). As regras foram escritas olhando
`glosas_desenvolvimento.csv`; o numero que vale e o do `glosas_teste.csv`:

```bash
python avaliar_frases.py --csv glosas_teste.csv
python avaliar_frases.py --csv glosas_teste.csv --app --pausa 2   # LLM (precisa das chaves)
```

Salva `resultados/bleu_glosas_teste.json` e `resultados/frases_glosas_teste.csv`.

O resultado com LLM (`--app`) nao e deterministico: varia entre execucoes e com a
disponibilidade das APIs. Se uma resposta do LLM nao for uma traducao valida (por
exemplo, pedir os sinais ao usuario), o `frases.py` pede outra, e se nenhuma servir
a frase sai pelo fallback local e e contada como "sem LLM". O log mostra o BLEU com
as 26 frases e so com as que o LLM respondeu. Rode mais de uma vez e reporte a
media e quantas frases cairam no fallback.

As 3 rodadas com LLM do teste, com media e desvio, estao em
`resultados/bleu_llm_rodadas.md`.

## 8. Rodar a demo

```bash
python app.py
```

Abre em `http://localhost:5000` (ou a porta exposta pelo Codespace). A pagina
tem um seletor entre modo "Alfabeto (estatico)" e "Palavras (dinamico)".

