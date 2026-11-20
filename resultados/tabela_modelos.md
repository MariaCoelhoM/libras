| Modelo | n treino | n teste | Acurácia | F1 macro | MINDS-Libras (n=200) | V-Librasil Núcleo Comum (n=35) |
|---|---|---|---|---|---|---|
| LSTM final (treino+validacao, 19 epocas) | 703 | 235 | 0,753 | 0,721 | 0,745 | 0,800 |
| Random Forest (baseline) | 703 | 235 | 0,844 ± 0,007 | 0,804 ± 0,010 | 0,890 | 0,581 |
| k-NN (baseline) | 703 | 235 | 0,464 | 0,475 | 0,455 | 0,514 |
| LSTM etapa 1 (so treino, EarlyStopping) | 476 | 235 | 0,579 | 0,572 | 0,560 | 0,686 |
| LSTM etapa 1 sem pesos de classe | 476 | 235 | 0,600 | 0,554 | 0,595 | 0,629 |

Palavras: mesmo teste para todos os modelos (divisão por sinalizador, por base). Média ± desvio entre sementes quando há mais de uma (Random Forest); a LSTM tem uma semente. Com 35 amostras no Núcleo Comum, as diferenças nessa coluna são frágeis.

Alfabeto (CNN, imagens, divisão original train/test do dataset, sem sinalizador): acurácia 0,999 ± 0,000 em 11402 imagens de teste (média de 3 sementes).
