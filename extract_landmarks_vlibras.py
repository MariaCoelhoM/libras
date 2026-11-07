import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import json
import os

# 1. Carregar os dados de teste utilizados no script anterior
# (Certifique-se de que os dados ou variáveis X_test, y_test e encoder ainda estão acessíveis na sessão)
# Caso esteja rodando em uma nova célula limpa, recarregue os dados e o label encoder:
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

caminho_dados = "landmarks_nucleo_comum.npz"
data = np.load(caminho_dados, allow_pickle=True)

features_list = []
if "X_hands" in data:
    X_hands = data["X_hands"]
    features_list.append(X_hands.reshape(X_hands.shape[0], X_hands.shape[1], -1))
if "X_face_norm" in data:
    X_face = data["X_face_norm"]
    features_list.append(X_face.reshape(X_face.shape[0], X_face.shape[1], -1))
elif "X_face" in data:
    X_face = data["X_face"]
    features_list.append(X_face.reshape(X_face.shape[0], X_face.shape[1], -1))
if "X_traj" in data:
    X_traj = data["X_traj"]
    features_list.append(X_traj.reshape(X_traj.shape[0], X_traj.shape[1], -1))

X = np.concatenate(features_list, axis=-1)
y_raw = data["y"]
encoder = LabelEncoder()
y = encoder.fit_transform(y_raw)

# Mesma divisão de teste do script de treino
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Carregar o modelo salvo
modelo_path = "modelo_libras_nucleo_comum.keras"
if os.path.exists(modelo_path):
    model = tf.keras.models.load_model(modelo_path)
    print("Modelo carregado com sucesso!\n")
else:
    raise FileNotFoundError("Modelo salvo não encontrado. Execute o treinamento primeiro.")

# 3. Fazer as predições no conjunto de teste
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# 4. Exibir o Relatório de Classificação por Palavra
print("=" * 60)
print(" RELATÓRIO DE DESEMPENHO POR PALAVRA (CLASSE)")
print("=" * 60)
print(classification_report(y_test, y_pred, target_names=encoder.classes_, digits=4))

# 5. Exibir a Acurácia exata por classe individualmente (em porcentagem)
print("=" * 60)
print(" ACURÁCIA ISOLADA POR SINAL:")
print("=" * 60)
cm = confusion_matrix(y_test, y_pred)
for i, classe_nome in enumerate(encoder.classes_):
    total_classe = np.sum(cm[i])
    acertos_classe = cm[i, i]
    if total_classe > 0:
        acc_classe = (acertos_classe / total_classe) * 100
        print(f" - {classe_nome:<15}: {acc_classe:.2f}%  ({acertos_classe}/{total_classe} amostras corretas)")
    else:
        print(f" - {classe_nome:<15}: Sem amostras de teste.")
print("=" * 60)