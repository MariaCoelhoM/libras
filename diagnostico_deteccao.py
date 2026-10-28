import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "landmarks_minds_libras_final.npz"
data = np.load(path, allow_pickle=True)
y = data["y"]

# Frame "ausente" = tudo zero. Fracao de quadros sem mao, sem rosto e sem trajetoria, por classe.
hands = data["X_hands"].reshape(len(y), data["X_hands"].shape[1], -1).any(axis=-1)    # (N, T): alguma mao
face = data["X_face"].reshape(len(y), data["X_face"].shape[1], -1).any(axis=-1)       # (N, T)
n_hands = data["X_hands"].reshape(len(y), data["X_hands"].shape[1], 2, -1).any(axis=-1).sum(axis=-1)  # maos por quadro

print(f"{'classe':<12} {'sem mao':>8} {'so 1 mao':>9} {'2 maos':>7} {'sem rosto':>10}")
for c in sorted(set(y.tolist())):
    m = y == c
    print(f"{c:<12} {np.mean(n_hands[m] == 0):8.1%} {np.mean(n_hands[m] == 1):9.1%} "
          f"{np.mean(n_hands[m] == 2):7.1%} {np.mean(~face[m]):10.1%}")
