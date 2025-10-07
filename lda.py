import numpy as np
import matplotlib.pyplot as plt
from extract_features import *

def lda_fit(X, y):
    y = y.astype(int)
    classes = np.unique(y)
    K = len(classes)
    N, D = X.shape

    pis = np.zeros(K, dtype=np.float64)
    mus = np.zeros((K, D), dtype=np.float64)
    S = np.zeros((D, D), dtype=np.float64)

    for idx, c in enumerate(classes):
        Xi = X[y == c]
        pis[idx] = Xi.shape[0] / N
        mus[idx] = Xi.mean(axis=0)
    for idx, c in enumerate(classes):
        Xi = X[y == c]
        Z = Xi - mus[idx]
        S += Z.T @ Z
    S /= (N - K)
    Sinv = np.linalg.inv(S)

    Ak = (Sinv @ mus.T).T
    bk = np.empty(K, dtype=np.float64)
    for i in range(K):
        bk[i] = -0.5 * mus[i].T @ Sinv @ mus[i] + np.log(pis[i])

    return {"classes": classes, "A": Ak, "b": bk, "S": S, "Sinv": Sinv, "mu": mus, "pi": pis}

def lda_scores(X, model):
    A = model["A"]
    b = model["b"]
    return X @ A.T + b

def lda_predict(X, model):
    G = lda_scores(X, model)
    idx = np.argmax(G, axis=1)
    return model["classes"][idx]

def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))

def confusion_matrix(y_true, y_pred, classes=None):
    if classes is None:
        classes = np.unique(np.concatenate([y_true, y_pred]))
    K = len(classes)
    M = np.zeros((K, K), dtype=int)
    for i, c in enumerate(classes):
        for j, d in enumerate(classes):
            M[i, j] = np.sum((y_true == c) & (y_pred == d))
    return M, classes

def plot_confusion(M, classes, title="Confusion"):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(M, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("pred")
    ax.set_ylabel("true")
    ax.set_xticks(np.arange(len(classes))); ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.show()

if __name__ == "__main__":

    DIR_PATH = "data/skeletonized/"

    features_by_class, X, y, idx_by_class = extract_features_from_dir(dir_path=DIR_PATH)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    n_tr = int(0.8 * len(y))
    tr, te = idx[:n_tr], idx[n_tr:]

    Xtr, ytr = X[tr], y[tr]
    Xte, yte = X[te], y[te]

    # z-score normalizacija
    Xtr_n, Xte_n, mu, sigma = zscore_normalize(Xtr, Xte)

    model = lda_fit(Xtr, ytr)
    yhat_tr = lda_predict(Xtr, model)
    yhat_te = lda_predict(Xte, model)

    acc_tr = accuracy(ytr, yhat_tr)
    acc_te = accuracy(yte, yhat_te)
    print(f"acc_train={acc_tr:.4f}  acc_test={acc_te:.4f}")

    M, cls = confusion_matrix(yte, yhat_te)
    plot_confusion(M, cls, title="LDA na test skupu")
