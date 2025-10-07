import numpy as np
import matplotlib.pyplot as plt
from extract_features import *



def mahalanobis_fit(X):
    S = np.cov(X, rowvar=False)
    Sinv = np.linalg.pinv(S)
    return Sinv

def knn_fit(X, y, use_mahalanobis=False):
    model = {"X": X.astype(np.float64), "y": y.astype(int)}
    if use_mahalanobis:
        model["Sinv"] = mahalanobis_fit(model["X"])
    return model


def _pairwise_dist2(Xtr, x, Sinv=None):
    D = Xtr - x
    if Sinv is None:
        return np.einsum('ij,ij->i', D, D)
    return np.einsum('ij,jk,ik->i', D, Sinv, D)

def knn_predict(X, model, k=3, weighted=False):
    Xtr = model["X"]; ytr = model["y"]
    Sinv = model.get("Sinv", None)
    yhat = np.empty(X.shape[0], dtype=int)
    for i in range(X.shape[0]):
        d2 = _pairwise_dist2(Xtr, X[i], Sinv)
        idx = np.argpartition(d2, k)[:k]
        if not weighted:
            votes = np.bincount(ytr[idx])
        else:
            w = 1.0 / (np.sqrt(d2[idx]) + 1e-8)
            K = int(ytr.max()) + 1
            votes = np.zeros(K, dtype=np.float64)
            for j, cls in zip(w, ytr[idx]):
                if cls >= K:  # ako su klase 0..9, ovo je bezbedno
                    K = cls + 1
                    votes = np.pad(votes, (0, K - votes.size))
                votes[cls] += j
        yhat[i] = int(np.argmax(votes))
    return yhat

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

    Xtr_n, Xte_n, mu, sigma = zscore_normalize(Xtr, Xte)

    model = knn_fit(Xtr_n, ytr, use_mahalanobis=False)
    for k in [1,3,5,7,9,11,15]:
        yhat = knn_predict(Xte_n, model, k=k, weighted=True)
        print(f"k={k:>2}  acc_test={accuracy(yte, yhat):.4f}")

    modelM = knn_fit(Xtr_n, ytr, use_mahalanobis=True)
    k = 11
    yhat = knn_predict(Xte_n, modelM, k=k, weighted=True)
    M, cls = confusion_matrix(yte, yhat)
    print(f"[Mahalanobis] k={k} acc_test={accuracy(yte, yhat):.4f}")
    plot_confusion(M, cls, title=f"kNN Mahalanobis (k={k})")


