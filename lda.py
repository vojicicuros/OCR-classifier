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

    # procena priora i sredina
    for idx, c in enumerate(classes):
        Xi = X[y == c]
        pis[idx] = Xi.shape[0] / N
        mus[idx] = Xi.mean(axis=0)

    # zajednička kovarijaciona matrica
    for idx, c in enumerate(classes):
        Xi = X[y == c]
        Z = Xi - mus[idx]
        S += Z.T @ Z
    S /= (N - K)

    # invers
    Sinv = np.linalg.inv(S)

    # koeficijenti diskriminantnih funkcija
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


# -----------------------------
#   Metrics helpers
# -----------------------------
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


def _per_class_prf(M):
    K = M.shape[0]
    tp = np.diag(M).astype(float)
    support = M.sum(axis=1).astype(float)
    pred_count = M.sum(axis=0).astype(float)

    precision = np.zeros(K, dtype=float)
    recall = np.zeros(K, dtype=float)
    f1 = np.zeros(K, dtype=float)

    for k in range(K):
        denom_p = pred_count[k]
        denom_r = support[k]
        precision[k] = (tp[k] / denom_p) if denom_p > 0 else 0.0
        recall[k] = (tp[k] / denom_r) if denom_r > 0 else 0.0
        denom_f = precision[k] + recall[k]
        f1[k] = (2.0 * precision[k] * recall[k] / denom_f) if denom_f > 0 else 0.0

    return precision, recall, f1, support


def classification_report(y_true, y_pred, classes=None):
    M, classes = confusion_matrix(y_true, y_pred, classes)
    p, r, f1, support = _per_class_prf(M)
    total = int(np.sum(support))

    lines = []
    header = f"{'klasa':>8} | {'preciznost':>10} {'odziv':>8} {'f1':>8} {'broj':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for i, c in enumerate(classes):
        lines.append(f"{str(c):>8} | {p[i]:10.4f} {r[i]:8.4f} {f1[i]:8.4f} {int(support[i]):8d}")
    lines.append("-" * len(header))

    acc = accuracy(y_true, y_pred)
    lines.append(f"{'tačnost':>8} | {acc:10.4f}")
    return "\n".join(lines)


def plot_confusion(M, classes, title="Matrica konfuzije"):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(M, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("predikcija")
    ax.set_ylabel("stvarna klasa")
    ax.set_xticks(np.arange(len(classes))); ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Broj primera")
    plt.tight_layout(); plt.show()


# -----------------------------
#   Pairwise LDA viz (PCA→2D)
# -----------------------------
def _pca_2d(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:2].T
    Z = Xc @ W
    return Z, X.mean(axis=0), W


def _lda_pairwise_params_2d(Z, y_ab, a, b):
    Za, Zb = Z[y_ab == a], Z[y_ab == b]
    Na, Nb = len(Za), len(Zb)
    N = Na + Nb

    mu_a = Za.mean(axis=0)
    mu_b = Zb.mean(axis=0)

    Sa = (Za - mu_a).T @ (Za - mu_a)
    Sb = (Zb - mu_b).T @ (Zb - mu_b)
    S = (Sa + Sb) / (N - 2)
    Sinv = np.linalg.inv(S)

    pi_a = Na / N
    pi_b = Nb / N

    w = Sinv @ (mu_a - mu_b)
    b0 = -0.5 * (mu_a.T @ Sinv @ mu_a - mu_b.T @ Sinv @ mu_b) + np.log(pi_a / pi_b)
    return w, b0, mu_a, mu_b


def plot_pairwise_lda(X, y, cls_a, cls_b, title_note=""):
    """
    Nacrtaj PCA→2D za dve klase i LDA granicu između njih.
    """
    mask = (y == cls_a) | (y == cls_b)
    Xab = X[mask]
    yab = y[mask]

    if Xab.shape[0] < 2:
        print(f"Premalo uzoraka za par ({cls_a}, {cls_b}).")
        return

    # PCA→2D
    Z, mu, W = _pca_2d(Xab)

    # LDA u 2D prostoru
    w, b0, mu_a, mu_b = _lda_pairwise_params_2d(Z, yab, cls_a, cls_b)

    x_min, x_max = Z[:, 0].min() - 1.0, Z[:, 0].max() + 1.0
    xx = np.linspace(x_min, x_max, 200)

    plt.figure(figsize=(6.5, 4.8))
    plt.scatter(Z[yab == cls_a, 0], Z[yab == cls_a, 1], s=40, label=f"Klasa {cls_a}")
    plt.scatter(Z[yab == cls_b, 0], Z[yab == cls_b, 1], s=40, label=f"Klasa {cls_b}")

    if np.abs(w[1]) < 1e-12:
        x0 = -b0 / w[0]
        plt.axvline(x=x0, color="k", linewidth=2, label="Granica odlučivanja")
    else:
        yy = -(w[0] / w[1]) * xx - b0 / w[1]
        plt.plot(xx, yy, "k-", linewidth=2, label="Granica odlučivanja")

    plt.scatter(mu_a[0], mu_a[1], marker="x", s=120, linewidths=2, color="k", label="Sredine klasa")
    plt.scatter(mu_b[0], mu_b[1], marker="x", s=120, linewidths=2, color="k")

    plt.title(f"LDA (par {cls_a} vs {cls_b}){title_note}")
    plt.xlabel("Obeležje 1 (PCA)")
    plt.ylabel("Obeležje 2 (PCA)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def plot_many_pairs_lda(X, y, pairs=None, max_pairs=6):
    classes = np.unique(y)
    if pairs is None:
        pairs = []
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                pairs.append((int(classes[i]), int(classes[j])))
        pairs = pairs[:max_pairs]

    for (a, b) in pairs:
        plot_pairwise_lda(X, y, a, b, title_note="  (PCA 2D)")


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
    model = lda_fit(Xtr_n, ytr)

    yhat_tr = lda_predict(Xtr_n, model)
    yhat_te = lda_predict(Xte_n, model)

    print("=== TRAIN ===")
    print(classification_report(ytr, yhat_tr))

    print("\n=== TEST ===")
    print(classification_report(yte, yhat_te))

    M, cls = confusion_matrix(yte, yhat_te)
    plot_confusion(M, cls, title="LDA na test skupu")

    plot_many_pairs_lda(Xtr_n, ytr, pairs=None, max_pairs=6)



