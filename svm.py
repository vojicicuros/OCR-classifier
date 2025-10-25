import numpy as np
import matplotlib.pyplot as plt
from extract_features import *
from sklearn.svm import SVC


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
    """
    Izračunaj per-class precision/recall/F1 i support iz matrice konfuzije M (KxK).
    Vraća:
        precision (K,), recall (K,), f1 (K,), support (K,)
    """
    tp = np.diag(M).astype(float)
    support = M.sum(axis=1).astype(float)       # stvarni primeri po klasi (red)
    pred_count = M.sum(axis=0).astype(float)    # predikcije po klasi (kolona)

    K = M.shape[0]
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


def precision_recall_f1(y_true, y_pred, average="macro", classes=None):
    """
    Računa precision, recall, f1.
    average: 'macro' | 'weighted' | 'micro' | None  (None => per-class vektori)
    """
    M, classes = confusion_matrix(y_true, y_pred, classes)

    if average == "micro":
        # u multi-klasi: precision = recall = F1 = accuracy
        tp_sum = float(np.trace(M))
        total = float(M.sum())
        acc = tp_sum / total if total > 0 else 0.0
        return acc, acc, acc, classes

    precision, recall, f1, support = _per_class_prf(M)

    if average is None:
        return precision, recall, f1, classes

    if average == "macro":
        return float(np.mean(precision)), float(np.mean(recall)), float(np.mean(f1)), classes

    if average == "weighted":
        total = float(np.sum(support))
        w = (support / total) if total > 0 else np.zeros_like(support)
        return float(np.sum(precision * w)), float(np.sum(recall * w)), float(np.sum(f1 * w)), classes

    raise ValueError("average mora biti u {'macro','weighted','micro',None}")


def classification_report(y_true, y_pred, classes=None):
    """
    Tekstualni izveštaj sličan sklearn-ovom: per-class P/R/F1 + macro/weighted/micro + accuracy.
    """
    M, classes = confusion_matrix(y_true, y_pred, classes)
    p, r, f1, support = _per_class_prf(M)
    total = int(np.sum(support))

    lines = []
    header = f"{'class':>8} | {'precision':>9} {'recall':>9} {'f1-score':>9} {'support':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for i, c in enumerate(classes):
        lines.append(f"{str(c):>8} | {p[i]:9.4f} {r[i]:9.4f} {f1[i]:9.4f} {int(support[i]):8d}")
    lines.append("-" * len(header))

    pm, rm, f1m, _ = precision_recall_f1(y_true, y_pred, average="macro", classes=classes)
    pw, rw, f1w, _ = precision_recall_f1(y_true, y_pred, average="weighted", classes=classes)
    pmi, rmi, f1mi, _ = precision_recall_f1(y_true, y_pred, average="micro", classes=classes)
    acc = accuracy(y_true, y_pred)

    lines.append(f"{'macro avg':>8} | {pm:9.4f} {rm:9.4f} {f1m:9.4f} {total:8d}")
    lines.append(f"{'weighted':>8} | {pw:9.4f} {rw:9.4f} {f1w:9.4f} {total:8d}")
    lines.append(f"{'micro avg':>8} | {pmi:9.4f} {rmi:9.4f} {f1mi:9.4f} {total:8d}")
    lines.append(f"{'accuracy':>8} | {acc:9.4f}")
    return "\n".join(lines)


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


# -----------------------------
#   SVM fit/predict API
# -----------------------------
def svm_fit(
    X,
    y,
    kernel="rbf",
    C=10.0,
    gamma="scale",
    degree=3,
    coef0=0.0,
    probability=False,
    class_weight=None,
    random_state=0,
):
    """
    Treniraj SVM (SVC) nad datim obeležjima.
    """
    y = y.astype(int)
    classes = np.unique(y)

    clf = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        degree=degree,
        coef0=coef0,
        probability=probability,
        class_weight=class_weight,
        random_state=random_state,
    )
    clf.fit(X, y)
    return {"clf": clf, "classes": classes}


def svm_predict(X, model):
    clf = model["clf"]
    return clf.predict(X).astype(int)


# -----------------------------
#   Vizuelizacija: pairwise linear SVM u 2D (PCA)
# -----------------------------
def _pca_2d(X):
    """
    Vraća Z (N×2), mean (D,), W (D×2) — PCA projekcija u 2D preko SVD.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:2].T                    # D×2
    Z = Xc @ W                      # N×2
    return Z, X.mean(axis=0), W


def plot_pairwise_linear_svm(X, y, cls_a, cls_b, C=1.0, title_note=""):
    """
    Nacrtaj 2D PCA projekciju za dve klase, zajedno sa linearnim SVM-om,
    hiper-ravni, marginama i support vektorima.
    """
    mask = (y == cls_a) | (y == cls_b)
    Xab = X[mask]
    yab = y[mask]

    if Xab.shape[0] < 2:
        print(f"Premalo uzoraka za par ({cls_a}, {cls_b}).")
        return

    # PCA->2D
    Z, mu, W = _pca_2d(Xab)

    # Linearni SVM samo za crtanje granice
    clf_lin = SVC(kernel="linear", C=C, random_state=0)
    clf_lin.fit(Z, yab)

    # Koeficijenti hiper-ravni: w·z + b = 0
    w = clf_lin.coef_[0]
    b = clf_lin.intercept_[0]

    # Granice x-ose
    x_min, x_max = Z[:, 0].min() - 1.0, Z[:, 0].max() + 1.0
    xx = np.linspace(x_min, x_max, 200)

    # y = -(w1/w2) x - b/w2; margine: w·z + b = ±1  =>  y = -(w1/w2)x - (b±1)/w2
    if np.abs(w[1]) < 1e-12:
        # izbegni deljenje nulom: nacrtaj vertikalne linije
        x0 = -b / w[0]
        plt.figure(figsize=(6.5, 4.8))
        plt.scatter(Z[yab == cls_a, 0], Z[yab == cls_a, 1], s=40, label=f"Class {cls_a}")
        plt.scatter(Z[yab == cls_b, 0], Z[yab == cls_b, 1], s=40, label=f"Class {cls_b}")
        for c, ls in [(0, "-"), (1, "--"), (-1, "--")]:
            plt.axvline(x=(-b + c) / w[0], linewidth=2 if c == 0 else 1.5,
                        linestyle="-" if c == 0 else "--", color="k")
    else:
        yy = -(w[0] / w[1]) * xx - b / w[1]
        yy_m1 = -(w[0] / w[1]) * xx - (b - 1.0) / w[1]
        yy_p1 = -(w[0] / w[1]) * xx - (b + 1.0) / w[1]

        plt.figure(figsize=(6.5, 4.8))
        plt.scatter(Z[yab == cls_a, 0], Z[yab == cls_a, 1], s=40, label=f"Class {cls_a}")
        plt.scatter(Z[yab == cls_b, 0], Z[yab == cls_b, 1], s=40, label=f"Class {cls_b}")
        plt.plot(xx, yy, "k-", linewidth=2, label="Hyperplane")
        plt.plot(xx, yy_m1, "k--", linewidth=1.5, label="Margin")
        plt.plot(xx, yy_p1, "k--", linewidth=1.5)

    # Support vektori
    sv = Z[clf_lin.support_]
    plt.scatter(sv[:, 0], sv[:, 1], facecolors="none", edgecolors="k", s=120, linewidths=1.5, label="Support Vectors")

    plt.title(f"Linear SVM (pair {cls_a} vs {cls_b}){title_note}")
    plt.xlabel("Feature 1 (PCA)")
    plt.ylabel("Feature 2 (PCA)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def plot_many_pairs(X, y, pairs=None, C=1.0, max_pairs=6):
    """
    Nacrtaj više pairwise plotova.
    - pairs: lista npr. [(0,1), (3,8)], ako None uzmi prvih max_pairs kombinacija po redosledu klasa.
    """
    classes = np.unique(y)
    if pairs is None:
        pairs = []
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                pairs.append((int(classes[i]), int(classes[j])))
        pairs = pairs[:max_pairs]

    for (a, b) in pairs:
        plot_pairwise_linear_svm(X, y, a, b, C=C, title_note="  (PCA 2D)")






if __name__ == "__main__":


    DIR_PATH = "data/skeletonized/"
    # DIR_PATH = "mnist_dataset/"
    features_by_class, X, y, idx_by_class = extract_features_from_dir(dir_path=DIR_PATH)

    # 2) Podela na trening/test (80/20) sa istom semenom za fer poređenje
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    n_tr = int(0.8 * len(y))
    tr, te = idx[:n_tr], idx[n_tr:]
    Xtr, ytr = X[tr], y[tr]
    Xte, yte = X[te], y[te]

    # 3) Z-score normalizacija (μ, σ iz treninga)
    Xtr_n, Xte_n, mu, sigma = zscore_normalize(Xtr, Xte)

    # 4) Treniraj SVM za evaluaciju (može RBF)
    kernel = "rbf"
    C_main = 5.0
    gamma = "scale"
    model = svm_fit(
        Xtr_n,
        ytr,
        kernel=kernel,
        C=C_main,
        gamma=gamma,
        probability=False,
        random_state=0,
    )

    # 5) Predikcija
    yhat_tr = svm_predict(Xtr_n, model)
    yhat_te = svm_predict(Xte_n, model)

    # 6) Osnovna tačnost + izveštaji
    acc_tr = accuracy(ytr, yhat_tr)
    acc_te = accuracy(yte, yhat_te)
    print(f"SVM (kernel={kernel}, C={C_main}, gamma='{gamma}')  acc_train={acc_tr:.4f}  acc_test={acc_te:.4f}")

    print("\n=== TRAIN REPORT ===")
    print(classification_report(ytr, yhat_tr, classes=np.unique(y)))

    print("\n=== TEST REPORT ===")
    print(classification_report(yte, yhat_te, classes=np.unique(y)))

    # 7) Konfuziona matrica (TEST)
    M, cls = confusion_matrix(yte, yhat_te, classes=np.unique(y))
    plot_confusion(M, cls, title=f"SVM na test skupu (kernel={kernel}, C={C_main})")

    # 8) Pairwise grafici — linearni SVM u 2D (PCA) za nekoliko parova klasa
    #    (možeš promeniti pairs=[(0,1),(3,8),...] ili max_pairs)
    plot_many_pairs(Xtr_n, ytr, pairs=None, C=1.0, max_pairs=6)

    # Cs = [0.5, 1.0, 5.0, 10.0, 20.0]
    # gammas = ['scale', 0.1, 0.05, 0.02, 0.01]
    # best = (-1.0, None, None)
    # for C_ in Cs:
    #     for g_ in gammas:
    #         m = svm_fit(Xtr_n, ytr, kernel='rbf', C=C_, gamma=g_, random_state=0)
    #         yte_hat = svm_predict(Xte_n, m)
    #         acc = accuracy(yte, yte_hat)
    #         print(f"C={C_:>5}, gamma={str(g_):>6} => acc_test={acc:.4f}")
    #         if acc > best[0]:
    #             best = (acc, C_, g_)
    # print(f"Best RBF grid: acc_test={best[0]:.4f}  C={best[1]}  gamma={best[2]}")



