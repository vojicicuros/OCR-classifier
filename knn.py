import numpy as np
import matplotlib.pyplot as plt
from extract_features import *

# =============================
#   KNN: FIT / PREDICT
# =============================
def _stat_dist_inv_cov(X):
    # Kov. matrica i moore-pseudoinverz (robustnije za D>N/degeneraciju)
    S = np.cov(X, rowvar=False)
    return np.linalg.pinv(S)

def knn_fit(X, y, metric="euclidean"):
    """
    Memorijski kNN 'model' — samo čuva X, y i opciono S^{-1} za Statističku distancu.
    metric: 'euclidean' | 'statisticka'
    """
    X = X.astype(np.float64, copy=False)
    y = y.astype(int, copy=False)
    model = {"X": X, "y": y, "metric": metric, "n_classes": int(y.max()) + 1}
    if metric == "statisticka":
        model["Sinv"] = _stat_dist_inv_cov(X)
    else:
        model["Sinv"] = None
    return model

def _dist2(Xtr, x, Sinv=None):
    D = Xtr - x
    if Sinv is None:  # Euklidska kvadrirana
        return np.einsum("ij,ij->i", D, D)
    # Statistička kvadrirana (Mahalanobis)
    return np.einsum("ij,jk,ik->i", D, Sinv, D)

def knn_predict(X, model, k=3, weighted=False):
    """
    Standardni kNN: nalazi k najbližih po izabranoj metrici i glasa (po potrebi ponderisano).
    """
    Xtr, ytr, Sinv = model["X"], model["y"], model["Sinv"]
    C = model["n_classes"]
    yhat = np.empty(X.shape[0], dtype=int)

    for i in range(X.shape[0]):
        d2 = _dist2(Xtr, X[i], Sinv)
        idx = np.argpartition(d2, k)[:k]
        if not weighted:
            votes = np.bincount(ytr[idx], minlength=C)
        else:
            w = 1.0 / (np.sqrt(d2[idx]) + 1e-8)   # veća težina bližim komšijama
            votes = np.bincount(ytr[idx], weights=w, minlength=C)
        yhat[i] = int(np.argmax(votes))
    return yhat

# =============================
#   METRIKE
# =============================
def accuracy(y_true, y_pred):
    """Ukupna tačnost klasifikatora."""
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, classes=None):
    """
    Ručno izračunata konfuziona matrica, redovi = stvarne klase, kolone = predikcije.
    Vraća: (M, classes) gde je M oblika (K,K).
    """
    if classes is None:
        classes = np.unique(np.concatenate([y_true, y_pred]))
    K = len(classes)
    M = np.zeros((K, K), dtype=int)
    for i, c in enumerate(classes):
        for j, d in enumerate(classes):
            M[i, j] = np.sum((y_true == c) & (y_pred == d))
    return M, classes


def _precision_per_class(M):
    """
    Preciznost po klasi (tp / (tp + fp)).
    Ako za neku klasu nema nijedne predikcije, preciznost je 0.
    """
    tp = np.diag(M).astype(float)
    pred_count = M.sum(axis=0).astype(float)
    precision = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count > 0)
    return precision


def classification_report(y_true, y_pred, classes=None):
    """
    Ispisuje tabelu: preciznost po klasi + ukupna tačnost.
    """
    M, classes = confusion_matrix(y_true, y_pred, classes)
    precision = _precision_per_class(M)

    lines = []
    header = f"{'klasa':>8} | {'preciznost':>10}"
    lines += [header, "-" * len(header)]
    for i, c in enumerate(classes):
        lines.append(f"{str(c):>8} | {precision[i]:10.4f}")
    lines.append("-" * len(header))
    acc = accuracy(y_true, y_pred)
    lines.append(f"{'tačnost':>13} | {acc:10.4f}")
    return "\n".join(lines)


def plot_confusion(M, classes, title="Matrica konfuzije"):
    """Vizuelizacija konfuzione matrice."""
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
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Broj primera")
    plt.tight_layout(); plt.show()

# =============================
#   STRATIFIKOVANI K-FOLD (CV)
# =============================
def stratified_kfold_indices(y, folds=5, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    classes = np.unique(y)
    buckets = [[] for _ in range(folds)]
    for c in classes:
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        parts = np.array_split(idx_c, folds)
        for f in range(folds):
            buckets[f].extend(parts[f].tolist())
    folds_idx = []
    for b in buckets:
        arr = np.array(b, dtype=int); rng.shuffle(arr)
        folds_idx.append(arr)
    return folds_idx

def select_k_by_cv(X, y, k_list, folds=5, metric="euclidean", weighted=True, seed=0):
    """
    Stratifikovani k-fold izbor k (na TRAIN-u) — bez curenja:
      - normalizacija se uvek računa SAMO na train delu svakog folda.
    Vraća: best_k, {k:(mean,std)}
    """
    folds_idx = stratified_kfold_indices(y, folds=folds, seed=seed)
    scores = {k: [] for k in k_list}
    for f in range(folds):
        val_idx = folds_idx[f]
        tr_idx = np.concatenate([folds_idx[i] for i in range(folds) if i != f])
        Xtr_f, ytr_f = X[tr_idx], y[tr_idx]
        Xvl_f, yvl_f = X[val_idx], y[val_idx]

        # normalizacija bez curenja
        Xtr_f_n, Xvl_f_n, _, _ = zscore_normalize(Xtr_f, Xvl_f)

        model = knn_fit(Xtr_f_n, ytr_f, metric=metric)
        for k in k_list:
            yhat = knn_predict(Xvl_f_n, model, k=k, weighted=weighted)
            scores[k].append(accuracy(yvl_f, yhat))

    stats = {k: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v)>1 else 0.0) for k, v in scores.items()}
    best_k = max(stats.keys(), key=lambda kk: stats[kk][0])
    return best_k, stats

def plot_kfold_results(stats, metric_name="Euklidska", title_prefix="k-fold greška"):
    """
    Crta k-fold grešku (1 - tačnost) bez prikaza standardne devijacije.
    """
    k_vals = np.array(sorted(stats.keys()))
    acc = np.array([stats[k][0] for k in k_vals])
    err = 1.0 - acc

    best_k = k_vals[np.argmin(err)]

    plt.figure(figsize=(7, 4.5))
    plt.plot(k_vals, err, "-o", linewidth=2, markersize=6)
    plt.axvline(best_k, linestyle="--", linewidth=1.5, label=f"optimalno k={best_k}")
    plt.title(f"{title_prefix} — {metric_name}")
    plt.xlabel("k")
    plt.ylabel("Greška (1 − tačnost)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# =============================
#   PCA PROJEKCIJA (samo za vizualizaciju u 2D)
# =============================
def _pca_2d(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:2].T
    Z = Xc @ W
    return Z

# =============================
#   VIZ — PODSKUP KLASA: 2D (za optimalno k)
# =============================
def plot_knn_subset_2d(X, y, classes_to_plot, k=11, metric="euclidean", weighted=True,
                       grid_step=300, title_note=""):
    """
    Višeklasna vizualizacija kNN granica odlučivanja u 2D (PCA projekcija samo za prikaz)
    za proizvoljan podskup klasa. Ne menja trening/predikciju — isključivo vizuelizacija.
    """
    classes_to_plot = np.array(sorted(classes_to_plot))
    mask = np.isin(y, classes_to_plot)
    Xs, ys = X[mask], y[mask]
    if Xs.shape[0] < len(classes_to_plot):
        print("[WARN] Premalo uzoraka za zadati podskup klasa.")
        return

    # 1) PCA → 2D samo za prikaz
    Z = _pca_2d(Xs)

    # 2) mapiranje klasa na 0..C-1
    cls_to_int = {c: i for i, c in enumerate(classes_to_plot)}
    y_int = np.vectorize(cls_to_int.get)(ys)
    C = len(classes_to_plot)

    # 3) kNN u 2D prostoru za crtanje granica
    model2d = knn_fit(Z, y_int, metric=metric)

    # 4) Mreža
    x_min, x_max = Z[:, 0].min() - 1.0, Z[:, 0].max() + 1.0
    y_min, y_max = Z[:, 1].min() - 1.0, Z[:, 1].max() + 1.0
    xs = np.linspace(x_min, x_max, grid_step)
    ys_lin = np.linspace(y_min, y_max, grid_step)
    XX, YY = np.meshgrid(xs, ys_lin)
    ZZ = np.c_[XX.ravel(), YY.ravel()]

    # 5) Predikcije
    yhat_grid = knn_predict(ZZ, model2d, k=k, weighted=weighted).reshape(XX.shape)

    # 6) Plot
    plt.figure(figsize=(7.2, 5.6))
    plt.contourf(XX, YY, yhat_grid, alpha=0.18,
                 levels=np.arange(-0.5, C-0.5+1, 1), antialiased=True)
    if C >= 2:
        boundary_lvls = np.arange(0.5, C-0.5+1e-9, 1.0)
        plt.contour(XX, YY, yhat_grid, levels=boundary_lvls, colors="k", linewidths=1.6)

    for c in classes_to_plot:
        ci = cls_to_int[c]
        plt.scatter(Z[y_int == ci, 0], Z[y_int == ci, 1], s=28, label=f"Klasa {c}")

    met = "Statistička distanca" if metric == "statisticka" else "Euklidska"
    wnm = "ponderisano" if weighted else "neponderisano"
    subset_str = ", ".join(map(str, classes_to_plot))
    plt.title(f"kNN — k={k}, {wnm}, {met}  (klase: {subset_str}){title_note}")
    plt.xlabel("Obeležje 1 (PCA)")
    plt.ylabel("Obeležje 2 (PCA)")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.show()


# =============================
#   PRIMER UPOTREBE
# =============================
if __name__ == "__main__":
    DIR_PATH = "data/new-skeletonized/"
    # DIR_PATH = "mnist_dataset/"
    features_by_class, X, y, idx_by_class = extract_features_from_dir(dir_path=DIR_PATH)

    # Train/Test podela (zaključan test)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    n_tr = int(0.8*len(y))
    tr, te = idx[:n_tr], idx[n_tr:]
    Xtr, ytr = X[tr], y[tr]
    Xte, yte = X[te], y[te]

    # Normalizacija (μ,σ iz TRAIN) — teorijski korektno
    Xtr_n, Xte_n, mu, sigma = zscore_normalize(Xtr, Xte)

    # lista k nad kojima se trazi optimalni
    k_list = list(range(1, 21))
    folds  = 5
    seed   = 0

    # --- EUKLIDSKA ---
    print("== k-fold izbor k (Euklidska) ==")
    best_k_euc, stats_euc = select_k_by_cv(Xtr, ytr, k_list, folds, metric="euclidean", weighted=True, seed=seed)
    plot_kfold_results(stats_euc, metric_name="Euklidska")
    for k in k_list:
        m, s = stats_euc[k]; print(f"k={k:>2}  CV-tačnost={m:.4f} ± {s:.4f}")
    print(f"[CV-EUKLIDSKA] najbolji k={best_k_euc}")

    model_euc = knn_fit(Xtr_n, ytr, metric="euclidean")
    yhat_euc = knn_predict(Xte_n, model_euc, k=best_k_euc, weighted=True)
    print("\n[TEST-EUKLIDSKA]")
    print(classification_report(yte, yhat_euc))
    M, cls = confusion_matrix(yte, yhat_euc)
    plot_confusion(M, cls, title=f"Matrica konfuzije — kNN (Euklidska, k={best_k_euc})")

    # --- STATISTIČKA DISTANCA ---
    print("\n== k-fold izbor k (Statistička distanca) ==")
    best_k_stat, stats_stat = select_k_by_cv(Xtr, ytr, k_list, folds, metric="statisticka", weighted=True, seed=seed)
    plot_kfold_results(stats_stat, metric_name="Statistička distanca")
    for k in k_list:
        m, s = stats_stat[k]; print(f"k={k:>2}  CV-tačnost={m:.4f} ± {s:.4f}")
    print(f"[CV-STATISTIČKA] najbolji k={best_k_stat}")

    model_stat = knn_fit(Xtr_n, ytr, metric="statisticka")
    yhat_stat = knn_predict(Xte_n, model_stat, k=best_k_stat, weighted=True)
    print("\n[TEST-STATISTIČKA]")
    print(classification_report(yte, yhat_stat))
    M, cls = confusion_matrix(yte, yhat_stat)
    plot_confusion(M, cls, title=f"Matrica konfuzije — kNN (Statistička, k={best_k_stat})")

    # --- Vizuelizacije SAMO za optimalno k (2D) ---
    train_classes = np.unique(ytr)

    # (A) EUKLIDSKA — optimalni k (2D)
    if len(train_classes) >= 2:
        plot_knn_subset_2d(Xtr_n, ytr, train_classes[:2], k=best_k_euc, metric="euclidean", weighted=True,
                           title_note="  (TRAIN, PCA 2D)")
    if len(train_classes) >= 3:
        plot_knn_subset_2d(Xtr_n, ytr, train_classes[:3], k=best_k_euc, metric="euclidean", weighted=True,
                           title_note="  (TRAIN, PCA 2D)")
    plot_knn_subset_2d(Xtr_n, ytr, train_classes, k=best_k_euc, metric="euclidean", weighted=True,
                       title_note="  (TRAIN, PCA 2D)")

    # (B) STATISTIČKA DISTANCA — optimalni k (2D)
    if len(train_classes) >= 2:
        plot_knn_subset_2d(Xtr_n, ytr, train_classes[:2], k=best_k_stat, metric="statisticka", weighted=True,
                           title_note="  (TRAIN, PCA 2D)")
    if len(train_classes) >= 3:
        plot_knn_subset_2d(Xtr_n, ytr, train_classes[:3], k=best_k_stat, metric="statisticka", weighted=True,
                           title_note="  (TRAIN, PCA 2D)")
    plot_knn_subset_2d(Xtr_n, ytr, train_classes, k=best_k_stat, metric="statisticka", weighted=True,
                       title_note="  (TRAIN, PCA 2D)")
