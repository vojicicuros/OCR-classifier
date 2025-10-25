import numpy as np
import matplotlib.pyplot as plt
from extract_features import *

# =============================
#   KNN: FIT / PREDICT
# =============================
def _stat_dist_inv_cov(X):
    # Kov. matrica i moor-pseudoinverz (robustnije za D>N/degeneraciju)
    S = np.cov(X, rowvar=False)
    return np.linalg.pinv(S)

def knn_fit(X, y, metric="euclidean"):
    """
    Memorijski kNN 'model' — samo čuva X, y i opciono S^{-1} za Mahalanobis.
    metric: 'euclidean' | 'mahalanobis'
    """
    X = X.astype(np.float64, copy=False)
    y = y.astype(int, copy=False)
    model = {"X": X, "y": y, "metric": metric, "n_classes": int(y.max()) + 1}
    if metric == "mahalanobis":
        model["Sinv"] = _stat_dist_inv_cov(X)
    else:
        model["Sinv"] = None
    return model

def _dist2(Xtr, x, Sinv=None):
    D = Xtr - x
    if Sinv is None:                      # Euklidska kvadrirana
        return np.einsum("ij,ij->i", D, D)
    # Mahalanobis kvadrirana
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
#   METRIKE (kratko i jasno)
# =============================
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
    precision = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count>0)
    recall    = np.divide(tp, support,    out=np.zeros_like(tp), where=support>0)
    denom     = precision + recall
    f1        = np.divide(2*precision*recall, denom, out=np.zeros_like(denom), where=denom>0)
    return precision, recall, f1, support

def classification_report(y_true, y_pred, classes=None):
    M, classes = confusion_matrix(y_true, y_pred, classes)
    p, r, f1, sup = _per_class_prf(M)
    total = int(sup.sum())
    lines = []
    header = f"{'klasa':>8} | {'preciznost':>10} {'odziv':>9} {'f1-rez.':>9} {'broj':>6}"
    lines += [header, "-"*len(header)]
    for i, c in enumerate(classes):
        lines.append(f"{str(c):>8} | {p[i]:10.4f} {r[i]:9.4f} {f1[i]:9.4f} {int(sup[i]):6d}")
    lines.append("-"*len(header))
    acc = accuracy(y_true, y_pred)
    lines.append(f"{'tačnost':>13} | {acc:10.4f}")
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

        # z-score bez curenja
        Xtr_f_n, Xvl_f_n, _, _ = zscore_normalize(Xtr_f, Xvl_f)

        model = knn_fit(Xtr_f_n, ytr_f, metric=metric)
        for k in k_list:
            yhat = knn_predict(Xvl_f_n, model, k=k, weighted=weighted)
            scores[k].append(accuracy(yvl_f, yhat))

    stats = {k: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v)>1 else 0.0) for k, v in scores.items()}
    best_k = max(stats.keys(), key=lambda kk: stats[kk][0])
    return best_k, stats

# =============================
#   PCA→2D PROJEKCIJA
# =============================
def _pca_2d(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:2].T
    Z = Xc @ W
    return Z

# =============================
#   VIZ — PAIRWISE (2 KLASE) SA GRANICOM
# =============================
def plot_pairwise_knn(X, y, cls_a, cls_b, k=11, metric="euclidean", weighted=True,
                      grid_step=200, title_note=""):
    """
    Vizualizacija odluka kNN u 2D (PCA projekcija samo za prikaz) za dve klase.
    Crta i eksplicitnu granicu odlučivanja između te dve klase.
    """
    mask = (y == cls_a) | (y == cls_b)
    Xab, yab = X[mask], y[mask]
    if Xab.shape[0] < 2:
        print(f"Premalo uzoraka za par ({cls_a}, {cls_b})."); return

    Z = _pca_2d(Xab)   # 2D prikaz

    # mapiranje etiketa na {0,1} radi jasnog crtanja granice na 0.5
    classes = np.array([cls_a, cls_b])
    cls_to_int = {cls_a: 0, cls_b: 1}
    y_bin = np.vectorize(cls_to_int.get)(yab)

    # treniramo kNN u 2D na binarnim etiketama
    model2d = knn_fit(Z, y_bin, metric=metric)

    # mreža
    x_min, x_max = Z[:,0].min()-1.0, Z[:,0].max()+1.0
    y_min, y_max = Z[:,1].min()-1.0, Z[:,1].max()+1.0
    xs = np.linspace(x_min, x_max, grid_step)
    ys = np.linspace(y_min, y_max, grid_step)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = np.c_[XX.ravel(), YY.ravel()]
    yhat_grid = knn_predict(ZZ, model2d, k=k, weighted=weighted).reshape(XX.shape)

    # crtanje
    plt.figure(figsize=(6.5, 4.8))
    plt.contourf(XX, YY, yhat_grid, alpha=0.15, levels=np.array([-0.5,0.5,1.5]), antialiased=True)

    # granica odlučivanja: kontura binarne mape na 0.5
    plt.contour(XX, YY, yhat_grid, levels=[0.5], colors="k", linewidths=2)

    # originalne tačke sa originalnim oznakama
    plt.scatter(Z[y_bin==0,0], Z[y_bin==0,1], s=40, label=f"Klasa {cls_a}")
    plt.scatter(Z[y_bin==1,0], Z[y_bin==1,1], s=40, label=f"Klasa {cls_b}")

    met = "Mahalanobisova" if metric=="mahalanobis" else "Euklidska"
    wnm = "ponderisano" if weighted else "neponderisano"
    plt.title(f"kNN (par {cls_a} vs {cls_b}) — k={k}, {wnm}, {met}{title_note}")
    plt.xlabel("Obeležje 1 (PCA)"); plt.ylabel("Obeležje 2 (PCA)")
    plt.legend(); plt.tight_layout(); plt.show()

# =============================
#   VIZ — VIŠEKLASNO (SVE KLASE) SA GRANICAMA
# =============================
def plot_knn_multiclass_2d(X, y, k=11, metric="euclidean", weighted=True,
                           grid_step=300, title_note=""):
    """
    Višeklasna (multi-class) vizualizacija kNN granica odlučivanja u 2D (PCA projekcija).
    Crta sve diskriminacione krive između klasa.
    """
    # 1) PCA → 2D
    Z = _pca_2d(X)
    classes = np.unique(y)
    C = len(classes)

    # mapiranje klasa na 0..C-1 radi kontura između susednih celobrojnih nivoa
    cls_to_int = {c: i for i, c in enumerate(classes)}
    y_int = np.vectorize(cls_to_int.get)(y)

    # 2) kNN u 2D
    model2d = knn_fit(Z, y_int, metric=metric)

    # 3) Mreža tačaka
    x_min, x_max = Z[:, 0].min() - 1.0, Z[:, 0].max() + 1.0
    y_min, y_max = Z[:, 1].min() - 1.0, Z[:, 1].max() + 1.0
    xs = np.linspace(x_min, x_max, grid_step)
    ys = np.linspace(y_min, y_max, grid_step)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = np.c_[XX.ravel(), YY.ravel()]

    # 4) Predikcije na mreži
    yhat_grid = knn_predict(ZZ, model2d, k=k, weighted=weighted).reshape(XX.shape)

    # 5) Pozadinski regioni (diskretne klase)
    plt.figure(figsize=(7.2, 5.6))
    plt.contourf(XX, YY, yhat_grid, alpha=0.18,
                 levels=np.arange(-0.5, C-0.5+1, 1), antialiased=True)

    # 6) Granice: konture na polu-integer nivoima 0.5, 1.5, ..., C-1.5
    if C >= 2:
        boundary_lvls = np.arange(0.5, C-0.5+1e-9, 1.0)
        plt.contour(XX, YY, yhat_grid, levels=boundary_lvls, colors="k", linewidths=1.6)

    # 7) Tačke podataka
    for c in classes:
        ci = cls_to_int[c]
        plt.scatter(Z[y_int == ci, 0], Z[y_int == ci, 1], s=28, label=f"Klasa {c}")

    # 8) Oznake
    met = "Mahalanobisova" if metric == "mahalanobis" else "Euklidska"
    wnm = "ponderisano" if weighted else "neponderisano"
    plt.title(f"kNN višeklasno — k={k}, {wnm}, {met}{title_note}")
    plt.xlabel("Obeležje 1 (PCA)")
    plt.ylabel("Obeležje 2 (PCA)")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.show()

def compare_knn_k_values(X, y, k_list=(1, 5, 20), metric="euclidean", weighted=True):
    """
    Uporedna vizuelizacija za više vrednosti k — crta po JEDAN graf za svaki k.
    """
    for k in k_list:
        plot_knn_multiclass_2d(X, y, k=k, metric=metric, weighted=weighted,
                               title_note=f"  (poređenje K)")

# =============================
#   PRIMER UPOTREBE
# =============================
if __name__ == "__main__":
    DIR_PATH = "data/skeletonized/"
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

    k_list = [1,3,5,7,9,11,15]
    folds  = 5
    seed   = 0

    # --- EUKLIDSKA ---
    print("== k-fold izbor k (Euklidska) ==")
    best_k_euc, stats_euc = select_k_by_cv(Xtr, ytr, k_list, folds, metric="euclidean", weighted=True, seed=seed)
    for k in k_list:
        m, s = stats_euc[k]; print(f"k={k:>2}  CV-tačnost={m:.4f} ± {s:.4f}")
    print(f"[CV-EUKLIDSKA] najbolji k={best_k_euc}")

    model_euc = knn_fit(Xtr_n, ytr, metric="euclidean")
    yhat_euc = knn_predict(Xte_n, model_euc, k=best_k_euc, weighted=True)
    print("\n[TEST-EUKLIDSKA]")
    print(classification_report(yte, yhat_euc))
    M, cls = confusion_matrix(yte, yhat_euc)
    plot_confusion(M, cls, title=f"Matrica konfuzije — kNN (Euklidska, k={best_k_euc})")

    # --- MAHALANOBIS ---
    print("\n== k-fold izbor k (Mahalanobisova) ==")
    best_k_maha, stats_maha = select_k_by_cv(Xtr, ytr, k_list, folds, metric="mahalanobis", weighted=True, seed=seed)
    for k in k_list:
        m, s = stats_maha[k]; print(f"k={k:>2}  CV-tačnost={m:.4f} ± {s:.4f}")
    print(f"[CV-MAHALANOBISOVA] najbolji k={best_k_maha}")

    model_maha = knn_fit(Xtr_n, ytr, metric="mahalanobis")
    yhat_maha = knn_predict(Xte_n, model_maha, k=best_k_maha, weighted=True)
    print("\n[TEST-MAHALANOBISOVA]")
    print(classification_report(yte, yhat_maha))
    M, cls = confusion_matrix(yte, yhat_maha)
    plot_confusion(M, cls, title=f"Matrica konfuzije — kNN (Mahalanobisova, k={best_k_maha})")

    # --- Vizuelizacije na TRAIN podacima (PCA→2D) ---
    # 1) Par-klasa granica (primer za prve dve klase)
    classes = np.unique(ytr)
    if len(classes) >= 2:
        plot_pairwise_knn(Xtr_n, ytr, classes[0], classes[1], k=best_k_euc, metric="euclidean", weighted=True,
                          title_note="  (PCA 2D)")

    # 2) Višeklasne granice (sve klase odjednom) za najbolji k
    plot_knn_multiclass_2d(Xtr_n, ytr, k=best_k_euc, metric="euclidean", weighted=True,
                           title_note="  (TRAIN, PCA 2D)")

    # 3) Poređenje više K vrednosti (npr. 1, 5, 11, 21)
    compare_knn_k_values(Xtr_n, ytr, k_list=[1, 5, 11, 21], metric="euclidean", weighted=True)
