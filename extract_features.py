import cv2
import numpy as np
from matplotlib.gridspec import GridSpec
from skimage.morphology import skeletonize
from matplotlib import pyplot as plt
import os, glob
import numpy as np
from skimage.feature import hog as skimage_hog

def horizontal_black_density(img, k=4, dtype=np.float32):
    """
    Funkcija deli već binarnu sliku na k horizontalnih segmenata i računa
    gustinu crnih piksela u svakom segmentu.
    Očekuje se da je 'img' 2D numpy niz sa vrednostima 0/1 ili 0/255.
    Rezultat: numpy niz dužine k sa vrednostima [0,1].
    """
    img = np.asarray(img)
    if img.ndim != 2:
        raise ValueError("Očekujem 2D sliku (H,W).")

    H, W = img.shape
    edges = np.linspace(0, H, num=k+1, dtype=int)

    dens = np.empty(k, dtype=dtype)
    for i in range(k):
        r0, r1 = edges[i], edges[i + 1]
        seg = img[r0:r1, :]
        dens[i] = np.count_nonzero(seg == 0) / seg.size if seg.size > 0 else 0.0
    return dens

def vertical_black_density(img, k=4, dtype=np.float32):
    """
    Deli binarnu sliku (0=crno, >0=belo) na k vertikalnih segmenata
    i vraća gustinu crnih piksela po segmentu kao niz dužine k.
    """
    H, W = img.shape
    edges = np.linspace(0, W, k+1, dtype=int)

    dens = np.empty(k, dtype=dtype)
    for i in range(k):
        c0, c1 = edges[i], edges[i+1]
        seg = img[:, c0:c1]
        dens[i] = np.count_nonzero(seg == 0) / seg.size if seg.size > 0 else 0.0
    return dens

def plot_image_with_densities(img, kx = None, ky = None):
    """
    Prikazuje sliku, ispod gustinu crnog po koloni (x), desno po redu (y).
    (0=crno, >0=belo).
    """
    H, W = img.shape
    kx = W if kx is None else int(kx)
    ky = H if ky is None else int(ky)
    kx = max(1, kx)
    ky = max(1, ky)

    # izračun gustina po segmentima (nezavisno od piksela)
    x_density = vertical_black_density(img, k=kx)  # dužina = kx
    y_density = horizontal_black_density(img, k=ky)  # dužina = ky

    # === CRTANJE ===
    fig = plt.figure(figsize=(8, 6), constrained_layout=True)

    # Fiksne srazmere (ne zavise od H/W) i bez deljenja osa!
    gs = GridSpec(2, 2, figure=fig, height_ratios=[4, 1], width_ratios=[4, 1])

    ax_img = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])  # bez sharey
    ax_bot = fig.add_subplot(gs[1, 0])  # bez sharex

    # 1) Slika
    ax_img.imshow(img, cmap="gray", interpolation="nearest")
    ax_img.set_title("Slika (binarna)")
    ax_img.axis("off")

    # 2) Gustina po kolonama (x) — osa 0..kx-1
    ax_bot.plot(np.arange(kx), x_density)
    ax_bot.set_xlim(0, kx - 1)
    ax_bot.set_xlabel("segment po x (0..kx-1)")
    ax_bot.set_ylabel("gustina crnog")
    ax_bot.grid(True, alpha=0.3)

    # 3) Gustina po redovima (y) — osa 0..ky-1 (invertovana da odgovara vrhu slike)
    ax_right.plot(y_density, np.arange(ky))
    ax_right.set_ylim(ky - 1, 0)
    ax_right.set_xlabel("gustina crnog")
    ax_right.set_ylabel("segment po y (0..ky-1)")
    ax_right.grid(True, alpha=0.3)

    plt.show()
    return x_density, y_density

##classes

def feature_aspect_ratio(img, threshold=127):
    """
    Odnos širina/visina foreground-a (0 pikseli = crno = foreground).
    Računa se na bounding-boxu crnih piksela. Ako nema foregrounda, vraća 0.0.
    """

    mask = (img == 0)
    if not np.any(mask):
        return np.float32(0.0)

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r_idx = np.where(rows)[0]
    c_idx = np.where(cols)[0]
    r0, r1 = r_idx[0], r_idx[-1]
    c0, c1 = c_idx[0], c_idx[-1]

    height = (r1 - r0 + 1)
    width  = (c1 - c0 + 1)
    if height <= 0:
        return np.float32(0.0)
    return np.float32(width / float(height))


def feature_center_of_mass(img, threshold=127):
    """
    Vraća radijalnu udaljenost centra mase crnih piksela (0=crno)
    od centra slike, normalizovanu na [0, 1].
    Ako nema foregrounda, vraća 0.0.
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    mask = (bin_img == 0).astype(np.float64)
    mass = mask.sum()
    H, W = mask.shape
    if mass <= 0:
        return np.float32(0.0)

    xs = np.arange(W, dtype=np.float64)
    ys = np.arange(H, dtype=np.float64)
    cx = (mask.sum(axis=0) * xs).sum() / mass
    cy = (mask.sum(axis=1) * ys).sum() / mass

    cx_norm = cx / max(W - 1, 1.0)
    cy_norm = cy / max(H - 1, 1.0)

    # radijalna udaljenost od centra (0.5, 0.5)
    r = np.sqrt((cx_norm - 0.5) ** 2 + (cy_norm - 0.5) ** 2)
    return np.float32(r)


def count_transitions_xy(img, threshold=127, normalize=False):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    bw = (bin_img == 0).astype(np.uint8)

    H, W = bw.shape
    h_trans = np.sum(bw[:, 1:] != bw[:, :-1])
    v_trans = np.sum(bw[1:, :] != bw[:-1, :])

    if not normalize:
        return float(h_trans), float(v_trans)

    h_norm = h_trans / max(H * (W - 1), 1)
    v_norm = v_trans / max(W * (H - 1), 1)
    return float(h_norm), float(v_norm)

def feature_image_moments(img, threshold=127):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    w = (bin_img == 0).astype(np.float64)

    H, W = w.shape
    mass = w.sum()
    if mass == 0:
        return np.float32(0.0)

    xs = np.arange(W, dtype=np.float64)
    ys = np.arange(H, dtype=np.float64)
    cx = (w.sum(axis=0) * xs).sum() / mass
    cy = (w.sum(axis=1) * ys).sum() / mass
    X, Y = np.meshgrid(xs - cx, ys - cy)

    mu20 = (w * (X**2)).sum()
    mu02 = (w * (Y**2)).sum()
    mu11 = (w * (X*Y)).sum()

    # "energija" centralnih momenata drugog reda
    moment_feature = np.sqrt(mu20**2 + mu02**2 + 2*mu11**2) / (mass**2)
    return np.float32(moment_feature)


def compute_features(img, kx=4, ky=4, threshold=127):
    # obezbedi binarnu 2D sliku (0=crno, 255=belo)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if not (np.array_equal(img, img.astype(bool)*255) or np.array_equal(img, img.astype(bool)*0)):
        _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    x_density = horizontal_black_density(img, k=kx).astype(np.float32)
    mean_x = float(np.mean(x_density))
    var_x = np.var(x_density)
    y_density = vertical_black_density(img, k=ky).astype(np.float32)
    var_y = np.var(y_density)

    moment_feature = feature_image_moments(img)
    h, v = count_transitions_xy(img)
    r = feature_center_of_mass(img, threshold=threshold)

    return np.array([mean_x, var_x, var_y, moment_feature, h, v, r], dtype=np.float32)

def zscore_normalize(X_train, X_test=None):
    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0) + 1e-8
    X_train_n = (X_train - mu) / sigma
    if X_test is not None:
        X_test_n = (X_test - mu) / sigma
        return X_train_n, X_test_n, mu, sigma
    return X_train_n, mu, sigma

def extract_features_from_dir(dir_path="data/skeletonized/", classes=range(10), kx=4, ky=4,
                              exts=".png", threshold=127, binarize=True):
    """
    Prolazi kroz podfoldere '0'..'9' (ili dati 'classes'), učitava slike pomoću load_image,
    opciono binarizuje binarize_image, računa feature-e compute_features
    i vraća praktičnu strukturu za dalje korišćenje.
    """
    import os, glob
    features_by_class = {}
    X_list, y_list = [], []

    for cls in classes:
        cls_dir = os.path.join(dir_path, str(cls))
        if not os.path.isdir(cls_dir):
            print(f"[WARN] Preskačem (ne postoji): {cls_dir}")
            continue

        files = sorted(glob.glob(os.path.join(cls_dir, f"*{exts}")))
        if not files:
            print(f"[WARN] Nema slika u: {cls_dir}")
            continue

        per_class = []
        for fp in files:
            try:
                img = load_image(fp)
            except FileNotFoundError:
                print(f"[WARN] Ne mogu da učitam: {fp}")
                continue

            if binarize:
                img = binarize_image(img, threshold=threshold, invert=False)

            f = compute_features(img, kx=kx, ky=ky, threshold=threshold)
            per_class.append(f)
            X_list.append(f)
            y_list.append(int(cls))

        if per_class:
            features_by_class[int(cls)] = np.vstack(per_class).astype(np.float32)

    if not X_list:
        raise RuntimeError("Nije učitana nijedna slika / nema feature-a.")

    X = np.vstack(X_list).astype(np.float32)           # (N, D)
    y = np.asarray(y_list, dtype=np.uint8)             # (N,)
    idx_by_class = {c: np.where(y == c)[0] for c in np.unique(y)}

    return features_by_class, X, y, idx_by_class


##plotting

def plot_feature_corr(X, feature_names=None, show_cov=False, figsize=(7, 6), annotate=False):
    """
    Crta Pearsonovu korelacionu matricu feature-a; opciono i kovarijacionu.
    - X: (N, D) matrica obeležja
    - feature_names: lista dužine D (ako None, generišu se F0..F{D-1})
    - show_cov: ako True, nacrta i kovarijacionu matricu u posebnoj figuri
    - annotate: ako True, upisuje brojeve na ćelije (za male D)
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError("X mora biti np.ndarray oblika (N, D).")
    N, D = X.shape
    if feature_names is None:
        feature_names = [f"F{i}" for i in range(D)]

    # 1) Pearsonova korelacija
    corr = np.corrcoef(X, rowvar=False)  # (D, D)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr, interpolation="nearest")
    ax.set_title("Pearson correlation matrix")
    ax.set_xticks(np.arange(D)); ax.set_yticks(np.arange(D))
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_yticklabels(feature_names)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if annotate and D <= 20:
        for i in range(D):
            for j in range(D):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.show()

    # 2) (opciono) kovarijaciona matrica
    if show_cov:
        cov = np.cov(X, rowvar=False)  # (D, D)
        fig2, ax2 = plt.subplots(figsize=figsize)
        im2 = ax2.imshow(cov, interpolation="nearest")
        ax2.set_title("Covariance matrix")
        ax2.set_xticks(np.arange(D)); ax2.set_yticks(np.arange(D))
        ax2.set_xticklabels(feature_names, rotation=45, ha="right")
        ax2.set_yticklabels(feature_names)
        fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        if annotate and D <= 20:
            for i in range(D):
                for j in range(D):
                    ax2.text(j, i, f"{cov[i, j]:.2e}", ha="center", va="center", fontsize=8)
        plt.tight_layout()
        plt.show()

def plot_feature_space_2d(X, y, idx_by_class, feat_i=0, feat_j=1,
                          title=None, figsize=(7, 6), alpha=0.75, s=18,
                          show_centroids=True, equal_axes=False):
    N, D = X.shape
    classes = np.unique(y)

    fig, ax = plt.subplots(figsize=figsize)
    for c in classes:
        idx = idx_by_class[c]
        color = plt.cm.tab10(int(c) % 10)  # jedinstvena boja za klasu

        xi, xj = X[idx, feat_i], X[idx, feat_j]
        ax.scatter(xi, xj, s=s, alpha=alpha, label=str(c), c=[color])
        if show_centroids:
            cx, cy = float(np.mean(xi)), float(np.mean(xj))
            ax.scatter([cx], [cy], marker="x", s=s * 3, linewidths=1.5, c=[color])

    ax.set_xlabel(f"Feature {feat_i}")
    ax.set_ylabel(f"Feature {feat_j}")
    ax.grid(True, alpha=0.3)
    ax.set_title(title or f"2D feature space: feat {feat_i} vs feat {feat_j}")
    ax.legend(title="Class", ncols=min(5, len(classes)), fontsize=9)

    if equal_axes:
        ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.show()

def plot_class_densities(class_label, dataset_root="data/skeletonized", br_kolona=5, br_redova=5, threshold=127, binarize=True):
    import glob, os
    paths = sorted(glob.glob(os.path.join(dataset_root, str(class_label), "*.png")))
    if not paths:
        print(f"Nema slika za klasu {class_label}.")
        return None

    x_list, y_list = [], []
    ref_shape = None
    kx = ky = None

    for p in paths:
        img = load_image(p)
        if binarize:
            img = binarize_image(img, threshold=threshold, invert=False)

        H, W = img.shape
        if ref_shape is None:
            ref_shape = (H, W)
            kx = W if br_kolona is None else int(br_kolona)
            ky = H if br_redova is None else int(br_redova)
        elif (H, W) != ref_shape:
            continue

        xd = vertical_black_density(img, k=kx).astype(np.float32)
        yd = horizontal_black_density(img, k=ky).astype(np.float32)
        x_list.append(xd)
        y_list.append(yd)

    if not x_list:
        print(f"Nema validnih slika (dimenzije ne poklapaju) za klasu {class_label}.")
        return None

    X = np.vstack(x_list)
    Y = np.vstack(y_list)
    mean_x = X.mean(axis=0)
    mean_y = Y.mean(axis=0)
    std_x  = X.std(axis=0, ddof=1) if X.shape[0] > 1 else np.zeros_like(mean_x)
    std_y  = Y.std(axis=0, ddof=1) if Y.shape[0] > 1 else np.zeros_like(mean_y)

    peak_x_idx = int(np.argmax(mean_x))
    peak_y_idx = int(np.argmax(mean_y))

    overall_black_ratio_x = float(mean_x.mean())
    overall_black_ratio_y = float(mean_y.mean())

    stats = {
        "class": int(class_label),
        "n_samples": int(X.shape[0]),
        "kx": int(kx),
        "ky": int(ky),
        "mean_x": mean_x.astype(np.float32),
        "mean_y": mean_y.astype(np.float32),
        "std_x": std_x.astype(np.float32),
        "std_y": std_y.astype(np.float32),
        "peak_x_idx": peak_x_idx,
        "peak_y_idx": peak_y_idx,
        "overall_black_ratio_x": overall_black_ratio_x,
        "overall_black_ratio_y": overall_black_ratio_y,
    }

    print(f"\n[Kvantitativno] Klasa {class_label} | uzoraka: {stats['n_samples']}")
    print(f"- kx={stats['kx']}, ky={stats['ky']}")
    print(f"- peak_x_idx={peak_x_idx}, peak_y_idx={peak_y_idx}")
    print(f"- overall_black_ratio_x={overall_black_ratio_x:.4f}, overall_black_ratio_y={overall_black_ratio_y:.4f}")
    print(f"- std_x_mean={std_x.mean():.4f}, std_y_mean={std_y.mean():.4f}")

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), constrained_layout=True)

    ax = axes[0]
    for yd in Y:
        ax.plot(np.arange(ky), yd, alpha=0.15)
    ax.plot(np.arange(ky), mean_y, linewidth=3, label="mean")
    ax.set_title(f"Klasa {class_label} – y_densities (po redovima)")
    ax.set_xlabel("segment po y (0..ky-1)")
    ax.set_ylabel("gustina crnog")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    for xd in X:
        ax.plot(np.arange(kx), xd, alpha=0.15)
    ax.plot(np.arange(kx), mean_x, linewidth=3, label="mean")
    ax.set_title(f"Klasa {class_label} – x_densities (po kolonama)")
    ax.set_xlabel("segment po x (0..kx-1)")
    ax.set_ylabel("gustina crnog")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.show()
    return stats

def plot_cov_corr_matrix(X, feature_names=None, show_corr=True, figsize=(6, 5)):

    N, D = X.shape
    if feature_names is None:
        feature_names = [f"F{i}" for i in range(D)]

    cov = np.cov(X, rowvar=False)
    corr = np.corrcoef(X, rowvar=False)

    abs_corr = np.abs(corr)
    mean_corr = abs_corr.mean(axis=1)
    order = np.argsort(-mean_corr)

    cov_sorted = cov[order][:, order]
    corr_sorted = corr[order][:, order]
    feat_sorted = [feature_names[i] for i in order]

    data_orig = corr if show_corr else cov
    data_sorted = corr_sorted if show_corr else cov_sorted
    title_base = "Korelaciona" if show_corr else "Kovarijaciona"

    # Figura 1: pre sortiranja
    fig1, ax1 = plt.subplots(figsize=figsize)
    im1 = ax1.imshow(data_orig, cmap="coolwarm", interpolation="nearest")
    ax1.set_xticks(np.arange(D))
    ax1.set_yticks(np.arange(D))
    ax1.set_xticklabels(feature_names, rotation=45, ha="right")
    ax1.set_yticklabels(feature_names)
    ax1.set_title(f"{title_base} matrica (pre sortiranja)")
    fig1.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    for i in range(D):
        for j in range(D):
            ax1.text(j, i, f"{data_orig[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.show()

    # Figura 2: nakon sortiranja
    fig2, ax2 = plt.subplots(figsize=figsize)
    im2 = ax2.imshow(data_sorted, cmap="coolwarm", interpolation="nearest")
    ax2.set_xticks(np.arange(D))
    ax2.set_yticks(np.arange(D))
    ax2.set_xticklabels(feat_sorted, rotation=45, ha="right")
    ax2.set_yticklabels(feat_sorted)
    ax2.set_title(f"{title_base} matrica (nakon sortiranja)")
    fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    for i in range(D):
        for j in range(D):
            ax2.text(j, i, f"{data_sorted[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.show()

##utils

def load_image(filepath):
    """
    Učitava sliku sa zadate putanje pomoću OpenCV-a i vraća je kao numpy niz.
    Ako je as_gray=True, slika se učitava u grayscale modu (2D niz).
    """
    flag = cv2.IMREAD_GRAYSCALE
    img = cv2.imread(filepath, flag)
    if img is None:
        raise FileNotFoundError(f"Ne mogu da učitam sliku sa putanje: {filepath}")
    return img

def show_image(img):
    cv2.imshow('image', img)
    cv2.waitKey()

def binarize_image(img, threshold=127, invert=False):
    """
    Binarizuje grayscale sliku. Ako invert=True, crno i belo se zamene.
    Rezultat: 0 i 255.
    """
    if img.ndim != 2:
        raise ValueError("Ocekujem grayscale sliku.")
    ttype = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(img, threshold, 255, ttype)
    return binary

if __name__ == "__main__":
    DIR_PATH = "data/skeletonized/"

    br_kolona, br_redova = 4, 4

    # for i in range(0,10):
    #
    #     image_example = load_image(DIR_PATH+f'{i}/{i}_4_0.png')
    #     image_example = binarize_image(image_example)
    #     plot_image_with_densities(image_example, kx= br_kolona, ky=br_redova)
    #
    # for i in range(0,10):
    #     plot_class_densities(class_label=f'{i}', dataset_root='data/skeletonized')
    #     plot_class_densities(class_label=f'{i}', dataset_root='data/non-skeletonized')


    features_by_class, X, y, idx_by_class = extract_features_from_dir(dir_path=DIR_PATH)
    num_of_feats = len(features_by_class[0][0])


    features_names = ["mean_x", "var_x", "var_y","moment_feature", "hor_prelazi", "vert_prelazi", "r"]
    plot_cov_corr_matrix(X, feature_names=features_names, show_corr=True)

    # for i in range(0, num_of_feats-1):
    #     for j in range(i+1, num_of_feats):
    #         plot_feature_space_2d(X, y, idx_by_class, feat_i=i, feat_j=j, title=f"obelezje_{j}/obelezje_{i}")



