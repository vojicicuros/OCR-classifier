import cv2
import numpy as np
from skimage.morphology import skeletonize
from matplotlib import pyplot as plt
import os

def read_img(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error: Could not load image from {path}")
    else:
        print("Image loaded successfully.")

    return img

def crop_padding_image(image):
    height, width = image.shape

    up, down, left, right = None, None, None, None

    for i in range(0,height):
        line = image[i][:]
        if 0 in line:
            up = i
            break

    for i in range(height-1,0,-1):
        line = image[i][:]
        if 0 in line:
            down = i
            break

    for j in range(0,width):
        line = image[:, j]
        if 0 in line:
            left = j
            break

    for j in range(width-1, 0, -1):
        line = image[:, j]
        if 0 in line:
            right = j
            break

    cropped_image = image[up : down, left: right]
    return cropped_image

def split_digits_grid(image, save_dir=None, label=None, show_plots=False, target_size=(28, 28)):
    """
    Ako je save_dir zadat, svaka obrađena pod-slika se snima u taj folder
    kao {label}_{i}_{j}.png. Ako je show_plots=True, prikazuje 1x2 subplot.
    """
    height, width = image.shape
    single_height = height // 10
    single_width = width // 12

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for i in range(0, 10):
        for j in range(0, 12):
            single_img = image[i*single_height:i*single_height + single_height,
                               j*single_width:j*single_width + single_width]

            processed_img = processing_pipeline(single_img, target_size=target_size)

            # snimi ako je traženo
            if save_dir is not None:
                fname = f"{label}_{i}_{j}.png" if label is not None else f"{i}_{j}.png"
                out_path = os.path.join(save_dir, fname)
                # processed_img = (processed_img > 0).astype(np.uint8) * 255
                cv2.imwrite(out_path, processed_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])

            if show_plots:
                plt.figure(figsize=(8, 4))
                plt.subplot(1, 2, 1)
                plt.imshow(single_img, cmap="gray")
                plt.title("Original")
                plt.axis("off")

                plt.subplot(1, 2, 2)
                plt.imshow(processed_img, cmap="gray")
                plt.title("Processed")
                plt.axis("off")

                plt.tight_layout()
                plt.show()

def binary_mask(image, threshold=230):
    """Vrati ČISTO binarnu sliku: 0 ili 255 (ništa između)."""
    bin_img = (image > threshold).astype(np.uint8) * 255  # True->255, False->0
    return bin_img

def image_erosion(image, kernel_dim = (3, 3), iterations_num=2):
    kernel = np.ones(kernel_dim , np.uint8)
    dilated_img = cv2.erode(image, kernel, iterations=iterations_num)
    return dilated_img

def image_dilatation(image, kernel_dim = (3, 3), iterations_num=2):
    kernel = np.ones(kernel_dim , np.uint8)
    eroded_img = cv2.erode(255 - image, kernel, iterations=iterations_num)
    return 255 - eroded_img

def resize_image(image, target_size=(32, 32)):
    # NEAREST bez sivljenja + re-binarizacija na 0/1
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)
    # resized01 = (resized > 0).astype(np.uint8)  # 0/1 umesto 0/255
    return resized

def skeletonize_image(image):
    image = image < 150   # crno postaje True
    skeleton = skeletonize(image)

    # Pretvori nazad u uint8: True=linija=crno (0), False=pozadina=belo (255)
    skeleton_img = np.where(skeleton, 0, 255).astype(np.uint8)

    return skeleton_img

def processing_pipeline(image, target_size=(32, 32)):

    img = cv2.GaussianBlur(image, 3)
    img = binary_mask(img)
    # img = image_erosion(img,iterations_num=3)
    # img = image_dilatation(img,iterations_num=2)    ##AKO POSTOJI PROBLEM PROVERITi DILATACIJU I EROZIJU
    img = crop_padding_image(img)
    img = resize_image(img, target_size)
    # img = skeletonize_image(img)

    return img

# novo

def save_comparison_plot(original, processed, step_name, save_dir):
    """Prikazuje originalnu i obrađenu sliku u 1x2 poretku i čuva plot."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{step_name}.png")

    plt.figure(figsize=(7, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Originalna slika")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(processed, cmap='gray')
    plt.title(step_name)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Sačuvan korak: {save_path}")


def visualize_processing_pipeline(image_path, save_dir="pipeline_steps", target_size=(32, 32)):
    """Izvršava processing_pipeline i čuva 1x2 poređenje za svaki korak."""

    # Učitavanje originalne slike
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise FileNotFoundError(f"Nije moguće učitati sliku: {image_path}")

    img = original.copy()
    save_comparison_plot(original, img, "01_original", save_dir)

    # Gaussian blur
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    save_comparison_plot(original, blurred, "02_Gaussian_Blur", save_dir)
    img = blurred

    # Binarizacija
    img_bin = binary_mask(img)
    save_comparison_plot(original, img_bin, "03_Binarizacija", save_dir)
    img = img_bin

    # (opciono) Erozija
    # img_er = image_erosion(img, iterations_num=1)
    # save_comparison_plot(original, img_er, "04_Dilatacija", save_dir)
    # img = img_er
    #
    # # (opciono) Dilatacija
    # img_dil = image_dilatation(img, iterations_num=1)
    # save_comparison_plot(original, img_dil, "05_Erozija", save_dir)
    # img = img_dil

    # Crop/padding
    img_crop = crop_padding_image(img)
    save_comparison_plot(original, img_crop, "06_Crop_padding", save_dir)
    img = img_crop

    # Resize
    img_resized = resize_image(img, target_size)
    save_comparison_plot(original, img_resized, "07_Resize", save_dir)
    img = img_resized

    # (opciono) Skeletonizacija
    img_skel = skeletonize_image(img)
    save_comparison_plot(original, img_skel, "08_Skeletonizacija", save_dir)
    img = img_skel

    print(f"[INFO] Obrada završena. Svi rezultati sačuvani u: {save_dir}")
    return img

if __name__ == "__main__":

    # for d in range(10):
    #     image_path = f"data/{d}/cifra_{d}.jpg"
    #     raw_img = read_img(path=image_path)
    #
    #     # Snimaj obrađene isečke u isti folder (data/d/)
    #     split_digits_grid(
    #         raw_img,
    #         save_dir=f"data/{d}",
    #         label=str(d),
    #         show_plots=False,
    #         target_size=(28, 28)
    #     )

    # folder_path = 'mnist_dataset'
    # out_dir = 'big_processed_images'  # folder za čuvanje novih slika
    # os.makedirs(out_dir, exist_ok=True)  # napravi folder ako ne postoji

    # for i in range(0, 10):
    #     for j in range(1, 113):
    #         # ulazni path
    #         image_path = os.path.join(folder_path, f"{i}{j:03}.png")
    #         raw_img = read_img(path=image_path)
    #         processed_img = processing_pipeline(raw_img)
    #
    #         # napravi ime fajla: new_000.png
    #         base = os.path.basename(image_path)  # npr. "000.png"
    #         name, _ = os.path.splitext(base)  # "000"
    #         fname = f"new_{name}.png"  # "new_000.png"
    #
    #         # izlazni path
    #         out_path = os.path.join(out_dir, fname)
    #
    #         # snimi fajl
    #         cv2.imwrite(out_path, processed_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    #
    #         print(f"Done: {out_path}")

    # for class_label in range(10):  # prolazak kroz klase 0–9
    #     class_dir = os.path.join(folder_path, str(class_label))
    #     if not os.path.isdir(class_dir):
    #         continue  # preskoči ako folder ne postoji (za svaki slučaj)
    #
    #     # napravi izlazni podfolder za tu klasu
    #     out_class_dir = os.path.join(out_dir, str(class_label))
    #     os.makedirs(out_class_dir, exist_ok=True)
    #
    #     count = 0  # brojač slika po klasi
    #
    #     for filename in sorted(os.listdir(class_dir)):
    #         if not filename.endswith(".png"):
    #             continue
    #
    #         image_path = os.path.join(class_dir, filename)
    #         raw_img = read_img(path=image_path)
    #         processed_img = processing_pipeline(raw_img)
    #
    #         fname = f"{filename}"
    #         out_path = os.path.join(out_class_dir, fname)
    #
    #         cv2.imwrite(out_path, processed_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    #         print(f"Done: {out_path}")
    #
    #         count += 1
    #         if count >= 20:
    #             break

    visualize_processing_pipeline("cifra_5/cifra.png", save_dir="cifra_5")





