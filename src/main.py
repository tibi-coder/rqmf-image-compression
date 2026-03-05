import os
import time
import csv
import shutil
import numpy as np
import cv2

from skimage.metrics import structural_similarity as ssim
from skimage.restoration import denoise_tv_chambolle
from skimage.measure import shannon_entropy


patching = True
mi = True
patch_size = 8

alpha, beta = -16, 15
iterations = 10

dir_images = "images"
root = "./"
folder = os.path.join(root, dir_images)

Q_space = np.linspace(0.05, 0.5, 8)

csv_file = "rqmf_results.csv"


def psnr(orig_path, recon_path):
    orig = cv2.imread(orig_path)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB).astype(np.float32)

    rec = cv2.imread(recon_path)
    rec = cv2.cvtColor(rec, cv2.COLOR_BGR2RGB).astype(np.float32)

    if rec.shape != orig.shape:
        rec = cv2.resize(rec, (orig.shape[1], orig.shape[0]))

    mse = np.mean((orig - rec) ** 2)
    if mse == 0:
        return float("inf")

    return 10 * np.log10((255 ** 2) / mse)


def ssim_metric(orig_path, recon_path):
    orig = cv2.imread(orig_path)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

    rec = cv2.imread(recon_path)
    rec = cv2.cvtColor(rec, cv2.COLOR_BGR2RGB)

    if rec.shape != orig.shape:
        rec = cv2.resize(rec, (orig.shape[1], orig.shape[0]))

    return ssim(orig, rec, data_range=255, channel_axis=2)


def img_to_patches(img, ps):
    h, w = img.shape
    pad_h = (ps - h % ps) % ps
    pad_w = (ps - w % ps) % ps

    img = np.pad(img, ((0, pad_h), (0, pad_w)), mode="reflect")

    H, W = img.shape

    patches = img.reshape(H // ps, ps, W // ps, ps)
    patches = patches.transpose(0, 2, 1, 3).reshape(-1, ps * ps)

    return patches, H, W


def patches_to_image(patches, h, w, ps):
    hb = h // ps
    wb = w // ps

    img = patches.reshape(hb, wb, ps, ps).transpose(0, 2, 1, 3)
    img = img.reshape(h, w)

    return img[:h, :w]


def compute_mi_y(Y):
    H = shannon_entropy(Y)
    Hmax = np.log2(256)

    gamma = 60
    val = gamma * (1 - H / Hmax)

    return float(np.clip(val, 0.5, 50))


def compute_mi_c(C):
    H = shannon_entropy(C)
    Hmax = np.log2(256)

    gamma = 80
    val = gamma * (1 - H / Hmax)

    return float(np.clip(val, 0.5, 70))


def qmf_single_channel(X, rank, iterations, alpha, beta, mi_local):
    X = X.astype(np.float32)

    U_svd, S, Vt = np.linalg.svd(X, full_matrices=False)

    s = np.sqrt(S[:rank])
    U = U_svd[:, :rank] * s
    V = (Vt[:rank].T) * s

    U = np.clip(np.rint(U), alpha, beta)
    V = np.clip(np.rint(V), alpha, beta)

    eps = 1e-8

    for _ in range(iterations):

        A = X @ V
        B = V.T @ V
        Vnorm = (V ** 2).sum(axis=0) + eps

        U_old = U.copy()

        for r in range(rank):

            left = A[:, r].copy()

            if r > 0:
                left -= U[:, :r] @ B[:r, r]

            if r + 1 < rank:
                left -= U[:, r + 1:] @ B[r + 1:, r]

            u = (left + mi_local * U_old[:, r]) / (Vnorm[r] + mi_local)

            U[:, r] = np.clip(np.rint(u), alpha, beta)

        A = X.T @ U
        B = U.T @ U
        Unorm = (U ** 2).sum(axis=0) + eps

        V_old = V.copy()
        V_new = V.copy()

        for r in range(rank):

            left = A[:, r].copy()

            if r > 0:
                left -= V_new[:, :r] @ B[:r, r]

            if r + 1 < rank:
                left -= V[:, r + 1:] @ B[r + 1:, r]

            v = (left + mi_local * V_old[:, r]) / (Unorm[r] + mi_local)

            V_new[:, r] = np.clip(np.rint(v), alpha, beta)

        V = V_new

    return U.astype(np.int16), V.astype(np.int16)


def preprocess(img_path):

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

    Y, Cr, Cb = cv2.split(img)

    Cb = cv2.resize(Cb, (Cb.shape[1] // 2, Cb.shape[0] // 2))
    Cr = cv2.resize(Cr, (Cr.shape[1] // 2, Cr.shape[0] // 2))

    if patching:

        XY, HY, WY = img_to_patches(Y, patch_size)
        XCb, HC, WC = img_to_patches(Cb, patch_size)
        XCr, _, _ = img_to_patches(Cr, patch_size)

        return XY, XCb, XCr, HY, WY, HC, WC

    else:

        return Y, Cb, Cr, *Y.shape, *Cb.shape


def reconstruct(UY, VY, UCb, VCb, UCr, VCr, HY, WY, HC, WC):

    Y = np.clip(UY @ VY.T, 0, 255).astype(np.uint8)
    Cb = np.clip(UCb @ VCb.T, 0, 255).astype(np.uint8)
    Cr = np.clip(UCr @ VCr.T, 0, 255).astype(np.uint8)

    if patching:

        Y = patches_to_image(Y, HY, WY, patch_size)
        Cb = patches_to_image(Cb, HC, WC, patch_size)
        Cr = patches_to_image(Cr, HC, WC, patch_size)

    Cb = cv2.resize(Cb, (WY, HY))
    Cr = cv2.resize(Cr, (WY, HY))

    img = cv2.merge((Y, Cr, Cb))
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)

    return img


def compress_image(img_path, Q, output_file):

    XY, XCb, XCr, HY, WY, HC, WC = preprocess(img_path)

    rankY = max(int(Q * XY.shape[1]), 1)
    rankC = max(int(Q * XCb.shape[1]), 1)

    miY = compute_mi_y(XY) if mi else 0
    miC = compute_mi_c(XCb) if mi else 0

    UY, VY = qmf_single_channel(XY, rankY, iterations, alpha, beta, miY)
    UCb, VCb = qmf_single_channel(XCb, rankC, iterations, alpha, beta, miC)
    UCr, VCr = qmf_single_channel(XCr, rankC, iterations, alpha, beta, miC)

    np.savez_compressed(
        output_file,
        UY=UY,
        VY=VY,
        UCb=UCb,
        VCb=VCb,
        UCr=UCr,
        VCr=VCr,
        HY=HY,
        WY=WY,
        HC=HC,
        WC=WC,
    )

    size = os.path.getsize(output_file)
    bpp = (size * 8) / (HY * WY)

    return bpp


def decompress_image(npz_path, output_file):

    data = np.load(npz_path)

    img = reconstruct(
        data["UY"],
        data["VY"],
        data["UCb"],
        data["VCb"],
        data["UCr"],
        data["VCr"],
        int(data["HY"]),
        int(data["WY"]),
        int(data["HC"]),
        int(data["WC"]),
    )

    cv2.imwrite(output_file, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main():

    results_dir = "results"
    compressed_dir = "compressed"

    for d in [results_dir, compressed_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Q",
                "psnr_mean",
                "ssim_mean",
                "bpp_mean",
            ]
        )

    for Q in Q_space:

        psnrs, ssims, bpps = [], [], []

        for fname in files:

            name, _ = os.path.splitext(fname)

            img_path = os.path.join(folder, fname)

            cmp_path = os.path.join(compressed_dir, name + ".npz")
            rec_path = os.path.join(results_dir, name + ".png")

            bpp = compress_image(img_path, Q, cmp_path)

            decompress_image(cmp_path, rec_path)

            ps = psnr(img_path, rec_path)
            ss = ssim_metric(img_path, rec_path)

            psnrs.append(ps)
            ssims.append(ss)
            bpps.append(bpp)

            print(f"{name} | PSNR {ps:.2f} | SSIM {ss:.4f}")

        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    Q,
                    np.mean(psnrs),
                    np.mean(ssims),
                    np.mean(bpps),
                ]
            )


if __name__ == "__main__":
    main()