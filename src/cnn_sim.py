import copy
import csv
import io
import os
import re
import time
from pathlib import Path
import torch
from PIL import Image
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

INPUT_DIR = ""
CALIB_DIR = ""
OUT_CSV = "sequential_results.csv"

PRECISION = "fp32" # int8 for PTQ
N_CALIB = 256 # calibration samples for PTQ

DEVICE = torch.device("cpu") if PRECISION == "int8" else \
    torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif")
LABEL_RE = re.compile(r"label_(\d+)")

TRANSFORM = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BILINEAR, antialias=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

HEADER = ["model", "precision", "n_images", "top1", "top5", "model_MB", "ms_per_frame"]


def list_frames(folder):
    items = []
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(IMG_EXT):
            continue
        m = LABEL_RE.search(name)
        if m:
            items.append((name, int(m.group(1))))
    return items


def run_frame(model, folder, fname, label, host_buf, dev_buf):
    t0 = time.perf_counter()

    with open(folder / fname, "rb") as f:
        raw = f.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    chw = TRANSFORM(img)

    host_buf.copy_(chw.unsqueeze(0))
    dev_buf.copy_(host_buf, non_blocking=(DEVICE.type == "cuda"))

    with torch.inference_mode():
        logits = model(dev_buf)

    top5 = logits.cpu().topk(5, dim=1).indices[0].tolist()

    return int(top5[0] == label), int(label in top5), time.perf_counter() - t0


def run_folder(model, folder, items):
    host_buf = torch.empty((1, 3, 224, 224), pin_memory=(DEVICE.type == "cuda"))
    dev_buf = torch.empty((1, 3, 224, 224), device=DEVICE)

    c1 = c5 = 0
    total = 0.0
    for fname, label in items:
        a, b, dt = run_frame(model, folder, fname, label, host_buf, dev_buf)
        c1 += a
        c5 += b
        total += dt

    n = len(items)
    return dict(n=n, top1=100 * c1 / n, top5=100 * c5 / n, ms=1e3 * total / n)


def model_size_mb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / (1024 ** 2)


def quantize_ptq(model, calib_folder):
    from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig
    from torch.ao.quantization.qconfig_mapping import QConfigMapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    torch.backends.quantized.engine = "fbgemm"

    qconfig = QConfig(
        activation=MinMaxObserver.with_args(dtype=torch.quint8,
                                            qscheme=torch.per_tensor_affine,
                                            reduce_range=True),
        weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8,
                                                  qscheme=torch.per_channel_symmetric),
    )
    qmap = QConfigMapping().set_global(qconfig)

    m = copy.deepcopy(model).eval().cpu()
    prepared = prepare_fx(m, qmap, example_inputs=(torch.randn(1, 3, 224, 224),))

    calib = list_frames(calib_folder)[:N_CALIB]
    if not calib:
        raise SystemExit(f"no calibration images in {calib_folder}")
    with torch.inference_mode():
        for fname, _ in calib:
            img = Image.open(calib_folder / fname).convert("RGB")
            prepared(TRANSFORM(img).unsqueeze(0))

    return convert_fx(prepared)


def main():
    folder = Path(INPUT_DIR)
    if not folder.is_dir():
        raise SystemExit(f"INPUT_DIR is not a folder: {folder}")

    items = list_frames(folder)
    if not items:
        raise SystemExit(f"no labeled images in {folder}")

    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print("GPU:", torch.cuda.get_device_name(0))
    print(f"Device: {DEVICE}   precision: {PRECISION}")
    print(f"Folder: {folder}  ({len(items)} images)")

    nets = {
        "AlexNet": models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1),
        "ResNet50": models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2),
        "MobileNetV2": models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2),
    }

    if PRECISION == "int8":
        calib_folder = Path(CALIB_DIR) if CALIB_DIR else folder
        print(f"Calibrating on {calib_folder} ({N_CALIB} images)")
        for name in nets:
            t0 = time.perf_counter()
            nets[name] = quantize_ptq(nets[name], calib_folder)
            print(f"  {name:12s} quantized in {time.perf_counter() - t0:.1f}s")

    sizes = {k: model_size_mb(m) for k, m in nets.items()}
    nets = {k: m.eval().to(DEVICE) for k, m in nets.items()}

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for name, model in nets.items():
            r = run_folder(model, folder, items)
            print(f"  {name:12s} Top1={r['top1']:6.2f}% Top5={r['top5']:6.2f}% "
                  f"{r['ms']:7.2f} ms/frame  {sizes[name]:6.1f} MB")
            w.writerow([name, PRECISION, r["n"], f"{r['top1']:.3f}", f"{r['top5']:.3f}",
                        f"{sizes[name]:.2f}", f"{r['ms']:.3f}"])
            f.flush()

    print("\nDone:", OUT_CSV)


if __name__ == "__main__":
    main()
