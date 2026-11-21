"""Genera imágenes sintéticas de cognitive targets para probar el detector.

Cada archivo se llama  <ESTADO>_<n>_<SECUENCIA>.jpg  donde SECUENCIA son las
letras de los 5 anillos del centro hacia afuera (K negro, R rojo, Y amarillo,
G verde, B azul). Así evaluate.py sabe la respuesta correcta.

Uso:
    python tools/generate_synthetic.py --out sinteticas --n 400
    python tools/generate_synthetic.py --out dificiles --n 400 --hard --distractors
    python tools/generate_synthetic.py --out fondos --n 400 --bg random --distractors
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cognitive.scoring import COLORS, FAKE, HARMED, STABLE, UNHARMED, label_for, sequence_code  # noqa: E402

# Colores "de impresión" base (BGR) y variación aleatoria
PRINT_BGR = {
    "black": (30, 28, 28),
    "red": (45, 40, 205),
    "yellow": (40, 210, 240),
    "green": (70, 165, 45),
    "blue": (175, 95, 30),
}

LABELS = [HARMED, STABLE, UNHARMED, FAKE]


def random_sequence(label, rng):
    """5 colores (centro -> afuera) cuya suma corresponda a la etiqueta pedida."""
    for _ in range(10000):
        seq = [rng.choice(COLORS) for _ in range(5)]
        if label_for(seq)[0] == label:
            return seq
    raise RuntimeError("no se encontró secuencia")


def jitter_color(bgr, rng, amount=18):
    return tuple(int(np.clip(c + rng.uniform(-amount, amount), 0, 255)) for c in bgr)


def draw_target(seq, size, rng):
    """Dibuja el target en un lienzo cuadrado con fondo blanco. Devuelve (img, máscara)."""
    img = np.full((size, size, 3), 255, np.uint8)
    mask = np.zeros((size, size), np.uint8)
    c = size // 2
    R = size * 0.45
    # Variación de impresión: mismo color base para todo el target, más leve por anillo
    base = {k: jitter_color(v, rng, 22) for k, v in PRINT_BGR.items()}
    for k in range(4, -1, -1):  # de afuera hacia adentro
        r = R * (k + 1) / 5
        col = jitter_color(base[seq[k]], rng, 6)
        cv2.circle(img, (c, c), int(round(r)), col, -1, cv2.LINE_AA)
    cv2.circle(mask, (c, c), int(round(R)) + 1, 255, -1, cv2.LINE_AA)
    return img, mask


def draw_smiley(img, cx, cy, r, rng):
    ink = (35, 35, 35)
    cv2.circle(img, (cx, cy), r, (245, 245, 245), -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), r, ink, max(2, r // 9), cv2.LINE_AA)
    ex, ey = int(r * 0.35), int(r * 0.3)
    for sx in (-1, 1):
        cv2.ellipse(img, (cx + sx * ex, cy - ey), (max(2, r // 9), max(3, r // 5)), 0, 0, 360, ink, -1, cv2.LINE_AA)
    kind = rng.choice([-1, 0, 1])
    my = cy + int(r * 0.35)
    if kind == 0:
        cv2.line(img, (cx - r // 2, my), (cx + r // 2, my), ink, max(2, r // 10), cv2.LINE_AA)
    else:
        cv2.ellipse(img, (cx, my - (r // 5 if kind > 0 else -r // 5)), (r // 2, r // 4), 0,
                    0 if kind > 0 else 180, 180 if kind > 0 else 360, ink, max(2, r // 10), cv2.LINE_AA)


def make_wall(w, h, rng):
    base = rng.randint(190, 245)
    tint = np.array([rng.randint(-8, 8) for _ in range(3)])
    wall = np.clip(np.full((h, w, 3), base, np.int16) + tint, 0, 255).astype(np.float32)
    wall += np.random.normal(0, rng.uniform(1, 4), (h, w, 1))
    wall = np.clip(wall, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        x = rng.randint(0, w - 1)
        cv2.line(wall, (x, 0), (x + rng.randint(-20, 20), h), (base - 45,) * 3, rng.randint(1, 3))
    return wall


def _rand_color(rng):
    return tuple(int(rng.randint(0, 255)) for _ in range(3))


def make_random_background(w, h, rng):
    kind = rng.choice(["noise", "gradient", "stripes", "checker", "plain"])
    if kind == "noise":
        s = rng.choice([4, 8, 16, 32])
        small = np.random.rand(max(2, h // s), max(2, w // s), 3).astype(np.float32) * 255
        bg = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    elif kind == "gradient":
        c1, c2 = np.array(_rand_color(rng), np.float32), np.array(_rand_color(rng), np.float32)
        t = np.linspace(0, 1, w)[None, :, None]
        bg = np.broadcast_to(c1 * (1 - t) + c2 * t, (h, w, 3)).copy()
    elif kind in ("stripes", "checker"):
        bg = np.zeros((h, w, 3), np.float32)
        c1, c2 = _rand_color(rng), _rand_color(rng)
        yy, xx = np.mgrid[0:h, 0:w]
        if kind == "stripes":
            p = rng.randint(6, 40)
            ang = rng.uniform(0, np.pi)
            band = ((xx * np.cos(ang) + yy * np.sin(ang)) // p).astype(int) % 2
        else:
            sq = rng.randint(8, 40)
            band = ((xx // sq) + (yy // sq)) % 2
        bg[band == 0] = c1
        bg[band == 1] = c2
    else:
        bg = np.full((h, w, 3), _rand_color(rng), np.float32)
    return np.clip(bg, 0, 255).astype(np.uint8)


def add_distractors(bg, rng, avoid):
    """Caritas, figuras de color, letras y líneas que NO son targets."""
    h, w = bg.shape[:2]
    ax, ay, ar = avoid
    for _ in range(rng.randint(2, 6)):
        for _try in range(20):
            x, y = rng.randint(0, w - 1), rng.randint(0, h - 1)
            sz = rng.randint(10, max(11, min(w, h) // 5))
            if np.hypot(x - ax, y - ay) > ar + sz + 6:
                break
        else:
            continue
        kind = rng.choice(["smiley", "rect", "tri", "letter", "line", "ring"])
        col = tuple(int(v) for v in jitter_color(rng.choice(list(PRINT_BGR.values())), rng, 25))
        if kind == "smiley":
            draw_smiley(bg, x, y, sz, rng)
        elif kind == "rect":
            cv2.rectangle(bg, (x, y), (x + rng.randint(8, sz * 2), y + rng.randint(8, sz * 2)), col, -1)
        elif kind == "tri":
            pts = np.array([[x, y], [x + rng.randint(-sz, sz), y + sz], [x + sz, y + rng.randint(-sz, sz)]], np.int32)
            cv2.fillPoly(bg, [pts], col)
        elif kind == "letter":
            cv2.putText(bg, rng.choice("HSUX"), (x, y), cv2.FONT_HERSHEY_SIMPLEX, sz / 20, col, max(2, sz // 8))
        elif kind == "line":
            cv2.line(bg, (x, y), (rng.randint(0, w), rng.randint(0, h)), col, rng.randint(1, 4))
        else:  # aro de color (hueco): no debe confundirse con target
            cv2.circle(bg, (x, y), sz, col, max(2, sz // 4), cv2.LINE_AA)


def compose(target, tmask, out_w, out_h, rng, bg_mode="wall", hard=False, distractors=False):
    mode = bg_mode if bg_mode != "mix" else rng.choice(["wall", "random"])
    bg = make_random_background(out_w, out_h, rng) if mode == "random" else make_wall(out_w, out_h, rng)

    s = target.shape[0]
    diameter = rng.uniform(0.18, 0.6) * min(out_w, out_h)
    scale = diameter / (s * 0.9)
    margin = diameter * 0.6
    cx = rng.uniform(margin, out_w - margin)
    cy = rng.uniform(margin, out_h - margin)
    if distractors:
        add_distractors(bg, rng, (cx, cy, diameter / 2))

    M = cv2.getRotationMatrix2D((s / 2, s / 2), rng.uniform(0, 360), scale)
    M[0, 2] += cx - s / 2
    M[1, 2] += cy - s / 2
    warped = cv2.warpAffine(target, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    m = cv2.warpAffine(tmask, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderValue=0).astype(np.float32) / 255
    img = warped.astype(np.float32) * m[..., None] + bg.astype(np.float32) * (1 - m[..., None])

    if hard:
        if rng.random() < 0.8:  # perspectiva
            d = min(out_w, out_h) * rng.uniform(0.0, 0.10)
            src = np.float32([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]])
            dst = src + np.float32([[rng.uniform(-d, d), rng.uniform(-d, d)] for _ in range(4)])
            P = cv2.getPerspectiveTransform(src, dst)
            img = cv2.warpPerspective(img, P, (out_w, out_h), borderMode=cv2.BORDER_REPLICATE)
            m = cv2.warpPerspective(m, P, (out_w, out_h))
        img *= rng.uniform(0.65, 1.15)  # brillo
        if rng.random() < 0.6:  # gradiente de luz
            g = np.linspace(rng.uniform(0.75, 1.0), rng.uniform(0.75, 1.0), out_w)[None, :, None]
            img *= g
        img *= np.array([rng.uniform(0.92, 1.08) for _ in range(3)])[None, None, :]  # balance de blancos
        k = rng.choice([1, 1, 3, 5])
        if k > 1:
            img = cv2.GaussianBlur(img, (k, k), 0)
        img += np.random.normal(0, rng.uniform(1, 7), img.shape)
    ys, xs = np.nonzero(m > 0.5)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1) if len(xs) else None
    return np.clip(img, 0, 255).astype(np.uint8), bbox


def generate(out_dir: Path, n: int, size="320x240", hard=False, bg="wall", distractors=False, seed=0):
    rng = random.Random(seed)
    np.random.seed(seed)
    out_w, out_h = (int(v) for v in size.lower().split("x"))
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    rows = ["archivo,estado,secuencia,x1,y1,x2,y2"]
    for i in range(n):
        label = LABELS[i % len(LABELS)]
        seq = random_sequence(label, rng)
        target, tmask = draw_target(seq, 400, rng)
        img, bbox = compose(target, tmask, out_w, out_h, rng, bg, hard, distractors)
        name = f"{label}_{i:04d}_{sequence_code(seq)}.jpg"
        cv2.imwrite(str(out_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(70, 95)])
        counts[label] = counts.get(label, 0) + 1
        rows.append(f"{name},{label},{sequence_code(seq)}," + ",".join(str(v) for v in bbox))
    # Cajas reales: sirven para simular un modelo (YOLO/FOMO) perfecto o para entrenar uno.
    (out_dir / "labels.csv").write_text("\n".join(rows) + "\n")
    return counts


def main():
    ap = argparse.ArgumentParser(description="Genera imágenes sintéticas de cognitive targets.")
    ap.add_argument("--out", default="sinteticas")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--size", default="320x240", help="ANCHOxALTO (por defecto 320x240, QVGA)")
    ap.add_argument("--hard", action="store_true", help="perspectiva, luz desigual, blur, ruido")
    ap.add_argument("--bg", choices=["wall", "random", "mix"], default="wall")
    ap.add_argument("--distractors", action="store_true", help="agrega caritas, figuras, letras y aros")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    counts = generate(Path(a.out), a.n, a.size, a.hard, a.bg, a.distractors, a.seed)
    print(f"{a.n} imágenes en {a.out}/  ->  " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
