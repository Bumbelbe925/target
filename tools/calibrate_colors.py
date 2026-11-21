"""Calibra los colores con fotos reales de targets cuya secuencia conoces.

Toma fotos donde aparece un target, lo detecta (o usa la caja que le des),
y guarda el color LAB típico de cada uno de los 5 colores y de la pared.
El JSON resultante se usa con  --colors colores.json  y hace al detector
mucho más resistente a la luz del lugar y a la tinta de la impresora.

Cada foto debe decir su secuencia (centro -> afuera, letras K R Y G B):
  - en el nombre: lo_que_sea_<SECUENCIA>.jpg   p. ej.  calib_01_BRYGK.jpg
  - o con --seq BRYGK  (la misma para todas las fotos)

Uso:
    python tools/calibrate_colors.py fotos_calibracion/ --out colores.json
    python tools/calibrate_colors.py foto.jpg --seq KRYGB --out colores.json

Consejo: usa 2-3 targets que juntos tengan los 5 colores, fotografiados con
la misma cámara y la misma luz con las que vas a detectar.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cognitive import CognitiveDetector, Config  # noqa: E402
from cognitive.rings import N_RINGS, ellipse_point  # noqa: E402
from cognitive.scoring import COLORS, parse_sequence  # noqa: E402

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def sample_lab(lab, disc, radii, n=64):
    """Valores LAB sobre elipses de radio normalizado `radii` (1.0 = borde del target)."""
    vals = []
    H, W = lab.shape[:2]
    for f in radii:
        for i in range(n):
            x, y = ellipse_point(disc, f, 2 * np.pi * i / n)
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < W and 0 <= yi < H:
                vals.append(lab[yi, xi])
    return vals


def sample_ring_lab(lab, disc, k):
    return sample_lab(lab, disc, np.linspace((k + 0.35) / N_RINGS, (k + 0.65) / N_RINGS, 3))


def sample_surround_lab(lab, disc):
    return sample_lab(lab, disc, (1.25, 1.4))


def main():
    ap = argparse.ArgumentParser(description="Calibra los colores LAB con fotos de targets conocidos.")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--seq", help="secuencia común a todas las fotos (si no viene en el nombre)")
    ap.add_argument("--out", default="colores.json")
    a = ap.parse_args()

    det = CognitiveDetector(Config())
    samples = {c: [] for c in COLORS}
    samples["background"] = []
    files = []
    for p in a.inputs:
        p = Path(p)
        files += sorted(f for f in p.rglob("*") if f.suffix.lower() in EXTS) if p.is_dir() else [p]

    used = 0
    for f in files:
        m = re.search(r"([KRYGB]{5})$", f.stem)
        code = m.group(1) if m else a.seq
        if not code:
            print(f"[aviso] {f.name}: sin secuencia (ponla en el nombre o usa --seq)")
            continue
        img = cv2.imread(str(f))
        if img is None:
            continue
        dets = det.detect(img, include_uncertain=True)
        if not dets:
            print(f"[aviso] {f.name}: no encontré el target")
            continue
        t = max(dets, key=lambda d: d.disc.r)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        for k, color in enumerate(parse_sequence(code)):
            samples[color] += sample_ring_lab(lab, t.disc, k)
        samples["background"] += sample_surround_lab(lab, t.disc)
        used += 1
        print(f"{f.name}: OK ({code})")

    missing = [c for c in COLORS if not samples[c]]
    if missing:
        sys.exit(f"Faltan muestras de: {missing}. Agrega fotos con esos colores.")
    out = {c: [round(float(v), 1) for v in np.median(np.array(samples[c]), axis=0)] for c in samples if samples[c]}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\n{used} fotos usadas -> {a.out}")
    for c, v in out.items():
        print(f"  {c:10s} LAB = {v}")


if __name__ == "__main__":
    main()
