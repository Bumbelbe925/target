"""Mide la precisión del detector sobre imágenes etiquetadas.

Nombres aceptados:
  <ESTADO>_<lo que sea>.jpg            -> se evalúa solo el estado
  <ESTADO>_<n>_<SECUENCIA>.jpg         -> también se evalúa anillo por anillo
ESTADO: HARMED, STABLE, UNHARMED, FAKE (o herida, estable, ilesa, falso)
SECUENCIA: 5 letras del centro hacia afuera: K negro, R rojo, Y amarillo, G verde, B azul

Uso:
    python tools/evaluate.py sinteticas/
    python tools/evaluate.py fotos_reales/ --errors errores/
"""

import argparse
import random
import re
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cognitive import CognitiveDetector, Config  # noqa: E402
from cognitive.draw import annotate  # noqa: E402

ALIASES = {
    "harmed": "HARMED", "herida": "HARMED",
    "stable": "STABLE", "estable": "STABLE",
    "unharmed": "UNHARMED", "ilesa": "UNHARMED",
    "fake": "FAKE", "falso": "FAKE",
}
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def expected(path: Path):
    parts = path.stem.split("_")
    label = ALIASES.get(parts[0].lower()) or ALIASES.get(path.parent.name.lower())
    seq = None
    if len(parts) >= 3 and re.fullmatch(r"[KRYGB]{5}", parts[-1]):
        seq = parts[-1]
    return label, seq


def jitter_box(box, j, rng):
    """Mueve cada borde de la caja hasta ±j veces su tamaño (simula un modelo imperfecto)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return (x1 + rng.uniform(-j, j) * w, y1 + rng.uniform(-j, j) * h,
            x2 + rng.uniform(-j, j) * w, y2 + rng.uniform(-j, j) * h)


def main():
    ap = argparse.ArgumentParser(description="Mide la precisión del detector sobre imágenes etiquetadas.")
    ap.add_argument("folder")
    ap.add_argument("--errors", help="carpeta para guardar errores anotados")
    ap.add_argument("--colors", help="prototipos de color calibrados (JSON)")
    ap.add_argument("--roi-jitter", type=float, default=0.0,
                    help="con --roi-labels: error aleatorio de la caja (0.1 = ±10%% del tamaño), para simular un modelo real")
    ap.add_argument("--roi-labels", metavar="CSV",
                    help="usar las cajas de este CSV (labels.csv del generador o de un modelo) y analizar solo ese ROI")
    a = ap.parse_args()

    cfg = Config()
    if a.colors:
        cfg.prototypes_file = a.colors
    det = CognitiveDetector(cfg)
    err_dir = Path(a.errors) if a.errors else None
    if err_dir:
        err_dir.mkdir(parents=True, exist_ok=True)

    files = [p for p in sorted(Path(a.folder).rglob("*")) if p.suffix.lower() in EXTS]
    boxes = {}
    if a.roi_labels:
        for line in Path(a.roi_labels).read_text().splitlines()[1:]:
            parts = line.split(",")
            if len(parts) >= 7:
                boxes[parts[0]] = tuple(int(v) for v in parts[3:7])
    total = ok = seq_total = seq_ok = 0
    rings_total = rings_ok = 0
    extras = imgs_extra = 0
    dangerous = 0  # se reportó víctima cuando era FAKE, o víctima con estado equivocado
    confusion = Counter()

    for p in files:
        exp, exp_seq = expected(p)
        if exp is None:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        if a.roi_labels:
            box = boxes.get(p.name)
            if box and a.roi_jitter > 0:
                box = jitter_box(box, a.roi_jitter, random.Random(p.name))
            t = det.analyze_roi(img, box) if box else None
            dets = [t] if t is not None else []
        else:
            dets = det.detect(img, include_uncertain=True)
        dets = sorted(dets, key=lambda t: -t.disc.r)
        best = dets[0] if dets else None
        got = "NO_DETECTADO" if best is None else (best.label or "INSEGURO")
        extra = max(0, len(dets) - 1)
        extras += extra
        imgs_extra += 1 if extra else 0

        total += 1
        confusion[(exp, got)] += 1
        if got == exp:
            ok += 1
        if got in ("HARMED", "STABLE", "UNHARMED") and got != exp:
            dangerous += 1

        if exp_seq and best is not None:
            seq_total += 1
            if best.sequence == exp_seq:
                seq_ok += 1
            for a_ch, b_ch in zip(best.sequence, exp_seq):
                rings_total += 1
                rings_ok += a_ch == b_ch

        if err_dir and (got != exp or extra):
            cv2.imwrite(str(err_dir / f"{p.stem}__{got}{'_EXTRA' + str(extra) if extra else ''}.jpg"),
                        annotate(img, dets))

    if total == 0:
        print("No se encontraron imágenes con etiqueta en el nombre.")
        return 1
    print(f"Estado correcto:        {ok}/{total} = {100 * ok / total:.1f}%")
    if seq_total:
        print(f"Secuencia exacta:       {seq_ok}/{seq_total} = {100 * seq_ok / seq_total:.1f}%")
        print(f"Anillos correctos:      {rings_ok}/{rings_total} = {100 * rings_ok / rings_total:.1f}%")
    print(f"Errores peligrosos:     {dangerous}  (reportó víctima con estado equivocado o un FAKE como víctima)")
    print(f"Falsos positivos:       {extras} detecciones extra en {imgs_extra} imágenes\n")
    print("Esperado -> Obtenido: cantidad")
    for (e, g), c in sorted(confusion.items()):
        print(f"  {e:9s} -> {g:13s}: {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
