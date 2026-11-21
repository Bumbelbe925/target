"""Línea de comandos.

Ejemplos:
    python -m cognitive foto.jpg                   # guarda en detectadas/ si encuentra targets
    python -m cognitive carpeta/ --save salida/
    python -m cognitive foto.jpg --show
    python -m cognitive carpeta/ --no-save
    python -m cognitive carpeta/ --json resultados.json
    python -m cognitive --camera 0                 # webcam en vivo (guarda al detectar)
    python -m cognitive fotos/ --colors colores.json   # con colores calibrados (tools/calibrate_colors.py)
    python -m cognitive fotos/ --yolo target.pt         # un modelo YOLO busca el target y aquí se leen los anillos
    python -m cognitive foto.jpg --roi 120,80,200,160   # analizar solo esa caja (x1,y1,x2,y2)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

from .config import Config
from .draw import annotate
from .pipeline import CognitiveDetector

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
NOMBRE = {"HARMED": "HARMED", "STABLE": "STABLE", "UNHARMED": "UNHARMED", "FAKE": "FALSO", None: "INSEGURO"}


def iter_images(paths):
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.suffix.lower() in EXTS:
                    yield f
        elif p.suffix.lower() in EXTS:
            yield p
        else:
            print(f"[aviso] se ignora {p}", file=sys.stderr)


def summary(t) -> str:
    if t.label is None:
        return f"INSEGURO [{t.sequence}]"
    return f"{NOMBRE[t.label]} suma={t.total:+d} [{t.sequence}]"


def output_name(stem, dets) -> str:
    tags = "-".join(NOMBRE[t.label] for t in dets) or "SIN_TARGETS"
    return f"{stem}_{tags}.jpg"


def build_config(a) -> Config:
    cfg = Config()
    if a.colors:
        cfg.prototypes_file = a.colors
    if a.min_diameter is not None:
        cfg.min_diameter_px = a.min_diameter
    if a.max_diameter is not None:
        cfg.max_diameter_px = a.max_diameter
    if a.min_conf is not None:
        cfg.min_ring_conf = a.min_conf
    return cfg


class YoloBoxes:
    """Envuelve un modelo YOLO (Ultralytics) que solo dice DÓNDE hay targets."""

    def __init__(self, path, conf=0.4, classes=None):
        try:
            from ultralytics import YOLO
        except ImportError:
            sys.exit("Para --yolo instala ultralytics:  pip install ultralytics")
        self.model = YOLO(path)
        self.conf = conf
        self.classes = [c.strip().lower() for c in classes.split(",")] if classes else None

    def boxes(self, img):
        res = self.model(img, conf=self.conf, verbose=False)[0]
        names = res.names
        out = []
        for b in res.boxes:
            name = str(names.get(int(b.cls[0]), "")).lower() if isinstance(names, dict) else ""
            if self.classes and not any(k in name for k in self.classes):
                continue
            out.append(tuple(float(v) for v in b.xyxy[0].tolist()))
        return out


def detect_any(det, img, a, yolo):
    """Detección normal, o por ROI si hay modelo YOLO o caja manual."""
    if a.roi:
        box = tuple(float(v) for v in a.roi.split(","))
        t = det.analyze_roi(img, box, include_uncertain=a.all)
        return [t] if t is not None else []
    if yolo is not None:
        out = []
        for box in yolo.boxes(img):
            t = det.analyze_roi(img, box, include_uncertain=a.all)
            if t is not None:
                out.append(t)
        return out
    return det.detect(img, include_uncertain=a.all)


def run_camera(det, index, save_dir, every, a, yolo=None):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(f"No se pudo abrir la cámara {index}")
    print("q = salir   s = guardar este cuadro")
    last, n = 0.0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets = detect_any(det, frame, a, yolo)
        view = annotate(frame, dets)
        cv2.imshow("cognitive", view)
        key = cv2.waitKey(1) & 0xFF
        now = time.time()
        if save_dir and ((dets and now - last >= every) or key == ord("s")):
            name = output_name(f"camara_{time.strftime('%Y%m%d_%H%M%S')}_{n:03d}", dets)
            cv2.imwrite(str(save_dir / name), view)
            print(f"guardada {save_dir / name}")
            last, n = now, n + 1
        if key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m cognitive", description="Detecta cognitive targets y su estado.")
    ap.add_argument("inputs", nargs="*", help="imágenes o carpetas")
    ap.add_argument("--save", metavar="CARPETA", default="detectadas",
                    help="dónde guardar las fotos anotadas (por defecto: detectadas/)")
    ap.add_argument("--no-save", action="store_true", help="no guardar imágenes")
    ap.add_argument("--save-all", action="store_true", help="guardar también fotos sin targets")
    ap.add_argument("--show", action="store_true", help="mostrar cada imagen anotada en una ventana")
    ap.add_argument("--json", metavar="ARCHIVO", help="guardar resultados en JSON")
    ap.add_argument("--camera", type=int, metavar="N", help="webcam N en vivo")
    ap.add_argument("--every", type=float, default=1.0, help="con --camera: segundos entre guardados")
    ap.add_argument("--all", action="store_true", help="incluir targets con algún anillo dudoso")
    ap.add_argument("--colors", metavar="JSON", help="prototipos de color calibrados (tools/calibrate_colors.py)")
    ap.add_argument("--min-diameter", type=int)
    ap.add_argument("--max-diameter", type=int)
    ap.add_argument("--min-conf", type=float, help="votos mínimos del color ganador por anillo (0-1)")
    ap.add_argument("--yolo", metavar="MODELO.pt", help="usar un modelo YOLO para encontrar targets (requiere ultralytics)")
    ap.add_argument("--yolo-conf", type=float, default=0.4)
    ap.add_argument("--yolo-classes", default="target,bullseye",
                    help="clases del modelo que cuentan como target (texto contenido en el nombre); vacío = todas")
    ap.add_argument("--roi", metavar="x1,y1,x2,y2", help="analizar solo esta caja")
    a = ap.parse_args(argv)

    det = CognitiveDetector(build_config(a))
    save_dir = None if a.no_save else Path(a.save)
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    yolo = YoloBoxes(a.yolo, a.yolo_conf, a.yolo_classes) if a.yolo else None

    if a.camera is not None:
        run_camera(det, a.camera, save_dir, a.every, a, yolo)
        return 0
    if not a.inputs:
        ap.print_help()
        return 1

    results = {}
    for path in iter_images(a.inputs):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"{path}: no se pudo leer", file=sys.stderr)
            continue
        dets = detect_any(det, img, a, yolo)
        results[str(path)] = [t.to_dict() for t in dets]
        print(f"{path}: " + (", ".join(summary(t) for t in dets) if dets else "sin targets"))

        if (save_dir and (dets or a.save_all)) or a.show:
            view = annotate(img, dets)
            if save_dir and (dets or a.save_all):
                out = save_dir / output_name(path.stem, dets)
                cv2.imwrite(str(out), view)
                print(f"    -> {out}")
            if a.show:
                cv2.imshow(path.name, view)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
