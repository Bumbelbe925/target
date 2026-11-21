"""Une los pasos: imagen -> lista de cognitive targets con su estado."""

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from .candidates import Disc, dedupe, disc_from_bbox, edge_mask, find_discs, target_mask
from .colors import ColorClassifier
from .config import Config
from .rings import RingReading, read_rings
from .scoring import FAKE, label_for, sequence_code


def to_bgr(image: np.ndarray) -> np.ndarray:
    """Convierte gris o BGRA a BGR (sin copiar si ya es BGR)."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


@dataclass
class TargetDetection:
    label: Optional[str]          # "HARMED", "STABLE", "UNHARMED", "FAKE" o None (lectura insegura)
    total: Optional[int]          # suma de los 5 anillos
    colors: List[Optional[str]]   # centro -> afuera (None = anillo incierto)
    rings: List[RingReading]
    disc: Disc

    @property
    def sequence(self) -> str:
        if any(c is None for c in self.colors):
            return "".join("?" if c is None else sequence_code([c]) for c in self.colors)
        return sequence_code(self.colors)

    @property
    def is_victim(self) -> bool:
        """Solo HARMED/STABLE/UNHARMED se deben identificar (FAKE resta puntos)."""
        return self.label is not None and self.label != FAKE

    def to_dict(self):
        return {
            "label": self.label,
            "sum": self.total,
            "rings_center_to_out": self.colors,
            "sequence": self.sequence,
            "confidence": [round(r.confidence, 3) for r in self.rings],
            "center": [round(self.disc.cx, 1), round(self.disc.cy, 1)],
            "radius": round(self.disc.r, 1),
        }


class CognitiveDetector:
    """Uso:
        det = CognitiveDetector()
        for t in det.detect(imagen_bgr):
            print(t.label, t.total, t.colors)
    """

    def __init__(self, config: Optional[Config] = None):
        self.cfg = config or Config()
        self.clf = ColorClassifier(self.cfg)

    def _smooth(self, image):
        k = self.cfg.blur_ksize
        return cv2.GaussianBlur(image, (k | 1, k | 1), 0) if k and k > 1 else image

    def _surround_colored(self, colored: np.ndarray, disc: Disc) -> float:
        """Fracción de píxeles de color en un anillo justo afuera del target (1.12R a 1.35R)."""
        outer = np.zeros(colored.shape, np.uint8)
        inner = np.zeros(colored.shape, np.uint8)
        (cx, cy), (a1, a2), ang = disc.ellipse
        cv2.ellipse(outer, ((cx, cy), (a1 * 1.35, a2 * 1.35), ang), 1, -1)
        cv2.ellipse(inner, ((cx, cy), (a1 * 1.12, a2 * 1.12), ang), 1, -1)
        ring = (outer > 0) & (inner == 0)
        n = int(ring.sum())
        return np.count_nonzero(colored[ring]) / n if n else 1.0

    def _detect_core(self, image, include_uncertain, surround_check, use_edges):
        smooth = self._smooth(to_bgr(image))

        classes = self.clf.classify_bgr(smooth)
        colored = target_mask(classes, 0)
        discs = []
        for k in self.cfg.open_ksizes:
            discs += find_discs(target_mask(classes, k), self.cfg, colored, open_k=k)
        discs = dedupe(discs)
        if use_edges:
            # Respaldo por bordes: solo donde el método por color no encontró nada.
            for e in dedupe(find_discs(edge_mask(smooth), self.cfg, colored)):
                if not any(np.hypot(e.cx - d.cx, e.cy - d.cy) < max(e.r, d.r) for d in discs):
                    discs.append(e)

        out: List[TargetDetection] = []
        for disc in discs:
            rings = read_rings(smooth, disc, self.clf, self.cfg)
            colors = [r.color for r in rings]
            unsafe = surround_check and self._surround_colored(colored, disc) > self.cfg.max_surround_colored
            if unsafe or any(c is None for c in colors):
                if include_uncertain:
                    out.append(TargetDetection(None, None, colors, rings, disc))
                continue
            label, total = label_for(colors)
            out.append(TargetDetection(label, total, colors, rings, disc))
        out.sort(key=lambda t: (round(t.disc.cy / max(1.0, t.disc.r * 2)), t.disc.cx))
        return out

    def detect(self, image: np.ndarray, include_uncertain: bool = False) -> List[TargetDetection]:
        """Busca targets en toda la imagen."""
        return self._detect_core(image, include_uncertain,
                                 surround_check=self.cfg.require_plain_surround,
                                 use_edges=self.cfg.use_edges)

    def analyze_roi(self, image: np.ndarray, bbox, pad: float = 0.15,
                    include_uncertain: bool = True) -> Optional[TargetDetection]:
        """Analiza solo la caja (x1, y1, x2, y2) que dio un modelo (YOLO, FOMO, etc.).

        Como el modelo ya dijo que ahí hay un target, se usan también los bordes
        y no se exige pared lisa alrededor. Devuelve la detección en coordenadas
        de la imagen completa, o None.
        """
        image = to_bgr(image)
        H, W = image.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        bw, bh = x2 - x1, y2 - y1
        p = int(round(max(bw, bh) * pad))
        ox, oy = max(0, x1 - p), max(0, y1 - p)
        ex, ey = min(W, x2 + p), min(H, y2 + p)
        crop = image[oy:ey, ox:ex]
        if crop.size == 0:
            return None
        bx1, by1, bx2, by2 = x1 - ox, y1 - oy, x2 - ox, y2 - oy
        box_r = (bw + bh) / 4.0

        # 1) Si el detector normal encuentra un disco que coincide con la caja, se usa ese.
        dets = self._detect_core(crop, include_uncertain=True, surround_check=False, use_edges=False)
        match = [t for t in dets if bx1 <= t.disc.cx <= bx2 and by1 <= t.disc.cy <= by2
                 and abs(t.disc.r - box_r) < 0.3 * box_r]
        if match:
            best = max(match, key=lambda t: t.disc.r)
        else:
            # 2) Si no (por ejemplo, fondo de colores), se ajusta el borde desde la caja.
            disc = disc_from_bbox(crop, (bx1, by1, bx2, by2))
            if disc is None:
                return None
            best = self._read(crop, disc, self.cfg.min_ring_conf_bbox)
        if best.label is None and not include_uncertain:
            return None
        return _shift(best, ox, oy)

    def _read(self, image, disc: Disc, min_conf: Optional[float] = None) -> TargetDetection:
        rings = read_rings(self._smooth(image), disc, self.clf, self.cfg)
        if min_conf is not None:
            for r in rings:
                if r.confidence < min_conf:
                    r.color = None
        colors = [r.color for r in rings]
        if any(c is None for c in colors):
            return TargetDetection(None, None, colors, rings, disc)
        label, total = label_for(colors)
        return TargetDetection(label, total, colors, rings, disc)


def _shift(t: TargetDetection, dx: int, dy: int) -> TargetDetection:
    """Pasa una detección de coordenadas del recorte a la imagen completa."""
    d = t.disc
    disc = Disc(d.cx + dx, d.cy + dy, d.axes, d.angle, d.contour + np.array([dx, dy]))
    for r in t.rings:
        r.samples = [(x + dx, y + dy, c) for x, y, c in r.samples]
    return TargetDetection(t.label, t.total, t.colors, t.rings, disc)
