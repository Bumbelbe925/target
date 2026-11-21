"""Paso 2: leer el color de cada uno de los 5 anillos.

La regla dice que los 5 anillos tienen el mismo ancho (0.5 cm de radio cada
uno, el centro incluido). Por eso cada anillo k (0 = centro, 4 = afuera) ocupa
la franja de radio normalizado [k/5, (k+1)/5] de la elipse exterior.

Se muestrea solo la parte central de cada franja (band_lo..band_hi) para no
caer en la frontera entre dos anillos, y se vota por mayoría. Si el ganador no
tiene suficientes votos, el anillo queda "incierto".
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .candidates import Disc
from .colors import BACKGROUND, ColorClassifier
from .config import Config
from .scoring import COLORS

N_RINGS = 5


@dataclass
class RingReading:
    index: int                 # 0 = centro ... 4 = exterior
    color: Optional[str]       # None si no hubo mayoría clara
    confidence: float          # fracción de votos del ganador
    votes: dict                # color -> votos (incluye "fondo")
    samples: list = field(default_factory=list)  # [(x, y, clase), ...] para dibujar


def ellipse_point(disc: Disc, f: float, phi: float):
    """Punto a radio normalizado f (0 centro, 1 borde) y ángulo phi sobre la elipse."""
    a = disc.axes[0] / 2.0
    b = disc.axes[1] / 2.0
    t = math.radians(disc.angle)
    x = a * f * math.cos(phi)
    y = b * f * math.sin(phi)
    return (disc.cx + x * math.cos(t) - y * math.sin(t),
            disc.cy + x * math.sin(t) + y * math.cos(t))


def read_rings(image_bgr: np.ndarray, disc: Disc, clf: ColorClassifier, cfg: Config) -> List[RingReading]:
    H, W = image_bgr.shape[:2]
    readings: List[RingReading] = []
    for k in range(N_RINGS):
        lo = (k + cfg.band_lo) / N_RINGS
        hi = (k + cfg.band_hi) / N_RINGS
        if k == 0:
            lo = 0.25 / N_RINGS  # el centro es un disco: se evita el punto exacto del centro
        if cfg.n_radii == 1:
            radii = [(lo + hi) / 2]
        else:
            radii = [lo + (hi - lo) * i / (cfg.n_radii - 1) for i in range(cfg.n_radii)]

        pts = []
        for f in radii:
            # En el centro hay pocos píxeles: menos ángulos para no repetir el mismo píxel.
            n = cfg.n_angles if k > 0 else max(8, cfg.n_angles // 3)
            for i in range(n):
                phi = 2 * math.pi * i / n + (0.5 * k)  # desfase por anillo
                x, y = ellipse_point(disc, f, phi)
                xi, yi = int(round(x)), int(round(y))
                if 0 <= xi < W and 0 <= yi < H:
                    pts.append((xi, yi))
        if not pts:
            readings.append(RingReading(k, None, 0.0, {}, []))
            continue

        arr = np.array(pts)
        px = image_bgr[arr[:, 1], arr[:, 0]]
        cls = clf.classify_bgr(px.reshape(-1, 1, 3)).reshape(-1)

        votes = {}
        for cidx in cls:
            key = "fondo" if cidx == BACKGROUND else COLORS[int(cidx)]
            votes[key] = votes.get(key, 0) + 1
        color_votes = {c: v for c, v in votes.items() if c != "fondo"}
        total = len(cls)
        if color_votes:
            best = max(color_votes, key=color_votes.get)
            conf = color_votes[best] / total
        else:
            best, conf = None, 0.0
        best_out = best if conf >= cfg.min_ring_conf else None
        samples = [(int(x), int(y), int(c)) for (x, y), c in zip(pts, cls)]
        readings.append(RingReading(k, best_out, float(conf), votes, samples))
    return readings
