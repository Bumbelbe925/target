"""Clasificación de píxeles en los 5 colores del target (o fondo).

Dos modos:
  - HSV con reglas (por defecto, no necesita calibración).
  - Prototipos LAB calibrados con fotos reales (tools/calibrate_colors.py).
    Más robusto ante cambios de luz.

Índices: 0 negro, 1 rojo, 2 amarillo, 3 verde, 4 azul, -1 fondo/desconocido.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .scoring import COLORS

BACKGROUND = -1

# Colores BGR para dibujar cada clase
DRAW_BGR = {
    "black": (30, 30, 30),
    "red": (40, 40, 220),
    "yellow": (40, 220, 240),
    "green": (60, 170, 40),
    "blue": (200, 90, 30),
}

NOMBRE_ES = {"black": "NEGRO", "red": "ROJO", "yellow": "AMARILLO", "green": "VERDE", "blue": "AZUL"}


class ColorClassifier:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.prototypes = None
        if cfg.prototypes_file:
            self.load_prototypes(cfg.prototypes_file)

    # ---------- prototipos LAB ----------
    def load_prototypes(self, path: str):
        data = json.loads(Path(path).read_text())
        names = [c for c in COLORS if c in data]
        missing = [c for c in COLORS if c not in data]
        if missing:
            raise ValueError(f"Faltan colores en {path}: {missing}")
        self.proto_names = names + (["background"] if "background" in data else [])
        self.prototypes = np.array([data[n] for n in self.proto_names], np.float32)

    # ---------- clasificación ----------
    def classify_bgr(self, bgr: np.ndarray) -> np.ndarray:
        """bgr: (..., 3) uint8. Devuelve índices int8 con la misma forma sin el canal."""
        shape = bgr.shape[:-1]
        flat = bgr.reshape(-1, 1, 3)
        if self.prototypes is not None:
            out = self._classify_lab(flat)
        else:
            out = self._classify_hsv(flat)
        return out.reshape(shape)

    def _classify_hsv(self, flat_bgr):
        c = self.cfg
        bgr = flat_bgr.reshape(-1, 3).astype(np.int16)
        chroma = bgr.max(axis=1) - bgr.min(axis=1)
        hsv = cv2.cvtColor(flat_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int16)
        h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
        out = np.full(h.shape, BACKGROUND, np.int8)

        colored = (s >= c.sat_min) & (chroma >= c.chroma_min) & (v >= c.color_v_min)
        out[colored & ((h <= c.red_h_low_max) | (h >= c.red_h_high_min))] = 1
        out[colored & (h >= c.yellow_h[0]) & (h <= c.yellow_h[1])] = 2
        out[colored & (h >= c.green_h[0]) & (h <= c.green_h[1])] = 3
        out[colored & (h >= c.blue_h[0]) & (h <= c.blue_h[1])] = 4

        # Negro: sin color y oscuro. Lo claro sin color es pared/fondo.
        black = (~colored & (v < c.black_v_max)) | (v < c.black_v_hard)
        out[black] = 0
        return out

    def _classify_lab(self, flat_bgr):
        lab = cv2.cvtColor(flat_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
        wgt = np.array([self.cfg.prototype_l_weight, 1.0, 1.0], np.float32)
        d = np.linalg.norm((lab[:, None, :] - self.prototypes[None, :, :]) * wgt, axis=2)
        idx = np.argmin(d, axis=1)
        best = d[np.arange(len(idx)), idx]
        out = np.full(len(idx), BACKGROUND, np.int8)
        for j, name in enumerate(self.proto_names):
            if name == "background":
                continue
            sel = (idx == j) & (best <= self.cfg.prototype_max_dist)
            out[sel] = COLORS.index(name)
        return out
