"""Dibujo de resultados: elipse del target, fronteras de anillos, puntos de muestreo y tabla.

annotate(imagen, detecciones) devuelve la imagen completa anotada y, debajo,
un acercamiento de cada target con sus 5 anillos leídos.
"""

import math
from typing import List

import cv2
import numpy as np

from .colors import BACKGROUND, DRAW_BGR, NOMBRE_ES
from .pipeline import TargetDetection, to_bgr
from .rings import N_RINGS
from .scoring import COLORS, VALUES

LABEL_BGR = {
    "HARMED": (0, 0, 220),
    "STABLE": (0, 170, 220),
    "UNHARMED": (0, 160, 0),
    "FAKE": (120, 120, 120),
    None: (160, 160, 160),
}
LABEL_ES = {"HARMED": "HARMED (herida)", "STABLE": "STABLE (estable)",
            "UNHARMED": "UNHARMED (ilesa)", "FAKE": "FALSO (no es victima)", None: "LECTURA INSEGURA"}


def _text(img, text, org, scale, color, t=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), t + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, t, cv2.LINE_AA)


def _short(t: TargetDetection) -> str:
    if t.label is None:
        return f"? {t.sequence}"
    return f"{t.label} (suma {t.total:+d})"


def draw_detections(image: np.ndarray, dets: List[TargetDetection]) -> np.ndarray:
    img = to_bgr(image).copy()
    th = max(1, int(round(min(img.shape[:2]) / 300)))
    for t in dets:
        col = LABEL_BGR.get(t.label, LABEL_BGR[None])
        d = t.disc
        cv2.ellipse(img, d.ellipse, col, th + 1, cv2.LINE_AA)
        scale = max(0.35, min(0.9, d.r / 110))
        _text(img, _short(t), (int(d.cx - d.r), int(max(15, d.cy - d.r - 8))), scale, col, th)
    return img


def target_zoom(image: np.ndarray, t: TargetDetection, size: int = 300) -> np.ndarray:
    """Acercamiento con fronteras de anillos, puntos de muestreo y tabla de lectura."""
    img = to_bgr(image).copy()
    d = t.disc
    half = max(d.axes) / 2 * 1.2
    s = size / (2 * half)
    x0, y0 = d.cx - half, d.cy - half
    M = np.float32([[s, 0, -x0 * s], [0, s, -y0 * s]])
    crop = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

    def P(x, y):
        return int(round((x - x0) * s)), int(round((y - y0) * s))

    cxy = P(d.cx, d.cy)
    # Fronteras de los anillos (k/5 del borde exterior)
    for k in range(1, N_RINGS + 1):
        f = k / N_RINGS
        axes = (max(1, int(d.axes[0] / 2 * f * s)), max(1, int(d.axes[1] / 2 * f * s)))
        cv2.ellipse(crop, cxy, axes, d.angle, 0, 360, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.ellipse(crop, cxy, axes, d.angle, 0, 360, (0, 0, 0), 1, cv2.LINE_AA)

    # Puntos de muestreo, pintados con el color que se leyó
    for r in t.rings:
        for x, y, c in r.samples:
            p = P(x, y)
            if c == BACKGROUND:
                cv2.drawMarker(crop, p, (150, 150, 150), cv2.MARKER_TILTED_CROSS, 5, 1)
            else:
                cv2.circle(crop, p, 3, DRAW_BGR[COLORS[c]], -1, cv2.LINE_AA)
                cv2.circle(crop, p, 3, (255, 255, 255), 1, cv2.LINE_AA)

    # Número de anillo sobre un rayo
    for k in range(N_RINGS):
        a = d.axes[0] / 2 * (k + 0.5) / N_RINGS
        tt = math.radians(d.angle)
        x = d.cx + a * math.cos(tt)
        y = d.cy + a * math.sin(tt)
        _text(crop, str(k + 1), P(x, y), 0.45, (0, 0, 0), 1)

    # Tabla de lectura
    col = LABEL_BGR.get(t.label, LABEL_BGR[None])
    table_h = 40 + 22 * N_RINGS + 34
    table = np.full((table_h, size, 3), 255, np.uint8)
    cv2.putText(table, LABEL_ES.get(t.label, "?"), (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    for r in t.rings:
        y = 40 + 22 * r.index + 14
        name = "centro" if r.index == 0 else ("exterior" if r.index == N_RINGS - 1 else "")
        if r.color is None:
            txt = f"{r.index + 1} {name:8s} ?      ({r.confidence * 100:.0f}%)"
            sw = (180, 180, 180)
        else:
            txt = f"{r.index + 1} {name:8s} {NOMBRE_ES[r.color]:8s} {VALUES[r.color]:+d}  ({r.confidence * 100:.0f}%)"
            sw = DRAW_BGR[r.color]
        cv2.rectangle(table, (8, y - 11), (22, y + 3), sw, -1)
        cv2.rectangle(table, (8, y - 11), (22, y + 3), (90, 90, 90), 1)
        cv2.putText(table, txt, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)
    total = "?" if t.total is None else f"{t.total:+d}"
    cv2.putText(table, f"SUMA = {total}", (8, table_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    return np.vstack([crop, table])


def _legend(width: int) -> np.ndarray:
    items = [(NOMBRE_ES[c] + f" {VALUES[c]:+d}", DRAW_BGR[c]) for c in COLORS] + [("fondo", (150, 150, 150))]
    rows, row, x = [], [], 8
    for txt, c in items:
        w = 16 + int(len(txt) * 8) + 14
        if row and x + w > width:
            rows.append(row)
            row, x = [], 8
        row.append((x, txt, c))
        x += w
    rows.append(row)
    leg = np.full((24 * len(rows) + 6, width, 3), 255, np.uint8)
    for i, r in enumerate(rows):
        y = 15 + 24 * i
        for x, txt, c in r:
            cv2.circle(leg, (x + 6, y), 6, c, -1, cv2.LINE_AA)
            cv2.putText(leg, txt, (x + 16, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1, cv2.LINE_AA)
    return leg


def annotate(image: np.ndarray, dets: List[TargetDetection], zoom_size: int = 300) -> np.ndarray:
    full = draw_detections(image, dets)
    if not dets:
        return full
    row = np.hstack([target_zoom(image, t, zoom_size) for t in dets])
    W = max(full.shape[1], row.shape[1])

    def pad(im):
        if im.shape[1] == W:
            return im
        return np.hstack([im, np.full((im.shape[0], W - im.shape[1], 3), 255, np.uint8)])

    sep = np.full((6, W, 3), 220, np.uint8)
    return np.vstack([pad(full), sep, pad(row), pad(_legend(W))])
