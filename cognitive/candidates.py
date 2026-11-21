"""Paso 1: encontrar discos de color (posibles targets) en la imagen."""

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from .colors import BACKGROUND
from .config import Config

_NO_CONTOUR = np.zeros((0, 1, 2), np.int32)


@dataclass
class Disc:
    cx: float
    cy: float
    axes: tuple      # (eje1, eje2) diámetros completos de la elipse
    angle: float     # grados, como lo da cv2.fitEllipse
    contour: np.ndarray

    @property
    def r(self) -> float:
        return (self.axes[0] + self.axes[1]) / 4.0

    @property
    def ellipse(self):
        return ((self.cx, self.cy), self.axes, self.angle)


def target_mask(classes: np.ndarray, k: int = 3) -> np.ndarray:
    """255 donde el píxel es de alguno de los 5 colores del target.

    k: tamaño de la apertura. Más grande corta líneas más gruesas pegadas al target,
    pero borra targets muy chicos; por eso se prueban varios tamaños.
    """
    mask = (classes != BACKGROUND).astype(np.uint8) * 255
    if k and k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def edge_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Regiones cerradas por bordes fuertes (para cuando el fondo también es de color)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    e1 = cv2.Canny(gray, 40, 120)
    e2 = cv2.Canny(hsv[:, :, 1], 40, 120)
    e3 = cv2.Canny(hsv[:, :, 0], 40, 120)
    edges = cv2.dilate(e1 | e2 | e3, np.ones((3, 3), np.uint8))
    # Rellenar cada contorno cerrado
    filled = np.zeros_like(edges)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(filled, cnts, -1, 255, -1)
    return filled


def ellipse_iou(contour, ellipse) -> float:
    """Qué tanto coincide el contorno relleno con la elipse ajustada (1.0 = idénticos)."""
    x, y, w, h = cv2.boundingRect(contour)
    pad = 4
    ox, oy = x - pad, y - pad
    W, H = w + 2 * pad, h + 2 * pad
    a = np.zeros((H, W), np.uint8)
    b = np.zeros((H, W), np.uint8)
    cv2.drawContours(a, [contour - [ox, oy]], -1, 1, -1)
    (ex, ey), axes, ang = ellipse
    cv2.ellipse(b, ((ex - ox, ey - oy), axes, ang), 1, -1)
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return inter / union if union else 0.0


def find_discs(mask: np.ndarray, cfg: Config, colored: Optional[np.ndarray] = None,
               open_k: int = 0) -> List[Disc]:
    """Contornos de mask que parecen un disco de color lleno.

    colored: máscara de píxeles de color para verificar que la elipse esté llena (por defecto, mask).
    open_k: apertura usada para crear mask; redondea figuras chicas, así que exige targets grandes.
    """
    min_area = np.pi * (cfg.min_diameter_px / 2) ** 2
    if colored is None:
        colored = mask
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out: List[Disc] = []
    for c in cnts:
        if len(c) < 5:
            continue
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        (ex, ey), (a1, a2), ang = cv2.fitEllipse(c)
        if min(a1, a2) <= 0:
            continue
        d = (a1 + a2) / 2
        if not (cfg.min_diameter_px <= d <= cfg.max_diameter_px):
            continue
        if open_k > 3 and d * cfg.max_open_frac < open_k:
            continue
        if max(a1, a2) / min(a1, a2) > cfg.max_aspect:
            continue
        ell_area = np.pi * a1 * a2 / 4
        fill = area / ell_area
        if not (cfg.min_ellipse_fill <= fill <= cfg.max_ellipse_fill):
            continue
        if ellipse_iou(c, ((ex, ey), (a1, a2), ang)) < cfg.min_ellipse_iou:
            continue
        hull = cv2.convexHull(c)
        hull_a = cv2.contourArea(hull)
        if hull_a <= 0 or area / hull_a < cfg.min_solidity:
            continue

        # La elipse debe estar llena de color (descarta aros con interior blanco, como las caritas).
        em = np.zeros(mask.shape, np.uint8)
        cv2.ellipse(em, ((ex, ey), (a1 * 0.92, a2 * 0.92), ang), 255, -1)
        inside = em > 0
        n_in = int(inside.sum())
        if n_in == 0:
            continue
        frac = np.count_nonzero(colored[inside]) / n_in
        if frac < cfg.min_colored_inside:
            continue

        out.append(Disc(float(ex), float(ey), (float(a1), float(a2)), float(ang), c))
    return out


def dedupe(discs: List[Disc]) -> List[Disc]:
    """Une candidatos repetidos (mismo target hallado con distintos cortes) y quita los anidados."""
    discs = sorted(discs, key=lambda d: -d.r)
    kept: List[Disc] = []
    for d in discs:
        dup = False
        for k in kept:
            dist = np.hypot(d.cx - k.cx, d.cy - k.cy)
            if dist < 0.3 * k.r and abs(d.r - k.r) < 0.25 * k.r:
                dup = True  # el mismo target
                break
            if dist + d.r <= k.r * 1.05:
                dup = True  # está dentro de otro (por ejemplo, un anillo interior)
                break
        if not dup:
            kept.append(d)
    return kept


def disc_from_bbox(image_bgr: np.ndarray, bbox, n_rays: int = 48, window=(0.85, 1.2)) -> Optional[Disc]:
    """Ajusta la elipse del target partiendo de la caja de un modelo.

    Desde el centro de la caja, en cada rayo busca el cambio de color más fuerte
    entre 0.85 y 1.2 veces el borde de la caja (el borde exterior del target) y
    ajusta una elipse a esos puntos, descartando los raros.
    """
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    a, b = (x2 - x1) / 2.0, (y2 - y1) / 2.0
    if a < 4 or b < 4:
        return None
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    H, W = lab.shape[:2]
    scales = np.linspace(window[0], window[1], 40)
    pts = []
    for i in range(n_rays):
        phi = 2 * np.pi * i / n_rays
        xs = cx + a * scales * np.cos(phi)
        ys = cy + b * scales * np.sin(phi)
        ok = (xs >= 0) & (xs < W - 1) & (ys >= 0) & (ys < H - 1)
        if ok.sum() < 6:
            continue
        mapx = xs[ok].astype(np.float32).reshape(1, -1)
        mapy = ys[ok].astype(np.float32).reshape(1, -1)
        prof = cv2.remap(lab, mapx, mapy, cv2.INTER_LINEAR).reshape(-1, 3)
        diff = np.linalg.norm(np.diff(prof, axis=0), axis=1)
        j = int(np.argmax(diff))
        if diff[j] < 8:  # sin borde claro en este rayo
            continue
        pts.append(((xs[ok][j] + xs[ok][j + 1]) / 2, (ys[ok][j] + ys[ok][j + 1]) / 2))
    if len(pts) < 8:
        return Disc(cx, cy, (2 * a, 2 * b), 0.0, _NO_CONTOUR)
    P = np.array(pts, np.float32)
    # Quitar puntos raros (bordes de otras cosas): radio normalizado lejos de la mediana
    rn = np.hypot((P[:, 0] - cx) / a, (P[:, 1] - cy) / b)
    med = np.median(rn)
    P = P[np.abs(rn - med) < 0.08]
    if len(P) < 6:
        return Disc(cx, cy, (2 * a * med, 2 * b * med), 0.0, _NO_CONTOUR)
    (ex, ey), (a1, a2), ang = cv2.fitEllipse(P.reshape(-1, 1, 2))
    if max(a1, a2) / max(1e-6, min(a1, a2)) > 1.6:
        return Disc(cx, cy, (2 * a * med, 2 * b * med), 0.0, _NO_CONTOUR)
    return Disc(float(ex), float(ey), (float(a1), float(a2)), float(ang), P.astype(np.int32).reshape(-1, 1, 2))
