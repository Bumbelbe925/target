"""Parámetros ajustables del detector de cognitive targets.

Todos los umbrales viven aquí para poder calibrar sin tocar la lógica.
Escala de HSV de OpenCV: H 0-179, S 0-255, V 0-255.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # --- Clasificación de color por HSV + croma (modo por defecto) ---
    # croma = max(B,G,R) - min(B,G,R): cuánto color tiene de verdad el píxel.
    # El negro impreso puede tener "tono" azul y saturación HSV alta (porque es oscuro),
    # pero su croma es bajo. Por eso el croma decide entre color y negro/fondo.
    chroma_min: int = 45           # croma mínimo para ser rojo/amarillo/verde/azul
    sat_min: int = 70              # además, saturación HSV mínima para ser color
    black_v_max: int = 110         # sin color y con V menor a esto = negro (si no: pared/fondo)
    black_v_hard: int = 35         # V menor a esto = negro siempre
    color_v_min: int = 75          # un color impreso es más claro que esto; lo más oscuro es negro
    # Rangos de tono (H de OpenCV, 0-179)
    red_h_low_max: int = 9         # rojo: H <= esto ...
    red_h_high_min: int = 165      # ... o H >= esto
    yellow_h: tuple = (14, 36)
    green_h: tuple = (37, 88)
    blue_h: tuple = (89, 135)

    # --- Clasificación por prototipos LAB (opcional, ver tools/calibrate_colors.py) ---
    prototypes_file: Optional[str] = None
    prototype_max_dist: float = 60.0   # distancia LAB máxima para aceptar un color
    prototype_l_weight: float = 0.35   # peso del brillo (L) frente al color (a, b): la luz cambia más que el tono

    # --- Preprocesado ---
    blur_ksize: int = 3            # desenfoque leve antes de clasificar (impar; 0/1 = nada)
    open_ksizes: tuple = (3, 7, 11)  # aperturas a probar: cortan líneas pegadas al target (px)
    max_open_frac: float = 0.15      # una apertura de k px solo vale para targets de diámetro >= k / esto

    # --- Candidatos (discos de color) ---
    min_diameter_px: int = 18
    max_diameter_px: int = 3000
    max_aspect: float = 1.45           # eje mayor / eje menor de la elipse
    min_ellipse_fill: float = 0.80     # área del contorno / área de la elipse ajustada
    max_ellipse_fill: float = 1.20
    min_solidity: float = 0.90         # área / área de la envolvente convexa
    min_colored_inside: float = 0.82   # fracción de píxeles "de color" dentro de la elipse
    min_ellipse_iou: float = 0.90      # coincidencia contorno vs. elipse ajustada (un cuadrado da ~0.85)
    use_edges: bool = False            # respaldo por bordes (se activa solo en analyze_roi)
    require_plain_surround: bool = True  # alrededor del target debe haber pared (no colores)
    max_surround_colored: float = 0.35   # fracción máxima de píxeles de color en el anillo exterior

    # --- Lectura de anillos ---
    n_angles: int = 48                 # rayos por anillo
    band_lo: float = 0.30              # se muestrea entre band_lo y band_hi de cada franja
    band_hi: float = 0.70              #   (lejos de las fronteras entre anillos)
    n_radii: int = 3                   # radios por franja
    min_ring_conf: float = 0.55        # fracción mínima de votos del color ganador
    min_ring_conf_bbox: float = 0.75   # más exigente cuando el borde salió de la caja de un modelo
