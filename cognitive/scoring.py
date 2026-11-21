"""Reglas del cognitive target.

Python puro (sin numpy ni OpenCV): funciona tal cual en MicroPython.

- 5 anillos concéntricos de igual ancho (el círculo central cuenta como uno).
- Valores: negro -2, rojo -1, amarillo 0, verde +1, azul +2.
- Se suman los 5 anillos por separado, aunque los colores se repitan.
- Suma 2 -> HARMED, 1 -> STABLE, 0 -> UNHARMED, cualquier otra -> FAKE (no es víctima).
"""

BLACK, RED, YELLOW, GREEN, BLUE = "black", "red", "yellow", "green", "blue"
COLORS = [BLACK, RED, YELLOW, GREEN, BLUE]

VALUES = {BLACK: -2, RED: -1, YELLOW: 0, GREEN: 1, BLUE: 2}

HARMED = "HARMED"
STABLE = "STABLE"
UNHARMED = "UNHARMED"
FAKE = "FAKE"

SUM_TO_LABEL = {2: HARMED, 1: STABLE, 0: UNHARMED}

# Letra corta por color (para nombres de archivo y resúmenes): K = negro
LETTER = {BLACK: "K", RED: "R", YELLOW: "Y", GREEN: "G", BLUE: "B"}
FROM_LETTER = {v: k for k, v in LETTER.items()}


def ring_sum(colors):
    """colors: lista de 5 colores (centro -> afuera)."""
    return sum(VALUES[c] for c in colors)


def label_for(colors):
    """Devuelve (etiqueta, suma)."""
    s = ring_sum(colors)
    return SUM_TO_LABEL.get(s, FAKE), s


def sequence_code(colors):
    """['blue','red',...] -> 'BR...' (centro -> afuera)."""
    return "".join(LETTER[c] for c in colors)


def parse_sequence(code):
    return [FROM_LETTER[ch] for ch in code]
