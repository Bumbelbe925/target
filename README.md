# Detector de cognitive targets

Detecta targets de 5 anillos de color y los clasifica como HARMED, STABLE, UNHARMED o FAKE. Está pensado para targets impresos sobre una pared o fondo claro.

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Comandos básicos

```bash
# Procesar una imagen
python -m cognitive ejemplos/hoja_targets.png

# Procesar todas las imágenes de una carpeta
python -m cognitive mis_fotos/

# Usar la cámara
python -m cognitive --camera 0
```

Las imágenes procesadas se guardan en `detectadas/`. Para no guardarlas:

```bash
python -m cognitive mis_fotos/ --no-save
```

## Cómo funciona

1. Clasifica cada píxel como negro, rojo, amarillo, verde, azul o fondo.
2. Encuentra los discos de color y divide cada uno en 5 anillos iguales.
3. Lee el color de cada anillo por votación y suma sus valores: negro −2, rojo −1, amarillo 0, verde +1, azul +2.
4. Usa la suma para decidir el estado: 2 es HARMED, 1 es STABLE, 0 es UNHARMED y cualquier otra es FAKE.