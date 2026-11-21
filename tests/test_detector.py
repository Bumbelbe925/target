"""Pruebas automáticas. Correr con:  python -m unittest discover tests -v"""

import random
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import generate_synthetic as gen  # noqa: E402
from cognitive import FAKE, HARMED, STABLE, UNHARMED, CognitiveDetector, label_for  # noqa: E402
from cognitive.scoring import parse_sequence  # noqa: E402

HOJA = ROOT / "ejemplos" / "hoja_targets.png"


def run_set(folder, det, use_boxes=False):
    boxes = {}
    if use_boxes:
        for line in (folder / "labels.csv").read_text().splitlines()[1:]:
            p = line.split(",")
            boxes[p[0]] = tuple(int(v) for v in p[3:7])
    ok = dangerous = 0
    files = sorted(folder.glob("*.jpg"))
    for f in files:
        exp = f.stem.split("_")[0]
        img = cv2.imread(str(f))
        if use_boxes:
            t = det.analyze_roi(img, boxes[f.name])
            dets = [t] if t else []
        else:
            dets = sorted(det.detect(img), key=lambda t: -t.disc.r)
        got = dets[0].label if dets else None
        ok += got == exp
        if got in (HARMED, STABLE, UNHARMED) and got != exp:
            dangerous += 1
    return ok / len(files), dangerous


class TestReglas(unittest.TestCase):
    def test_sumas(self):
        self.assertEqual(label_for(parse_sequence("BBKYY")), (HARMED, 2))
        self.assertEqual(label_for(parse_sequence("GYYYY")), (STABLE, 1))
        self.assertEqual(label_for(parse_sequence("BRYGK")), (UNHARMED, 0))
        self.assertEqual(label_for(parse_sequence("YYYYY")), (UNHARMED, 0))
        self.assertEqual(label_for(parse_sequence("GYKBB")), (FAKE, 3))
        self.assertEqual(label_for(parse_sequence("KKKKK")), (FAKE, -10))


@unittest.skipUnless(HOJA.exists(), "falta ejemplos/hoja_targets.png")
class TestEjemplo(unittest.TestCase):
    def test_hoja(self):
        dets = CognitiveDetector().detect(cv2.imread(str(HOJA)))
        self.assertEqual([t.sequence for t in dets], ["BBKYY", "GYYYY", "BRYGK", "GYKBB"])
        self.assertEqual([t.label for t in dets], [HARMED, STABLE, UNHARMED, FAKE])

    def test_anillos_vecinos_iguales_no_se_juntan(self):
        """En BBKYY el centro y el anillo 2 son azules: cuentan como dos anillos (+2 +2)."""
        t = CognitiveDetector().detect(cv2.imread(str(HOJA)))[0]
        self.assertEqual(t.colors[:2], ["blue", "blue"])
        self.assertEqual(t.total, 2)


class TestSinteticas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        base = Path(cls.tmp.name)
        cls.easy = base / "easy"
        cls.hard = base / "hard"
        cls.dist = base / "dist"
        cls.rand = base / "rand"
        gen.generate(cls.easy, 120, seed=11)
        gen.generate(cls.hard, 120, hard=True, seed=12)
        gen.generate(cls.dist, 120, hard=True, distractors=True, seed=13)
        gen.generate(cls.rand, 80, hard=True, distractors=True, bg="random", seed=14)
        cls.det = CognitiveDetector()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_pared_limpia(self):
        acc, bad = run_set(self.easy, self.det)
        self.assertGreaterEqual(acc, 0.97)
        self.assertEqual(bad, 0)

    def test_dificiles(self):
        acc, bad = run_set(self.hard, self.det)
        self.assertGreaterEqual(acc, 0.95)
        self.assertEqual(bad, 0)

    def test_con_distractores(self):
        acc, bad = run_set(self.dist, self.det)
        self.assertGreaterEqual(acc, 0.88)
        self.assertEqual(bad, 0)

    def test_fondo_de_colores_no_inventa(self):
        """Sin modelo, con fondo de colores puede no encontrarlo, pero nunca debe reportar mal."""
        _, bad = run_set(self.rand, self.det)
        self.assertEqual(bad, 0)

    def test_fondo_de_colores_con_caja_de_modelo(self):
        acc, bad = run_set(self.rand, self.det, use_boxes=True)
        self.assertGreaterEqual(acc, 0.90)
        self.assertEqual(bad, 0)


class TestFalsosPositivos(unittest.TestCase):
    def setUp(self):
        self.det = CognitiveDetector()

    def blank(self):
        return np.full((240, 320, 3), 232, np.uint8)

    def test_pared_vacia(self):
        self.assertEqual(self.det.detect(self.blank()), [])

    def test_carita(self):
        img = self.blank()
        gen.draw_smiley(img, 160, 120, 60, random.Random(1))
        self.assertEqual(self.det.detect(img), [])

    def test_cuadro_de_color(self):
        img = self.blank()
        cv2.rectangle(img, (100, 60), (220, 180), (40, 210, 240), -1)  # amarillo
        self.assertEqual(self.det.detect(img), [])

    def test_aro_de_color(self):
        img = self.blank()
        cv2.circle(img, (160, 120), 60, (175, 95, 30), 14)  # aro azul con centro de pared
        self.assertEqual(self.det.detect(img), [])

    def test_letra(self):
        img = self.blank()
        cv2.putText(img, "H", (110, 190), cv2.FONT_HERSHEY_SIMPLEX, 5, (45, 40, 205), 16)
        self.assertEqual(self.det.detect(img), [])


if __name__ == "__main__":
    unittest.main()
