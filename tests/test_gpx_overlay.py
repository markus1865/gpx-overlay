"""Automatic tests for gpx-overlay.py."""

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO_GPX = ROOT / "examples" / "demo.gpx"
DEMO_FIT = ROOT / "examples" / "demo.fit"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "gpx_overlay", str(ROOT / "gpx-overlay.py")
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load gpx-overlay.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gpx_overlay = load_module()


class HelperTests(unittest.TestCase):
    def test_haversine_zero(self):
        self.assertEqual(gpx_overlay.haversine(48.0, 11.0, 48.0, 11.0), 0.0)

    def test_haversine_one_degree_of_latitude(self):
        metres = gpx_overlay.haversine(0.0, 0.0, 1.0, 0.0)
        self.assertAlmostEqual(metres / 1000.0, 111.19, delta=0.2)

    def test_fit_semicircles_to_degrees(self):
        self.assertAlmostEqual(gpx_overlay._fit_semicircles_to_deg(2**31), 180.0)
        self.assertEqual(gpx_overlay._fit_semicircles_to_deg(0), 0.0)
        self.assertIsNone(gpx_overlay._fit_semicircles_to_deg(None))

    def test_format_time(self):
        self.assertEqual(gpx_overlay.fmt_time(0), "00:00")
        self.assertEqual(gpx_overlay.fmt_time(59.9), "00:59")
        self.assertEqual(gpx_overlay.fmt_time(3661), "1:01:01")

    def test_format_pace(self):
        self.assertEqual(gpx_overlay.fmt_pace(None), "--:--")
        self.assertEqual(gpx_overlay.fmt_pace(31), "--:--")
        self.assertEqual(gpx_overlay.fmt_pace(5.5), "5:30")

    def test_format_speed(self):
        self.assertEqual(gpx_overlay.fmt_speed(None), "--")
        self.assertEqual(gpx_overlay.fmt_speed(12.345), "12.3 km/h")

    def test_format_slope(self):
        self.assertEqual(gpx_overlay.fmt_slope(None), "--%")
        self.assertEqual(gpx_overlay.fmt_slope(3.21), "+3.2%")
        self.assertEqual(gpx_overlay.fmt_slope(-3.21), "-3.2%")
        self.assertEqual(gpx_overlay.fmt_slope(0.0), "0.0%")

    def test_slope_colors(self):
        self.assertEqual(gpx_overlay.slope_color(None), gpx_overlay.ACCENT_COLOR)
        self.assertEqual(gpx_overlay.slope_color(0.0), (100, 220, 100, 255))
        self.assertEqual(gpx_overlay.slope_color(3.0), (255, 210, 0, 255))
        self.assertEqual(gpx_overlay.slope_color(6.0), (255, 140, 0, 255))
        self.assertEqual(gpx_overlay.slope_color(10.0), (255, 60, 60, 255))

    def test_heart_rate_colors(self):
        self.assertEqual(
            gpx_overlay.hr_color(None, 100, 191), gpx_overlay.ACCENT_COLOR
        )
        self.assertEqual(
            gpx_overlay.hr_color(100, 100, 191), (255, 255, 255, 255)
        )
        self.assertEqual(gpx_overlay.hr_color(191, 100, 191), (255, 0, 0, 255))
        self.assertEqual(
            gpx_overlay.hr_color(150, 100, 100), gpx_overlay.ACCENT_COLOR
        )

    def test_corner_positions(self):
        args = (1000, 800, 100, 50, 40)
        self.assertEqual(gpx_overlay.corner_xy("top-left", *args), (40, 40))
        self.assertEqual(gpx_overlay.corner_xy("top-right", *args), (860, 40))
        self.assertEqual(gpx_overlay.corner_xy("bottom-left", *args), (40, 710))
        self.assertEqual(gpx_overlay.corner_xy("bottom-right", *args), (860, 710))

    def test_spring_easing(self):
        self.assertAlmostEqual(gpx_overlay.ease_out_spring(0.0), 0.0)
        self.assertAlmostEqual(gpx_overlay.ease_out_spring(1.0), 1.0)
        self.assertAlmostEqual(gpx_overlay.ease_out_spring(-1.0), 0.0)
        self.assertAlmostEqual(gpx_overlay.ease_out_spring(2.0), 1.0)
        self.assertGreater(gpx_overlay.ease_out_spring(0.28), 1.0)


class DemoFileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gpx_data, cls.gpx_coords, cls.gpx_has_ele = gpx_overlay.parse_gpx(
            str(DEMO_GPX)
        )
        cls.fit_data, cls.fit_coords, cls.fit_has_ele = gpx_overlay.parse_gpx(
            str(DEMO_FIT)
        )

    def test_gpx_demo(self):
        self.assertEqual(len(self.gpx_data), 121)
        self.assertEqual(len(self.gpx_coords), 121)
        self.assertTrue(self.gpx_has_ele)
        self.assertEqual(self.gpx_data[0]["elapsed_s"], 0.0)
        self.assertAlmostEqual(self.gpx_data[-1]["elapsed_s"], 600.0)
        self.assertAlmostEqual(self.gpx_data[-1]["distance_km"], 4.38, delta=0.05)
        self.assertIsNotNone(self.gpx_data[0]["hr"])
        self.assertIsNotNone(self.gpx_data[-1]["slope_pct"])

    def test_fit_demo(self):
        self.assertEqual(len(self.fit_data), 121)
        self.assertEqual(len(self.fit_coords), 121)
        self.assertTrue(self.fit_has_ele)
        self.assertEqual(self.fit_data[0]["elapsed_s"], 0.0)
        self.assertAlmostEqual(self.fit_data[-1]["elapsed_s"], 600.0)
        self.assertAlmostEqual(self.fit_data[-1]["distance_km"], 4.38, delta=0.05)
        self.assertIsNotNone(self.fit_data[0]["hr"])
        self.assertIsNotNone(self.fit_data[-1]["slope_pct"])

    def test_demo_files_agree(self):
        self.assertEqual(len(self.gpx_data), len(self.fit_data))
        for gpx_point, fit_point in zip(self.gpx_data, self.fit_data):
            self.assertAlmostEqual(gpx_point["elapsed_s"], fit_point["elapsed_s"])
            self.assertAlmostEqual(
                gpx_point["distance_km"], fit_point["distance_km"], delta=0.01
            )

    def test_interpolate(self):
        first = gpx_overlay.interpolate(self.gpx_data, -1.0)
        last = gpx_overlay.interpolate(self.gpx_data, 700.0)
        middle = gpx_overlay.interpolate(self.gpx_data, 300.0)

        self.assertIs(first, self.gpx_data[0])
        self.assertIs(last, self.gpx_data[-1])
        self.assertAlmostEqual(middle["elapsed_s"], 300.0)
        self.assertGreater(middle["distance_km"], 0.0)
        self.assertLess(middle["distance_km"], self.gpx_data[-1]["distance_km"])


class RenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.coords, cls.has_ele = gpx_overlay.parse_gpx(str(DEMO_GPX))
        cls.font_big = gpx_overlay.load_font(20)
        cls.font_small = gpx_overlay.load_font(10)
        cls.minimap = gpx_overlay.MiniMap(cls.coords, 80, 160, 2)
        cls.graph = gpx_overlay.ElevationGraph(
            cls.data, 160, 60, 160, 2, cls.font_small
        )
        cls.state = gpx_overlay.interpolate(cls.data, 300.0)

    def test_minimap_render_and_cache(self):
        image = self.minimap.render(0.0)
        self.assertEqual(image.size, (80, 80))
        self.assertIs(image, self.minimap.render(0.0))
        self.assertEqual(self.minimap.render(1.0).size, (80, 80))

    def test_elevation_graph_render_and_cache(self):
        image = self.graph.render(0.5)
        self.assertEqual(image.size, (160, 60))
        self.assertIs(image, self.graph.render(0.5))

    def render(self, background):
        return gpx_overlay.render_frame(
            self.state,
            320,
            180,
            self.font_big,
            self.font_small,
            "top-left",
            160,
            2,
            self.has_ele,
            True,
            True,
            self.minimap,
            "bottom-right",
            self.graph,
            "bottom-left",
            0.5,
            100,
            191,
            background=background,
            text_cache=gpx_overlay.TextCache(),
        )

    def test_render_frame_with_alpha(self):
        frame, has_alpha = self.render("transparent")
        self.assertTrue(has_alpha)
        self.assertEqual(frame.shape, (180, 320, 4))
        self.assertGreater(int(frame[:, :, 3].max()), 0)

    def test_render_frame_without_alpha(self):
        frame, has_alpha = self.render("black")
        self.assertFalse(has_alpha)
        self.assertEqual(frame.shape, (180, 320, 3))


if __name__ == "__main__":
    unittest.main()
