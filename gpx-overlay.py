#!/usr/bin/env python3
"""
GPX Overlay Video Generator  (with live mini-map + optional slope)
-----------------------------------------------------------
Generates an overlay video (H.265/HEVC MP4, or ProRes 4444 .mov with alpha)
displaying:
  - Elapsed time
  - Distance (km)
  - Pace (min/km) or speed (km/h)
  - Heart rate (bpm)         [if present in GPX]
  - Slope / gradient (%)     [if elevation data present in GPX]
  - Mini-map of the full track with a start dot, a checkered finish
    marker and a moving position dot

All text is rendered with a solid black outline for readability on any
background.

Dependencies:
    pip install gpxpy pillow numpy opencv-python-headless
    pip install fitparse          # for .fit support

FFmpeg with libx265 is required.  Transparent mode also needs prores_ks.

Usage:
    python gpx-overlay.py your_activity.gpx [options]
    python gpx-overlay.py your_activity.fit [options]
    python gpx-overlay.py ride.gpx --stat-mode speed

The output file name is the input file name with a .mp4 ending
(your_activity.gpx -> your_activity.mp4).  Use --output to set another path.

All text uses a bold font.  The stat titles (TIME, PACE/SPEED, ...) and the
elevation numbers in the graph are drawn smaller than the stat values.

Options:
    FILE             Path to the GPX or FIT file (required)
    --output         Output MP4 filename          (default: input file name with .mp4 ending)
    --distance-source {gps,footpod}  Distance source for FIT files (default: gps)
    --pace-source    {gps,footpod}  Pace/speed source for FIT files (default: gps)
    --stat-mode      {pace,speed}   Show pace (min/km) or speed (km/h) (default: pace)
    --width          Video width in pixels         (default: 1920)
    --height         Video height in pixels        (default: 1080)
    --fps            Frames per second             (default: 30)
    --duration       Override total duration (s)   (default: auto from GPX)
    --bg-alpha       Background box opacity 0-255  (default: 160)
    --position       Stats box corner              (default: top-left)
    --map-position   Mini-map corner               (default: bottom-right)
    --map-size       Mini-map side length (px)     (default: 320)
    --no-map         Disable the mini-map
    --no-avg-pace    Disable the average pace/speed stat
    --avg-pace-transition {slide,fade}  AVG PACE / AVG SPEED entrance effect (default: slide)
    --no-slope       Disable the slope / gradient display
    --hr-color       Colour the HR value by intensity   (default: disabled)
    --ele-graph      Show the elevation graph panel     (default: enabled)
    --no-ele-graph   Disable the elevation graph panel
    --graph-position Corner for the elevation graph      (default: bottom-left)
    --graph-width    Elevation graph width in pixels      (default: 480)
    --graph-height   Elevation graph height in pixels     (default: 160)
    --text-color     Stat text colour as R,G,B      (default: white; black on white)
    --font-size      Base font size (px)           (default: 52)
    --outline-width  Text outline thickness (px)   (default: 2)
    --speed          Playback speed multiplier      (default: 1.0)

Corner choices: top-left | top-right | bottom-left | bottom-right

Combine with source footage:
    ffmpeg -i source_video.mp4 -i overlay.mp4 \\
           -filter_complex "[0:v][1:v]overlay=0:0" \\
           -c:v libx265 -crf 18 final_output.mp4

"""

import argparse
import math
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

try:
    import gpxpy
    import gpxpy.gpx
except ImportError:
    sys.exit("❌  Missing dependency: pip install gpxpy")

try:
    import fitparse
except ImportError:
    fitparse = None

try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit("❌  Missing dependency: pip install pillow numpy")

try:
    import cv2
except ImportError:
    sys.exit("❌  Missing dependency: pip install opencv-python-headless")


# ============================================================
# GPX Parsing
# ============================================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _extract_hr(pt):
    if not pt.extensions:
        return None
    for ext in pt.extensions:
        tag = ext.tag.split("}")[-1] if "}" in ext.tag else ext.tag
        if tag.lower() in ("hr", "heartrate"):
            try:
                return int(ext.text)
            except (TypeError, ValueError):
                pass
        for child in ext:
            ctag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if ctag.lower() in ("hr", "heartrate"):
                try:
                    return int(child.text)
                except (TypeError, ValueError):
                    pass
    return None


def _fit_semicircles_to_deg(value):
    try:
        return float(value) * (180.0 / 2**31)
    except Exception:
        return None


def _fit_field_value(record, *field_names):
    for name in field_names:
        value = record.get_value(name)
        if value is not None:
            return value
    return None


def parse_fit(path: str, distance_source: str = "gps", pace_source: str = "gps"):
    if fitparse is None:
        sys.exit("❌  Missing dependency: pip install fitparse")

    fitfile = fitparse.FitFile(path)
    raw = []
    for record in fitfile.get_messages("record"):
        time = record.get_value("timestamp")
        lat = record.get_value("position_lat")
        lon = record.get_value("position_long")
        if time is None or lat is None or lon is None:
            continue

        ele = record.get_value("enhanced_altitude")
        if ele is None:
            ele = record.get_value("altitude")

        hr = record.get_value("heart_rate")
        distance_m = None
        speed_mps = None
        if distance_source == "footpod" or pace_source == "footpod":
            distance_m = _fit_field_value(record, "distance", "total_distance")
            speed_mps = _fit_field_value(record, "enhanced_speed", "speed")

        raw.append({
            "time": time,
            "lat": _fit_semicircles_to_deg(lat),
            "lon": _fit_semicircles_to_deg(lon),
            "ele": ele,
            "hr": hr,
            "distance_m": distance_m,
            "speed_mps": speed_mps,
        })

    if len(raw) < 2:
        sys.exit("❌  FIT file has fewer than 2 timed track points.")

    return _process_raw_track(raw, distance_source=distance_source, pace_source=pace_source)


def parse_gpx(path: str, distance_source: str = "gps", pace_source: str = "gps"):
    """
    Returns:
        data   – list of dicts:
                   elapsed_s, distance_km, pace_min_km, hr, slope_pct, lat, lon
        coords – list of (lat, lon) for all raw track points (for the map)
    """
    path_obj = Path(path)
    if path_obj.suffix.lower() == ".fit":
        return parse_fit(path, distance_source=distance_source, pace_source=pace_source)

    with open(path, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)

    raw = []
    for track in gpx.tracks:
        for segment in track.segments:
            for pt in segment.points:
                if pt.time is None:
                    continue
                raw.append({
                    "time": pt.time,
                    "lat":  pt.latitude,
                    "lon":  pt.longitude,
                    "ele":  pt.elevation,       # may be None
                    "hr":   _extract_hr(pt),
                })

    if len(raw) < 2:
        sys.exit("❌  GPX file has fewer than 2 timed track points.")

    return _process_raw_track(raw)


def _process_raw_track(raw, distance_source: str = "gps", pace_source: str = "gps"):
    raw.sort(key=lambda p: p["time"])
    t0 = raw[0]["time"]

    has_ele = any(p["ele"] is not None for p in raw)

    # ── Smooth elevation with a 10-point centred moving average ──────────
    # GPS barometric elevation data has ~0.2 m quantisation noise that
    # inflates raw gain figures. A 10-point window reduces this to match
    # real-world gain values while preserving genuine gradient changes.
    if has_ele:
        _SMOOTH = 10
        raw_eles = [p["ele"] for p in raw]
        smoothed_eles = []
        for i in range(len(raw_eles)):
            lo = max(0, i - _SMOOTH // 2)
            hi = min(len(raw_eles), i + _SMOOTH // 2 + 1)
            vals = [e for e in raw_eles[lo:hi] if e is not None]
            smoothed_eles.append(sum(vals) / len(vals) if vals else raw_eles[i])
        for i, p in enumerate(raw):
            p["ele"] = smoothed_eles[i]

    result     = []
    cum_dist_m = 0.0

    # Segment lengths and times are used again by the rolling pace and
    # slope windows, so compute each one only once.
    seg_dist = [0.0] * len(raw)
    seg_time = [0.0] * len(raw)
    for i in range(1, len(raw)):
        seg_dist[i] = haversine(raw[i-1]["lat"], raw[i-1]["lon"],
                                raw[i]["lat"],   raw[i]["lon"])
        seg_time[i] = (raw[i]["time"] - raw[i-1]["time"]).total_seconds()

    for i, pt in enumerate(raw):
        elapsed = (pt["time"] - t0).total_seconds()

        if i > 0:
            cum_dist_m += seg_dist[i]

            if pace_source == "footpod":
                speed_mps = pt.get("speed_mps")
                speed_kmh = (speed_mps * 3.6) if (speed_mps is not None and speed_mps > 0.5) else None
                pace = (60 / speed_kmh) if speed_kmh is not None else None
            else:
                # ── Pace / speed: rolling 30-second window ──────────
                wdist, wtime = 0.0, 0.0
                j = i
                while j > 0 and wtime < 30:
                    wdist += seg_dist[j]
                    wtime += seg_time[j]
                    j -= 1
                spd_kmh = (wdist / wtime) * 3.6 if (wdist > 0 and wtime > 0) else 0
                speed_kmh = spd_kmh if spd_kmh > 0.5 else None
                pace      = (60 / speed_kmh) if speed_kmh is not None else None

            # ── Slope: rolling ~100 m horizontal distance window ─────────
            # Walk backwards until we have covered ~100 m of horizontal distance.
            # sele_end   = elevation at the CURRENT point (fixed before the loop).
            # sele_start = elevation at the OLDEST point in the window
            #              (updated each step so it always reflects the far end).
            slope_pct = None
            if has_ele and pt["ele"] is not None:
                sdist      = 0.0
                sele_end   = pt["ele"]          # current point — does not change
                sele_start = None
                k = i
                while k > 0 and sdist < 100:
                    sdist     += seg_dist[k]
                    sele_start = raw[k-1]["ele"] # oldest point seen so far
                    k -= 1
                if (sdist > 5
                        and sele_start is not None
                        and sele_end   is not None):
                    slope_pct = (sele_end - sele_start) / sdist * 100
        else:
            pace      = None
            speed_kmh = None
            slope_pct = None

        if distance_source == "footpod" and pt.get("distance_m") is not None:
            dist_km = pt["distance_m"] / 1000.0
        else:
            dist_km = cum_dist_m / 1000.0
        # Average pace = elapsed time / total distance so far (min/km)
        avg_pace = (elapsed / 60.0 / dist_km) if dist_km > 0.05 else None
        # Average speed = total distance so far / elapsed time (km/h)
        avg_speed = ((dist_km / (elapsed / 3600.0))
                     if (dist_km > 0.05 and elapsed > 0) else None)

        result.append({
            "elapsed_s":      elapsed,
            "distance_km":    dist_km,
            "pace_min_km":    pace,
            "avg_pace_min_km": avg_pace,
            "speed_kmh":      speed_kmh,
            "avg_speed_kmh":  avg_speed,
            "hr":             pt["hr"],
            "slope_pct":      slope_pct,
            "ele_m":          pt["ele"],
            "lat":            pt["lat"],
            "lon":            pt["lon"],
        })

    coords = [(p["lat"], p["lon"]) for p in raw]
    return result, coords, has_ele


def interpolate(data, t_s):
    if t_s <= data[0]["elapsed_s"]:
        return data[0]
    if t_s >= data[-1]["elapsed_s"]:
        return data[-1]

    lo, hi = 0, len(data) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if data[mid]["elapsed_s"] <= t_s:
            lo = mid
        else:
            hi = mid

    a, b = data[lo], data[hi]
    frac = (t_s - a["elapsed_s"]) / (b["elapsed_s"] - a["elapsed_s"] + 1e-9)

    def lerp(va, vb):
        if va is None or vb is None:
            return va if vb is None else vb
        return va + frac * (vb - va)

    return {
        "elapsed_s":   t_s,
        "distance_km": lerp(a["distance_km"], b["distance_km"]),
        "pace_min_km":     lerp(a["pace_min_km"],     b["pace_min_km"]),
        "avg_pace_min_km": lerp(a.get("avg_pace_min_km"), b.get("avg_pace_min_km")),
        "speed_kmh":       lerp(a["speed_kmh"],       b["speed_kmh"]),
        "avg_speed_kmh":   lerp(a.get("avg_speed_kmh"), b.get("avg_speed_kmh")),
        "hr":   (round(lerp(a["hr"], b["hr"])) if (a["hr"] and b["hr"])
                 else (a["hr"] or b["hr"])),
        "slope_pct": lerp(a["slope_pct"], b["slope_pct"]),
        "ele_m": lerp(a.get("ele_m"), b.get("ele_m")),
        "lat":  lerp(a["lat"], b["lat"]),
        "lon":  lerp(a["lon"], b["lon"]),
    }


# ============================================================
# Mini-Map
# ============================================================

def _merc_x(lon): return (lon + 180) / 360

def _merc_y(lat):
    r = math.radians(max(-85.0, min(85.0, lat)))
    return (1 - math.log(math.tan(r) + 1 / math.cos(r)) / math.pi) / 2


class MiniMap:
    #COL_TRACK_DONE = (255, 255, 255, 255)   # done track – solid white
    COL_TRACK_DONE = (252,  82,   2, 255)   # done track - strava orange
    COL_OUTLINE    = (  0,   0,   0, 255)   # black outline for track lines
    COL_DOT        = (255,  70,  70, 255)
    COL_START      = ( 80, 220,  80, 255)

    # Track line widths
    TRACK_W      = 3   # dim (remaining) track
    DONE_W       = 4   # completed track

    def __init__(self, coords, size: int, bg_alpha: int, outline_width: int = 2,
                 base_color=(255, 255, 255, 255)):
        self.size          = size
        self.n             = len(coords)
        self.outline_width = outline_width
        self.col_track     = base_color
        self.col_ring      = base_color
        pad                = max(16, int(size * 0.08))
        inner              = size - 2 * pad

        xs = [_merc_x(lon) for _, lon in coords]
        ys = [_merc_y(lat) for lat, _ in coords]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        rx, ry     = xmax - xmin or 1e-9, ymax - ymin or 1e-9
        scale      = min(inner / rx, inner / ry)
        ox         = pad + (inner - rx * scale) / 2
        oy         = pad + (inner - ry * scale) / 2

        self.px = [ox + (x - xmin) * scale for x in xs]
        self.py = [oy + (y - ymin) * scale for y in ys]

        # Marker radii scale with the map size.
        self._start_r  = max(4, round(size * 0.016))
        self._finish_r = max(5, round(size * 0.020))
        self._dot_r    = max(8, round(size * 0.025))

        # Static background: panel + outlined dim full track
        self._bg = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        d = ImageDraw.Draw(self._bg)
        pts = list(zip(self.px, self.py))
        if len(pts) >= 2:
            ow = self.outline_width
            # Black outline pass (wider line drawn first)
            for i in range(len(pts) - 1):
                d.line([pts[i], pts[i+1]],
                       fill=self.COL_OUTLINE,
                       width=self.TRACK_W + ow * 2)
            # Coloured track on top
            for i in range(len(pts) - 1):
                d.line([pts[i], pts[i+1]],
                       fill=self.col_track,
                       width=self.TRACK_W)
        self._cache_cur = -1
        self._cache_img = None

    def _dot(self, draw, x, y, color, r=8):
        ring = max(2, r // 4)
        draw.ellipse([(x-r-ring, y-r-ring), (x+r+ring, y+r+ring)],
                     fill=self.col_ring)
        draw.ellipse([(x-r,   y-r),   (x+r,   y+r)],   fill=color)

    @staticmethod
    def _finish_marker(img, x, y, r):
        """Draw a black/white checkered finish disc with a white rim."""
        r  = int(max(4, r))
        n  = 2 * r + 1
        grid = 3
        tile = n / grid
        plate = Image.new("RGBA", (n, n), (255, 255, 255, 255))
        pd = ImageDraw.Draw(plate)
        for i in range(grid):
            for j in range(grid):
                if (i + j) % 2:
                    pd.rectangle([i * tile, j * tile,
                                  (i + 1) * tile - 1, (j + 1) * tile - 1],
                                 fill=(0, 0, 0, 255))

        mask = Image.new("L", (n, n), 0)
        ImageDraw.Draw(mask).ellipse([0, 0, n - 1, n - 1], fill=255)

        x0, y0 = int(round(x - r)), int(round(y - r))
        rim = max(2, r // 6)
        d = ImageDraw.Draw(img)
        # White rim behind the checker plate
        d.ellipse([x0 - rim, y0 - rim, x0 + n + rim - 1, y0 + n + rim - 1],
                  fill=(255, 255, 255, 255))
        img.paste(plate, (x0, y0), mask)
        d.ellipse([x0, y0, x0 + n - 1, y0 + n - 1],
                  outline=(0, 0, 0, 255), width=max(1, r // 5))

    def render(self, elapsed_frac: float) -> Image.Image:
        # The image only changes when the track index changes. At 30 fps a
        # 10 km run has thousands of frames per track point, so caching the
        # last render avoids redrawing the whole completed track each frame.
        cur = min(int(elapsed_frac * (self.n - 1)), self.n - 1)
        if cur == self._cache_cur:
            return self._cache_img

        img = self._bg.copy()
        d   = ImageDraw.Draw(img)
        ow  = self.outline_width

        if cur >= 1:
            done = list(zip(self.px[:cur+1], self.py[:cur+1]))
            # Black outline pass
            for i in range(len(done) - 1):
                d.line([done[i], done[i+1]],
                       fill=self.COL_OUTLINE,
                       width=self.DONE_W + ow * 2)
            # Coloured done track on top
            for i in range(len(done) - 1):
                d.line([done[i], done[i+1]],
                       fill=self.COL_TRACK_DONE,
                       width=self.DONE_W)

        # Markers sit above the track so the lines never hide them.
        self._dot(d, self.px[0], self.py[0], self.COL_START, r=self._start_r)
        self._dot(d, self.px[cur], self.py[cur], self.COL_DOT, r=self._dot_r)
        # Finish stays on top so it remains visible when the dot arrives.
        self._finish_marker(img, self.px[-1], self.py[-1], self._finish_r)
        self._cache_cur = cur
        self._cache_img = img
        return img




# ============================================================
# Elevation Graph
# ============================================================

class ElevationGraph:
    """
    A widescreen panel showing the full elevation profile as a filled
    area chart with a vertical playhead that tracks current position.

    The profile is colour-coded by slope severity (same palette as the
    slope text stat) so steep sections stand out at a glance.

    Layout (inside rounded panel):
        ┌──────────────────────────────────┐
        │  label                           │
        │  ╱╲___╱╲__________________________  ← area chart
        │                          playhead│
        └──────────────────────────────────┘
    """

    # Slope-colour breakpoints (mirrors slope_color())
    _SLOPE_COLORS = [
        ( 2, (100, 220, 100, 255)),   # flat    – green
        ( 5, (255, 210,   0, 255)),   # gentle  – yellow
        (10, (255, 140,   0, 255)),   # steep   – orange
        (99, (255,  60,  60, 255)),   # severe  – red
    ]
    COL_FILL_DEFAULT = (100, 160, 220, 255)   # blue when no slope data
    COL_LINE         = (255, 255, 255, 255)   # profile line on top of fill
    COL_TRACK_DONE   = (252,  82,   2, 255)   # done track - strava orange
    COL_GRID         = (  0,   0,   0, 255)
    COL_PLAYHEAD     = COL_TRACK_DONE 

    def __init__(self, data: list, width: int, height: int,
                 bg_alpha: int, outline_width: int, font_small,
                 base_color=COL_LINE):
        self.width         = width
        self.height        = height
        self.outline_width = outline_width
        self.col_line      = base_color
        self.col_label     = base_color
        self.col_ring      = base_color

        pad_x = max(14, int(width  * 0.04))
        pad_y = max(12, int(height * 0.10))
        self._pad_x = pad_x
        self._pad_y = pad_y
        self._inner_w = width  - 2 * pad_x
        self._inner_h = height - 2 * pad_y

        # ── Collect elevation + slope samples ──────────────────
        # Use every data point; slope_pct may be None for the first few.
        eles   = [d["distance_km"] for d in data]   # x axis = distance
        dists  = [d["distance_km"] for d in data]
        slopes = [d["slope_pct"]   for d in data]
        elevs  = []
        for d in data:
            # elevation isn't stored in the interpolated data dict —
            # back-calculate from the raw GPX via slope_pct is impractical,
            # so we carry it through data as "ele_m" if present.
            elevs.append(d.get("ele_m"))

        self._n      = len(data)
        self._dists  = dists
        self._elevs  = elevs
        self._slopes = slopes
        self._has_ele = any(e is not None for e in elevs)

        if self._has_ele:
            valid = [e for e in elevs if e is not None]
            self._ele_min = min(valid)
            self._ele_max = max(valid)
            ele_range = self._ele_max - self._ele_min or 1.0
            # 5 % headroom top and bottom
            self._ele_lo = self._ele_min - ele_range * 0.05
            self._ele_hi = self._ele_max + ele_range * 0.15   # extra room for label

        dist_max = dists[-1] if dists else 1.0
        self._dist_max = dist_max or 1.0

        # ── Pre-render static background ───────────────────────
        self._bg = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        d = ImageDraw.Draw(self._bg)

        if self._has_ele:
            self._profile_pts = self._compute_pts()
            self._draw_static(d, font_small)
        self._cache_cur = -1
        self._cache_img = None

    # ── helpers ──────────────────────────────────────────────────

    def _ele_to_y(self, ele):
        """Map elevation value → pixel y inside the panel."""
        frac = (ele - self._ele_lo) / (self._ele_hi - self._ele_lo)
        return self._pad_y + self._inner_h * (1.0 - frac)

    def _dist_to_x(self, dist):
        """Map distance value → pixel x inside the panel."""
        return self._pad_x + self._inner_w * (dist / self._dist_max)

    def _compute_pts(self):
        """Return list of (px, py) profile points, skipping None elevs."""
        pts = []
        for i in range(self._n):
            if self._elevs[i] is None:
                continue
            pts.append((self._dist_to_x(self._dists[i]),
                        self._ele_to_y(self._elevs[i])))
        return pts

    @staticmethod
    def _slope_col(slope):
        if slope is None:
            return ElevationGraph.COL_FILL_DEFAULT
        v = abs(slope)
        for thresh, col in ElevationGraph._SLOPE_COLORS:
            if v < thresh:
                return col
        return ElevationGraph._SLOPE_COLORS[-1][1]

    def _draw_static(self, d, font_small):
        """Draw the full profile, gridlines and label onto the background."""
        px, py = self._pad_x, self._pad_y
        iw, ih = self._inner_w, self._inner_h
        pts    = self._profile_pts
        n      = len(pts)

        # Tick positions are shared by the grid lines and the labels.
        ele_range = self._ele_hi - self._ele_lo
        step_raw  = ele_range / 4
        magnitude = 10 ** math.floor(math.log10(max(step_raw, 1)))
        step      = round(step_raw / magnitude) * magnitude or magnitude
        tick      = math.ceil(self._ele_lo / step) * step
        ticks     = []
        while tick <= self._ele_hi:
            gy = self._ele_to_y(tick)
            if py <= gy <= py + ih:
                ticks.append((gy, tick))
            tick += step

        # ── Grid lines stay behind the filled area ───────────────
        for gy, _ in ticks:
            d.line([(px, gy), (px + iw, gy)], fill=self.COL_GRID, width=2)

        # ── Filled area, coloured per slope segment ──────────────
        # This never changes during playback, so bake it once.
        if n >= 2:
            bottom = self._pad_y + self._inner_h
            for i in range(n - 1):
                x0, y0 = pts[i]
                x1, y1 = pts[i + 1]

                data_i = min(int(i / (n - 1) * (self._n - 1)), self._n - 1)
                col    = self._slope_col(self._slopes[data_i])

                poly = [(x0, y0), (x1, y1), (x1, bottom), (x0, bottom)]
                d.polygon(poly, fill=col)

        # ── Elevation markers stay readable above the fill ───────
        if font_small:
            text_h = getattr(font_small, "size", 12)
            for gy, marker in ticks:
                draw_outlined_text(
                    d, (px + 3, gy - int(text_h * 0.55)),
                    f"{marker:.0f}m", font_small, self.col_label,
                    OUTLINE_COLOR, self.outline_width
                )

        # Full profile line (white); the completed part is overpainted orange.
        if n >= 2:
            ow = self.outline_width
            if ow > 0:
                d.line(pts, fill=(0, 0, 0, 255), width=ow * 2 + 1)
            d.line(pts, fill=self.col_line, width=2)

    # ── public render ─────────────────────────────────────────────

    def render(self, elapsed_frac: float) -> Image.Image:
        if not self._has_ele or len(self._profile_pts) < 2:
            return self._bg

        pts = self._profile_pts
        cur = min(int(elapsed_frac * (len(pts) - 1)), len(pts) - 1)

        # Only the playhead position and the orange done-track change.
        # Cache the last frame to avoid redrawing them for every video frame.
        if cur == self._cache_cur:
            return self._cache_img

        img = self._bg.copy()
        d   = ImageDraw.Draw(img)

        # ── Completed part of the profile line in orange ──────────
        if cur >= 1:
            d.line(pts[:cur + 1], fill=self.COL_TRACK_DONE, width=2)

        # ── Playhead ──────────────────────────────────────────────
        px_head = pts[cur][0]
        d.line([(px_head, self._pad_y),
                (px_head, self._pad_y + self._inner_h)],
               fill=self.COL_PLAYHEAD, width=2)
        # Small dot on the profile line
        py_head = pts[cur][1]
        r = 5
        d.ellipse([(px_head-r-1, py_head-r-1),
                    (px_head+r+1, py_head+r+1)],
                   fill=self.col_ring)
        d.ellipse([(px_head-r, py_head-r),
                    (px_head+r, py_head+r)],
                   fill=self.COL_PLAYHEAD)

        self._cache_cur = cur
        self._cache_img = img
        return img

# ============================================================
# Text rendering helpers
# ============================================================

PADDING      = 20
LINE_SPACING = 1.25
AVG_PACE_TRANSITION_S = 0.8
ACCENT_COLOR = (255, 210,   0, 255)
LABEL_COLOR  = (200, 200, 200, 255)
OUTLINE_COLOR = (0, 0, 0, 255)


def load_font(size: int):
    """Load a bold TrueType face. Every overlay text uses bold."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    # Last resort: ask fontconfig for any installed bold sans-serif face.
    try:
        import subprocess
        fc = subprocess.run(
            ["fc-match", "-f", "%{file}", "sans-serif:bold"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if fc and Path(fc).exists():
            return ImageFont.truetype(fc, size)
    except Exception:
        pass
    return ImageFont.load_default()


def draw_outlined_text(draw, xy, text, font, fill, outline_fill, outline_width):
    """Draw text with a solid outline around the glyphs."""
    ow = outline_width
    if ow <= 0:
        draw.text(xy, text, font=font, fill=fill)
        return
    try:
        # One stroke pass is much faster than stamping shifted copies.
        draw.text(xy, text, font=font, fill=fill,
                  stroke_width=ow, stroke_fill=outline_fill)
        return
    except TypeError:
        pass  # Pillow too old for stroke_width - use the shifted-copy fallback
    x, y = xy
    for dx in range(-ow, ow + 1):
        for dy in range(-ow, ow + 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, font=font, fill=outline_fill)
    draw.text(xy, text, font=font, fill=fill)


class TextCache:
    """Cache outlined text sprites. Repeated strings become a cheap paste.

    Stats refresh 30 times per second, but most values repeat across
    several frames. Rendering each string once and compositing the sprite
    avoids the FreeType stroke pass on every frame.
    """

    def __init__(self, max_entries: int = 128):
        self._cache = OrderedDict()
        self._max = max_entries

    def draw(self, img, xy, text, font, fill, outline_fill, outline_width,
             opacity: float = 1.0):
        key = (text, id(font), fill, outline_fill, outline_width)
        entry = self._cache.get(key)
        if entry is not None:
            self._cache.move_to_end(key)
        else:
            try:
                x0, y0, x1, y1 = font.getbbox(text, stroke_width=outline_width)
            except (AttributeError, TypeError):
                draw_outlined_text(ImageDraw.Draw(img), xy, text, font,
                                   fill, outline_fill, outline_width)
                return
            w, h = x1 - x0, y1 - y0
            if w <= 0 or h <= 0:
                return
            sprite = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            ImageDraw.Draw(sprite).text(
                (-x0, -y0), text, font=font, fill=fill,
                stroke_width=outline_width, stroke_fill=outline_fill)
            entry = (sprite, x0, y0)
            self._cache[key] = entry
            if len(self._cache) > self._max:
                self._cache.popitem(last=False)

        sprite, x0, y0 = entry
        if opacity < 1.0:
            sprite = sprite.copy()
            sprite.putalpha(sprite.getchannel("A").point(
                lambda a: int(a * opacity + 0.5)))
        img.alpha_composite(sprite, (int(xy[0] + x0), int(xy[1] + y0)))


def fmt_pace(pace):
    if pace is None or pace > 30:
        return "--:--"
    m = int(pace)
    s = int((pace - m) * 60)
    return f"{m}:{s:02d}"


def fmt_speed(speed_kmh):
    if speed_kmh is None:
        return "--"
    return f"{speed_kmh:.1f} km/h"


def fmt_time(seconds):
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def fmt_slope(slope_pct):
    if slope_pct is None:
        return "--%"
    sign = "+" if slope_pct > 0 else ""
    return f"{sign}{slope_pct:.1f}%"


def slope_color(slope_pct):
    """Return an RGBA colour that reflects slope severity."""
    if slope_pct is None:
        return ACCENT_COLOR
    v = abs(slope_pct)
    if v < 2:
        return (100, 220, 100, 255)   # flat – green
    if v < 5:
        return (255, 210,   0, 255)   # moderate – yellow
    if v < 8:
        return (255, 140,   0, 255)   # steep – orange
    return     (255,  60,  60, 255)   # very steep – red


def hr_color(hr, min_hr, max_hr):
    """Return an RGBA colour from white to red as HR moves from min to max."""
    if hr is None or min_hr is None or max_hr is None:
        return ACCENT_COLOR
    if max_hr <= min_hr:
        return ACCENT_COLOR
    if hr <= min_hr:
        return (255, 255, 255, 255)
    ratio = max(0.0, min(1.0, (hr - min_hr) / (max_hr - min_hr)))
    r = 255
    g = int(255 * (1.0 - ratio))
    b = int(255 * (1.0 - ratio))
    return (r, g, b, 255)


def corner_xy(pos, W, H, w, h, margin=40):
    if pos == "top-left":    return margin,         margin
    if pos == "top-right":   return W - w - margin, margin
    if pos == "bottom-left": return margin,         H - h - margin
    return                          W - w - margin, H - h - margin


# ============================================================
# Frame Renderer
# ============================================================

def measure_stat_block(lines, draw, font_big, font_small):
    """Return the pixel width of a stat block including its padding."""
    max_w = 0
    for entry in lines:
        lbl, val = entry[0], entry[1]
        max_w = max(max_w,
                    draw.textlength(lbl, font=font_small),
                    draw.textlength(val, font=font_big))
    return int(max_w) + PADDING * 2


def draw_stat_block(img, draw, lines, x, y,
                    font_big, font_small, outline_width,
                    text_color, text_cache, opacity: float = 1.0):
    """Draw one stat block with its labels and values."""
    lh_big   = font_big.size   if hasattr(font_big,   "size") else 48
    lh_small = font_small.size if hasattr(font_small, "size") else 24
    # Labels use lower contrast than the values: darker on a light base,
    # lighter on a dark base.
    label_color = tuple(
        min(255, c + 60) if c < 128 else max(0, c - 60)
        for c in text_color[:3]
    ) + (255,)
    cy = y + PADDING

    for entry in lines:
        lbl, val = entry[0], entry[1]
        value_color = entry[2] if len(entry) > 2 else text_color
        if text_cache is not None:
            text_cache.draw(img, (x + PADDING, cy),
                            lbl, font_small, label_color, OUTLINE_COLOR, outline_width,
                            opacity=opacity)
        else:
            draw_outlined_text(draw, (x + PADDING, cy),
                               lbl, font_small, label_color, OUTLINE_COLOR, outline_width)
        cy += lh_small + 4
        if text_cache is not None:
            text_cache.draw(img, (x + PADDING, cy),
                            val, font_big, value_color, OUTLINE_COLOR, outline_width,
                            opacity=opacity)
        else:
            draw_outlined_text(draw, (x + PADDING, cy),
                               val, font_big, value_color, OUTLINE_COLOR, outline_width)
        cy += int(lh_big * LINE_SPACING) + 4


# Extrema of the AVG PACE slide: (time fraction, position multiplier).
# First overshoot 8 %, then a smaller second wave, with damping between.
_SPRING_KEYS = [
    (0.00, 0.000),
    (0.28, 1.080),
    (0.50, 0.956),
    (0.70, 1.030),
    (0.86, 0.990),
    (1.00, 1.000),
]
_SPRING_START_SLOPE = 3.8


def ease_out_spring(progress: float) -> float:
    """Two damped overshoot waves: +8 %, then a weaker second wave."""
    p = max(0.0, min(1.0, progress))

    for i in range(len(_SPRING_KEYS) - 1):
        p0, x0 = _SPRING_KEYS[i]
        p1, x1 = _SPRING_KEYS[i + 1]
        if p <= p1:
            t  = (p - p0) / (p1 - p0)
            m0 = _SPRING_START_SLOPE if i == 0 else 0.0
            # Cubic Hermite with zero slope at both extrema.
            h00 = 2*t**3 - 3*t**2 + 1
            h10 = t**3 - 2*t**2 + t
            h01 = -2*t**3 + 3*t**2
            return x0*h00 + (p1 - p0)*m0*h10 + x1*h01
    return 1.0


def render_frame(state, W, H,
                 font_big, font_small,
                 stats_pos, bg_alpha, outline_width,
                 has_ele, show_slope, show_avg_pace,
                 minimap, map_pos,
                 ele_graph, graph_pos,
                 elapsed_frac,
                 min_hr,
                 max_hr,
                 text_color=(255, 255, 255, 255),
                 background="transparent",
                 text_cache=None,
                 show_hr_color=False,
                 avg_pace_progress=1.0,
                 avg_pace_transition="slide",
                 show_speed=False):
    """
    background: "transparent" | "greenscreen" | "black" | "white" | "blue"
      transparent  – fully clear canvas; caller gets BGRA with alpha
      greenscreen  – pure (0,255,0) canvas; caller gets BGR
      black/white  – solid fill; caller gets BGR
      blue         – solid blue fill; caller gets BGR
    """
    canvas_color = {
        "transparent": (  0,   0,   0,   0),
        "greenscreen": (  0, 255,   0, 255),
        "black":       (  0,   0,   0, 255),
        "blue":        (  0,   0, 255, 255),
        "white":       (255, 255, 255, 255),
    }.get(background, (0, 0, 0, 0))

    img  = Image.new("RGBA", (W, H), canvas_color)
    draw = ImageDraw.Draw(img)

    # ── Build stats lines ──────────────────────────────────────
    # Each entry: (label, value_string, value_color?)
    # The pace/speed stat moves to its own block that sits between this
    # block and the elevation graph.
    lines = [
        ("TIME", fmt_time(state["elapsed_s"])),
        ("DIST", f"{state['distance_km']:.2f} km"),
    ]
    if has_ele and show_slope:
        sc = slope_color(state["slope_pct"])
        lines.append(("SLOPE", fmt_slope(state["slope_pct"]), sc))
    if state["hr"] is not None:
        hc = (hr_color(state['hr'], min_hr, max_hr)
              if show_hr_color else text_color)
        lines.append(("HR", f"{state['hr']} bpm", hc))

    if show_speed:
        stat_label = "SPEED"
        avg_label  = "AVG SPEED"
        stat_value = fmt_speed(state.get("speed_kmh"))
        avg_value  = fmt_speed(state.get("avg_speed_kmh"))
        avg_ready  = (show_avg_pace
                      and state.get("avg_speed_kmh") is not None
                      and avg_pace_progress > 0)
    else:
        stat_label = "PACE"
        avg_label  = "AVG PACE"
        stat_value = f"{fmt_pace(state['pace_min_km'])} /km"
        avg_value  = f"{fmt_pace(state.get('avg_pace_min_km'))} /km"
        avg_ready  = (show_avg_pace
                      and state.get("avg_pace_min_km") is not None
                      and avg_pace_progress > 0)

    lh_big   = font_big.size   if hasattr(font_big,   "size") else 48
    lh_small = font_small.size if hasattr(font_small, "size") else 24
    row_h    = int(lh_big * LINE_SPACING) + lh_small + 8

    # ── Upper block: TIME, DIST, SLOPE, HR ─────────────────────
    block_w = measure_stat_block(lines, draw, font_big, font_small)
    block_h = row_h * len(lines) + PADDING * 2
    sx, sy  = corner_xy(stats_pos, W, H, block_w, block_h)
    draw_stat_block(img, draw, lines, sx, sy,
                    font_big, font_small, outline_width,
                    text_color, text_cache)

    # ── Pace / speed block ─────────────────────────────────────
    # Reserve the second row while the average stat is enabled, so the
    # current value keeps its position when the average appears below it.
    stat_rows = 2 if show_avg_pace else 1
    measure_lines = [(stat_label, stat_value)]
    if show_avg_pace:
        measure_lines.append((avg_label, avg_value))
    stat_w = measure_stat_block(measure_lines, draw, font_big, font_small)
    stat_h = row_h * stat_rows + PADDING * 2

    px = sx + block_w - stat_w if stats_pos.endswith("right") else sx

    graph_top = None
    if ele_graph is not None:
        _, graph_top = corner_xy(graph_pos, W, H,
                                 ele_graph.width, ele_graph.height)

    if stats_pos.startswith("top"):
        upper_bottom = sy + block_h
        lower_bound  = graph_top if graph_top is not None else H - 40
        if lower_bound > upper_bottom + stat_h + 24:
            py = upper_bottom + (lower_bound - upper_bottom - stat_h) / 2
        else:
            py = upper_bottom + 12
    else:
        # Stats at the bottom: put the pace/speed block above it.
        py = max(40, sy - stat_h - 12)

    py = int(py)
    draw_stat_block(img, draw, [(stat_label, stat_value)], px, py,
                    font_big, font_small, outline_width,
                    text_color, text_cache)

    if avg_ready:
        if avg_pace_transition == "slide":
            # Run in from the screen side with a slight overshoot.
            ease = ease_out_spring(avg_pace_progress)
            start_x = W if stats_pos.endswith("right") else -stat_w
            avg_x = start_x + (px - start_x) * ease
            avg_opacity = 1.0
        else:  # fade
            avg_x = px
            avg_opacity = min(max(avg_pace_progress, 0.0), 1.0)

        draw_stat_block(img, draw, [(avg_label, avg_value)], int(avg_x), py + row_h,
                        font_big, font_small, outline_width,
                        text_color, text_cache,
                        opacity=avg_opacity)

    # ── Mini-map ───────────────────────────────────────────────
    if minimap is not None:
        map_img = minimap.render(elapsed_frac)
        ms      = minimap.size
        mx, my  = corner_xy(map_pos, W, H, ms, ms)
        img.alpha_composite(map_img, (mx, my))

    # ── Elevation graph ────────────────────────────────────────
    if ele_graph is not None:
        graph_img = ele_graph.render(elapsed_frac)
        gx, gy    = corner_xy(graph_pos, W, H,
                               ele_graph.width, ele_graph.height)
        img.alpha_composite(graph_img, (gx, gy))

    # Convert to OpenCV format (single C++ pass instead of split/merge)
    arr = np.asarray(img)
    if background == "transparent":
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA), True
    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR), False


# ============================================================
# Video Writer
# ============================================================


def _ffmpeg_has_encoder(ffmpeg: str, name: str) -> bool:
    try:
        res = subprocess.run([ffmpeg, "-hide_banner", "-encoders"],
                             capture_output=True, text=True)
        out = (res.stdout or "") + (res.stderr or "")
        return name in out
    except OSError:
        return False


def _ffmpeg_error(err_file) -> str:
    err_file.seek(0)
    return err_file.read().decode(errors="replace").strip()[-400:]


def write_video(data, coords, output_path,
                W, H, fps, total_duration,
                font_size, stats_pos, bg_alpha, outline_width, speed,
                has_ele, show_slope, show_avg_pace, show_map, map_pos, map_size,
                show_graph, graph_pos, graph_w, graph_h,
                min_hr, max_hr,
                text_color=(255, 255, 255, 255),
                background="transparent",
                show_hr_color=False,
                avg_pace_transition="slide",
                show_speed=False):

    import shutil, tempfile

    total_frames = int(total_duration * fps)
    gpx_dur      = data[-1]["elapsed_s"]
    use_alpha    = (background == "transparent")

    font_big   = load_font(max(16, round(font_size * 1.12)))
    font_small = load_font(max(12, round(font_size * 0.30)))
    graph_font = load_font(max(11, round(font_size * 0.28)))
    text_cache = TextCache()
    base_color = (0, 0, 0, 255) if background == "white" else (255, 255, 255, 255)
    minimap    = (MiniMap(coords, map_size, bg_alpha, outline_width, base_color=base_color)
                  if show_map else None)
    ele_graph  = (ElevationGraph(data, graph_w, graph_h, bg_alpha, outline_width,
                                 graph_font, base_color=base_color)
                  if show_graph else None)

    avg_start = None
    if show_avg_pace:
        avg_key = "avg_speed_kmh" if show_speed else "avg_pace_min_km"
        avg_start = next((d["elapsed_s"] for d in data
                          if d.get(avg_key) is not None), None)

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        need = "libx265 and prores_ks" if use_alpha else "libx265"
        sys.exit(f"❌  FFmpeg not found. Install FFmpeg with {need}.")

    encoder = "prores_ks" if use_alpha else "libx265"
    if not _ffmpeg_has_encoder(ffmpeg, encoder):
        sys.exit(f"❌  FFmpeg has no {encoder} encoder. "
                 f"Install an FFmpeg build with {encoder}.")

    if use_alpha:
        mov_path = output_path.replace(".mp4", "_transparent.mov")
        pipe_cmd = [
            ffmpeg, "-y", "-nostdin", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgra",
            "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
            "-c:v", "prores_ks", "-profile:v", "4444",
            "-pix_fmt", "yuva444p10le",
            mov_path,
        ]
    else:
        # Keep full chroma fidelity for green-screen keying
        pix = "yuv444p" if background == "greenscreen" else "yuv420p"
        pipe_cmd = [
            ffmpeg, "-y", "-nostdin", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
            "-c:v", "libx265", "-crf", "18", "-preset", "fast",
            "-pix_fmt", pix,
            output_path,
        ]

    print(f"   Encoder   : {encoder} via raw-frame pipe")
    print(f"🎬  Rendering {total_frames} frames  ({total_duration:.1f}s @ {fps} fps)…")
    print(f"   Background: {background}")

    err_file = tempfile.TemporaryFile()
    proc = subprocess.Popen(pipe_cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=err_file)

    for fi in range(total_frames):
        t_s          = (fi / fps) * speed
        elapsed_frac = min(t_s / gpx_dur, 1.0)
        state        = interpolate(data, t_s)

        if avg_start is not None:
            avg_progress = min(max((t_s - avg_start) / AVG_PACE_TRANSITION_S, 0.0), 1.0)
        else:
            avg_progress = 1.0

        frame, _ = render_frame(
            state, W, H,
            font_big, font_small,
            stats_pos, bg_alpha, outline_width,
            has_ele, show_slope, show_avg_pace,
            minimap, map_pos,
            ele_graph, graph_pos,
            elapsed_frac,
            min_hr,
            max_hr,
            text_color=text_color,
            background=background,
            text_cache=text_cache,
            show_hr_color=show_hr_color,
            avg_pace_progress=avg_progress,
            avg_pace_transition=avg_pace_transition,
            show_speed=show_speed,
        )

        try:
            proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, ValueError):
            rc  = proc.wait()
            err = _ffmpeg_error(err_file)
            err_file.close()
            sys.exit(f"❌  FFmpeg encoder stopped (exit {rc}):\n{err}")

        if fi % fps == 0:
            print(f"   {fi/total_frames*100:5.1f}%  frame {fi}/{total_frames}", end="\r", flush=True)

    try:
        proc.stdin.close()
    except BrokenPipeError:
        pass
    rc  = proc.wait()
    err = _ffmpeg_error(err_file)
    err_file.close()

    if use_alpha:
        if rc == 0:
            print(f"\n✅  ProRes 4444 (alpha) → {mov_path}")
            # H.265 with alpha needs libx265 + yuva420p10le support.
            cmd_hevc = [
                ffmpeg, "-y", "-loglevel", "error", "-i", mov_path,
                "-c:v", "libx265", "-crf", "18", "-preset", "fast",
                "-pix_fmt", "yuva420p10le",
                "-tag:v", "hvc1",
                output_path,
            ]
            res = subprocess.run(cmd_hevc, capture_output=True, text=True)
            if res.returncode == 0:
                print(f"✅  H.265 with alpha    → {output_path}")
            else:
                Path(output_path).unlink(missing_ok=True)
                print("⚠️  H.265 alpha encode failed (needs FFmpeg with libx265 + yuva420p10le).")
                print(f"   Use the .mov file for full alpha support.")
        else:
            print(f"⚠️  ProRes encode failed:\n{err}")
            sys.exit(1)
    elif rc == 0:
        print(f"\n✅  H.265 overlay saved → {output_path}")
    else:
        print(f"⚠️  FFmpeg encode failed:\n{err}")
        sys.exit(1)


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="GPX stats + slope + mini-map overlay video (H.265 MP4).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("input",           metavar="FILE",
                    help="GPX or FIT file (extracts stats + route)")
    ap.add_argument("--output",        default=None,
                    help="output MP4 path (default: input file name with .mp4 ending)")
    ap.add_argument("--distance-source", default="gps",
                    choices=["gps", "footpod"],
                    help="Use GPS-derived or FIT footpod-derived distance (default: gps)")
    ap.add_argument("--pace-source",   default="gps",
                    choices=["gps", "footpod"],
                    help="Use GPS-derived or FIT footpod-derived pace/speed (default: gps)")
    ap.add_argument("--stat-mode",     default="pace",
                    choices=["pace", "speed"],
                    help="Show pace (min/km) or speed (km/h) in the overlay (default: pace)")
    ap.add_argument("--width",         type=int,   default=1920)
    ap.add_argument("--height",        type=int,   default=1080)
    ap.add_argument("--fps",           type=int,   default=30)
    ap.add_argument("--duration",      type=float, default=None)
    ap.add_argument("--bg-alpha",      type=int,   default=160,
                    help="Background panel opacity 0-255 (default 160)")
    ap.add_argument("--position",      default="top-left",
                    choices=["top-left","top-right","bottom-left","bottom-right"])
    ap.add_argument("--map-position",  default="bottom-right",
                    choices=["top-left","top-right","bottom-left","bottom-right"])
    ap.add_argument("--map-size",      type=int,   default=320)
    ap.add_argument("--no-map",        action="store_true")
    ap.add_argument("--no-avg-pace",   action="store_true",
                    help="Disable the average pace stat in the overlay")
    ap.add_argument("--avg-pace-transition", default="slide",
                    choices=["slide", "fade"],
                    help="AVG PACE / AVG SPEED entrance effect (default: slide)")
    ap.add_argument("--no-slope",      action="store_true",
                    help="Disable the slope / gradient display even if elevation data is present")
    ap.add_argument("--ele-graph",     action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Show elevation profile graph panel (default: enabled; use --no-ele-graph to hide)")
    ap.add_argument("--graph-position", default="bottom-left",
                    choices=["top-left","top-right","bottom-left","bottom-right"],
                    help="Corner for the elevation graph (default: bottom-left)")
    ap.add_argument("--graph-width",   type=int, default=480,
                    help="Elevation graph panel width in pixels (default: 480)")
    ap.add_argument("--graph-height",  type=int, default=160,
                    help="Elevation graph panel height in pixels (default: 160)")
    ap.add_argument("--text-color",    default=None,
                    help="Stat text colour as R,G,B (default: white; black on a white background)")
    ap.add_argument("--font-size",     type=int,   default=52)
    ap.add_argument("--outline-width", type=int,   default=2,
                    help="Black text outline thickness in pixels (default 2)")
    ap.add_argument("--background", "--back", "--bg", default="transparent",
                    choices=["transparent", "greenscreen", "black", "white", "blue"],
                    help="Canvas background (default: transparent). "
                         "'transparent' → HEVC+alpha .mp4 and ProRes 4444 .mov; "
                         "'greenscreen' → pure green (0,255,0) H.265 MP4 for chroma keying; "
                         "'blue' → solid blue background (useful for YouTube green-screen keying); "
                         "'black' / 'white' → opaque H.265 MP4.")
    ap.add_argument("--min-hr",        type=int,   default=100,
                    help="Min heart rate for HR colour scaling (default: 100)")
    ap.add_argument("--max-hr",        type=int,   default=191,
                    help="Max heart rate for HR colour scaling (default: 191)")
    ap.add_argument("--hr-color",      action=argparse.BooleanOptionalAction,
                    default=False,
                    help="Colour the HR value from white to red by intensity (default: disabled)")
    ap.add_argument("--speed",         type=float, default=1.0)
    args = ap.parse_args()

    output_path = args.output or str(Path(args.input).with_suffix(".mp4"))

    print(f"📂  Parsing GPX: {args.input}")
    data, coords, has_ele = parse_gpx(args.input, distance_source=args.distance_source, pace_source=args.pace_source)
    gpx_dur        = data[-1]["elapsed_s"]
    total_duration = args.duration if args.duration else gpx_dur / args.speed
    has_hr         = any(d["hr"] for d in data)

    print(f"   Duration  : {gpx_dur:.0f}s  ({gpx_dur/60:.1f} min)")
    print(f"   Distance  : {data[-1]['distance_km']:.2f} km")
    print(f"   Distance source: {args.distance_source}")
    print(f"   Pace source   : {args.pace_source}")
    print(f"   Stat mode     : {args.stat_mode} ({'km/h' if args.stat_mode == 'speed' else 'min/km'})")
    print(f"   Track pts : {len(coords)}")
    print(f"   Heart rate: {'yes' if has_hr else 'not found in GPX'}")
    print(f"   Min HR    : {args.min_hr}")
    print(f"   Max HR    : {args.max_hr}")
    print(f"   HR colour : {'enabled' if args.hr_color else 'disabled'}")
    avg_name = "Avg speed" if args.stat_mode == "speed" else "Avg pace "
    print(f"   {avg_name} : {'disabled' if args.no_avg_pace else f'enabled ({args.avg_pace_transition})'}")
    print(f"   Elevation : {'yes – slope enabled' if (has_ele and not args.no_slope) else ('disabled via --no-slope' if args.no_slope else 'not found in GPX')}")
    print(f"   Mini-map  : {'disabled' if args.no_map else f'{args.map_size}×{args.map_size}px @ {args.map_position}'}")
    show_graph = args.ele_graph and has_ele
    if args.ele_graph and not has_ele:
        print("⚠️   --ele-graph requested but GPX has no elevation data — skipping graph.")
    print(f"   Ele-graph : {'enabled' if show_graph else 'disabled'}" +
          (f"  {args.graph_width}×{args.graph_height}px @ {args.graph_position}" if show_graph else ""))
    print(f"   Outline   : {args.outline_width}px")
    print(f"   Background: {args.background}")
    print(f"   Video     : {args.width}×{args.height} @ {args.fps} fps  speed={args.speed}×")

    color_arg = args.text_color
    if color_arg is None:
        color_arg = "0,0,0" if args.background == "white" else "255,255,255"

    try:
        r, g, b = [int(v.strip()) for v in color_arg.split(",")]
        text_color = (r, g, b, 255)
    except Exception:
        sys.exit("❌  --text-color must be R,G,B integers e.g. 255,255,255")

    write_video(
        data=data, coords=coords,
        output_path=output_path,
        W=args.width, H=args.height,
        fps=args.fps, total_duration=total_duration,
        font_size=args.font_size,
        stats_pos=args.position,
        bg_alpha=args.bg_alpha,
        outline_width=args.outline_width,
        speed=args.speed,
        has_ele=has_ele,
        show_slope=not args.no_slope,
        show_avg_pace=not args.no_avg_pace,
        show_map=not args.no_map,
        map_pos=args.map_position,
        map_size=args.map_size,
        show_graph=show_graph,
        graph_pos=args.graph_position,
        graph_w=args.graph_width,
        graph_h=args.graph_height,
        min_hr=args.min_hr,
        max_hr=args.max_hr,
        text_color=text_color,
        background=args.background,
        show_hr_color=args.hr_color,
        avg_pace_transition=args.avg_pace_transition,
        show_speed=(args.stat_mode == "speed"),
    )


if __name__ == "__main__":
    main()
