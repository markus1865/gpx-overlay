# GPX Overlay

GPX Overlay makes a video overlay from a GPS activity file. The overlay shows
live activity statistics, a mini-map, and an elevation graph. Use it as a
transparent layer over cycling or running footage.

The script reads GPX and FIT files. It writes an H.265/HEVC MP4. In transparent
mode it also writes a ProRes 4444 MOV with an alpha channel.

## Features

1. Elapsed time.
2. Distance in kilometres.
3. Pace in minutes per kilometre, or speed in kilometres per hour.
4. Heart rate, if the file contains it.
5. Slope or gradient in percent, if the file contains elevation data.
6. Average pace or average speed, with a slide or fade entrance.
7. A mini-map of the full track. It shows the start point, a checkered finish
   marker, and a moving position dot.
8. An elevation profile graph with the current position.
9. Optional heart-rate colour scaling from white to red.
10. A solid text outline for readability on any background.
11. A transparent, green-screen, blue, black, or white canvas.

## Requirements

1. Python 3.9 or later.
2. The Python packages `gpxpy`, `pillow`, `numpy`, and
   `opencv-python-headless`. Install `fitparse` to read FIT files.
3. FFmpeg with `libx265`. Transparent mode also needs `prores_ks`.
4. A bold TrueType font. The script tries DejaVu Sans Bold, Liberation Sans
   Bold, FreeSans Bold, Noto Sans Bold, and Arial Bold. If it finds no font, it
   asks fontconfig for a bold sans-serif face.

## Install

```bash
pip install gpxpy pillow numpy opencv-python-headless
pip install fitparse
```

Check that FFmpeg has the encoders:

```bash
ffmpeg -hide_banner -encoders | grep -E 'libx265|prores_ks'
```

## Usage

```bash
python gpx-overlay.py activity.gpx
python gpx-overlay.py activity.fit --stat-mode speed
python gpx-overlay.py ride.gpx --background greenscreen --output ride_overlay.mp4
```

The default output name is the input name with an `.mp4` ending. For example,
`ride.gpx` becomes `ride.mp4`.

The default background is `transparent`. In this mode the script writes two
files:

1. `activity.mp4` with H.265 and an alpha channel, if FFmpeg supports it.
2. `activity_transparent.mov` with ProRes 4444 and an alpha channel.

## Options

| Option | Default | Description |
| --- | --- | --- |
| `FILE` | required | GPX or FIT input file. |
| `--output PATH` | input name with `.mp4` | Output path. |
| `--distance-source {gps,footpod}` | `gps` | Distance source for FIT files. |
| `--pace-source {gps,footpod}` | `gps` | Pace or speed source for FIT files. |
| `--stat-mode {pace,speed}` | `pace` | Show pace (min/km) or speed (km/h). |
| `--width PX` | `1920` | Video width. |
| `--height PX` | `1080` | Video height. |
| `--fps N` | `30` | Frames per second. |
| `--duration SECONDS` | from the file | Override the total duration. |
| `--speed FACTOR` | `1.0` | Playback speed. `2.0` halves the video duration. |
| `--bg-alpha N` | `160` | Opacity of the statistic panel, from 0 to 255. |
| `--position CORNER` | `top-left` | Corner for the statistic panel. |
| `--map-position CORNER` | `bottom-right` | Corner for the mini-map. |
| `--map-size PX` | `320` | Side length of the mini-map. |
| `--no-map` | off | Hide the mini-map. |
| `--no-avg-pace` | off | Hide the average pace or average speed. |
| `--avg-pace-transition {slide,fade}` | `slide` | Entrance effect for the average value. |
| `--no-slope` | off | Hide the slope, even if elevation data is present. |
| `--ele-graph`, `--no-ele-graph` | enabled | Show or hide the elevation graph. |
| `--graph-position CORNER` | `bottom-left` | Corner for the elevation graph. |
| `--graph-width PX` | `480` | Elevation graph width. |
| `--graph-height PX` | `160` | Elevation graph height. |
| `--text-color R,G,B` | white | Statistic text colour. The default is black on a white background. |
| `--font-size PX` | `52` | Base font size. |
| `--outline-width PX` | `2` | Text outline thickness. |
| `--background MODE` | `transparent` | Canvas background. The aliases are `--back` and `--bg`. |
| `--min-hr BPM` | `100` | Lowest heart rate for colour scaling. |
| `--max-hr BPM` | `191` | Highest heart rate for colour scaling. |
| `--hr-color`, `--no-hr-color` | disabled | Colour the heart rate from white to red by intensity. |

Corner values are `top-left`, `top-right`, `bottom-left`, and `bottom-right`.

## Background modes

| Mode | Output |
| --- | --- |
| `transparent` | HEVC MP4 with alpha, and ProRes 4444 MOV with alpha. |
| `greenscreen` | Opaque H.265 MP4 with pure green `(0,255,0)`. |
| `blue` | Opaque H.265 MP4 with a solid blue background. |
| `black` | Opaque H.265 MP4 with a black background. |
| `white` | Opaque H.265 MP4 with a white background. The text becomes black. |

## Combine with source footage

```bash
ffmpeg -i source_video.mp4 -i overlay.mp4 \
       -filter_complex "[0:v][1:v]overlay=0:0" \
       -c:v libx265 -crf 18 final_output.mp4
```

For an alpha channel, use the ProRes file as the second input:

```bash
ffmpeg -i source_video.mp4 -i overlay_transparent.mov \
       -filter_complex "[0:v][1:v]overlay=0:0" \
       -c:v libx265 -crf 18 final_output.mp4
```

## License

MIT. See [LICENSE](LICENSE).
