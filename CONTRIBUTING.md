# Contributing

Thank you for your help. This project is small. Keep each change focused.

## Before you start

1. Install the dependencies. See [README.md](README.md).
2. Check that FFmpeg has the `libx265` and `prores_ks` encoders.
3. Open an issue before you start a large change.

## Run the tests

Run the automatic tests:

```bash
python -m unittest discover -s tests -v
```

## Run the demo files

The `examples/` directory contains a synthetic GPX file and a synthetic FIT
file. Run both files after your change:

```bash
python gpx-overlay.py examples/demo.gpx --output /tmp/demo-gpx.mp4
python gpx-overlay.py examples/demo.fit --output /tmp/demo-fit.mp4 --stat-mode speed
```

Check the messages in the terminal. Check that the output files exist.

## Make a pull request

1. Fork the repository and create a branch.
2. Make your change.
3. Run the automatic tests and the two demo commands above.
4. Fill in the pull request template. Say which tests you ran and the result.
5. Wait for the `Automatic tests` and `Smoke test` checks. Both must pass
   before merge.

## Style

1. Use short sentences in text.
2. Keep the code style of `gpx-overlay.py`.
3. Do not add a dependency unless the change needs it.
4. Update `README.md` when you change an option or the output.
