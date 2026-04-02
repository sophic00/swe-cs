# SWE CS Case Study

[![Latest Release](https://img.shields.io/github/v/release/sophic00/swe-cs?label=latest%20release)](https://github.com/sophic00/swe-cs/releases/latest)

Latest PDF release: [github.com/sophic00/swe-cs/releases/latest](https://github.com/sophic00/swe-cs/releases/latest)

This repository contains the data-mining scripts, plotting pipeline, LaTeX report source, and automation for the software engineering case study.

## Layout

- `scripts/` for the Python pipeline scripts
- `data/` for generated CSV datasets
- `plots/` for generated PNG plots
- `repos/` for the cloned `django` and `transformers` repositories
- `report.tex` for the LaTeX report source

## Prerequisites

- `uv`
- `git`
- A LaTeX toolchain such as `pdflatex` if you want to build the report locally

Install Python dependencies with `uv`:

```bash
uv sync
```

## Generate CSVs And Plots

The pipeline script can clone both subject repositories, generate the CSV datasets, and render all plots in one command:

```bash
uv run python scripts/run_pipeline.py
```

Useful variants:

```bash
uv run python scripts/run_pipeline.py --refresh
uv run python scripts/run_pipeline.py --skip-plots
uv run python scripts/run_pipeline.py --years-back 3
```

What it does:

- clones `django/django` into `./repos/django` if missing
- clones `huggingface/transformers` into `./repos/transformers` if missing
- writes `data/django_complexity.csv` and `data/transformers_complexity.csv`
- writes `data/django_history.csv` and `data/transformers_history.csv`
- generates plot images under `./plots`

## Build The Report Locally

After the plots exist, build the PDF with:

```bash
pdflatex -interaction=nonstopmode report.tex
pdflatex -interaction=nonstopmode report.tex
```

The output file is `report.pdf`.

## License

This project is licensed under the MIT License. See `LICENSE`.
