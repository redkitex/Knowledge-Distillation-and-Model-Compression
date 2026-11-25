# Slide deck (Beamer, Overleaf)

The 60-minute talk for Wed 2 Dec 2026. Slides lead: T3.

## Open it in Overleaf

1. Overleaf: **New Project, Upload Project**, and upload `b3-slides-overleaf.zip`
   (or zip this `report/slides/` folder yourself).
2. Menu: Compiler **pdfLaTeX**, main document **main.tex**.
3. Share the project with the team (Share, "Can edit").

When the deck changes in Overleaf, copy the files back into `report/slides/`
and commit them, so the repository keeps the slide source as the brief requires.

## Who edits which file

| File | Presenter | Minutes |
| --- | --- | --- |
| `sections/00-opening.tex` | T3 | title, roadmap |
| `sections/01-motivation.tex` | T3 | 6 |
| `sections/02-kd-basics.tex` | T2 | 8 |
| `sections/03-kl-onpolicy.tex` | T1 | 6 |
| `sections/04-literature.tex` | E2 | 8 |
| `sections/05-design.tex` | T3, T2, T1 | 4 + 2 + 4 |
| `sections/06-evaluation.tex` | E2, E1 | 2 + 2 |
| `sections/07-results.tex` | E1, E3 | 6 + 4 |
| `sections/08-interpretation.tex` | E1, E3 | 2 + 3 |
| `sections/09-conclusion.tex` | E3 | 3 |
| `sections/10-backup.tex` | T3, E2 | backup, references |
| `preamble.tex` | T3 | colours and helper commands, change only after telling the team |

## Conventions

- **Orange "To add" boxes** mark content that is missing from the repository
  (seed 1, `ref_teacher`, figures script) or still has to be written (theory
  notes, final interpretation). Replace each box when the content exists.
- **Badges in frame titles**: `\est` (from the literature), `\ours` (from our
  runs), `\who{T3}` (presenter).
- **Numbers**: every result comes from `results/` via `python -m kd.stats`.
  The charts in `07-results.tex` hold the numbers as pgfplots coordinates;
  update them when seed 1 and `ref_teacher` arrive, or swap in the PNGs from
  `scripts/40_make_figures.py` once it exists.
- **Overlays** (`<2->` and `\item<2->`) reveal content step by step; to print
  handouts without them, add `handout` to the `\documentclass` options.
- Plain ASCII only in the `.tex` files (pdfLaTeX); write `--`, `\%`, `\_`, `$\times$`.
