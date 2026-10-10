# What to upload, and the steps before submitting

## Upload

1. **Code repository** (GitHub link or zip of this folder). Include `README.md`, `catsuggest/`, `scripts/`, `app/`, `tests/`, `docs/`, `config.yaml`, `requirements.txt`, `requirements.lock`, `run_all.py`, `run_all.sh` and `LICENSE`.
   - Include `reports/`: `RESULTS.md` and the `*.json` files are the evidence judges check.
   - **Do not** include `data/` or `artifacts/`. Both are large and regenerable, and `.gitignore` already excludes them.
2. **Results**: `reports/RESULTS.md`, produced by `scripts/s11_report.py`.
3. **Demo video**, 2–3 minutes. Script below.
4. **Slides**, about 10. The outline is in `docs/SLIDES.md`.
5. Optional: `docs/MODEL_CARD.md` as a PDF if the form asks for a technical write-up.

## Checklist before submitting

- [ ] `python3 -m unittest discover -s tests`: all tests pass, none skipped (the full dependency set is installed).
- [ ] `python run_all.py` completed without errors.
- [ ] `reports/RESULTS.md` exists and every table is filled.
- [ ] Optional but strong: mark 100–200 rows of `reports/review_sample.csv` y or n, run `python3 scripts/s12_apply_review.py`, then `python3 scripts/s11_report.py`. This adds the strict-vs-reviewed table.
- [ ] `python3 app/demo.py` works. Try 3 images: an object, an event, and a scene with a geotag.
- [ ] `LICENSE` has your team name.
- [ ] No secrets in the repo. `COMMONS_CONTACT` is an environment variable, not in the code.
- [ ] Every number in the slides and video is copied from `reports/RESULTS.md`.

## Demo video script (about 2.5 min)

1. **Problem (15 s).** Commons has millions of categories. New uploads need categories, and volunteers do it by hand.
2. **Live demo (75 s).** Upload a festival photo with no metadata and show the suggestions. Add the title and date, and show the event and year categories move up. Point at the "why" column and the neighbour gallery, which are the explanation. Tick the correct ones and submit feedback. Add a brand-new category name and show it can be suggested immediately.
3. **How it works (30 s).** Show the architecture diagram from the README: k-NN, text prototypes, ASL head and metadata, then calibrated fusion, then hierarchy.
4. **Evidence (30 s).** Show the `RESULTS.md` tables: fusion vs single scorers, the head/torso/tail/zero-shot bands, robustness, and latency.
5. **Honest limitations (10 s).** Read the limitations section of the README.
