# PREPUBLISH checklist

Machine-verified already (see CHANGELOG and the CI workflows):

- [x] pyproject `readme`, `license`, `[project.urls]` (OWNER placeholder), version 0.1.0.
- [x] `uv build` produces a clean sdist (124 KB, no repo artifacts) and wheel; `twine check` passes.
- [x] Wheel verified in a clean venv: import, `worldproof --help`, core-only
      (no optional stacks leak), and an end-to-end evaluate run.
- [x] `worldproof` is unclaimed on PyPI (checked 2026-08-01).
- [x] `publish.yml` workflow: builds, checks, smoke-tests, publishes on a `v*`
      tag via PyPI trusted publishing (no stored secrets).

- [x] GitHub repository created (`BuceaGeorgia/worldproof`); `OWNER` replaced in
      `pyproject.toml` and `README.md`, images point at raw.githubusercontent
      so the PyPI page renders them.

Remaining steps, in order. These need your accounts:

- [ ] Push `main`; confirm the CI matrix is green on GitHub.
- [ ] On pypi.org (account with 2FA): Publishing -> add a *pending* trusted
      publisher for project `worldproof`, repo `BuceaGeorgia/worldproof`,
      workflow `publish.yml`, environment `pypi`.
- [ ] In the GitHub repo settings, create the `pypi` environment (Settings ->
      Environments), optionally restricted to tags.
- [ ] Tag and push: `git tag v0.1.0 && git push origin v0.1.0`. The workflow
      builds, verifies, and publishes.
- [ ] Check `pip install worldproof` from PyPI on a clean machine, then create
      a GitHub Release from the tag with the CHANGELOG's Unreleased section.
