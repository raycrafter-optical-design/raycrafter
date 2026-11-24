# RayCrafter: instructions for coding agents (and humans)

RayCrafter is a lens-level optical design tool: users place off-the-shelf lenses on an optical
axis and see spots, MTF, aberrations and so on update live. Optiland does the optics. The code
is written mainly by coding agents, but it must be readable and maintainable by people who use
little or no AI. If something would confuse a careful human reader, it is wrong.


# Architecture rules

- **`raycrafter.core` never imports Qt.** It holds the optical model and the analyses. The GUI,
  the HTML report, scripts and (later) an MCP server for AI agents all use the same core API.
- **Model/view.** The GUI (`raycrafter.gui`) displays the core model and turns user actions into
  calls on it. It holds no optical state of its own beyond what it needs to draw.
- **Object-oriented, modular.** One clear responsibility per module. Subpackages must be usable
  on their own, e.g. `from raycrafter.lenscatalog import thorlabs` must not pull in Qt or the GUI.
- **Responsiveness first.** The GUI must respond immediately, on slow machines too (Optiland
  is noticeably slower on some). Be very careful about what is calculated on every change
  (each drag event or slider tick): only what the user is watching while they drag. Anything
  else goes on the slow path (after changes settle) or runs only once nothing is being
  dragged, as the f-number does (`MainWindow.is_dragging`). Measure what you add to the fast
  path; the side view shows the trace and draw times.
- **File size.** Aim for files under about 1000 lines. This is a soft limit: if a file has to be
  bigger, say why in `log.md`.
- **Optiland.** It is pinned to a tested minor version in `pyproject.toml`. Any workaround for an
  Optiland bug goes in one clearly marked place, with a comment that links to the upstream issue.
  Delete the workaround once the fix is released. When new versions of Optiland are released:
  test them and upgrade as soon as possible.

Package layout:

```
raycrafter/
  core/          Qt-free model, building the Optiland system, analyses
  lenscatalog/   lens catalogues (JSON data plus loaders)
  report/        HTML report (matplotlib figures)
  gui/           PyQt6 + pyqtgraph application
  palette.py     colours shared by the GUI and the report
```


# Code style

- Use descriptive variable names; avoid abbreviations that a newcomer would have to guess.
- Use white space well, including blank lines between functions and between logical blocks.
- Aim for lines of at most 100 chars. Up to 120 is allowed where it helps readability.
  `ruff format` wraps at 100 and the linter complains only above 120.
- Every public module, class and function gets a docstring in the NumPy style. Keep it short
  for simple functions.
- Add type hints to public functions and methods.
- Where the intent of a piece of code is not obvious (physics, Qt quirks, Optiland workarounds),
  explain it in a comment.
- Physical units go in names or comments (`_mm`, `_um`, degrees, dioptres).
- Once a feature is complete, remove any dead code.


# Tests

- Run them with `uv run pytest`. Tests live in `tests/test_unit` (fast, no display) and
  `tests/test_integration` (GUI, packaging, slower). GUI tests run headless.
- Every change to the optics must keep the physics checks passing (e.g. the ACT508-200-A back
  focal length). A new analysis needs at least one check against a known value.
- A bug fix comes with a test that would have caught the bug.


# Workflow

- **Branches.** Small, self-contained changes go straight onto `main`. A longer job that
  will take a series of commits is done on the long-lived `dev` branch; once it is complete
  and the local tests pass, `dev` is merged into `main`. Before starting on `dev`, merge
  `main` into it so it has any small fixes made there. A feature branch is used only for
  risky or experimental work: it branches from `dev`, merges back into `dev`, and is then
  deleted. Pull requests (from forks or feature branches) target `dev`, never `main`.
- **CI** runs on pushes to `main`, on every pull request, on tags, and by hand from the
  Actions tab. Pushes to `dev` or feature branches are not tested by CI, so the local checks
  below are what stand between them and `main`.
- Use GitHub issues for substantial bugs, features and design questions, so there is a public
  record of what was done and why. Reference the issue in commits and PRs.
- **Known problems.** For now, while the project is young, a feature is done when it does
  what its issue asks, even if not all of its ramifications are resolved. Each significant
  ramification left unsolved (e.g. something a file format cannot store, or an analysis that
  ignores a new setting) gets its own issue with the `known problem` label. The feature's
  issue is then closed with a comment linking to those known problems. This lets us move
  forward quickly. We will review this choice as the project develops: later we may be
  stricter and keep a feature's issue open until all its ramifications are resolved.
- Before committing, run `uv run pre-commit run --all-files` and `uv run pytest`.
- Agents may create or comment on issues, push `dev` and feature branches, and open pull
  requests into `dev`.
- Small changes on `main` may be committed and pushed without asking. Merging `dev` into
  `main`, tagging and releasing need explicit approval from Rob.
- Versions come from git tags via setuptools-scm, using semantic versioning. Pushing a
  `vX.Y.Z` tag publishes to PyPI. Always verify your proposed new version number with the
  human managing the project.
- Stop and ask at natural check-points rather than finishing large pieces of work in one go.
- Expect that people may have changed files you recently worked on or are working on now.
  Re-read a file before modifying it; otherwise you risk reverting a human developer's saved
  changes.
- When it is time for a new release, you will first conduct an adversarial code review of
  work down between now and the last release. The main task of the review will be to find
  bugs that may have been introduced. However, you should also verify that the changes made
  are in keeping with the design plan of the software. Once any fixes have been applied
  after the review ends, it is OK to increment the version and deploy the new release.


# Environment

- Use `uv`. The virtual environment lives **outside Dropbox**, because a venv cannot be shared
  between machines or operating systems. On the Mac, run:
  `export UV_PROJECT_ENVIRONMENT=~/uv_venvs/raycrafter_mac` and then `uv sync --extra dev`.
- `uv.lock` is committed so that every developer gets the same environment.
- `rm` and `mv` are aliased to interactive versions on Rob's machines, so use `rm -f` / `mv -f`.


# Decision log (`log.md`)

`log.md` records significant decisions and why they were made, so the history of the project
can be traced later (e.g. for a paper). Keep it short so it stays cheap to search.

- **Before starting work**, search the log for the topic, e.g.
  `grep -n -i -A6 "catalog" log.md`. Read the matching entries only, not the whole file.
- **Add an entry** for a decision someone might later ask "why?" about: architecture,
  dependencies, tooling, physics approximations, Optiland workarounds, or something tried and
  abandoned. Do not log routine changes or bug fixes; git and the issues record those.
- **Append only.** Add new entries at the end and never rewrite old ones. If a decision is
  reversed, add a new entry that says which entry it supersedes.
- **Format.** Each entry is at most about 8 lines:

  ```
  ## YYYY-MM-DD: Short title
  **Decision:** what was decided.
  **Why:** the reason, including alternatives rejected.
  **Supersedes:** (optional) the date and title of the earlier entry.
  ```
- Ask Rob before logging a decision that he has not made or agreed to.
