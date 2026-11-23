# Decision log

Significant decisions and the reasons for them, oldest first. Append only; see CLAUDE.md for
the rules. Entries marked "Pilot" summarise work done before this repo existed (in
`pilot/evaluation` and `pilot/minimal_GUI` of the planning folder).


## 2026-11-23: Pilot: Optiland as the optics engine
**Decision:** Use Optiland for ray tracing and analysis rather than writing our own engine.
**Why:** On the SMOD4.5 system it matches Zemax to 4 digits for ray-based quantities, and it
imports the Thorlabs ZMF catalogue. Wavefront quantities (RMS WFE, Strehl, PSF, MTF) are wrong
off axis because of an ideal-lens OPD bug; this is reported upstream and does not affect the
real-lens systems in this release.

## 2026-11-23: Pilot: pyqtgraph rather than matplotlib for the GUI
**Decision:** The interactive GUI is PyQt6 + pyqtgraph. matplotlib is used only for the
static HTML report.
**Why:** In the pilot, pyqtgraph took about 1 ms to draw where matplotlib took about 6 ms (a full
update took about 26 ms against about 65 ms). Its items can also be dragged directly.

## 2026-11-23: Pilot: rebuild the Optiland system on every change
**Decision:** The model keeps its own simple state (lenses, positions, flips, beam, image plane)
and builds a fresh Optiland `Optic` from it for each update, instead of editing a long-lived
one.
**Why:** It is simpler and avoids stale surfaces after adding, deleting or flipping lenses. At
about 8 ms per build it is fast enough. The surface-based Optiland model sits under a lens-level
layer of our own.

## 2026-11-23: Pilot: fast and slow update paths
**Decision:** The side view, spot, through-focus and status bar update on every change (about
17 ms). MTF, ray fan and field curvature are recomputed 120 ms after the last change, and only
for the visible tab.
**Why:** It keeps dragging smooth while still showing the costlier analyses.

## 2026-11-23: Pilot: modelling choices in the analyses
**Decision:** Through-focus and autofocus propagate the traced image-plane rays in straight
lines rather than re-tracing. Field curvature is a least-squares focus of ray fans, not
Optiland's `FieldCurvature` class. Beam vergence is an object at -1000/V mm. The Airy radius
uses the NA taken from the traced rays.
**Why:** Each is fast (well under 1 ms where vectorised) and correct for any vergence or tilt.

## 2026-11-23: Tooling for the first package
**Decision:** Start from the NIU python-cookiecutter. Keep pre-commit with ruff (format at 100
chars, lint limit 120), lenient mypy and codespell. Drop check-manifest. CI tests Linux with
Python 3.12 only, under XVFB for Qt. `uv.lock` is committed.
**Why:** The tools cost nothing when Claude runs them and keep diffs clean for future human
contributors. check-manifest adds nothing because setuptools-scm already includes all tracked
files; a test checks the catalogue is in the wheel. The CI matrix stays small until there are
users.

## 2026-11-23: Versioning and release
**Decision:** Semantic versioning from git tags (setuptools-scm). The first release is v0.0.1,
published to PyPI automatically by CI when the tag is pushed (Trusted Publishing).
**Why:** v0.0.1 is a proof of principle; the number should not suggest anything more.

## 2026-11-23: GUI included in the plain install; Qt Designer later
**Decision:** `pip install raycrafter` installs PyQt6 and pyqtgraph, and `raycrafter` launches
the app. The GUI layout stays in code for now.
**Why:** It is the simplest install for colleagues. Because the core never imports Qt, a
`[gui]` extra can be split out later. `.ui` files will come when someone wants to edit the
layout by hand.

## 2026-11-23: Ship only the lenses the GUI offers
**Decision:** The packaged Thorlabs catalogue holds only the 2" A-coated achromats and the
A-coated negative achromats shown in the drop-down, built from the full decoded catalogue by
`scripts/make_thorlabs_subset.py`.
**Why:** It keeps the package small (the full JSON is 4 MB and 2109 lenses). Open question: the
catalogues may later move to a separate, separately installed repository.

## 2026-11-23: Package layout for the first release
**Decision:** The pilot's three files became four subpackages: `core` (state, Optiland system
building, analyses, `OpticalModel`; no Qt), `lenscatalog`, `report` and `gui` (style, side
view, analysis panels, main window). Drawing geometry used by both the GUI and the report
(lens outlines, incoming rays, plot ranges) lives once, in `core/plot_geometry.py`.
**Why:** Each subpackage can be used alone (e.g. the catalogue or the report without Qt), no
file exceeds about 500 lines, and the GUI and report can no longer draw things differently.

## 2026-11-23: Work on main for now
**Decision:** Commit directly to `main`; use branches only when there is a real need.
**Why:** There are no users yet, so feature branches and PRs add overhead with no benefit. To
be revisited when the project has users or more contributors.

## 2026-11-23: Qt in CI without XVFB
**Decision:** CI runs the GUI tests with Qt's off-screen platform and installs Qt's system
libraries (libEGL etc.) with apt, instead of running under XVFB.
**Supersedes:** the XVFB part of "Tooling for the first package".
**Why:** The first CI run failed because the runner lacks libEGL. The tests never needed a
display, so XVFB only added a dependency (and a deprecated action).
