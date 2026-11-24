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

## 2026-11-23: dev branch and fewer CI runs
**Decision:** Small changes go straight onto `main`; longer jobs are done on a long-lived `dev`
branch and merged into `main` when complete. Feature branches (risky work only) and all pull
requests go into `dev`. CI runs only on pushes to `main`, on pull requests, on tags and by
hand; a newer push cancels an older run on the same branch.
**Supersedes:** "Work on main for now".
**Why:** Testing every push to every branch was excessive. Pre-commit and local pytest check
each commit; CI checks on a clean machine before anything is merged or released.

## 2026-11-23: Require Python 3.12 or newer
**Decision:** `requires-python` is `>=3.12`; Python 3.11 is dropped from the classifiers and tox.
**Why:** On 3.11 numba emits `NumbaIRAssumptionWarning` while compiling Optiland's conic code,
which our warnings-as-errors tests turn into a failure. CI only ever ran 3.12, so 3.11 support
was never real. Rob chose to drop it rather than filter the warning or report it upstream.

## 2026-11-24: Ship the whole Thorlabs catalogue, gzipped
**Decision:** The package ships every Thorlabs lens except the cylindrical ones (1564) as
`thorlabs.json.gz` (about 110 KB), made by `scripts/make_thorlabs_catalog.py`. All parts are
listed in the new lens picker (#1); near-duplicate focal lengths are no longer dropped.
Aspheres are in the file but not offered until `LensSurface` can represent them (#2).
**Why:** The picker's categories and filters need the full range. Coating variants share
prescriptions, so the 3 MB of JSON compresses about 27-fold.
**Supersedes:** 2026-11-23 "Ship only the lenses the GUI offers".

## 2026-11-24: Lens picker design (#1)
**Decision:** "Add lens" opens a dialog: category -> diameter -> lens tree, part-number search,
and filters for coating (default: all coatings), |f| range and sign. The |f| slider is
logarithmic and uses the magnitude; the sign is chosen with "hide positive/negative". After
Insert, a ghost follows the pointer and snaps to the nearest gap the lens fits in.
**Why:** An "Uncoated" default hid every achromat (Thorlabs sells none uncoated). A log scale
suits focal lengths from 4 mm to 2.5 m. Snapping means a placed lens never overlaps another.

## 2026-11-24: Lens surfaces have their clear aperture (#3)
**Decision:** Every lens surface blocks rays outside the lens's clear semi-aperture. Analyses
use only the rays that get through; with none, they report "no light". The MTF uses the part
of the beam that gets through on axis. Blocked rays are drawn up to the plane of the rim.
**Why:** Without apertures, a beam wider than a lens was traced through surfaces that do not
exist, giving wrong results or NaN (and a crash). Thorlabs' ZMX files give every surface the
same fixed aperture, so this matches the vendor model; Zemax agreement to be checked (#4).

## 2026-11-24: f-number from the transmitted beam
**Decision:** The f-number is |EFL| / the diameter of the on-axis beam that gets through every
lens, and is shown as e.g. "f/1.9 (beam clipped to 10.8 mm)" when a lens rim cuts the beam.
**Why:** When the beam overfills a lens, the rim sets the cone of light, so the nominal beam
overstates the speed. This probably differs from Zemax's paraxial F/#, which uses the system
aperture (#5); matching Zemax is not the aim, since a Zemax user would set the stop to the
lens. RayCrafter has no field stops yet. The probe costs about 3.5 ms per fast update.

## 2026-11-24: One wavelength (555 nm) for every lens, including IR designs
**Decision:** All tracing is at 555 nm, whatever a lens's design wavelength. The picker lists
the catalogue focal length.
**Why:** Recorded while adding the full catalogue: our prescriptions reproduce Optiland's own
ZMX import exactly, but IR-designed lenses (C, E, 1064 coatings etc.) come out a few percent
off their catalogue focal length at 555 nm because of dispersion. Wavelength choice is future
work.

## 2026-11-24: "Test in Zemax" issues
**Decision:** Checks that need a Zemax licence become GitHub issues with the "Test in Zemax"
label, each saying what to compare and how (e.g. #4, #5). Results can become reference tests.
**Why:** Not every developer has Zemax, and these checks are what support our claims of
agreement with it, so they need a visible queue rather than notes in a conversation.

## 2026-11-24: Responsiveness first: only essential work on every frame
**Decision:** The GUI must stay responsive, so anything computed on every change (each drag
event or slider tick) must be something the user is watching. Other work runs after changes
settle (the slow plots) or only once nothing is being dragged. The f-number is the first
case: it is computed only then, and is left out of the status bar during a drag.
**Supersedes:** the "about 3.5 ms per fast update" in "f-number from the transmitted beam".
**Why:** Optiland is noticeably slower on some machines (Rob's Windows PC), and nobody reads
the f-number while dragging.

## 2026-11-24: Slow plots fade during a drag and update on release
**Decision:** While a lens, the image plane or a slider handle is dragged, the MTF, ray fan
and field curvature are faded (40% opacity) and not recomputed; they are recomputed when
the mouse is released. The f-number keeps showing its last value until then.
**Supersedes:** the slow plots' "after changes settle" in "Pilot: fast and slow update
paths", for drags only (other changes still wait SLOW_PLOT_DELAY_MS), and hiding the f-number
during a drag in "Responsiveness first".
**Why:** A pause mid-drag used to trigger the slow plots, which stutters on slower machines.
Fading shows they are out of date; keeping the last f-number is less jarring than hiding it.

## 2026-11-24: Fields replace the beam sliders (#7)
**Decision:** The bench has a list of fields (beams), each with its own diameter, angle,
vergence, offset, wavelength and colour, edited live in a small non-modal field window. The
beam diameter, angle and vergence sliders and "Reset beam" are removed. The spot diagram and
side view show every field; through-focus, MTF, ray fan, field curvature, f-number and
autofocus use the selected field.
**Why:** The sliders were a single-beam prototype. Rob chose to remove them rather than keep
two ways of setting the same beam.

## 2026-11-24: Each field is traced at its own wavelength
**Decision:** Every field is traced at its own wavelength (default 555 nm), so glass
dispersion shows, e.g. as a chromatic focal shift between a blue and a red field.
**Supersedes:** "One wavelength (555 nm) for every lens, including IR designs".
**Why:** Rob's choice when fields gained a wavelength; using it only for the beam colour
would have been misleading.

## 2026-11-24: Each field's rays are launched by RayCrafter, not Optiland (#7)
**Decision:** The Optiland system is built once per update and each field's rays are made
in numpy (`core/ray_launch.py`) and traced with `optic.surfaces.trace`. A beam is defined
where it crosses the first lens's front vertex plane. The MTF still uses Optiland's own
object and pupil (it needs them), so it ignores a field's offset (#10).
**Supersedes:** "Beam vergence is an object at -1000/V mm" in "Pilot: modelling choices in
the analyses", except for the MTF and the ZMX export.
**Why:** Optiland's aperture, object and field angles are system-wide, but each field has its
own width, vergence, offset and wavelength. One build per field would cost ~8 ms each.

## 2026-11-24: Loading ZMX files matches lenses against the catalogue (#8)
**Decision:** Load ZMX turns each run of glass surfaces into a lens and looks it up in the
catalogue (curvatures, thicknesses and glasses, both ways round), so part numbers and flips
come back; unmatched lenses keep their prescription as "Imported lens N". Paraxial surfaces
become paraxial lenses. Aspheres, conics, mirrors and tilts are refused. Fields are rebuilt
from the EPD, object distance, angles and wavelengths, with any simplification reported.
**Why:** Matching keeps designs editable as catalogue parts. Lossy parts are parked (#11, #12).

## 2026-11-24: The side view fits the design (#17)
**Decision:** The side view and report keep 0-450 mm x ±32 mm for designs that fit, and
otherwise reach 10% beyond the image plane (and above the tallest lens). The view is refitted
after each update, but never during a drag, so the axes do not move under the mouse.
**Why:** Fixed axes cut off long systems such as SMOD4.5 (760 mm). Fitting to rays was
rejected: they diverge without limit after the last lens.

## 2026-11-24: ZMX fields are aimed through the stop (#18)
**Decision:** When a file's stop is in air in front of the first lens, each field's offset is
(stop to lens distance) x tan(angle), so its chief ray crosses the axis at the stop; the
EPD is scaled to the beam's width at the lens. Otherwise fields stay centred on the lens.
**Why:** Scanning systems pivot at the scan mirror. RayCrafter then matches Optiland's trace
of SMOD4.5 exactly. Field curvature and MTF still ignore the pivot (#24, #10).

## 2026-11-24: Imported lens sizes are kept, and differences from Zemax reported (#20)
**Decision:** A lens keeps its catalogue size, or the file's drawn size (DIAM) when it has no
aperture, even though Zemax blocks rays only at apertures. A note says when RayCrafter
therefore blocks rays that Zemax passes. SQAP, DIAM and COMM are read from the file text,
because Optiland 0.6.3 skips them (see `pilot/load/optiland_zmx_reader_gaps.md`).
**Why:** Rob's choice. Widening lenses to match Zemax was tried and rejected: the drawn size
is often the real one (SMOD4.5's objective back aperture), and resizing would hide clipping.

## 2026-11-24: Imported fields are coloured by angle (#21)
**Decision:** A ZMX file with several field angles loads with one colour per angle, in
Zemax's order (`palette.FIELD_COLOURS`). Extra wavelengths share the 0 degree field's colour.
**Why:** Rob's choice. Wavelength colours could not tell the fields apart (same wavelength,
or all infrared). Doing the same for fields made in RayCrafter is #25.
