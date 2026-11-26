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

## 2026-11-24: MCP server uses the official `mcp` SDK, in its own package (#26)
**Decision:** The agent interface is `raycrafter.mcp_server`, built on the official `mcp` Python
SDK (`MCPServer`), pinned to a minor version and installed with the `mcp` extra. `DesignTools`
holds the tools as plain methods on an `OpticalModel`; `server.py` only publishes them. File
loading and saving are not restricted, and undo/redo is not offered (Rob's choice).
**Why:** `fastmcp` was the alternative: a bigger dependency we do not need. Keeping the tools
free of the MCP library lets the same tools later drive the model inside the GUI. The package
is not called `mcp` so it cannot be confused with the library.

## 2026-11-24: Agents reach the open window through a local HTTP link (#28)
**Decision:** The GUI runs a localhost-only HTTP server (stdlib) and writes its port and a random
token to `~/.raycrafter/agent_link.json` (mode 600). `raycrafter-mcp` tries the window on every
tool call and falls back to its own headless model if none answers; results say which
(`connected_to`). Calls run on the Qt thread via a signal. On by default; `--no-agent-link` turns it off.
**Why:** The Claude app starts the MCP server itself over stdio, so it is a different process from
the window. Having the window host MCP over HTTP was rejected: the desktop app launches local
servers rather than connecting to running ones. The fallback keeps the server useful with no window.

## 2026-11-26: Undo keeps snapshots of the state (#37)
**Decision:** `core.history.UndoHistory` keeps whole `OpticalState` copies, recorded by the
front end once nothing is held, so a drag is one step. Small changes to one value in quick
succession merge into one step. New and Load clear the history.
**Why:** Snapshots cover anything later added to the state (locks, goals) with no extra code,
and are cheap. Merging stops a spun number box making dozens of steps. Clearing on New and
Load is simplest, and unsaved changes are already asked about. Merge and clearing: Rob agreed.

## 2026-11-26: Typed gaps are refused rather than adjusted (#35)
**Decision:** A gap that is too small (surfaces would overlap, or under 1 mm to the image) is
refused and the minimum is shown. With "Lock focus" on, the gap to the image plane cannot be
typed. The gap inside a Plossl pair is shown but fixed. Dragging is unchanged, so it can still
overlap concave lenses at the rim.
**Why:** Silently changing a typed number would mislead, and "Lock focus" would overwrite a
typed image gap at once. The Plossl and dragging rules follow the issue. Rob agreed.

## 2026-11-26: Project files embed the ZMX they were imported from (#40)
**Decision:** A `.raycrafter` project made by Import ZMX keeps the file's exact bytes (base64),
name, SHA-256, importing version and import notes in `imported_from`, on `OpticalState`. It is a
record only and is never loaded; it is replaced by the next import and cleared by New. Always
embedded, documented. "Lock focus" is not stored. After Import ZMX the design counts as saved.
**Why:** The import is lossy and still improving, so the source allows re-import later,
reproducible import bugs and importer regression tests. Storing only a path/hash was rejected
as files move. Rob agreed.

## 2026-11-26: Imported projects keep the design as imported, until forgotten (#40)
**Decision:** `imported_from` also stores the bench exactly as the importer made it, so
RayCrafter can tell whether the design was changed since import and what a newer importer
does differently. "Save Without Original ZMX" (`save_project(forget_import=True)`) drops the
record, and the design stops being an import. Bench file methods live in `design_files.py`.
**Why:** After edits the original file alone says only where a design started. A hash of the
imported bench was rejected: it says that something changed, not what. Rob's choice.

## 2026-11-26: Gap locks are a flag per lens; a Plossl pair is a locked pair (#36)
**Decision:** Each lens has `locked_to_next`, and the state has `image_gap_locked`. Lenses
joined by locked gaps form a block that drags and swaps as one. `group_id` stays as the identity
of a Plossl pair (to become the assembly id in #43), whose gap is always locked. A locked image
gap moves the image with the last lens and refuses autofocus, "Lock focus" and new lenses
behind it; typing a locked gap is refused. Rob chose this over replacing `group_id` with locks.
**Why:** One mechanism for moving things together, while a pair keeps an identity for flipping
and saving as a part. Refusing follows "Typed gaps are refused rather than adjusted".

## 2026-11-26: Assemblies replace Plossl groups; user parts are files in the user's folder (#43)
**Decision:** `group_id` became `assembly_id`, with a name per assembly in the state. A Plossl
pair is an assembly named "Plossl pair". Locking gaps then "Make assembly" groups a block;
flipping reverses and flips its lenses in place. "Save as user part" writes one JSON file
(lenses in the project-file form) to the platform data folder (`core/user_parts.py`). Scripts
lose `plossl_group` and `split_plossl` for `assembly` and `split_assembly` (0.0.x, no alias).
**Why:** Rob chose the user folder over the project file, so parts are shared between designs.
Bench assembly methods live in `assembly_methods.py` to keep `api.py` under 1000 lines.

## 2026-11-26: Fields are defined at a source plane with its own z (#41)
**Decision:** `OpticalState.source_z` is where each field's diameter, offset, angle and vergence
apply. Its gap to the first lens is locked at 0 mm by default and then follows whichever lens
is first, so designs behave as before; unlocked, it stays put and lenses stop at it. ZMX import
puts it at a stop in front of the first lens (offset 0, EPD as is). The MTF and ZMX export
carry the beam to the first lens (`beam_at_first_lens`). Rob chose an explicit source z.
**Supersedes:** "beam is defined where it crosses the first lens's front vertex plane" in
"Each field's rays are launched by RayCrafter" and the offset rule in "ZMX fields are aimed
through the stop" (2026-11-24).

## 2026-11-26: Replacing a lens keeps its rear principal plane (#63)
**Decision:** "Replace with same f…", "Swap lens…" and "Replace with paraxial" put the new lens
where the old one's rear principal plane was, so focus barely moves. A locked gap in front
keeps the front vertex, one behind keeps the back vertex; an overlap is refused. For a lens
of a Plossl pair the user replaces both or splits the pair. The picker's coating filter
defaults to "suits the beams" (uncoated always suits); "same f" means same sign, within ±10%.
**Why:** Rob's choice, for teaching. Keeping the front vertex or centre shifts the focus.

## 2026-11-26: Gap optimisation uses our own rays and a 1-D search (#38, #56)
**Decision:** "Collimate output" and "Focus" scan the allowed gap range with RayCrafter's own
launched rays of the selected field, then refine with SciPy's bounded scalar minimiser. The
Optiland system is built once per search; trials change thicknesses (`updater.set_thickness`).
Gaps that clip rays are never chosen. A flip of a neighbouring lens is suggested when it gives
at most half the merit and is better by at least 5 µrad or 1 µm.
**Why:** Optiland's optimiser ignores a field's offset (#10) and is unbounded; one variable
needs no more than a scan. Rebuilding per trial took about 1 s per search; this takes 0.1 s.
