# Optiland fixes and workarounds

Every place where RayCrafter works around an Optiland bug or gap, and every upstream fix we are
waiting for. Optiland lives at <https://github.com/optiland/optiland> (formerly
`HarrisonKramer/optiland`).

Pinned version: **0.6.3** (`pyproject.toml`). Last checked for a newer release: 2026-11-24
(0.6.3 is still the latest).

How this file is used: see "Optiland" under Workflow in `CLAUDE.md`. Add an entry when a
workaround is written; when an upstream fix is released, remove the workaround and move its
entry to "Retired".


## Active workarounds

### Zemax reader ignores SQAP, COMM and non-stop DIAM
- **Where:** `raycrafter/core/zmx_import.py`, section "what Optiland does not read"
  (`SurfaceExtras`, `_read_surface_extras`, and their callers).
- **Problem (0.6.3):** the reader drops rectangular apertures (`SQAP`), surface comments
  (`COMM`), and the semi-diameter (`DIAM`) of every surface except the stop. We read them from
  the file text.
- **Upstream status:** fixed on Optiland `main`, **not yet released** (after 0.6.3):
  - SQAP and ELAP: issue [#931](https://github.com/optiland/optiland/issues/931), PR
    [#954](https://github.com/optiland/optiland/pull/954) (merged 2026-10-24).
  - DIAM as a radius and fixed semi-diameters: issue
    [#928](https://github.com/optiland/optiland/issues/928), PR
    [#929](https://github.com/optiland/optiland/pull/929), refined by
    [#1017](https://github.com/optiland/optiland/pull/1017).
  - Also on `main`: `OBSC` ([#953](https://github.com/optiland/optiland/pull/953)) and `SPID`
    ([#1015](https://github.com/optiland/optiland/pull/1015)).
  - `COMM`: nothing found upstream (searched 2026-11-24). Probably stays a workaround; file an
    issue only if we still need it after the next release.
- **Remove when:** a release contains #954 and #929/#1017. Check that Optiland's imported
  apertures match `SurfaceExtras` before deleting the parser.

### Ideal-lens OPD error off axis
- **Where:** no code workaround; the wavefront quantities (RMS WFE, Strehl, PSF, MTF) are wrong
  off axis for ideal (paraxial) lenses. See `log.md`, 2026-11-23 entry "Optiland as the optics engine".
- **Upstream status:** reported upstream, but the issue link was not recorded. TODO: find it
  and put it here.
- **Remove when:** a release fixes it; add a physics check against a known value.


## Retired

Nothing yet.
