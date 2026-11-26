# Changelog

Each release also has a release message on
[GitHub](https://github.com/raycrafter-optical-design/raycrafter/releases).


## [0.0.4](https://github.com/raycrafter-optical-design/raycrafter/releases/tag/v0.0.4) (2026-11-26)

Version 0.0.3 was skipped.

- Project files (`.raycrafter`) keep the whole design, and the Zemax file it was imported from ([#40](https://github.com/raycrafter-optical-design/raycrafter/issues/40)).
- Undo and redo ([#37](https://github.com/raycrafter-optical-design/raycrafter/issues/37)).
- Gaps between lenses are shown, can be typed and locked ([#35](https://github.com/raycrafter-optical-design/raycrafter/issues/35), [#36](https://github.com/raycrafter-optical-design/raycrafter/issues/36)).
- Assemblies: group lenses, flip them as one, save them as user parts. Plossl pairs are assemblies ([#43](https://github.com/raycrafter-optical-design/raycrafter/issues/43)).
- Fields are defined at a source plane with its own position ([#41](https://github.com/raycrafter-optical-design/raycrafter/issues/41)).
- Python scripting API, `raycrafter.Bench` ([#31](https://github.com/raycrafter-optical-design/raycrafter/issues/31)).
- MCP server for AI agents, which can also drive the open window ([#26](https://github.com/raycrafter-optical-design/raycrafter/issues/26), [#28](https://github.com/raycrafter-optical-design/raycrafter/issues/28)).
- Bench viewer and part metadata ([#34](https://github.com/raycrafter-optical-design/raycrafter/issues/34), [#42](https://github.com/raycrafter-optical-design/raycrafter/issues/42)).
- Zemax import aims fields through the stop and reads apertures properly; the side view fits the design ([#17](https://github.com/raycrafter-optical-design/raycrafter/issues/17)–[#21](https://github.com/raycrafter-optical-design/raycrafter/issues/21)).


## [0.0.2](https://github.com/raycrafter-optical-design/raycrafter/releases/tag/v0.0.2) (2026-11-24)

- Lens picker dialog and the full Thorlabs catalogue ([#1](https://github.com/raycrafter-optical-design/raycrafter/issues/1)).
- Rays outside a lens's clear aperture are blocked; slow analyses wait until a drag ends ([#3](https://github.com/raycrafter-optical-design/raycrafter/issues/3)).
- Paraxial (ideal thin) lenses ([#6](https://github.com/raycrafter-optical-design/raycrafter/issues/6)).
- Several fields (beams), edited live in a field window ([#7](https://github.com/raycrafter-optical-design/raycrafter/issues/7)).
- File menu: new, save and load Zemax `.zmx` files ([#8](https://github.com/raycrafter-optical-design/raycrafter/issues/8)).
- Requires Python 3.12+ and PyQt6 6.8+.


## 0.0.1 (2026-11-23)

First preview release.
