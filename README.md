# RayCrafter

RayCrafter is an early proof of principle for a lens-level optical design tool. You place
off-the-shelf lenses on an optical axis, drag them around, and see the spot diagram,
through-focus curve, MTF, ray fan and field curvature update as you go. The ray tracing is
done by [Optiland](https://github.com/HarrisonKramer/optiland).

At present it offers a short list of Thorlabs achromats, a single wavelength (555 nm) and a
single beam. It can export the system as a Zemax `.zmx` file and write an HTML report.

**Status:** pre-alpha. Expect anything to change, including the API.


## Install and run

```
pip install raycrafter
raycrafter
```

Or, with [uv](https://docs.astral.sh/uv/): `uvx raycrafter`.


## An agent-generated project

The code in this repository is written by AI coding agents, working with the project's
developers. Due to time constraints, the project is only possible because of agentic
coding.

We nonetheless aim for well-designed, readable code that people who use little or no AI can
understand and contribute to. The rules the agents follow are in [CLAUDE.md](CLAUDE.md), and
significant decisions, with their reasons, are recorded in [log.md](log.md).
