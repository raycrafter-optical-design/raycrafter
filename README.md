# RayCrafter

[![Python Version](https://img.shields.io/pypi/pyversions/raycrafter.svg)](https://pypi.org/project/raycrafter)
[![PyPI Version](https://img.shields.io/pypi/v/raycrafter.svg)](https://pypi.org/project/raycrafter)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/raycrafter-optical-design/raycrafter/blob/main/LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/raycrafter-optical-design/raycrafter/test_and_deploy.yml?branch=main&label=CI)](https://github.com/raycrafter-optical-design/raycrafter/actions/workflows/test_and_deploy.yml)
[![codecov](https://codecov.io/gh/raycrafter-optical-design/raycrafter/branch/main/graph/badge.svg)](https://codecov.io/gh/raycrafter-optical-design/raycrafter)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

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
