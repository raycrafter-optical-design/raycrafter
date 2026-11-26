"""Optimising one air gap for a goal: collimate the output, or focus on the image plane (#38).

Finding the right spacing by eye is slow: e.g. an f = 100 mm lens then an f = 50 mm lens as
a beam reducer collimates only at one gap. ``optimise_gap`` finds that gap with real rays
across the selected field's beam (not paraxial focal lengths), so it allows for aberrations,
the beam's own width, angle, vergence and offset, and rays clipped by lens rims. It works on
a copy of the bench and does not change it; ``apply_gap`` sets the gap it found. It has no
Qt code, and runs only on request (never while something is dragged).

The goals
---------
- **collimate**: the rays leaving the last lens are parallel. The merit is the RMS angle
  of the rays to their mean direction, in microradians. The lenses behind the gap move
  together with the image plane, as when a gap is typed.
- **focus**: the smallest RMS spot on the image plane, which stays where it is (autofocus
  already covers moving the image plane instead). The merit is the RMS spot radius, in
  micrometres. The lenses behind the gap move; the image plane does not.

Guard rails
-----------
The gap is kept within what is physically possible: the facing surfaces may not touch, the
first lens keeps its distance from the start of the axis, and for "focus" the last lens
stays in front of the image plane. A gap at which fewer rays get through than at the best
gap in the search is never chosen, so a solution cannot gain by clipping rays. The result
says when the best gap is at one end of the allowed range (there is then probably no true
solution within it).

The search scans the allowed range coarsely, then refines around the best point, so it does
not stop in a local minimum. The Optiland system is built once per search; each trial only
changes the thicknesses the gap sets (see ``_GapTrials``).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import minimize_scalar

from raycrafter.core import gaps, optic_builder
from raycrafter.core.analysis import SPOT_RINGS, transmitted
from raycrafter.core.ray_launch import launch_rays, pupil_points, trace_rays
from raycrafter.core.state import OpticalState

COLLIMATE = "collimate"
FOCUS = "focus"
GOALS = (COLLIMATE, FOCUS)
MERIT_UNITS = {COLLIMATE: "µrad", FOCUS: "µm"}  # what each goal's merit is measured in

MAX_GAP_MM = 1000.0  # "collimate" looks for gaps up to this long
COARSE_STEPS = 41  # trial gaps in the first scan across the allowed range
GAP_TOLERANCE_MM = 1e-3  # the refined gap is found to about this precision
AT_BOUND_TOLERANCE_MM = 0.01  # a best gap this close to a limit counts as at that limit


@dataclass(frozen=True)
class GapOptimisation:
    """The result of ``optimise_gap``: the best gap found, and how good it is.

    Attributes
    ----------
    lens_index : int
        The lens in front of the gap, or ``gaps.SOURCE_GAP_INDEX`` for the gap from the
        source plane.
    goal : str
        COLLIMATE or FOCUS.
    gap_before_mm, gap_after_mm : float
        The gap now, and the best gap found.
    merit_before, merit_after : float
        The merit (see ``merit_unit``) at those gaps; smaller is better. NaN if no light
        gets through at the current gap.
    merit_unit : str
        "µrad" (collimate: RMS angle of the output rays) or "µm" (focus: RMS spot radius).
    shortest_gap_mm, longest_gap_mm : float
        The range that was searched.
    at_bound : bool
        True if the best gap is at one end of that range.
    rays_launched : int
        How many rays of the selected field were traced.
    rays_through_before, rays_through_after : int
        How many of them got through at each gap.
    """

    lens_index: int
    goal: str
    gap_before_mm: float
    gap_after_mm: float
    merit_before: float
    merit_after: float
    merit_unit: str
    shortest_gap_mm: float
    longest_gap_mm: float
    at_bound: bool
    rays_launched: int
    rays_through_before: int
    rays_through_after: int

    @property
    def improvement_factor(self) -> float:
        """How many times better the merit is after (inf if it is now zero)."""
        if self.merit_after == 0.0:
            return math.inf
        return self.merit_before / self.merit_after


def current_gap_mm(state: OpticalState, lens_index: int) -> float:
    """The width of the gap behind lens ``lens_index`` (or from the source plane), in mm."""
    if lens_index == gaps.SOURCE_GAP_INDEX:
        source_gap = gaps.source_gap(state)
        if source_gap is None:
            raise ValueError("There is no lens, so there is no gap to optimise")
        return source_gap.width_mm
    return gaps.air_gaps(state)[lens_index].width_mm


def check_can_optimise(state: OpticalState, lens_index: int, goal: str) -> None:
    """Raise ValueError, saying why, if this gap cannot be optimised for this goal."""
    if goal not in GOALS:
        raise ValueError(f"The goal must be one of {', '.join(GOALS)}, not {goal!r}")
    if lens_index == gaps.SOURCE_GAP_INDEX:
        source_gap = gaps.source_gap(state)
        if source_gap is None:
            raise ValueError("There is no lens, so there is no gap to optimise")
        if source_gap.locked:
            raise ValueError("The gap from the source plane is locked: unlock it to optimise it")
    else:
        if not 0 <= lens_index < len(state.lenses):
            raise IndexError(f"There is no lens {lens_index}")
        gap = gaps.air_gaps(state)[lens_index]
        if gap.to_image:
            raise ValueError(
                "The gap to the image plane does not change the output beam; "
                "use autofocus to focus on the image plane"
            )
        if gap.locked:
            raise ValueError("This gap is locked: unlock it to optimise it")
    if goal == FOCUS and state.image_gap_locked:
        raise ValueError(
            "The gap to the image plane is locked, so the image plane would move with the "
            "lenses: unlock it to focus by moving them"
        )


def apply_gap(state: OpticalState, lens_index: int, goal: str, gap_mm: float) -> None:
    """Set the gap as the goal requires: for FOCUS the image plane stays where it is.

    The lenses behind the gap (all of them, for the gap from the source plane) move
    together. Limits are the caller's business (``optimise_gap`` keeps within them).
    """
    image_z = state.image_z
    if lens_index == gaps.SOURCE_GAP_INDEX:
        gaps.set_source_gap(state, gap_mm)
    else:
        gaps.set_gap(state, lens_index, gap_mm)
    if goal == FOCUS:
        state.image_z = image_z


def gap_limits_mm(
    state: OpticalState, lens_index: int, goal: str, smallest_source_gap_mm: float
) -> tuple[float, float]:
    """The shortest and longest gap the search may try (mm).

    Parameters
    ----------
    smallest_source_gap_mm : float
        For the gap from the source plane: the shortest it may be (``OpticalModel`` keeps
        the first lens a little way from the start of the axis).
    """
    if lens_index == gaps.SOURCE_GAP_INDEX:
        shortest_mm = max(0.0, smallest_source_gap_mm)
    else:
        shortest_mm = gaps.minimum_gap_mm(state, lens_index)
    if goal == FOCUS:
        # The lenses behind the gap move towards the image plane, which stays put.
        room_mm = state.image_z - gaps.MIN_IMAGE_GAP_MM - state.last_back_z()
        longest_mm = current_gap_mm(state, lens_index) + room_mm
    else:
        longest_mm = MAX_GAP_MM
    return shortest_mm, max(shortest_mm, longest_mm)


class _GapTrials:
    """Traces the selected field with the gap set to trial values, for one search.

    The Optiland system is built once. Each trial changes only the thicknesses that the gap
    sets (the gap itself, and for FOCUS the gap from the last lens to the image), which is
    far quicker than rebuilding it: about 1 ms a trial instead of 15.
    """

    def __init__(self, state: OpticalState, lens_index: int, goal: str) -> None:
        self.state = copy.deepcopy(replace(state, imported_from=None))  # the import record is large
        self.lens_index = lens_index
        self.goal = goal
        self.optic = optic_builder.build_optic(self.state)
        self.pupil_x, self.pupil_y = pupil_points("hexapolar", SPOT_RINGS)
        # Optiland's thickness of a surface is the distance to the next one, so the gap
        # behind a lens is the thickness of its last surface.
        last_lens = len(self.state.lenses) - 1
        self.image_gap_surface = self.state.surface_range(last_lens)[1]
        self.gap_surface = None
        if lens_index != gaps.SOURCE_GAP_INDEX:
            self.gap_surface = self.state.surface_range(lens_index)[1]

    @property
    def rays_launched(self) -> int:
        """How many rays each trial traces."""
        return int(self.pupil_x.size)

    def evaluate(self, gap_mm: float) -> tuple[float, int]:
        """The merit and the number of rays through, with the gap set to ``gap_mm``."""
        trial = copy.deepcopy(self.state)
        apply_gap(trial, self.lens_index, self.goal, gap_mm)
        if self.gap_surface is not None:
            self.optic.updater.set_thickness(gap_mm, self.gap_surface)
        # Moving the source plane needs no change to the system: the rays are launched
        # further from (or nearer to) the first lens instead, by ``launch_rays``.
        self.optic.updater.set_thickness(
            trial.image_z - trial.last_back_z(), self.image_gap_surface
        )

        beam = trial.selected_field
        rays = trace_rays(
            self.optic, launch_rays(trial, beam, self.optic, self.pupil_x, self.pupil_y)
        )
        through = transmitted(rays)
        count = int(through.sum())
        if count == 0:
            return math.nan, 0
        if self.goal == COLLIMATE:
            # RMS angle to the mean direction, from the direction cosines (the sines of
            # the angles, which are small).
            direction_x = np.asarray(rays.L, dtype=float)[through]
            direction_y = np.asarray(rays.M, dtype=float)[through]
            spread = (direction_x - direction_x.mean()) ** 2 + (
                direction_y - direction_y.mean()
            ) ** 2
            return float(np.sqrt(spread.mean())) * 1e6, count
        x = np.asarray(rays.x, dtype=float)[through]
        y = np.asarray(rays.y, dtype=float)[through]
        spread = (x - x.mean()) ** 2 + (y - y.mean()) ** 2
        return float(np.sqrt(spread.mean())) * 1e3, count


def optimise_gap(
    state: OpticalState, lens_index: int, goal: str, smallest_source_gap_mm: float = 0.0
) -> GapOptimisation:
    """Find the gap behind ``lens_index`` that best meets the goal, without changing the bench.

    Parameters
    ----------
    state : OpticalState
        The bench. The selected field's beam is used.
    lens_index : int
        The lens in front of the gap, or ``gaps.SOURCE_GAP_INDEX`` for the gap from the
        source plane. The gap must be unlocked, and not the gap to the image plane.
    goal : str
        COLLIMATE or FOCUS (see the module docstring).
    smallest_source_gap_mm : float
        For the gap from the source plane: the shortest it may be.

    Returns
    -------
    GapOptimisation
        The best gap and its merit; apply it with ``apply_gap``.

    Raises
    ------
    ValueError
        If the gap cannot be optimised (``check_can_optimise``), or no light gets through
        at any gap in the allowed range.
    """
    check_can_optimise(state, lens_index, goal)
    gap_before_mm = current_gap_mm(state, lens_index)
    shortest_mm, longest_mm = gap_limits_mm(state, lens_index, goal, smallest_source_gap_mm)
    trials = _GapTrials(state, lens_index, goal)
    merit_before, through_before = trials.evaluate(gap_before_mm)

    # A coarse scan (which includes the current gap) finds the right valley, and the most
    # rays that get through anywhere in the range.
    trial_gaps = np.linspace(shortest_mm, longest_mm, COARSE_STEPS)
    if shortest_mm <= gap_before_mm <= longest_mm:
        trial_gaps = np.sort(np.append(trial_gaps, gap_before_mm))
    results = [trials.evaluate(float(gap)) for gap in trial_gaps]
    counts = np.array([count for _merit, count in results])
    most_through = int(counts.max())
    if most_through == 0:
        raise ValueError("No light gets through at any gap in the allowed range")

    # Gaps that lose rays are never chosen: their merit counts as infinite.
    merits = np.array(
        [merit if count == most_through else math.inf for merit, count in results], dtype=float
    )
    best = int(np.argmin(merits))

    def penalised_merit(gap_mm: float) -> float:
        merit, count = trials.evaluate(gap_mm)
        return merit if count == most_through else math.inf

    low = float(trial_gaps[max(best - 1, 0)])
    high = float(trial_gaps[min(best + 1, len(trial_gaps) - 1)])
    best_gap_mm, best_merit = float(trial_gaps[best]), float(merits[best])
    if high > low:
        refined = minimize_scalar(
            penalised_merit,
            bounds=(low, high),
            method="bounded",
            options={"xatol": GAP_TOLERANCE_MM},
        )
        if np.isfinite(refined.fun) and refined.fun < best_merit:
            best_gap_mm, best_merit = float(refined.x), float(refined.fun)

    at_bound = min(best_gap_mm - shortest_mm, longest_mm - best_gap_mm) < AT_BOUND_TOLERANCE_MM
    return GapOptimisation(
        lens_index=lens_index,
        goal=goal,
        gap_before_mm=gap_before_mm,
        gap_after_mm=best_gap_mm,
        merit_before=merit_before,
        merit_after=best_merit,
        merit_unit=MERIT_UNITS[goal],
        shortest_gap_mm=shortest_mm,
        longest_gap_mm=longest_mm,
        at_bound=bool(at_bound),
        rays_launched=trials.rays_launched,
        rays_through_before=through_before,
        rays_through_after=most_through,
    )
