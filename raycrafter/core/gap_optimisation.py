"""Optimising one air gap for a goal: collimate the light, or focus on the image plane (#38).

Finding the right spacing by eye is slow: e.g. an f = 100 mm lens then an f = 50 mm lens as
a beam reducer collimates only at one gap. ``optimise_gap`` finds that gap with real rays
across the selected field's beam (not paraxial focal lengths), so it allows for aberrations,
the beam's own width, angle, vergence and offset, and rays clipped by lens rims. It works on
a copy of the bench and does not change it; ``apply_gap`` sets the gap it found. It has no
Qt code, and runs only on request (never while something is dragged).

The goals
---------
- **collimate**: the rays leaving a chosen lens are parallel. By default that is the last
  lens (the system's output); any lens behind the gap can be chosen instead (#66), e.g. to
  set the first lens so that collimated light leaves it before the next one is adjusted.
  Lenses behind the chosen one then play no part. The merit is the RMS angle of the rays to
  their mean direction, in microradians. The lenses behind the gap move together with the
  image plane, as when a gap is typed.
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

from raycrafter.core import assemblies, gaps, optic_builder, placement
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
MEASUREMENT_IMAGE_GAP_MM = 10.0  # "collimate" traces to a plane this far behind the lens

# Flipping a lens next to the gap is suggested (#56) only if it is clearly better: the merit
# at most this fraction of the unflipped one, and better by at least this much.
FLIP_BETTER_FRACTION = 0.5
FLIP_MINIMUM_GAIN = {COLLIMATE: 5.0, FOCUS: 1.0}  # µrad, µm


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
    measured_after_lens_index : int or None
        For COLLIMATE, the lens after which the rays were measured (the last lens for the
        system's output). None for FOCUS, which is measured on the image plane.
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
    flip_lens_index : int or None
        A lens next to the gap that does clearly better turned round (#56), or None. A lens
        the wrong way round cannot be fixed by spacing.
    flipped_gap_mm, flipped_merit : float or None
        The best gap and its merit with that lens flipped; None if there is no suggestion.
    """

    lens_index: int
    goal: str
    measured_after_lens_index: int | None
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
    flip_lens_index: int | None = None
    flipped_gap_mm: float | None = None
    flipped_merit: float | None = None


def current_gap_mm(state: OpticalState, lens_index: int) -> float:
    """The width of the gap behind lens ``lens_index`` (or from the source plane), in mm."""
    if lens_index == gaps.SOURCE_GAP_INDEX:
        source_gap = gaps.source_gap(state)
        if source_gap is None:
            raise ValueError("There is no lens, so there is no gap to optimise")
        return source_gap.width_mm
    return gaps.air_gaps(state)[lens_index].width_mm


def element_end_behind_gap(state: OpticalState, lens_index: int) -> int:
    """The last lens of the element behind a gap: the next lens, or all of its assembly.

    ``lens_index`` is the lens in front of the gap, or ``gaps.SOURCE_GAP_INDEX``. The gap
    must not be the one to the image plane. This is where "collimate after the element"
    measures the rays.
    """
    first_behind = 0 if lens_index == gaps.SOURCE_GAP_INDEX else lens_index + 1
    return assemblies.assembly_indices(state, first_behind)[-1]


def check_can_optimise(
    state: OpticalState,
    lens_index: int,
    goal: str,
    measure_after_lens_index: int | None = None,
) -> None:
    """Raise ValueError, saying why, if this gap cannot be optimised for this goal.

    The parameters are those of ``optimise_gap``.
    """
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
    if measure_after_lens_index is None:
        return
    if goal == FOCUS:
        raise ValueError(
            "Focus is measured on the image plane: only collimate can be measured after a lens"
        )
    first_behind = 0 if lens_index == gaps.SOURCE_GAP_INDEX else lens_index + 1
    if not first_behind <= measure_after_lens_index < len(state.lenses):
        raise ValueError(
            f"The light can be measured after lenses {first_behind} to "
            f"{len(state.lenses) - 1}, which are behind this gap, not lens "
            f"{measure_after_lens_index}"
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


def apply_optimisation(
    state: OpticalState, result: GapOptimisation, lock: bool = False, flip: bool = False
) -> None:
    """Set the gap ``optimise_gap`` found and, if ``lock``, lock it.

    With ``flip``, first turn round the lens the result suggests flipping, and set the gap
    found for that orientation instead.

    Raises
    ------
    ValueError
        If ``flip`` is True but the result suggests no flip.
    """
    gap_mm = result.gap_after_mm
    if flip:
        if result.flip_lens_index is None or result.flipped_gap_mm is None:
            raise ValueError("Flipping a lens would not do clearly better here")
        assemblies.flip_lens(state, result.flip_lens_index)
        gap_mm = result.flipped_gap_mm
    apply_gap(state, result.lens_index, result.goal, gap_mm)
    if lock:
        if result.lens_index == gaps.SOURCE_GAP_INDEX:
            state.source_gap_locked = True
        else:
            gaps.set_gap_locked(state, result.lens_index, True)


def gap_limits_mm(state: OpticalState, lens_index: int, goal: str) -> tuple[float, float]:
    """The shortest and longest gap the search may try (mm).

    The gap from the source plane must keep the first lens FIRST_LENS_MIN_Z from the start
    of the axis (``raycrafter.core.placement``).
    """
    if lens_index == gaps.SOURCE_GAP_INDEX:
        shortest_mm = max(0.0, placement.FIRST_LENS_MIN_Z - state.source_z)
    else:
        shortest_mm = gaps.minimum_gap_mm(state, lens_index)
    if goal == FOCUS:
        # The lenses behind the gap move towards the image plane, which stays put.
        room_mm = state.image_z - gaps.MIN_IMAGE_GAP_MM - state.last_back_z()
        longest_mm = current_gap_mm(state, lens_index) + room_mm
    else:
        longest_mm = MAX_GAP_MM
    return shortest_mm, max(shortest_mm, longest_mm)


def _bench_up_to(state: OpticalState, last_lens_index: int) -> OpticalState:
    """A copy of the bench that ends with lens ``last_lens_index``, for measuring there.

    Light leaving that lens is not affected by anything behind it, and leaving those lenses
    out also stops their rims counting as clipping. The image plane is put just behind the
    lens: the directions measured do not depend on where it is.
    """
    shortened = copy.deepcopy(replace(state, imported_from=None))  # the import record is large
    del shortened.lenses[last_lens_index + 1 :]
    shortened.lenses[-1].locked_to_next = False
    shortened.image_z = shortened.last_back_z() + MEASUREMENT_IMAGE_GAP_MM
    shortened.image_gap_locked = False
    return shortened


class _GapTrials:
    """Traces the selected field with the gap set to trial values, for one search.

    The Optiland system is built once. Each trial changes only the thicknesses that the gap
    sets (the gap itself, and for FOCUS the gap from the last lens to the image), which is
    far quicker than rebuilding it: about 1 ms a trial instead of 15.
    """

    def __init__(
        self,
        state: OpticalState,
        lens_index: int,
        goal: str,
        measure_after_lens_index: int | None,
    ) -> None:
        if measure_after_lens_index is None:  # focus: on the image plane, behind every lens
            self.state = copy.deepcopy(replace(state, imported_from=None))  # the record is large
        else:
            self.state = _bench_up_to(state, measure_after_lens_index)
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
    state: OpticalState,
    lens_index: int,
    goal: str,
    consider_flips: bool = True,
    measure_after_lens_index: int | None = None,
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
    consider_flips : bool
        If True, also optimise with each lens next to the gap turned round, and suggest
        the flip in the result if it is clearly better (``suggest_flip``).
    measure_after_lens_index : int, optional
        For COLLIMATE only: the lens, behind the gap, after which the rays are to be
        parallel (``element_end_behind_gap`` gives the one next to the gap). By default the
        last lens: the output of the whole system.

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
    check_can_optimise(state, lens_index, goal, measure_after_lens_index)
    if goal == COLLIMATE and measure_after_lens_index is None:
        measure_after_lens_index = len(state.lenses) - 1
    gap_before_mm = current_gap_mm(state, lens_index)
    shortest_mm, longest_mm = gap_limits_mm(state, lens_index, goal)
    trials = _GapTrials(state, lens_index, goal, measure_after_lens_index)
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
    result = GapOptimisation(
        lens_index=lens_index,
        goal=goal,
        measured_after_lens_index=measure_after_lens_index,
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
    if consider_flips:
        result = suggest_flip(state, result)
    return result


def flippable_neighbours(state: OpticalState, lens_index: int) -> list[int]:
    """The lenses next to the gap that can be turned round on their own.

    Paraxial lenses are the same both ways round, and a lens in an assembly can only be
    flipped with the whole assembly, so neither is included.
    """
    if lens_index == gaps.SOURCE_GAP_INDEX:
        neighbours = [0]
    else:
        neighbours = [lens_index, lens_index + 1]
    return [
        index
        for index in neighbours
        if not state.lenses[index].is_paraxial and state.lenses[index].assembly_id is None
    ]


def suggest_flip(state: OpticalState, result: GapOptimisation) -> GapOptimisation:
    """Optimise again with each lens next to the gap flipped; add a suggestion if clearly better.

    A flip is suggested if, at its own best gap, its merit is at most FLIP_BETTER_FRACTION
    of ``result.merit_after`` and better by at least FLIP_MINIMUM_GAIN, with as many rays
    getting through. Of two such flips, the better one is suggested. The bench is not changed.

    Returns
    -------
    GapOptimisation
        ``result``, with ``flip_lens_index``, ``flipped_gap_mm`` and ``flipped_merit`` set
        if a flip is suggested.
    """
    best: GapOptimisation | None = None
    best_index = None
    for index in flippable_neighbours(state, result.lens_index):
        flipped = copy.deepcopy(replace(state, imported_from=None))
        flipped.lenses[index].flipped = not flipped.lenses[index].flipped
        try:
            trial = optimise_gap(
                flipped,
                result.lens_index,
                result.goal,
                consider_flips=False,
                measure_after_lens_index=result.measured_after_lens_index,
            )
        except ValueError:
            continue  # e.g. no light gets through this way round
        if trial.rays_through_after < result.rays_through_after:
            continue
        if best is None or trial.merit_after < best.merit_after:
            best, best_index = trial, index

    if best is None:
        return result
    clearly_better = (
        best.merit_after <= FLIP_BETTER_FRACTION * result.merit_after
        and result.merit_after - best.merit_after >= FLIP_MINIMUM_GAIN[result.goal]
    )
    if not clearly_better:
        return result
    return replace(
        result,
        flip_lens_index=best_index,
        flipped_gap_mm=best.gap_after_mm,
        flipped_merit=best.merit_after,
    )
