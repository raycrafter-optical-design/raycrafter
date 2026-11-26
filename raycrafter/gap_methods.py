"""The ``Bench`` methods for the air gaps and the source plane.

``Bench`` (``raycrafter.api``) inherits these, so a script calls them on a bench. A gap is
measured vertex to vertex along the axis: behind each lens, to the next lens or (behind the
last) to the image plane, and in front of the first lens, from the source plane. The source
plane is where every field's diameter, offset, angle and vergence are defined: a light
source, a scan mirror or a stop. By default its gap is locked at 0 mm, so the fields are
defined at the first lens and move with it.

>>> from raycrafter import Bench
>>> bench = Bench()
>>> bench.source_gap_mm
0.0
>>> bench.lock_source_gap(False)
>>> bench.set_source_gap(50.0)  # the lens moves 50 mm away from the source plane
>>> bench.source_gap_mm, bench.lenses[0].front_z_mm
(50.0, 150.0)

Any gap can be locked, so that it keeps its length while the rest of the design changes.
``optimise_gap`` finds the gap that collimates the light (leaving the system, or a chosen
lens) or focuses on the image plane.
They are kept apart from ``raycrafter.api`` to keep that file a manageable size.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from raycrafter.core.gap_optimisation import GapOptimisation

if TYPE_CHECKING:
    from raycrafter.api import LensInfo


class GapMethods:
    """Setting and locking gaps, and placing the source plane, for ``Bench`` to inherit.

    Every method goes through ``Bench._call``, so it works the same on a headless bench and
    on one connected to a RayCrafter window.
    """

    if TYPE_CHECKING:  # provided by Bench

        def _call(self, tool_name: str, **arguments) -> dict: ...

        @property
        def lenses(self) -> list[LensInfo]: ...

    def set_gap(self, lens_index: int, gap_mm: float) -> LensInfo:
        """Set the air gap behind a lens and return that lens's snapshot.

        The gap runs from the lens's back vertex to the next lens's front vertex, or to the
        image plane behind the last lens. All later lenses and the image plane shift
        together, so no other gap changes.

        Raises
        ------
        ValueError
            If there is no such lens, the gap is locked (always so inside an assembly), or
            it is so small that the surfaces would overlap (the image plane needs at least
            1 mm).

        Example
        -------
        >>> from raycrafter import Bench
        >>> bench = Bench(empty=True)
        >>> _ = bench.add_paraxial_lens(100.0)
        >>> _ = bench.add_paraxial_lens(50.0)
        >>> bench.set_gap(0, 150.0).gap_after_mm
        150.0
        >>> bench.lenses[1].front_z_mm
        250.0
        """
        self._call("set_gap", lens_index=int(lens_index), gap_mm=float(gap_mm))
        return self.lenses[int(lens_index)]

    def lock_gap(self, lens_index: int, locked: bool = True) -> LensInfo:
        """Lock (or, with ``locked=False``, unlock) the gap behind a lens; return its snapshot.

        A locked gap keeps its length while the rest of the design is edited: lenses joined
        by locked gaps move together, and ``set_gap`` refuses to change it. Behind the last
        lens it is the gap to the image plane, which then moves with the last lens and
        cannot be moved on its own (``set_image_z`` and ``autofocus`` refuse). The gap
        inside an assembly is always locked.

        Example
        -------
        >>> from raycrafter import Bench
        >>> bench = Bench(empty=True)
        >>> _ = bench.add_paraxial_lens(100.0)
        >>> _ = bench.add_paraxial_lens(50.0)
        >>> bench.lock_gap(0).gap_after_locked
        True
        >>> bench.move_lens(1, front_z_mm=200.0).front_z_mm  # moves lens 0 too
        200.0
        >>> bench.lenses[0].front_z_mm
        190.0
        """
        self._call("lock_gap", lens_index=int(lens_index), locked=bool(locked))
        return self.lenses[int(lens_index)]

    @property
    def source_z_mm(self) -> float:
        """z of the source plane, where every field is defined (mm)."""
        return float(self._call("get_design")["source"]["z_mm"])

    @property
    def source_gap_mm(self) -> float | None:
        """Distance from the source plane to the first lens (mm); None with no lenses."""
        gap = self._call("get_design")["source"]["gap_to_first_lens_mm"]
        return None if gap is None else float(gap)

    @property
    def source_gap_locked(self) -> bool:
        """True if the source plane keeps its distance from the first lens and moves with it."""
        return bool(self._call("get_design")["source"]["gap_locked"])

    def set_source_gap(self, gap_mm: float) -> None:
        """Set the distance from the source plane to the first lens.

        Every lens and the image plane shift together; the source plane stays put. Changing
        it moves where an angled beam lands on the first lens by the change times
        tan(angle).

        Raises
        ------
        ValueError
            If the gap is locked (see ``lock_source_gap``), negative, or there are no lenses.
        """
        self._call("set_source_gap", gap_mm=float(gap_mm))

    def lock_source_gap(self, locked: bool = True) -> None:
        """Lock (or, with ``locked=False``, unlock) the gap from the source plane.

        While locked (the default), the source plane moves with the first lens, keeping its
        distance. While unlocked it stays put when lenses move, and no lens can be moved in
        front of it.
        """
        self._call("lock_source_gap", locked=bool(locked))

    def set_source_z(self, source_z_mm: float) -> None:
        """Move the source plane to ``source_z_mm``, which must not be behind the first lens.

        Raises
        ------
        ValueError
            If the gap from the source plane is locked, or the position is not between the
            start of the axis and the first lens.
        """
        self._call("set_source_z", source_z_mm=float(source_z_mm))

    def optimise_gap(
        self,
        lens_index: int,
        goal: str = "collimate",
        apply: bool = True,
        lock: bool = False,
        flip_if_better: bool = False,
        measure_after_lens_index: int | None = None,
    ) -> GapOptimisation:
        """Find, and by default set, the gap behind a lens that best meets a goal.

        Real rays of the selected field are traced, so aberrations and the beam's own width,
        angle, vergence and offset are allowed for. Gaps at which rays are clipped are never
        chosen. Check ``at_bound`` in the result: if True, the best gap is at an end of the
        allowed range, and there is probably no true solution within it.

        Parameters
        ----------
        lens_index : int
            The lens in front of the gap; -1 for the gap from the source plane. The gap must
            be unlocked, and not the gap to the image plane (use ``autofocus`` for that).
        goal : str
            "collimate": the rays leaving the last lens (or the one given by
            ``measure_after_lens_index``) are parallel (merit: their RMS angle, µrad); the
            later lenses and the image plane move together. "focus": the smallest
            RMS spot (µm) on the image plane, which stays where it is.
        apply : bool
            If False, only report the best gap; the bench is not changed.
        lock : bool
            If True, lock the gap once it is set.
        flip_if_better : bool
            Each lens next to the gap is also tried turned round. If that is clearly better,
            the result's ``flip_lens_index`` names the lens; with this True, it is flipped
            and its own best gap set. Lenses are never flipped otherwise.
        measure_after_lens_index : int, optional
            "collimate" only: a lens behind the gap after which the light is to be parallel;
            lenses behind it are ignored. By default the last lens.

        Returns
        -------
        GapOptimisation
            The gap and merit before and after (see ``raycrafter.core.gap_optimisation``).

        Example
        -------
        >>> from raycrafter import Bench
        >>> bench = Bench(empty=True)
        >>> for focal_length_mm in (100.0, 50.0):  # a 2x beam reducer
        ...     _ = bench.add_paraxial_lens(focal_length_mm)
        >>> round(bench.optimise_gap(0, "collimate").gap_after_mm, 3)
        150.0

        A light source 80 mm in front of the first lens is moved to that lens's focus,
        whatever follows:

        >>> _ = bench.set_finite_object()
        >>> bench.lock_source_gap(False)
        >>> bench.set_source_gap(80.0)
        >>> result = bench.optimise_gap(-1, "collimate", measure_after_lens_index=0)
        >>> round(result.gap_after_mm, 3)
        100.0
        """
        result = self._call(
            "optimise_gap",
            lens_index=int(lens_index),
            goal=goal,
            apply=bool(apply),
            lock=bool(lock),
            flip_if_better=bool(flip_if_better),
            measure_after_lens_index=(
                None if measure_after_lens_index is None else int(measure_after_lens_index)
            ),
        )["optimisation"]
        # JSON has no NaN, so a merit with no light came back as None.
        for name in ("merit_before", "merit_after"):
            if result[name] is None:
                result[name] = math.nan
        return GapOptimisation(**result)
