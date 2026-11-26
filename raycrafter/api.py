"""Scripting RayCrafter from Python: ``Bench``, a lens bench you drive with method calls.

A ``Bench`` can do everything the GUI (except undo) and the AI agent tools can: search the lens
catalogue, place, move, flip and delete lenses, set and lock the gaps between them, make
Plossl pairs, edit the fields (beams), move the image plane, autofocus, run the analyses, draw
the plots, write the HTML report, save and open project files, and import and export Zemax
.zmx files.
It does not import Qt.

There are two kinds of bench:

- ``Bench()`` is *headless*: the design lives in your Python process and nothing is shown.
- ``Bench.connect()`` drives the RayCrafter window that is open on this computer. Every edit
  happens in the window, which redraws as it would for a drag, so you can watch a script
  work, and the script sees changes made by hand in the window. This uses the same local link
  as the AI agent server (``raycrafter.mcp_server.link``).

Both kinds behave the same. Edits go through the same code as the agent tools
(``raycrafter.mcp_server.tools.DesignTools``), so the rules are those of the GUI: lenses never
overlap, a dragged lens swaps places with a neighbour it is pushed past the middle of, and so
on. A request that makes no sense (an unknown part number, an index with no lens) raises
``ValueError`` with an explanation. Analyses always run in your process, on a copy of the
design, and return full-resolution NumPy results.

Conventions
-----------
- Distances are in mm, measured along the optical axis (z) from the start of the axis, where
  the beams start. Light travels towards larger z.
- Angles are in degrees, vergence in dioptres (1/m, positive converging), wavelengths in nm.
- Lenses are numbered from 0 in order along the axis; fields from 0 in the order they were
  added. A lens's index changes if it is moved past a neighbour or a lens is added in front
  of it, so read ``bench.lenses`` again after an edit rather than keeping indices.
- ``bench.lenses`` and ``bench.fields`` return snapshots (``LensInfo``, ``FieldInfo``):
  plain records that do not change when the design does. To change the design, call a
  ``Bench`` method.
- One field is *selected*. Autofocus works on it, and so do the single-field analyses
  (through-focus, MTF, ray fan, field curvature) unless you pass ``field_index``.

Example
-------
>>> from raycrafter import Bench
>>> bench = Bench(empty=True)
>>> first = bench.add_lens("AC254-100-A")  # placed at z = 100 mm
>>> second = bench.add_lens("AC254-050-A", front_z_mm=150)
>>> second
LensInfo(index=1, part_number='AC254-050-A', front_z_mm=150.0, back_z_mm=161.5, ...)
>>> bench.set_field(0, diameter_mm=8, wavelength_nm=633)
FieldInfo(index=0, diameter_mm=8.0, angle_deg=0.0, ...)
>>> round(bench.autofocus(), 1)  # image plane to the best focus
180.7
>>> summary = bench.summary()
>>> round(summary.effective_focal_length_mm, 1), round(summary.f_number, 2)
(48.5, 6.06)
>>> figure = bench.plot("spot_diagram")  # a matplotlib Figure
>>> figure.savefig("spot.png")  # doctest: +SKIP

Driving the window instead (RayCrafter must be running, with its agent link on):

>>> bench = Bench.connect()  # doctest: +SKIP
>>> bench.move_lens(0, front_z_mm=120)  # doctest: +SKIP
"""

from __future__ import annotations

import copy
import webbrowser
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from matplotlib.figure import Figure

from raycrafter.core import analysis, gaps
from raycrafter.core.model import OpticalModel
from raycrafter.core.state import OpticalState, state_from_dict
from raycrafter.design_files import DesignFileMethods
from raycrafter.lenscatalog import thorlabs
from raycrafter.lenscatalog.catalog import CatalogEntry
from raycrafter.lenscatalog.selection import search_catalog
from raycrafter.mcp_server.link import LinkClient, LinkUnavailable
from raycrafter.mcp_server.tools import DesignTools
from raycrafter.report import build_report_html, figures

PLOT_KINDS = figures.PLOT_KINDS


# ----------------------------------------------------------------------------- snapshots


@dataclass(frozen=True)
class LensInfo:
    """A snapshot of one lens on the bench, as returned by ``Bench.lenses``.

    Attributes
    ----------
    index : int
        Position in the row of lenses, from 0 at the left (smallest z).
    part_number : str
        Catalogue part number, e.g. ``"AC254-100-A"``. A paraxial lens is called e.g.
        ``"Paraxial f=+100 mm"``; a lens loaded from a ZMX file that matched no catalogue
        part is called e.g. ``"Imported lens 1"``.
    front_z_mm, back_z_mm : float
        z of the front and back vertices.
    thickness_mm : float
        Front-to-back vertex thickness (0 for a paraxial lens).
    clear_diameter_mm : float
        The diameter that lets light through; rays outside it are blocked.
    focal_length_mm : float or None
        Effective focal length (negative for a diverging lens), from the catalogue or the
        paraxial lens; None for an imported lens that is not in the catalogue.
    flipped : bool
        True if the lens is turned round relative to the catalogue.
    plossl_group : int or None
        Lenses that share a number form a Plossl pair and move together.
    is_paraxial : bool
        True for an ideal, aberration-free thin lens.
    gap_after_mm : float
        Air gap from the back vertex to the next lens's front vertex, or to the image plane
        for the last lens (see ``Bench.set_gap``).
    gap_after_locked : bool
        True if that gap is locked (see ``Bench.lock_gap``).
    vendor : str or None
        Who makes the lens, e.g. ``"Thorlabs"``; None for an imported or paraxial lens.
    mount_type : str or None
        ``"Mounted"`` or ``"Unmounted"``; None if unknown, as for most catalogue lenses.
    surface_range : tuple of (int, int)
        Numbers of the lens's first and last surface, counted as in the exported ZMX file
        (the object is surface 0).
    """

    index: int
    part_number: str
    front_z_mm: float
    back_z_mm: float
    thickness_mm: float
    clear_diameter_mm: float
    focal_length_mm: float | None
    flipped: bool
    plossl_group: int | None
    is_paraxial: bool
    gap_after_mm: float
    gap_after_locked: bool
    vendor: str | None
    mount_type: str | None
    surface_range: tuple[int, int]


@dataclass(frozen=True)
class FieldInfo:
    """A snapshot of one field (a beam entering from the left), as in ``Bench.fields``.

    The beam's diameter and offset are measured where it meets the front vertex of the first
    lens (or the start of the axis when there are no lenses).

    Attributes
    ----------
    index : int
        The field's number, from 0.
    diameter_mm : float
        Beam diameter.
    angle_deg : float
        Angle of the beam's central ray to the axis (positive rises).
    vergence_dioptres : float
        Positive is converging, negative diverging, 0 collimated.
    offset_mm : float
        Height of the beam centre above the axis.
    wavelength_nm : float
        The wavelength it is traced at.
    colour : str
        The hex colour it is drawn in, e.g. ``"#3b7dd8"``.
    colour_follows_wavelength : bool
        True if ``colour`` is the colour of the wavelength rather than one chosen for it.
    selected : bool
        True for the field that autofocus and the single-field analyses use.
    """

    index: int
    diameter_mm: float
    angle_deg: float
    vergence_dioptres: float
    offset_mm: float
    wavelength_nm: float
    colour: str
    colour_follows_wavelength: bool
    selected: bool


@dataclass(frozen=True)
class DesignSummary:
    """First-order properties and spot sizes of the design, as returned by ``Bench.summary``.

    A value is NaN when it cannot be computed because no light gets through, and the
    first-order values are None when there are no lenses.

    Attributes
    ----------
    effective_focal_length_mm, back_focal_length_mm : float or None
        Of the whole lens train; the back focal length is measured from the last vertex.
    f_number : float or None
        From the part of the beam that gets through every lens, for the selected field.
    transmitted_beam_diameter_mm : float or None
        How much of the selected field's beam gets through every lens.
    beam_is_clipped : bool or None
        True if a lens rim cuts the selected field's beam.
    image_z_mm : float
        Where the image plane is.
    rms_spot_radius_um : list of float
        RMS spot radius at the image plane, one per field.
    airy_radius_um : list of float or None
        Airy disc radius, one per field; None for a field that is not focused (parallel
        light reaching the image plane).
    """

    effective_focal_length_mm: float | None
    back_focal_length_mm: float | None
    f_number: float | None
    transmitted_beam_diameter_mm: float | None
    beam_is_clipped: bool | None
    image_z_mm: float
    rms_spot_radius_um: list[float]
    airy_radius_um: list[float | None]


# ----------------------------------------------------------------------------- the bench


class Bench(DesignFileMethods):
    """A lens bench: a row of lenses on an optical axis, beams of light and an image plane.

    ``Bench()`` makes a headless bench in this process. To drive the open RayCrafter window
    instead, use ``Bench.connect()``; to start from a file, ``Bench.from_project(path)`` or
    ``Bench.from_zmx(path)``. Saving, opening, importing and exporting are described in
    ``raycrafter.design_files``.

    Parameters
    ----------
    empty : bool
        If False (the default) the bench starts as a new RayCrafter window does: one
        ACT508-200-A achromat at z = 100 mm and one 10 mm collimated field at 555 nm. If
        True it starts with no lenses.

    Example
    -------
    >>> bench = Bench()
    >>> [lens.part_number for lens in bench.lenses]
    ['ACT508-200-A']
    >>> Bench(empty=True).lenses
    []
    """

    def __init__(self, empty: bool = False) -> None:
        model = OpticalModel(add_default_lens=not empty)
        self._tools: DesignTools | None = DesignTools(model)
        self._link: LinkClient | None = None
        self._catalog = model.catalog

    @classmethod
    def connect(cls, endpoint_path: str | Path | None = None) -> Bench:
        """A bench that drives the RayCrafter window open on this computer.

        Every edit is carried out by the window, which redraws so you can watch. Reading the
        design or running an analysis copies the window's current design, including any
        change made by hand, and works on the copy in this process.

        Parameters
        ----------
        endpoint_path : str or Path, optional
            The window's link file. Leave it out to use the usual one
            (``~/.raycrafter/agent_link.json``, or ``$RAYCRAFTER_LINK_FILE``).

        Raises
        ------
        ConnectionError
            If no RayCrafter window answers. Start one with ``raycrafter``; the link is on
            unless it was started with ``--no-agent-link``.

        Example
        -------
        >>> bench = Bench.connect()  # doctest: +SKIP
        >>> bench.is_connected  # doctest: +SKIP
        True
        """
        bench = cls.__new__(cls)
        bench._tools = None
        bench._link = LinkClient(None if endpoint_path is None else Path(endpoint_path))
        bench._catalog = thorlabs.load_lenses()  # the same packaged catalogue the window has
        bench._call("get_design")  # fails here, with a clear message, if there is no window
        return bench

    @classmethod
    def from_project(cls, file_path: str | Path) -> Bench:
        """A headless bench holding the design in a RayCrafter project (.raycrafter) file.

        Example
        -------
        >>> bench = Bench.from_project("my_design.raycrafter")  # doctest: +SKIP
        """
        bench = cls(empty=True)
        bench.load_project(file_path)
        return bench

    @classmethod
    def from_zmx(cls, file_path: str | Path) -> Bench:
        """A headless bench holding the design in a Zemax .zmx file.

        Lenses that match a catalogue part become that part; others keep their prescription
        as "Imported lens N". See ``load_zmx`` for the notes on anything simplified.

        Example
        -------
        >>> bench = Bench.from_zmx("my_design.zmx")  # doctest: +SKIP
        """
        bench = cls(empty=True)
        bench.load_zmx(file_path)
        return bench

    @property
    def is_connected(self) -> bool:
        """True if this bench drives a RayCrafter window, False if it is headless."""
        return self._link is not None

    def __repr__(self) -> str:
        """A short description, e.g. ``<Bench (headless): 1 lens, 1 field, image at 310 mm>``."""
        state = self._state()
        kind = "connected to the RayCrafter window" if self.is_connected else "headless"
        lens_count, field_count = len(state.lenses), len(state.fields)
        return (
            f"<Bench ({kind}): {lens_count} lens{'es' if lens_count != 1 else ''}, "
            f"{field_count} field{'s' if field_count != 1 else ''}, "
            f"image at {state.image_z:.1f} mm>"
        )

    # ------------------------------------------------------------------ the catalogue

    @property
    def catalog(self) -> dict[str, CatalogEntry]:
        """Every lens that can be placed: part number -> ``CatalogEntry``.

        Example
        -------
        >>> entry = Bench(empty=True).catalog["LA1509-A"]
        >>> entry.focal_length, entry.clear_diameter_mm
        (100.0, 25.4)
        """
        return self._catalog

    def search_lenses(
        self,
        query: str = "",
        min_focal_length_mm: float | None = None,
        max_focal_length_mm: float | None = None,
        min_diameter_mm: float | None = None,
        category: str | None = None,
    ) -> list[CatalogEntry]:
        """Catalogue lenses that pass every given filter, shortest |focal length| first.

        Parameters
        ----------
        query : str
            Text that must appear in the part number, product family or coating (case does
            not matter), e.g. ``"AC254"``, ``"plano-convex"`` or ``"B:"`` for B coating.
        min_focal_length_mm, max_focal_length_mm : float, optional
            Limits on the signed focal length: diverging lenses have negative ones.
        min_diameter_mm : float, optional
            Smallest acceptable clear diameter.
        category : str, optional
            One of ``"Spherical lenses"``, ``"Achromats"``, ``"Air-spaced doublets"``,
            ``"Achromatic triplets"``, ``"Matched achromatic pairs"``.

        Returns
        -------
        list of CatalogEntry
            Each has ``part_number``, ``focal_length``, ``clear_diameter_mm``, ``category``,
            ``family`` and ``coating``.

        Example
        -------
        >>> bench = Bench(empty=True)
        >>> found = bench.search_lenses("AC254", min_focal_length_mm=49, max_focal_length_mm=51)
        >>> sorted(entry.part_number for entry in found)
        ['AC254-050-A', 'AC254-050-AB', 'AC254-050-B', 'AC254-050-C']
        """
        return search_catalog(
            self._catalog.values(),
            query,
            min_focal_length_mm,
            max_focal_length_mm,
            min_diameter_mm,
            category,
        )

    # ------------------------------------------------------------------ reading the design

    @property
    def lenses(self) -> list[LensInfo]:
        """Snapshots of the lenses, left to right along the axis.

        Example
        -------
        >>> lens = Bench().lenses[0]
        >>> lens.part_number, lens.front_z_mm
        ('ACT508-200-A', 100.0)
        """
        state = self._state()
        return [self._lens_info(state, index) for index in range(len(state.lenses))]

    @property
    def fields(self) -> list[FieldInfo]:
        """Snapshots of the fields (beams), in order. There is always at least one.

        Example
        -------
        >>> field = Bench().fields[0]
        >>> field.diameter_mm, field.wavelength_nm, field.selected
        (10.0, 555.0, True)
        """
        state = self._state()
        return [self._field_info(state, index) for index in range(len(state.fields))]

    @property
    def image_z_mm(self) -> float:
        """z of the image plane. Change it with ``set_image_z`` or ``autofocus``."""
        return float(self._state().image_z)

    @property
    def selected_field_index(self) -> int:
        """The field that autofocus and the single-field analyses use (see ``select_field``)."""
        return self._state().selected_field_index

    def state(self) -> OpticalState:
        """A copy of the complete design as RayCrafter stores it, every surface included.

        For advanced use, e.g. with the functions in ``raycrafter.core.analysis``. Changing
        the copy does not change the bench.
        """
        return copy.deepcopy(self._state())

    # ------------------------------------------------------------------ editing lenses

    def add_lens(self, part_number: str, front_z_mm: float | None = None) -> LensInfo:
        """Put a catalogue lens on the bench and return its snapshot.

        Parameters
        ----------
        part_number : str
            A key of ``catalog``, e.g. ``"AC254-100-A"`` (see ``search_lenses``).
        front_z_mm : float, optional
            Where its front vertex should go. Lenses never overlap, so it goes to the
            nearest place with room, which may be in another gap. Left out, the lens goes
            10 mm after the last lens (or at z = 100 mm on an empty bench).

        If the lens ends up behind the image plane, the image plane is moved back.

        Raises
        ------
        ValueError
            If the part number is unknown or the lens fits nowhere.

        Example
        -------
        >>> bench = Bench(empty=True)
        >>> bench.add_lens("LA1509-A").front_z_mm
        100.0
        >>> bench.add_lens("LA1509-A", front_z_mm=40).index  # goes in front of the first
        0
        """
        if front_z_mm is None:
            result = self._call("add_lens", part_number=part_number)
        else:
            result = self._call(
                "insert_lens", part_number=part_number, front_z_mm=float(front_z_mm)
            )
        return self.lenses[result["new_lens_index"]]

    def add_paraxial_lens(
        self, focal_length_mm: float, front_z_mm: float | None = None
    ) -> LensInfo:
        """Put an ideal thin lens on the bench and return its snapshot.

        A paraxial lens focuses perfectly (no aberrations), has zero thickness and a 50.8 mm
        (2 inch) clear diameter. It is useful for sketching a design before choosing parts.

        Parameters
        ----------
        focal_length_mm : float
            Non-zero; negative for a diverging lens.
        front_z_mm : float, optional
            As for ``add_lens``.

        Example
        -------
        >>> bench = Bench(empty=True)
        >>> bench.add_paraxial_lens(150.0).part_number
        'Paraxial f=+150 mm'
        """
        arguments = {"focal_length_mm": float(focal_length_mm)}
        if front_z_mm is not None:
            arguments["front_z_mm"] = float(front_z_mm)
        result = self._call("add_paraxial_lens", **arguments)
        return self.lenses[result["new_lens_index"]]

    def move_lens(self, lens_index: int, front_z_mm: float) -> LensInfo:
        """Move a lens towards ``front_z_mm``, as dragging it in the window would.

        Lenses joined by locked gaps (such as a Plossl pair) move together. The lens stops
        when it meets a neighbour or the image plane, so check the returned ``front_z_mm``
        to see where it went. If the request takes the lens past the middle of a neighbour,
        the two swap places, and the returned snapshot has the lens's new index.

        Example
        -------
        >>> bench = Bench(empty=True)
        >>> _ = bench.add_lens("LA1509-A")
        >>> bench.move_lens(0, front_z_mm=150).front_z_mm
        150.0
        """
        result = self._call("move_lens", lens_index=int(lens_index), front_z_mm=float(front_z_mm))
        return self.lenses[result["moved_lens_index"]]

    def set_gap(self, lens_index: int, gap_mm: float) -> LensInfo:
        """Set the air gap behind a lens and return that lens's snapshot.

        The gap runs from the lens's back vertex to the next lens's front vertex, or to the
        image plane behind the last lens. All later lenses and the image plane shift
        together, so no other gap changes.

        Raises
        ------
        ValueError
            If there is no such lens, the gap is locked (always so inside a Plossl pair), or
            it is so small that the surfaces would overlap (the image plane needs at least
            1 mm).

        Example
        -------
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
        inside a Plossl pair is always locked.

        Example
        -------
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

    def flip_lens(self, lens_index: int) -> LensInfo:
        """Turn a lens round and return its snapshot.

        Paraxial lenses (the same both ways) and lenses in a Plossl pair are left as they are.

        Example
        -------
        >>> Bench().flip_lens(0).flipped
        True
        """
        self._call("flip_lens", lens_index=int(lens_index))
        return self.lenses[int(lens_index)]

    def delete_lens(self, lens_index: int) -> None:
        """Take a lens off the bench. If it was in a Plossl pair, its partner stays.

        Example
        -------
        >>> bench = Bench()
        >>> bench.delete_lens(0)
        >>> bench.lenses
        []
        """
        self._call("delete_lens", lens_index=int(lens_index))

    def make_plossl(self, lens_index: int) -> None:
        """Snap a lens and an identical neighbour into a Plossl pair.

        The neighbour must be the same positive catalogue lens, no further away than twice
        its thickness, and not already paired. The two are turned to face each other with
        their positive elements inside and put 1 mm apart; later lenses and the image plane
        shift to make room. The pair then moves as one.

        Raises
        ------
        ValueError
            If the lens has no such neighbour.

        Example
        -------
        >>> bench = Bench(empty=True)
        >>> _ = bench.add_lens("AC254-050-A")
        >>> _ = bench.add_lens("AC254-050-A")
        >>> bench.make_plossl(0)
        >>> [lens.plossl_group for lens in bench.lenses]
        [1, 1]
        """
        self._call("make_plossl", lens_index=int(lens_index))

    def split_plossl(self, lens_index: int) -> None:
        """Undo a Plossl pair so its two lenses move independently. They stay where they are."""
        self._call("split_plossl", lens_index=int(lens_index))

    # ------------------------------------------------------------------ editing fields

    def set_field(
        self,
        field_index: int,
        diameter_mm: float | None = None,
        angle_deg: float | None = None,
        vergence_dioptres: float | None = None,
        offset_mm: float | None = None,
        wavelength_nm: float | None = None,
        colour: str | None = None,
    ) -> FieldInfo:
        """Change any of a field's properties; those left out stay as they are.

        Parameters
        ----------
        field_index : int
            Which field (see ``fields``).
        diameter_mm : float, optional
            Beam diameter at the first lens; must be positive.
        angle_deg : float, optional
            Angle of the beam to the axis (positive rises).
        vergence_dioptres : float, optional
            Positive converging, negative diverging, 0 collimated.
        offset_mm : float, optional
            Height of the beam centre above the axis at the first lens.
        wavelength_nm : float, optional
            Must be positive.
        colour : str, optional
            A hex colour such as ``"#ff8800"``, or ``"wavelength"`` to go back to the colour
            of its wavelength.

        Returns
        -------
        FieldInfo
            The field after the change.

        Example
        -------
        >>> bench = Bench()
        >>> bench.set_field(0, angle_deg=2.0, colour="#ff8800").colour
        '#ff8800'
        """
        changes = {
            "diameter_mm": diameter_mm,
            "angle_deg": angle_deg,
            "vergence_dioptres": vergence_dioptres,
            "offset_mm": offset_mm,
            "wavelength_nm": wavelength_nm,
        }
        arguments: dict[str, float | str] = {
            name: float(value) for name, value in changes.items() if value is not None
        }
        if colour is not None:
            arguments["colour"] = colour
        self._call("set_field", field_index=int(field_index), **arguments)
        return self.fields[int(field_index)]

    def add_field(self, **changes) -> FieldInfo:
        """Add a field, a copy of the selected one, and select it.

        Starting from a copy means the new field already suits the bench; pass any of the
        ``set_field`` properties to change them at once. A new field is drawn in the colour
        of its wavelength.

        Example
        -------
        >>> bench = Bench()
        >>> new_field = bench.add_field(angle_deg=1.5)
        >>> new_field.index, new_field.angle_deg, new_field.diameter_mm, new_field.selected
        (1, 1.5, 10.0, True)
        """
        previously_selected = self.selected_field_index
        self._call("add_field")
        new_index = len(self._state().fields) - 1
        if not changes:
            return self.fields[new_index]
        try:
            return self.set_field(new_index, **changes)
        except (ValueError, TypeError):  # a bad value or property name: leave no half-made field
            self.remove_fields([new_index])
            self.select_field(previously_selected)
            raise

    def remove_fields(self, field_indices: list[int]) -> None:
        """Remove fields by index. At least one field always remains (the first, if all
        are listed).

        Example
        -------
        >>> bench = Bench()
        >>> _ = bench.add_field(wavelength_nm=486)
        >>> bench.remove_fields([0])
        >>> [field.wavelength_nm for field in bench.fields]
        [486.0]
        """
        self._call("remove_fields", field_indices=[int(index) for index in field_indices])

    def select_field(self, field_index: int) -> None:
        """Choose the field that autofocus and the single-field analyses use by default."""
        self._call("select_field", field_index=int(field_index))

    # ------------------------------------------------------------------ the image plane

    def set_image_z(self, image_z_mm: float) -> None:
        """Put the image plane at ``image_z_mm``, which must be behind the last lens.

        Refused if the gap to the image plane is locked (see ``lock_gap``).

        Example
        -------
        >>> bench = Bench()
        >>> bench.set_image_z(320.0)
        >>> bench.image_z_mm
        320.0
        """
        self._call("set_image_z", image_z_mm=float(image_z_mm))

    def autofocus(self) -> float:
        """Move the image plane to the smallest RMS spot of the selected field.

        Refused if the gap to the image plane is locked (see ``lock_gap``).

        Returns
        -------
        float
            The new image plane z, in mm.

        Example
        -------
        >>> bench = Bench()
        >>> round(bench.autofocus())  # near the back focal plane of the ACT508-200-A
        307
        """
        self._call("autofocus")
        return self.image_z_mm

    # ------------------------------------------------------------------ analyses

    def summary(self) -> DesignSummary:
        """First-order properties (focal lengths, f-number) and every field's spot size.

        Example
        -------
        >>> bench = Bench()
        >>> _ = bench.autofocus()
        >>> summary = bench.summary()
        >>> round(summary.effective_focal_length_mm), round(summary.f_number)
        (200, 20)
        """
        state = self._state()
        fast_trace = analysis.trace_fast(state)
        f_number = analysis.compute_f_number(state)
        first_order = fast_trace.summary
        airy_radii_mm = [analysis.airy_radius(spot) for spot in fast_trace.spots]

        def plain(value) -> float | None:  # NumPy scalars print badly; None stays None
            return None if value is None else float(value)

        return DesignSummary(
            effective_focal_length_mm=plain(first_order and first_order.effective_focal_length),
            back_focal_length_mm=plain(first_order and first_order.back_focal_length),
            f_number=plain(f_number and f_number.f_number),
            transmitted_beam_diameter_mm=plain(f_number and f_number.transmitted_beam_diameter),
            beam_is_clipped=None if f_number is None else bool(f_number.beam_is_clipped),
            image_z_mm=float(state.image_z),
            rms_spot_radius_um=[float(spot.rms_radius) * 1000.0 for spot in fast_trace.spots],
            airy_radius_um=[plain(radius and radius * 1000.0) for radius in airy_radii_mm],
        )

    def spots(self) -> list[analysis.SpotResult]:
        """The spot diagram of every field at the image plane, one ``SpotResult`` each.

        Positions are in mm: ``x`` and ``y`` relative to the spot's centroid, ``position_x``
        and ``position_y`` absolute. ``rms_radius`` is in mm. Blocked rays are left out.

        Example
        -------
        >>> spot = Bench().spots()[0]
        >>> spot.x.shape == spot.y.shape
        True
        """
        return analysis.trace_fast(self._state()).spots

    def through_focus(self, field_index: int | None = None) -> analysis.ThroughFocusResult:
        """RMS spot radius against image plane position for one field (default: selected).

        ``offsets`` (mm from the current image plane, positive further from the lenses),
        ``rms_current`` (mm), ``best_offset`` (mm) and ``airy_radius`` (mm or None).

        Example
        -------
        >>> result = Bench().through_focus()
        >>> result.offsets.shape == result.rms_current.shape
        True
        """
        state = self._state(field_index)
        return analysis.compute_through_focus(state, analysis.trace_fast(state))

    def mtf(self, field_index: int | None = None) -> analysis.MtfResult:
        """MTF of one field (default: selected) at the image plane.

        ``frequency`` (cycles/mm), ``tangential``, ``sagittal`` and ``diffraction_limit``;
        the arrays are empty if the beam is blocked.

        Example
        -------
        >>> bench = Bench()
        >>> _ = bench.autofocus()
        >>> mtf = bench.mtf()
        >>> float(mtf.tangential[0])  # 1 at zero frequency
        1.0
        """
        return analysis.compute_mtf(self._state(field_index))

    def ray_fan(self, field_index: int | None = None) -> analysis.RayFanResult:
        """Transverse ray aberration of one field (default: selected).

        ``pupil`` (-1 to 1), ``tangential_error`` and ``sagittal_error`` (mm from the chief
        ray; NaN where a ray is blocked).
        """
        return analysis.compute_ray_fan(self._state(field_index))

    def field_curvature(self, field_index: int | None = None) -> analysis.FieldCurvatureResult:
        """Best focus of tangential and sagittal fans as one field's beam is swung to 10 deg.

        ``angles`` (degrees), ``tangential_shift`` and ``sagittal_shift`` (mm from the image
        plane, positive behind it; NaN where too few rays get through).
        """
        return analysis.compute_field_curvature(self._state(field_index))

    def plot(self, kind: str, field_index: int | None = None) -> Figure:
        """Draw one analysis as a matplotlib ``Figure``, as the GUI and the report show it.

        Parameters
        ----------
        kind : str
            One of ``"ray_diagram"``, ``"spot_diagram"``, ``"through_focus"``, ``"mtf"``,
            ``"ray_fan"``, ``"field_curvature"`` (``PLOT_KINDS``).
        field_index : int, optional
            The field for the single-field plots; the selected one if left out.

        Example
        -------
        >>> figure = Bench().plot("mtf")
        >>> figure.savefig("mtf.png", dpi=150)  # doctest: +SKIP
        """
        return figures.draw_named_plot(self._state(field_index), kind)

    def write_report(self, file_path: str | Path, open_in_browser: bool = False) -> Path:
        """Write the HTML report (every plot, with explanations) and return where it went.

        Example
        -------
        >>> Bench().write_report("design.html", open_in_browser=True)  # doctest: +SKIP
        """
        model = OpticalModel(catalog=self._catalog, add_default_lens=False)
        model.state = self._state()
        report_path = Path(file_path).expanduser().resolve()
        report_path.write_text(build_report_html(model), encoding="utf-8")
        if open_in_browser:
            webbrowser.open(report_path.as_uri())
        return report_path

    # ------------------------------------------------------------------ internals

    def _call(self, tool_name: str, **arguments) -> dict:
        """Run one ``DesignTools`` call here (headless) or in the window (connected).

        Every call ``Bench`` makes returns a dictionary (only ``get_plot`` does not, and
        ``Bench`` draws its plots itself).
        """
        if self._tools is not None:
            return getattr(self._tools, tool_name)(**arguments)
        assert self._link is not None  # one or the other is always set
        try:
            return cast(dict, self._link.call(tool_name, arguments))
        except LinkUnavailable as error:
            raise ConnectionError(
                "No RayCrafter window is answering. Start RayCrafter (without "
                f"--no-agent-link), then call Bench.connect() again. ({error})"
            ) from error

    def _state(self, field_index: int | None = None) -> OpticalState:
        """The current design, with ``field_index`` selected if given.

        Headless, this is the model's own state, so callers must not change it. Connected,
        it is a fresh copy of the window's design. With ``field_index`` it is a shallow copy
        (the lenses and fields are shared) that differs only in the selected field.
        """
        if self._tools is not None:
            state = self._tools.model.state
        else:
            state = state_from_dict(self._call("get_state"))
        if field_index is None:
            return state
        if not 0 <= field_index < len(state.fields):
            raise ValueError(
                f"There is no field {field_index}; there are {len(state.fields)} field(s)"
            )
        return replace(state, selected_field_index=int(field_index))

    def _lens_info(self, state: OpticalState, index: int) -> LensInfo:
        """The snapshot of lens ``index`` in ``state``."""
        lens = state.lenses[index]
        gap = gaps.air_gaps(state)[index]
        if lens.is_paraxial:
            focal_length = lens.paraxial_focal_length_mm
        elif lens.part_number in self._catalog:
            focal_length = self._catalog[lens.part_number].focal_length
        else:
            focal_length = None  # imported with its own prescription
        return LensInfo(
            index=index,
            part_number=lens.part_number,
            front_z_mm=float(lens.z),
            back_z_mm=float(lens.back_z),
            thickness_mm=float(lens.thickness),
            clear_diameter_mm=float(2.0 * lens.semi_aperture),
            focal_length_mm=None if focal_length is None else float(focal_length),
            flipped=lens.flipped,
            plossl_group=lens.group_id,
            is_paraxial=lens.is_paraxial,
            gap_after_mm=float(gap.width_mm),
            gap_after_locked=gap.locked,
            vendor=lens.vendor,
            mount_type=lens.mount_type,
            surface_range=state.surface_range(index),
        )

    @staticmethod
    def _field_info(state: OpticalState, index: int) -> FieldInfo:
        """The snapshot of field ``index`` in ``state``."""
        beam = state.fields[index]
        return FieldInfo(
            index=index,
            diameter_mm=float(beam.diameter_mm),
            angle_deg=float(beam.angle_deg),
            vergence_dioptres=float(beam.vergence_dioptres),
            offset_mm=float(beam.offset_mm),
            wavelength_nm=float(beam.wavelength_nm),
            colour=beam.display_colour(),
            colour_follows_wavelength=beam.colour is None,
            selected=index == state.selected_field_index,
        )
