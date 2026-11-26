"""Replacing lenses in place (issue #63): where the new lens goes, locks, assemblies, coatings."""

import copy

import pytest

from raycrafter import Bench
from raycrafter.core import replacement
from raycrafter.core.assemblies import PLOSSL_NAME
from raycrafter.core.model import PLOSSL_GAP_MM, OpticalModel
from raycrafter.lenscatalog import thorlabs
from raycrafter.lenscatalog.selection import LensFilter
from raycrafter.mcp_server.tools import DesignTools

WAVELENGTH_NM = 555.0


@pytest.fixture(scope="module")
def catalog():
    return thorlabs.load_lenses()


def paraxial_bench(catalog, focal_lengths_mm=(200.0,), front_zs=(100.0,)) -> OpticalModel:
    """Paraxial lenses at the given z, image plane far behind."""
    model = OpticalModel(catalog, add_default_lens=False)
    for focal_length_mm, front_z in zip(focal_lengths_mm, front_zs):
        model.insert_lens(model.make_paraxial_lens(focal_length_mm, front_z))
    model.state.image_z = 600.0
    return model


def rear_plane_z(model: OpticalModel, lens_index: int) -> float:
    planes = replacement.principal_planes_z([model.state.lenses[lens_index]], WAVELENGTH_NM)
    return planes[1]


# ------------------------------------------------------------------ where the lens goes


def test_a_paraxial_lens_is_its_own_principal_plane(catalog):
    model = paraxial_bench(catalog)
    assert rear_plane_z(model, 0) == pytest.approx(100.0)


def test_swapping_a_paraxial_lens_for_a_real_one_keeps_the_focus(catalog):
    """Physics check: ACT508-200-A in place of a paraxial f = 200 mm lens focuses a collimated
    beam within 0.2 mm of where the paraxial lens did (its EFL is 199.95 mm at 555 nm)."""
    model = paraxial_bench(catalog)
    paraxial_focus_z = model.autofocus()
    assert paraxial_focus_z == pytest.approx(300.0, abs=0.01)

    model.replace_lens(0, model.make_lens("ACT508-200-A"))
    assert model.state.lenses[0].part_number == "ACT508-200-A"
    assert rear_plane_z(model, 0) == pytest.approx(100.0, abs=1e-6)
    assert model.autofocus() == pytest.approx(paraxial_focus_z, abs=0.2)


def test_replacing_with_paraxial_keeps_the_rear_principal_plane(catalog):
    model = OpticalModel(catalog)  # one ACT508-200-A
    plane_z = rear_plane_z(model, 0)
    focal_length_mm = model.lens_focal_length_mm(0)
    assert focal_length_mm == pytest.approx(200.0, abs=0.2)

    model.replace_lens(0, model.make_paraxial_lens(focal_length_mm))
    assert model.state.lenses[0].is_paraxial
    assert model.state.lenses[0].z == pytest.approx(plane_z)


def test_a_locked_gap_in_front_keeps_the_front_vertex(catalog):
    model = paraxial_bench(catalog, (100.0, 200.0), (100.0, 150.0))
    model.set_gap_locked(0, True)
    model.replace_lens(1, model.make_lens("ACT508-200-A"))
    assert model.state.lenses[1].z == pytest.approx(150.0)
    assert model.air_gaps()[0].locked


def test_a_locked_gap_behind_keeps_the_back_vertex(catalog):
    model = paraxial_bench(catalog, (200.0, 100.0), (100.0, 200.0))
    model.set_gap_locked(0, True)
    model.replace_lens(0, model.make_lens("ACT508-200-A"))
    lenses = model.state.lenses
    assert lenses[0].back_z == pytest.approx(100.0)
    assert lenses[1].z == pytest.approx(200.0)  # the lens behind does not move
    assert model.air_gaps()[0].locked


def test_with_both_gaps_locked_the_lenses_behind_shift(catalog):
    model = paraxial_bench(catalog, (100.0, 200.0, 100.0), (100.0, 150.0, 200.0))
    model.set_gap_locked(0, True)
    model.set_gap_locked(1, True)
    gaps_before = [gap.width_mm for gap in model.air_gaps()[:2]]
    new_lens = model.make_lens("ACT508-200-A")

    model.replace_lens(1, new_lens)
    lenses = model.state.lenses
    assert lenses[1].z == pytest.approx(150.0)
    assert lenses[2].z == pytest.approx(200.0 + new_lens.thickness)
    assert [gap.width_mm for gap in model.air_gaps()[:2]] == pytest.approx(gaps_before)


def test_a_lens_that_would_overlap_a_neighbour_is_refused(catalog):
    model = paraxial_bench(catalog, (200.0, 100.0), (100.0, 102.0))
    state_before = copy.deepcopy(model.state)
    with pytest.raises(ValueError, match="does not fit"):
        model.replace_lens(0, model.make_lens("ACT508-200-A"))
    assert model.state == state_before


def test_the_image_plane_is_pushed_back_if_the_last_lens_reaches_it(catalog):
    model = paraxial_bench(catalog)
    model.state.image_z = 101.0
    model.replace_lens(0, model.make_lens("ACT508-200-A"))
    assert model.state.image_z >= model.state.lenses[0].back_z + 1.0 - 1e-9


def test_a_locked_source_gap_follows_the_first_lens(catalog):
    model = paraxial_bench(catalog)
    model.state.source_z = 80.0  # locked (the default) 20 mm in front of the lens
    model.replace_lens(0, model.make_lens("ACT508-200-A"))
    assert model.source_gap().width_mm == pytest.approx(20.0)


# ------------------------------------------------------------------ what the new lens takes over


def test_a_flipped_lens_is_replaced_by_a_flipped_lens(catalog):
    model = OpticalModel(catalog)
    model.flip_lens(0)
    model.replace_lens(0, model.make_lens("AC254-200-A"))
    assert model.state.lenses[0].flipped
    model.replace_lens(0, model.make_paraxial_lens(200.0))
    assert not model.state.lenses[0].flipped  # a paraxial lens is never flipped


def test_a_lens_in_an_assembly_stays_in_it(catalog):
    model = paraxial_bench(catalog, (100.0, 200.0), (100.0, 150.0))
    model.make_assembly(0, 1, "relay")
    model.replace_lens(1, model.make_lens("ACT508-200-A"))
    assert model.assembly_name(1) == "relay" and model.assembly_indices(0) == [0, 1]
    assert model.air_gaps()[0].locked


# ------------------------------------------------------------------ Plossl pairs


@pytest.fixture
def plossl_model(catalog) -> OpticalModel:
    model = OpticalModel(catalog, add_default_lens=False)
    model.insert_lens(model.make_lens("AC254-050-A", 100.0))
    model.make_plossl(0)
    model.state.image_z = 300.0
    return model


def test_a_plossl_lens_is_not_replaced_without_saying_what_to_do(plossl_model):
    with pytest.raises(ValueError, match="Plossl pair"):
        plossl_model.replace_lens(0, plossl_model.make_lens("AC254-100-A"))


def test_splitting_a_plossl_pair_to_replace_one_lens(plossl_model):
    plossl_model.replace_lens(1, plossl_model.make_paraxial_lens(50.0), split_plossl_pair=True)
    lenses = plossl_model.state.lenses
    assert [lens.assembly_id for lens in lenses] == [None, None]
    assert not plossl_model.air_gaps()[0].locked
    assert lenses[1].is_paraxial


def test_a_refused_split_and_replace_leaves_the_pair_intact(plossl_model):
    # A lens just behind the pair leaves no room for a thick lens in place of its second.
    back_z = plossl_model.state.lenses[1].back_z
    plossl_model.insert_lens(plossl_model.make_paraxial_lens(100.0, back_z + 1.0))
    state_before = copy.deepcopy(plossl_model.state)
    with pytest.raises(ValueError, match="does not fit"):
        plossl_model.replace_lens(1, plossl_model.make_lens("ACT508-200-A"), split_plossl_pair=True)
    assert plossl_model.state == state_before
    assert plossl_model.in_plossl_pair(0)


def test_replacing_both_lenses_of_a_plossl_pair(plossl_model):
    plossl_model.replace_plossl_pair(0, "AC254-100-A")
    lenses = plossl_model.state.lenses
    assert [lens.part_number for lens in lenses] == ["AC254-100-A", "AC254-100-A"]
    assert plossl_model.in_plossl_pair(0) and plossl_model.assembly_indices(0) == [0, 1]
    assert [lens.flipped for lens in lenses] == [
        lenses[0].positive_element_is_first(),
        not lenses[1].positive_element_is_first(),
    ]
    assert plossl_model.state.assembly_names[lenses[0].assembly_id] == PLOSSL_NAME
    closest_gap_mm = lenses[1].z - lenses[0].back_z
    assert closest_gap_mm <= PLOSSL_GAP_MM + 1e-9


def test_a_plossl_pair_cannot_be_made_of_diverging_lenses(plossl_model):
    with pytest.raises(ValueError, match="positive"):
        plossl_model.replace_plossl_pair(0, "LC1060-A")


# ------------------------------------------------------------------ coatings


def test_coatings_suit_the_wavelengths_in_their_range(catalog):
    assert catalog["AC254-100-A"].suits_wavelengths([600.0])
    assert not catalog["AC254-100-B"].suits_wavelengths([600.0])
    assert catalog["AC254-100-B"].suits_wavelengths([700.0, 1000.0])
    assert not catalog["AC254-100-A"].suits_wavelengths([500.0, 800.0])  # one outside


def test_the_wavelength_filter_shows_suitable_coatings_and_uncoated_lenses(catalog):
    """A 600 nm beam: A and AB coatings, and uncoated lenses; not B or C (issue #63)."""
    shown = LensFilter(wavelengths_nm=(600.0,)).apply(catalog.values())
    coatings = {entry.coating for entry in shown}
    assert coatings == {"A: 350-700 nm", "AB: 400-1100 nm", "Uncoated"}


# ------------------------------------------------------------------ scripts and agents


def test_agents_replace_lenses(catalog):
    tools = DesignTools(OpticalModel(catalog))
    design = tools.replace_lens(0, paraxial_focal_length_mm=200.0)
    assert design["lenses"][0]["paraxial_focal_length_mm"] == 200.0
    with pytest.raises(ValueError, match="exactly one"):
        tools.replace_lens(0)
    with pytest.raises(ValueError, match="Unknown part number"):
        tools.replace_lens(0, part_number="NOT-A-LENS")


def test_scripts_replace_both_lenses_of_a_plossl_pair():
    bench = Bench(empty=True)
    bench.add_lens("AC254-050-A")
    bench.make_plossl(0)
    with pytest.raises(ValueError, match="Plossl pair"):
        bench.replace_lens(0, "AC254-100-A")
    bench.replace_lens(0, "AC254-100-A", plossl_pair="replace_both")
    assert [lens.part_number for lens in bench.lenses] == ["AC254-100-A", "AC254-100-A"]
    assert bench.lenses[0].assembly_name == PLOSSL_NAME
