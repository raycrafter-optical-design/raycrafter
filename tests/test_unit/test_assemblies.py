"""Assemblies and user parts (issue #43): group lenses, flip them as one, save and insert them."""

import copy

import pytest

from raycrafter import Bench
from raycrafter.core import user_parts
from raycrafter.core.model import OpticalModel
from raycrafter.mcp_server.tools import DesignTools


@pytest.fixture
def model(tmp_path) -> OpticalModel:
    """Three different lenses, the second flipped, with uneven gaps; parts go in tmp_path."""
    bench_model = OpticalModel(add_default_lens=False)
    bench_model.parts_folder = tmp_path / "parts"
    for part_number, front_z in (("AC254-050-A", 100.0), ("LA1509-A", 120.0), ("LC1060-A", 150.0)):
        bench_model.insert_lens(bench_model.make_lens(part_number, front_z))
    bench_model.flip_lens(1)
    bench_model.state.image_z = 300.0
    return bench_model


def layout(model: OpticalModel) -> list[tuple[str, float, bool]]:
    """Each lens's part number, front z and orientation."""
    return [(lens.part_number, round(lens.z, 9), lens.flipped) for lens in model.state.lenses]


def widths(model: OpticalModel) -> list[float]:
    return [gap.width_mm for gap in model.air_gaps()]


def test_making_an_assembly_locks_its_gaps_and_names_it(model):
    assembly_id = model.make_assembly(0, 1, "doublet pair")
    assert [gap.locked for gap in model.air_gaps()] == [True, False, False]
    assert model.assembly_indices(1) == [0, 1] and model.block_indices(1) == [0, 1]
    assert model.assembly_name(0) == "doublet pair"
    assert model.state.assembly_names == {assembly_id: "doublet pair"}
    assert model.assembly_name(2) is None


def test_an_assembly_absorbs_the_ones_inside_it_but_cannot_cut_one(model):
    model.make_assembly(0, 1)
    with pytest.raises(ValueError, match="reaches beyond"):
        model.make_assembly(1, 2)
    outer = model.make_assembly(0, 2, "all three")
    assert model.state.assembly_names == {outer: "all three"}
    with pytest.raises(ValueError, match="two or more"):
        model.make_assembly(2, 2)


def test_flipping_an_assembly_mirrors_it_in_place(model):
    model.make_assembly(0, 2)
    before = copy.deepcopy(model.state)
    gaps_before = widths(model)
    new_order = model.flip_assembly(1)

    assert new_order == [2, 1, 0]
    lenses = model.state.lenses
    assert [lens.part_number for lens in lenses] == ["LC1060-A", "LA1509-A", "AC254-050-A"]
    assert [lens.flipped for lens in lenses] == [True, False, True]
    assert lenses[0].z == pytest.approx(before.lenses[0].z)  # the same stretch of axis
    assert lenses[-1].back_z == pytest.approx(before.lenses[-1].back_z)
    assert widths(model)[:2] == pytest.approx(gaps_before[:2][::-1])
    assert all(gap.locked for gap in model.air_gaps()[:2])


def test_flipping_an_assembly_twice_restores_it(model):
    model.make_assembly(0, 2)
    before = copy.deepcopy(model.state)
    model.flip_assembly(0)
    model.flip_assembly(0)
    assert layout(model) == [
        (lens.part_number, round(lens.z, 9), lens.flipped) for lens in before.lenses
    ]
    assert model.state == before


def test_a_saved_user_part_is_inserted_with_the_same_spacings_and_orientations(model):
    model.make_assembly(0, 1, "pair")
    model.save_assembly_as_part(0, "My pair")
    saved_gap = widths(model)[0]
    saved_flips = [lens.flipped for lens in model.state.lenses[:2]]

    model.state.image_z = 500.0
    part = model.user_part("My pair")
    new_indices = model.insert_lenses(part.lenses_at(400.0), assembly_name=part.name)

    assert new_indices == [3, 4]
    new_lenses = model.state.lenses[3:5]
    assert [lens.part_number for lens in new_lenses] == ["AC254-050-A", "LA1509-A"]
    assert [lens.flipped for lens in new_lenses] == saved_flips
    assert new_lenses[0].z == pytest.approx(400.0)
    assert widths(model)[3] == pytest.approx(saved_gap)
    assert model.assembly_name(3) == "My pair" and model.air_gaps()[3].locked


def test_user_parts_refuse_duplicate_names_and_skip_bad_files(model):
    model.make_assembly(0, 1)
    model.save_assembly_as_part(0, "pair")
    with pytest.raises(ValueError, match="already"):
        model.save_assembly_as_part(0, "pair")
    model.save_assembly_as_part(0, "pair", overwrite=True)
    with pytest.raises(ValueError, match="Only an assembly"):
        model.save_assembly_as_part(2, "single")

    (model.parts_folder / f"broken{user_parts.PART_EXTENSION}").write_text("{", encoding="utf-8")
    parts, problems = user_parts.load_parts(model.parts_folder)
    assert [part.name for part in parts] == ["pair"]
    assert len(problems) == 1 and "broken" in problems[0]


def test_deleting_a_lens_dissolves_its_assembly(model):
    model.make_assembly(0, 2)
    model.delete_lens(2)
    assert model.state.assembly_names == {}
    assert [lens.assembly_id for lens in model.state.lenses] == [None, None]
    assert model.air_gaps()[0].locked  # the gaps not beside the deleted lens stay locked


def test_a_plossl_pair_is_an_assembly():
    plossl_model = OpticalModel()
    plossl_model.add_lens("ACT508-200-A")
    plossl_model.make_plossl(0)
    assert plossl_model.assembly_name(1) == "Plossl pair"
    plossl_model.split_assembly(0)
    assert plossl_model.state.assembly_names == {}


def test_scripts_and_agents_make_flip_save_and_insert_assemblies(tmp_path):
    bench = Bench(empty=True)
    bench._tools.model.parts_folder = tmp_path  # a headless bench keeps its own model
    bench.add_lens("AC254-050-A")
    bench.add_lens("LA1509-A")
    bench.make_assembly(0, 1, "pair")
    assert [lens.assembly_name for lens in bench.lenses] == ["pair", "pair"]
    bench.flip_assembly(0)
    assert [lens.part_number for lens in bench.lenses] == ["LA1509-A", "AC254-050-A"]
    bench.save_user_part(0, "flipped pair")
    assert bench.user_parts() == ["flipped pair"]
    inserted = bench.insert_user_part("flipped pair", front_z_mm=20.0)
    assert [lens.index for lens in inserted] == [0, 1]
    assert inserted[0].assembly_name == "flipped pair"

    tools = DesignTools(OpticalModel())
    with pytest.raises(ValueError, match="no assembly"):
        tools.flip_assembly(0)
