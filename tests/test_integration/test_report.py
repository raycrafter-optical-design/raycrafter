"""The HTML report builds from a model, without opening a browser."""

import base64
import json
import re

import pytest

from raycrafter.core.state import state_to_dict
from raycrafter.report import build_report_html, viewer_3d
from raycrafter.report.report_process import ReportProcess


def test_report_contains_every_plot(focused_model):
    focused_model.state.selected_field.angle_deg = 3.0
    page = build_report_html(focused_model)
    assert page.startswith("<!DOCTYPE html>")
    assert page.count("<svg") == 6  # ray diagram plus five analysis plots
    assert "ACT508-200-A" in page


def test_report_includes_a_paraxial_lens(focused_model):
    focused_model.insert_lens(focused_model.make_paraxial_lens(-150.0, z=40.0))
    focused_model.autofocus()
    page = build_report_html(focused_model)
    assert page.count("<svg") == 6
    assert "Paraxial f=-150 mm" in page


def script_data(page: str, script_start: str) -> dict:
    """The JSON held in the <script> element that begins with ``script_start``."""
    after_start = page[page.index(script_start) + len(script_start) :]
    return json.loads(after_start[: after_start.index("</script>")])


def test_report_carries_the_3d_view_and_everything_it_needs(focused_model):
    """Issue #51: the report is one file that can be shared and opened without the internet."""
    page = build_report_html(focused_model)
    assert 'id="viewer3d"' in page

    scene = script_data(page, '<script id="scene-data" type="application/json">')
    assert scene["format"] == "raycrafter-scene"
    assert "ACT508-200-A" in [item["name"] for item in scene["objects"]]
    assert set(scene["views"]) == {"3D", "Side", "Top", "End"}

    # three.js travels inside the page, complete, under the names the viewer imports.
    imports = script_data(page, '<script type="importmap">')["imports"]
    assert set(imports) == {"three", "three-orbit-controls"}
    for name, file_name in viewer_3d.THREE_MODULES.items():
        prefix, content = imports[name].split(",", 1)
        assert prefix == "data:text/javascript;base64"
        assert base64.b64decode(content).decode() == viewer_3d.read_asset(file_name)
        assert f"from '{name}'" in page

    # Nothing is fetched from elsewhere.
    assert not re.search(r"""(src|href)\s*=\s*["']https?:""", page)
    # The import map has to come before the script that uses it.
    assert page.index('type="importmap"') < page.index('type="module"')


def test_a_part_name_cannot_end_a_script_early(focused_model):
    focused_model.state.lenses[0].part_number = "</script><b>lens"
    page = build_report_html(focused_model)
    scene = script_data(page, '<script id="scene-data" type="application/json">')
    assert "</script><b>lens" in [item["name"] for item in scene["objects"]]


def test_the_report_process_builds_the_report_of_the_design_it_is_sent(focused_model):
    """A real worker process (issue #68): it builds, survives a bad request, and stops."""
    report_process = ReportProcess()
    try:
        page = report_process(state_to_dict(focused_model.state))
        assert "ACT508-200-A" in page and 'id="viewer3d"' in page
        worker = report_process._process

        with pytest.raises(RuntimeError, match="KeyError"):
            report_process({"nonsense": 1})
        focused_model.delete_lens(0)
        assert "There is no lens" in report_process(state_to_dict(focused_model.state))
        assert report_process._process is worker  # one worker is kept

        worker.kill()  # as if it had crashed: the next report starts another
        worker.wait(5.0)
        assert "There is no lens" in report_process(state_to_dict(focused_model.state))
        worker = report_process._process
    finally:
        report_process.close()
    assert worker.poll() is not None


def test_the_report_worker_ends_when_its_input_does():
    """When RayCrafter closes the pipe, or is killed, which closes it too, the worker ends."""
    report_process = ReportProcess()
    worker = report_process._start()
    try:
        worker.stdin.close()
        assert worker.wait(60.0) == 0
    finally:
        report_process.close()
