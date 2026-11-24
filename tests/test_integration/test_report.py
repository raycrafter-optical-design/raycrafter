"""The HTML report builds from a model, without opening a browser."""

from raycrafter.report import build_report_html


def test_report_contains_every_plot(focused_model):
    focused_model.state.field_angle = 3.0
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
