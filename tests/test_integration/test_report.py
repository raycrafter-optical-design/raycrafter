"""The HTML report builds from a model, without opening a browser."""

from raycrafter.report import build_report_html


def test_report_contains_every_plot(focused_model):
    focused_model.state.field_angle = 3.0
    page = build_report_html(focused_model)
    assert page.startswith("<!DOCTYPE html>")
    assert page.count("<svg") == 6  # ray diagram plus five analysis plots
    assert "ACT508-200-A" in page
