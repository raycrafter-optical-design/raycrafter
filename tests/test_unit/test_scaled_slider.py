"""Tests of the power-law scale of the context bar's sliders (no display needed)."""

import pytest

from raycrafter.gui.scaled_slider import position_of_value, value_at_position


def test_short_focal_lengths_get_more_of_the_slider():
    # #70: 0 to 250 mm should take clearly more than a quarter of a 1000 mm slider.
    assert position_of_value(250.0, 1000.0, 1.5) == pytest.approx(0.397, abs=0.001)
    assert position_of_value(50.0, 100.0, 1.3) == pytest.approx(0.587, abs=0.001)


def test_position_and_value_are_inverses_and_values_beyond_the_end_sit_at_the_end():
    for value in (0.0, 3.0, 99.5, 640.0, 1000.0):
        position = position_of_value(value, 1000.0, 1.5)
        assert value_at_position(position, 1000.0, 1.5) == pytest.approx(value)
    assert position_of_value(5000.0, 1000.0, 1.5) == 1.0
    assert position_of_value(-200.0, 1000.0, 1.5) == position_of_value(200.0, 1000.0, 1.5)
