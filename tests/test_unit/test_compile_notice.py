"""Tests for the message shown while numba compiles."""

from numba import njit

from raycrafter.gui.compile_notice import CompileNotice, announce_numba_compilation


def test_message_printed_once_when_numba_compiles(capsys):
    """Compiling two new functions inside the block prints the message a single time."""

    @njit
    def add_one(value):
        return value + 1

    @njit
    def add_two(value):
        return value + 2

    with announce_numba_compilation():
        add_one(1.0)
        add_two(1.0)

    assert capsys.readouterr().out.count(CompileNotice.MESSAGE) == 1


def test_no_message_when_nothing_compiles(capsys):
    """A block that triggers no compilation prints nothing."""
    with announce_numba_compilation():
        pass

    assert capsys.readouterr().out == ""
