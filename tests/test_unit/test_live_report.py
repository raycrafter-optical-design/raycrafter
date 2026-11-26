"""The report that stays up to date: when it is built, and how it is served (issue #68)."""

import socket
import threading
import time
import urllib.error
import urllib.request

import pytest

from raycrafter.core import OpticalModel
from raycrafter.report.live_report import LiveReport
from raycrafter.report.live_server import LiveReportServer

INTERVAL_S = 0.3  # the app builds at most every 10 s; the tests use this
QUIET_S = 3 * INTERVAL_S  # long enough for any build that was going to happen


class FakeBuild:
    """Stands in for building a report: it "shows" the design's image position.

    It records when each build started and how many ran at once, and can be held up.
    """

    def __init__(self) -> None:
        self.start_times: list[float] = []
        self.designs_built: list[float] = []
        self.running = 0
        self.most_at_once = 0
        self.may_finish = threading.Event()
        self.may_finish.set()
        self.fail = False

    def __call__(self, design: dict) -> str:
        self.start_times.append(time.monotonic())
        self.running += 1
        self.most_at_once = max(self.most_at_once, self.running)
        try:
            assert self.may_finish.wait(5.0)
            if self.fail:
                raise RuntimeError("no glass")
            self.designs_built.append(design["image_z"])
            return f"<html><body>image at {design['image_z']}</body></html>"
        finally:
            self.running -= 1


def wait_until(condition, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.005)


@pytest.fixture
def model() -> OpticalModel:
    return OpticalModel()


@pytest.fixture
def build() -> FakeBuild:
    return FakeBuild()


@pytest.fixture
def live_report(build):
    report = LiveReport(build, min_interval_s=INTERVAL_S)
    yield report
    build.may_finish.set()
    report.stop()


def show(live_report: LiveReport, model: OpticalModel, image_z: float) -> None:
    model.state.image_z = image_z
    live_report.set_design(model.state)


# ------------------------------------------------------------------ when it is built


def test_the_first_report_is_built_at_once(live_report, model, build):
    assert live_report.latest()[0] == 0 and "Building the report" in live_report.latest()[1]
    show(live_report, model, 300.0)
    wait_until(lambda: live_report.latest()[0] == 1)
    assert "image at 300.0" in live_report.latest()[1]


def test_nothing_more_is_built_while_no_page_watches(live_report, model, build):
    show(live_report, model, 300.0)
    wait_until(lambda: build.designs_built == [300.0])
    show(live_report, model, 301.0)
    time.sleep(QUIET_S)
    assert build.designs_built == [300.0]

    with live_report.watching():  # a page comes: it is brought up to date
        wait_until(lambda: build.designs_built == [300.0, 301.0])


def test_a_burst_of_changes_costs_one_build_of_the_last(live_report, model, build):
    with live_report.watching():
        show(live_report, model, 300.0)
        wait_until(lambda: build.designs_built == [300.0])
        for image_z in (301.0, 302.0, 303.0):
            show(live_report, model, image_z)
        wait_until(lambda: len(build.designs_built) == 2)
        time.sleep(QUIET_S)
    assert build.designs_built == [300.0, 303.0]
    assert build.start_times[1] - build.start_times[0] >= INTERVAL_S - 0.02


def test_a_change_made_while_building_is_not_lost(live_report, model, build):
    """The design moves on during a build: that report is published, then the newest one."""
    build.may_finish.clear()
    with live_report.watching():
        show(live_report, model, 300.0)
        wait_until(lambda: build.running == 1)
        show(live_report, model, 301.0)
        show(live_report, model, 302.0)
        time.sleep(QUIET_S)
        assert len(build.start_times) == 1  # no second build while one is under way
        build.may_finish.set()
        wait_until(lambda: "image at 302.0" in live_report.latest()[1])
    assert build.designs_built == [300.0, 302.0]
    assert build.most_at_once == 1
    assert live_report.latest()[0] == 3  # the revision of the newest design


def test_the_same_design_again_is_not_a_change(live_report, model, build):
    with live_report.watching():
        show(live_report, model, 300.0)
        wait_until(lambda: len(build.designs_built) == 1)
        show(live_report, model, 300.0)
        time.sleep(QUIET_S)
    assert build.designs_built == [300.0]


def test_the_design_is_copied_when_it_is_handed_over(live_report, model, build):
    build.may_finish.clear()
    show(live_report, model, 300.0)
    wait_until(lambda: build.running == 1)
    model.state.image_z = 999.0  # edited, but not handed over
    build.may_finish.set()
    wait_until(lambda: build.designs_built == [300.0])


def test_a_failed_build_is_reported_and_the_next_change_tried(live_report, model, build):
    build.fail = True
    with live_report.watching():
        show(live_report, model, 300.0)
        wait_until(lambda: "no glass" in live_report.latest()[1])
        time.sleep(QUIET_S)
        assert len(build.start_times) == 1  # not tried again and again
        build.fail = False
        show(live_report, model, 301.0)
        wait_until(lambda: "image at 301.0" in live_report.latest()[1])


def test_stopping_ends_the_building_thread(live_report, model, build):
    live_report.stop()
    live_report._thread.join(2.0)
    assert not live_report._thread.is_alive()
    show(live_report, model, 300.0)
    time.sleep(QUIET_S)
    assert build.start_times == []


# ------------------------------------------------------------------ how it is served


@pytest.fixture
def server(live_report):
    live_server = LiveReportServer(live_report)
    live_server.start()
    yield live_server
    live_server.stop()


def fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.read().decode()


def test_the_page_is_served_at_once_and_says_when_there_is_a_newer_one(
    live_report, server, model, build
):
    build.may_finish.clear()
    show(live_report, model, 300.0)
    page = fetch(server.url)  # answered although the first report is still being built
    assert "Building the report" in page and 'data-revision="0"' in page

    answers = []
    waiting = threading.Thread(
        target=lambda: answers.append(fetch(server.url + "revision?after=0"))
    )
    waiting.start()
    wait_until(lambda: live_report.is_watched)
    assert answers == []
    build.may_finish.set()
    waiting.join(5.0)
    assert answers == ["1"]

    page = fetch(server.url)
    assert "image at 300.0" in page and 'data-revision="1"' in page
    assert 'href="download"' in page and "location.reload()" in page


def test_the_downloaded_copy_needs_no_server(live_report, server, model, build):
    build.may_finish.clear()
    show(live_report, model, 300.0)
    with pytest.raises(urllib.error.HTTPError, match="404"):
        fetch(server.url + "download")  # there is no report yet
    build.may_finish.set()
    wait_until(lambda: live_report.latest()[0] == 1)
    with urllib.request.urlopen(server.url + "download", timeout=10) as response:
        assert response.headers["Content-Disposition"].startswith("attachment;")
        assert response.read().decode() == "<html><body>image at 300.0</body></html>"


def test_a_closed_tab_stops_counting_as_watching(live_report, server, model):
    show(live_report, model, 300.0)
    wait_until(lambda: live_report.latest()[0] == 1)
    host, port = server.url.split("/")[2].split(":")
    connection = socket.create_connection((host, int(port)))
    path = f"/{server.token}/revision?after=1"
    connection.sendall(f"GET {path} HTTP/1.1\r\nHost: {host}\r\n\r\n".encode())
    wait_until(lambda: live_report.is_watched)
    connection.close()
    wait_until(lambda: not live_report.is_watched, timeout_s=3.0)


@pytest.mark.parametrize("path", ["", "wrong-token/", "wrong-token/download"])
def test_without_the_token_nothing_is_served(server, path):
    address = server.url.removesuffix(f"{server.token}/")
    with pytest.raises(urllib.error.HTTPError, match="404"):
        fetch(address + path)
