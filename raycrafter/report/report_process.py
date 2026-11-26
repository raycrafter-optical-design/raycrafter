"""Building reports in a separate Python process, so that the window is not slowed (#68).

A report takes a few tenths of a second to build, much of it in Python code. A background
*thread* would not do: Python runs only one thread's code at a time, so the window's redraws
during a drag stalled for up to 100 ms while one was building. A separate process runs on its
own, and leaves them as they were.

``ReportProcess`` starts one worker (this module, run with ``python -m``) when first asked for
a report, and keeps it, as starting one takes a couple of seconds (it loads Optiland). The
design goes to the worker's standard input as a line of JSON, and the report comes back on its
standard output as another. The worker ends when its input does, so it does not outlive
RayCrafter, even if RayCrafter is killed. (``multiprocessing`` is not used: it runs the
program's main script again in the worker, and its pool workers are left behind when the
program is killed.)
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
from typing import TextIO

from raycrafter.report.html_report import build_report_from_data

# "-P" keeps the current folder off the worker's import path, as it is off RayCrafter's own.
WORKER_COMMAND = [sys.executable, "-P", "-m", "raycrafter.report.report_process"]
CLOSE_TIMEOUT_S = 5.0  # how long to wait for a killed worker to be gone


class ReportProcess:
    """Builds reports in a worker process; call it with a design to get the report.

    Meant to be called from one thread at a time (``raycrafter.report.live_report`` calls it
    from its building thread, which waits for the answer without holding Python up).
    ``close`` may be called from any thread.
    """

    def __init__(self) -> None:
        self._process: subprocess.Popen[str] | None = None

    def __call__(self, state_data: dict) -> str:
        """The report of a design given as ``state_to_dict`` data.

        Raises
        ------
        RuntimeError
            If the report could not be built, or the worker stopped. A new worker is started
            for the next report.
        """
        process = self._process
        if process is None or process.poll() is not None:
            self.close()
            process = self._start()
        assert process.stdin is not None and process.stdout is not None
        try:
            process.stdin.write(json.dumps(state_data) + "\n")
            process.stdin.flush()
            answer = process.stdout.readline()
        except (OSError, ValueError):  # the pipe broke, or was closed while in use
            answer = ""
        if not answer.endswith("\n"):  # the output ended: the worker has gone
            self.close()
            raise RuntimeError("The process that builds reports stopped")
        reply = json.loads(answer)
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply["report"]

    def close(self) -> None:
        """Stop the worker process, if there is one, even in the middle of a report."""
        process, self._process = self._process, None
        if process is None:
            return
        process.kill()
        process.wait(CLOSE_TIMEOUT_S)
        for pipe in (process.stdin, process.stdout):
            if pipe is not None:
                with contextlib.suppress(OSError):  # unsent text cannot be flushed to it
                    pipe.close()

    def _start(self) -> subprocess.Popen[str]:
        """Start a worker process with pipes to it."""
        self._process = subprocess.Popen(
            WORKER_COMMAND, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )
        return self._process


def serve(designs: TextIO, replies: TextIO) -> None:
    """The worker's loop: answer each line of ``designs`` with its report, until they end."""
    for line in designs:
        try:
            reply = {"report": build_report_from_data(json.loads(line))}
        except Exception as error:  # reported to RayCrafter; the worker carries on
            reply = {"error": f"{type(error).__name__}: {error}"}
        replies.write(json.dumps(reply) + "\n")
        replies.flush()


def main() -> None:
    """Run as the worker, talking to RayCrafter on standard input and output."""
    # Keep the real standard output for the replies, and send anything a library prints to
    # standard error instead, where it cannot be mistaken for one.
    replies = os.fdopen(os.dup(sys.stdout.fileno()), "w")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    try:
        serve(sys.stdin, replies)
    except BrokenPipeError:
        pass  # RayCrafter has gone


if __name__ == "__main__":
    main()
