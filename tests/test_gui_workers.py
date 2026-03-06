from __future__ import annotations

import time

from alphaomega_reporter_gui.app import create_application
from alphaomega_reporter_gui.controllers import GuiController, Worker


def test_worker_runs_function_without_progress_callback() -> None:
    results: list[object] = []
    errors: list[str] = []

    def add(a: int, b: int) -> int:
        return a + b

    worker = Worker(add, 2, 3)
    worker.signals.result.connect(results.append)
    worker.signals.error.connect(errors.append)
    worker.run()

    assert worker.autoDelete() is False
    assert results == [5]
    assert errors == []


def test_worker_injects_progress_callback_when_supported() -> None:
    results: list[object] = []
    errors: list[str] = []
    progress: list[str] = []

    def task(value: int, *, progress_callback) -> int:
        progress_callback("started")
        return value * 2

    worker = Worker(task, 4)
    worker.signals.result.connect(results.append)
    worker.signals.error.connect(errors.append)
    worker.signals.progress.connect(progress.append)
    worker.run()

    assert results == [8]
    assert errors == []
    assert progress == ["started"]


def test_controller_releases_worker_on_main_thread() -> None:
    app = create_application()
    controller = GuiController()
    results: list[object] = []
    errors: list[str] = []
    finished: list[bool] = []

    controller._submit(
        lambda: "done",
        on_result=results.append,
        on_error=errors.append,
        on_finished=lambda: finished.append(True),
    )

    deadline = time.monotonic() + 5.0
    while not finished and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)

    assert finished == [True]
    assert results == ["done"]
    assert errors == []
    assert controller._active_workers == {}


def test_controller_handles_many_short_lived_workers() -> None:
    app = create_application()
    controller = GuiController()
    results: list[int] = []
    errors: list[str] = []
    finished_count = 0

    def on_finished() -> None:
        nonlocal finished_count
        finished_count += 1

    for value in range(25):
        controller._submit(
            lambda current=value: current,
            on_result=results.append,
            on_error=errors.append,
            on_finished=on_finished,
        )

    deadline = time.monotonic() + 5.0
    while finished_count < 25 and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)

    for _ in range(20):
        app.processEvents()
        time.sleep(0.005)

    assert finished_count == 25
    assert sorted(results) == list(range(25))
    assert errors == []
    assert controller._active_workers == {}
