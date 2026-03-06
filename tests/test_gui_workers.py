from __future__ import annotations

from alphaomega_reporter_gui.controllers import Worker


def test_worker_runs_function_without_progress_callback() -> None:
    results: list[object] = []
    errors: list[str] = []

    def add(a: int, b: int) -> int:
        return a + b

    worker = Worker(add, 2, 3)
    worker.signals.result.connect(results.append)
    worker.signals.error.connect(errors.append)
    worker.run()

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
