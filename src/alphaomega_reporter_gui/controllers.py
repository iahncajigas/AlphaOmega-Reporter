from __future__ import annotations

import inspect
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6 import QtCore

from alphaomega_reporter import build_report, load_case

from .models import CaseModel, DepthModel, build_case_model
from .processing import (
    LfpPreviewResult,
    MerPreviewResult,
    ProcessingCache,
    compute_lfp_preview,
    compute_mer_preview,
)
from .report_config import GuiReportConfig


class WorkerSignals(QtCore.QObject):
    result = QtCore.Signal(object)
    error = QtCore.Signal(str)
    progress = QtCore.Signal(str)
    finished = QtCore.Signal()


class Worker(QtCore.QRunnable):
    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.setAutoDelete(False)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def dispose(self) -> None:
        self.fn = None
        self.args = ()
        self.kwargs = {}

    @QtCore.Slot()
    def run(self) -> None:
        try:
            call_kwargs = dict(self.kwargs)
            signature = inspect.signature(self.fn)
            accepts_var_kwargs = any(
                param.kind == inspect.Parameter.VAR_KEYWORD
                for param in signature.parameters.values()
            )
            if "progress_callback" in signature.parameters or accepts_var_kwargs:
                call_kwargs["progress_callback"] = self.signals.progress.emit
            result = self.fn(*self.args, **call_kwargs)
        except Exception:
            self.signals.error.emit(traceback.format_exc())
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


class GuiController(QtCore.QObject):
    log_message = QtCore.Signal(str)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.thread_pool = QtCore.QThreadPool.globalInstance()
        self.cache = ProcessingCache()
        self._active_workers: dict[int, Worker] = {}

    @QtCore.Slot()
    def _worker_finished(self) -> None:
        sender = self.sender()
        if not isinstance(sender, WorkerSignals):
            return
        worker = self._active_workers.pop(id(sender), None)
        if worker is not None:
            worker.dispose()
        sender.deleteLater()

    def _submit(
        self,
        fn: Callable[..., Any],
        *args: Any,
        on_result: Callable[[Any], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        on_finished: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> None:
        worker = Worker(fn, *args, **kwargs)
        worker.signals.setParent(self)
        self._active_workers[id(worker.signals)] = worker
        worker.signals.finished.connect(
            self._worker_finished,
            QtCore.Qt.ConnectionType.QueuedConnection,
        )
        worker.signals.progress.connect(
            self.log_message.emit,
            QtCore.Qt.ConnectionType.QueuedConnection,
        )
        if on_result is not None:
            worker.signals.result.connect(on_result, QtCore.Qt.ConnectionType.QueuedConnection)
        if on_error is not None:
            worker.signals.error.connect(on_error, QtCore.Qt.ConnectionType.QueuedConnection)
        else:
            worker.signals.error.connect(
                self.log_message.emit,
                QtCore.Qt.ConnectionType.QueuedConnection,
            )
        if on_finished is not None:
            worker.signals.finished.connect(
                on_finished,
                QtCore.Qt.ConnectionType.QueuedConnection,
            )
        self.thread_pool.start(worker)

    def clear_cache(self) -> None:
        self.cache.clear()

    def load_case_async(
        self,
        case_dir: str | Path,
        gui_config: GuiReportConfig,
        *,
        on_result: Callable[[CaseModel], None],
        on_error: Callable[[str], None],
        on_finished: Callable[[], None] | None = None,
    ) -> None:
        self.clear_cache()
        self._submit(
            self._load_case,
            case_dir,
            gui_config,
            on_result=on_result,
            on_error=on_error,
            on_finished=on_finished,
        )

    def mer_preview_async(
        self,
        current_depth: DepthModel,
        selected_depths: list[DepthModel],
        channel_index: int,
        gui_config: GuiReportConfig,
        *,
        on_result: Callable[[MerPreviewResult], None],
        on_error: Callable[[str], None],
    ) -> None:
        self._submit(
            compute_mer_preview,
            current_depth,
            selected_depths,
            channel_index,
            gui_config,
            self.cache,
            on_result=on_result,
            on_error=on_error,
        )

    def lfp_preview_async(
        self,
        selected_depths: list[DepthModel],
        channel_index: int,
        gui_config: GuiReportConfig,
        *,
        on_result: Callable[[LfpPreviewResult], None],
        on_error: Callable[[str], None],
    ) -> None:
        self._submit(
            compute_lfp_preview,
            selected_depths,
            channel_index,
            gui_config,
            self.cache,
            on_result=on_result,
            on_error=on_error,
        )

    def generate_report_async(
        self,
        case_model: CaseModel,
        gui_config: GuiReportConfig,
        out_pdf: str | Path,
        *,
        on_result: Callable[[Path], None],
        on_error: Callable[[str], None],
        on_finished: Callable[[], None] | None = None,
    ) -> None:
        self._submit(
            self._generate_report,
            case_model,
            gui_config,
            out_pdf,
            on_result=on_result,
            on_error=on_error,
            on_finished=on_finished,
        )

    @staticmethod
    def _load_case(
        case_dir: str | Path,
        gui_config: GuiReportConfig,
        *,
        progress_callback: Callable[[str], None] | None = None,
    ) -> CaseModel:
        if progress_callback is not None:
            progress_callback(f"Loading case directory: {case_dir}")
        report_config = gui_config.to_report_config()
        session = load_case(case_dir, report_config, require_depth=False)
        case_model = build_case_model(session, report_config)
        if progress_callback is not None:
            progress_callback(
                f"Loaded {len(case_model.trajectories)} trajectories from {case_model.case_name}"
            )
        return case_model

    @staticmethod
    def _generate_report(
        case_model: CaseModel,
        gui_config: GuiReportConfig,
        out_pdf: str | Path,
        *,
        progress_callback: Callable[[str], None] | None = None,
    ) -> Path:
        if progress_callback is not None:
            progress_callback(f"Generating report: {out_pdf}")
        report_config = gui_config.to_report_config()
        out_path = build_report(case_model.session, report_config, out_pdf)
        if progress_callback is not None:
            progress_callback(f"Finished report: {out_path}")
        return Path(out_path)
