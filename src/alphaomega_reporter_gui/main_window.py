from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtWidgets

from .controllers import GuiController
from .models import CaseModel, DepthModel, TrajectoryModel
from .plotting_mpl import MatplotlibWidget
from .plotting_pg import PG_AVAILABLE, PyQtGraphHeatmapWidget
from .report_config import GuiReportConfig


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AlphaOmega Reporter")
        self.resize(1600, 960)

        self.gui_config = GuiReportConfig()
        self.controller = GuiController(self)
        self.controller.log_message.connect(self.append_log)

        self.case_model: CaseModel | None = None
        self.current_trajectory: TrajectoryModel | None = None
        self._mer_request_id = 0
        self._lfp_request_id = 0
        self._depth_table_loading = False

        self._build_ui()
        self._apply_gui_config_to_widgets()
        self._set_idle_state()

    def _build_ui(self) -> None:
        splitter = QtWidgets.QSplitter()
        splitter.setChildrenCollapsible(False)
        self.setCentralWidget(splitter)

        sidebar = QtWidgets.QWidget()
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(8, 8, 8, 8)
        sidebar_layout.setSpacing(8)

        self.open_case_button = QtWidgets.QPushButton("Open Case...")
        self.open_case_button.clicked.connect(self.open_case_dialog)
        sidebar_layout.addWidget(self.open_case_button)

        self.case_summary = QtWidgets.QPlainTextEdit()
        self.case_summary.setReadOnly(True)
        self.case_summary.setMaximumHeight(150)
        sidebar_layout.addWidget(self.case_summary)

        sidebar_layout.addWidget(QtWidgets.QLabel("Group trajectories by"))
        self.group_by_combo = QtWidgets.QComboBox()
        self.group_by_combo.addItems(["trajectory", "side", "target"])
        self.group_by_combo.currentTextChanged.connect(self._on_group_by_changed)
        sidebar_layout.addWidget(self.group_by_combo)

        sidebar_layout.addWidget(QtWidgets.QLabel("Trajectories"))
        self.trajectory_list = QtWidgets.QListWidget()
        self.trajectory_list.currentRowChanged.connect(self._on_trajectory_selected)
        sidebar_layout.addWidget(self.trajectory_list, 1)

        sidebar_layout.addWidget(QtWidgets.QLabel("Depths"))
        self.depth_table = QtWidgets.QTableWidget(0, 2)
        self.depth_table.setHorizontalHeaderLabels(["Depth (mm)", "Segment"])
        self.depth_table.horizontalHeader().setStretchLastSection(True)
        self.depth_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.depth_table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.depth_table.itemSelectionChanged.connect(self._schedule_preview_refresh)
        self.depth_table.itemChanged.connect(self._on_depth_item_changed)
        sidebar_layout.addWidget(self.depth_table, 1)

        splitter.addWidget(sidebar)

        self.tabs = QtWidgets.QTabWidget()
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(1, 1)

        self._build_mer_tab()
        self._build_lfp_tab()
        self._build_report_tab()
        self.statusBar().showMessage("Ready")

    def _build_mer_tab(self) -> None:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        controls = QtWidgets.QGridLayout()
        self.mer_channel_combo = QtWidgets.QComboBox()
        self.mer_channel_combo.currentIndexChanged.connect(
            lambda *_args: self._schedule_preview_refresh()
        )
        self.mer_highpass_spin = self._double_spin(0.0, 20000.0, 300.0, 10.0)
        self.mer_lowpass_spin = self._double_spin(0.0, 20000.0, 0.0, 10.0)
        self.mer_filter_order_spin = self._int_spin(1, 8, 3)
        self.mer_preview_seconds_spin = self._double_spin(0.1, 10.0, 1.0, 0.1)
        self.mer_threshold_spin = self._double_spin(1.0, 10.0, 4.5, 0.1)
        self.mer_sort_checkbox = QtWidgets.QCheckBox("Fast clustering")
        self.mer_sort_checkbox.toggled.connect(lambda *_args: self._schedule_preview_refresh())
        self.mer_notch_combo, self.mer_notch_custom_spin = self._build_notch_controls()

        controls.addWidget(QtWidgets.QLabel("MER channel"), 0, 0)
        controls.addWidget(self.mer_channel_combo, 0, 1)
        controls.addWidget(QtWidgets.QLabel("Highpass (Hz)"), 0, 2)
        controls.addWidget(self.mer_highpass_spin, 0, 3)
        controls.addWidget(QtWidgets.QLabel("Lowpass (Hz)"), 0, 4)
        controls.addWidget(self.mer_lowpass_spin, 0, 5)
        controls.addWidget(QtWidgets.QLabel("Notch"), 1, 0)
        controls.addWidget(self.mer_notch_combo, 1, 1)
        controls.addWidget(self.mer_notch_custom_spin, 1, 2)
        controls.addWidget(QtWidgets.QLabel("Filter order"), 1, 3)
        controls.addWidget(self.mer_filter_order_spin, 1, 4)
        controls.addWidget(QtWidgets.QLabel("Preview (s)"), 1, 5)
        controls.addWidget(self.mer_preview_seconds_spin, 1, 6)
        controls.addWidget(QtWidgets.QLabel("Threshold MAD"), 0, 6)
        controls.addWidget(self.mer_threshold_spin, 0, 7)
        controls.addWidget(self.mer_sort_checkbox, 1, 7)
        layout.addLayout(controls)

        plots = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.mer_signal_plot = MatplotlibWidget(with_toolbar=False)
        self.mer_raster_plot = MatplotlibWidget(with_toolbar=False)
        self.waveform_plot = MatplotlibWidget(with_toolbar=False)
        plots.addWidget(self.mer_signal_plot)
        plots.addWidget(self.mer_raster_plot)
        plots.addWidget(self.waveform_plot)
        layout.addWidget(plots, 1)
        self.tabs.addTab(tab, "MER")

    def _build_lfp_tab(self) -> None:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        controls = QtWidgets.QGridLayout()
        self.lfp_channel_combo = QtWidgets.QComboBox()
        self.lfp_channel_combo.currentIndexChanged.connect(
            lambda *_args: self._schedule_preview_refresh()
        )
        self.band_preset_combo = QtWidgets.QComboBox()
        self.band_preset_combo.addItems(
            ["delta", "theta", "alpha", "beta", "highbeta", "gamma", "custom"]
        )
        self.band_preset_combo.currentTextChanged.connect(self._on_band_preset_changed)
        self.custom_band_low_spin = self._double_spin(0.0, 500.0, 13.0, 0.5)
        self.custom_band_high_spin = self._double_spin(0.0, 500.0, 30.0, 0.5)
        self.psd_method_combo = QtWidgets.QComboBox()
        self.psd_method_combo.addItems(["welch", "multitaper"])
        self.psd_method_combo.currentTextChanged.connect(
            lambda *_args: self._schedule_preview_refresh()
        )
        self.freq_min_spin = self._double_spin(0.0, 500.0, 0.0, 1.0)
        self.freq_max_spin = self._double_spin(1.0, 500.0, 100.0, 1.0)
        self.lfp_notch_combo, self.lfp_notch_custom_spin = self._build_notch_controls()
        self.lfp_notch_width_spin = self._double_spin(0.5, 20.0, 4.0, 0.5)

        controls.addWidget(QtWidgets.QLabel("LFP channel"), 0, 0)
        controls.addWidget(self.lfp_channel_combo, 0, 1)
        controls.addWidget(QtWidgets.QLabel("Band"), 0, 2)
        controls.addWidget(self.band_preset_combo, 0, 3)
        controls.addWidget(QtWidgets.QLabel("Custom low/high"), 0, 4)
        controls.addWidget(self.custom_band_low_spin, 0, 5)
        controls.addWidget(self.custom_band_high_spin, 0, 6)
        controls.addWidget(QtWidgets.QLabel("PSD method"), 1, 0)
        controls.addWidget(self.psd_method_combo, 1, 1)
        controls.addWidget(QtWidgets.QLabel("Freq min/max"), 1, 2)
        controls.addWidget(self.freq_min_spin, 1, 3)
        controls.addWidget(self.freq_max_spin, 1, 4)
        controls.addWidget(QtWidgets.QLabel("Notch"), 1, 5)
        controls.addWidget(self.lfp_notch_combo, 1, 6)
        controls.addWidget(self.lfp_notch_custom_spin, 1, 7)
        controls.addWidget(QtWidgets.QLabel("Width"), 1, 8)
        controls.addWidget(self.lfp_notch_width_spin, 1, 9)
        layout.addLayout(controls)

        plots = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        if PG_AVAILABLE:
            self.lfp_heatmap_pg = PyQtGraphHeatmapWidget()
            self.lfp_heatmap_plot = None
            plots.addWidget(self.lfp_heatmap_pg)
        else:
            self.lfp_heatmap_pg = None
            self.lfp_heatmap_plot = MatplotlibWidget(with_toolbar=False)
            plots.addWidget(self.lfp_heatmap_plot)
        self.bandpower_plot = MatplotlibWidget(with_toolbar=False)
        layout.addWidget(plots, 1)
        plots.addWidget(self.bandpower_plot)
        self.tabs.addTab(tab, "LFP")

    def _build_report_tab(self) -> None:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        panel_group = QtWidgets.QGroupBox("Include in PDF")
        panel_layout = QtWidgets.QGridLayout(panel_group)
        self.cover_checkbox = QtWidgets.QCheckBox("Cover page / metadata header")
        self.spike_raster_checkbox = QtWidgets.QCheckBox("Spike raster panel")
        self.lfp_heatmap_checkbox = QtWidgets.QCheckBox("LFP PSD heatmap panel")
        self.bandpower_checkbox = QtWidgets.QCheckBox("Bandpower vs depth panel")
        self.unit_fr_checkbox = QtWidgets.QCheckBox("Unit FR vs depth scatter + 2D hist")
        self.unit_amp_checkbox = QtWidgets.QCheckBox("Unit amplitude vs depth scatter + 2D hist")
        self.summary_table_checkbox = QtWidgets.QCheckBox("Per-depth summary table")
        self.mua_rms_checkbox = QtWidgets.QCheckBox("MUA/RMS vs depth")
        panel_widgets = [
            self.cover_checkbox,
            self.spike_raster_checkbox,
            self.lfp_heatmap_checkbox,
            self.bandpower_checkbox,
            self.unit_fr_checkbox,
            self.unit_amp_checkbox,
            self.summary_table_checkbox,
            self.mua_rms_checkbox,
        ]
        for index, widget in enumerate(panel_widgets):
            widget.toggled.connect(self._sync_report_panel_flags)
            panel_layout.addWidget(widget, index // 2, index % 2)
        layout.addWidget(panel_group)

        output_row = QtWidgets.QHBoxLayout()
        self.output_path_edit = QtWidgets.QLineEdit()
        self.output_browse_button = QtWidgets.QPushButton("Choose Output…")
        self.output_browse_button.clicked.connect(self._choose_output_path)
        output_row.addWidget(QtWidgets.QLabel("Output PDF"))
        output_row.addWidget(self.output_path_edit, 1)
        output_row.addWidget(self.output_browse_button)
        layout.addLayout(output_row)

        button_row = QtWidgets.QHBoxLayout()
        self.generate_report_button = QtWidgets.QPushButton("Generate Report")
        self.generate_report_button.clicked.connect(self._generate_report)
        self.save_config_button = QtWidgets.QPushButton("Save Config")
        self.save_config_button.clicked.connect(self._save_config)
        self.load_config_button = QtWidgets.QPushButton("Load Config")
        self.load_config_button.clicked.connect(self._load_config)
        button_row.addWidget(self.generate_report_button)
        button_row.addWidget(self.save_config_button)
        button_row.addWidget(self.load_config_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.progress_bar = QtWidgets.QProgressBar()
        layout.addWidget(self.progress_bar)
        self.log_window = QtWidgets.QPlainTextEdit()
        self.log_window.setReadOnly(True)
        layout.addWidget(self.log_window, 1)
        self.tabs.addTab(tab, "Report Builder")

    def _double_spin(
        self, minimum: float, maximum: float, value: float, step: float
    ) -> QtWidgets.QDoubleSpinBox:
        widget = QtWidgets.QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        widget.setSingleStep(step)
        widget.valueChanged.connect(lambda *_args: self._schedule_preview_refresh())
        return widget

    def _int_spin(self, minimum: int, maximum: int, value: int) -> QtWidgets.QSpinBox:
        widget = QtWidgets.QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        widget.valueChanged.connect(lambda *_args: self._schedule_preview_refresh())
        return widget

    def _build_notch_controls(self) -> tuple[QtWidgets.QComboBox, QtWidgets.QDoubleSpinBox]:
        combo = QtWidgets.QComboBox()
        combo.addItems(["Off", "50 Hz", "60 Hz", "Custom"])
        custom = self._double_spin(1.0, 500.0, 60.0, 1.0)
        custom.setEnabled(False)
        combo.currentTextChanged.connect(
            lambda text, spin=custom: spin.setEnabled(text == "Custom")
        )
        combo.currentTextChanged.connect(lambda *_args: self._schedule_preview_refresh())
        return combo, custom

    def _apply_gui_config_to_widgets(self) -> None:
        self.group_by_combo.setCurrentText(self.gui_config.report.render.group_by)
        self.mer_highpass_spin.setValue(self.gui_config.mer.highpass_hz)
        self.mer_lowpass_spin.setValue(self.gui_config.mer.lowpass_hz or 0.0)
        self.mer_filter_order_spin.setValue(self.gui_config.mer.filter_order)
        self.mer_preview_seconds_spin.setValue(self.gui_config.mer.preview_seconds)
        self.mer_threshold_spin.setValue(self.gui_config.mer.threshold_mad)
        self.mer_sort_checkbox.setChecked(self.gui_config.mer.run_fast_sorting)
        self._set_notch_controls(
            self.mer_notch_combo,
            self.mer_notch_custom_spin,
            self.gui_config.mer.notch_hz,
        )
        self.band_preset_combo.setCurrentText(self.gui_config.lfp_view.band_preset)
        self.custom_band_low_spin.setValue(self.gui_config.lfp_view.custom_band_low_hz)
        self.custom_band_high_spin.setValue(self.gui_config.lfp_view.custom_band_high_hz)
        self.psd_method_combo.setCurrentText(self.gui_config.lfp_view.psd_method)
        self.freq_min_spin.setValue(self.gui_config.lfp_view.freq_min_hz)
        self.freq_max_spin.setValue(self.gui_config.lfp_view.freq_max_hz)
        self._set_notch_controls(
            self.lfp_notch_combo,
            self.lfp_notch_custom_spin,
            self.gui_config.lfp_view.notch_hz,
        )
        self.lfp_notch_width_spin.setValue(self.gui_config.lfp_view.notch_width_hz)
        self.cover_checkbox.setChecked(self.gui_config.report.render.include_cover_page)
        self.spike_raster_checkbox.setChecked(
            self.gui_config.report.render.include_spike_raster_panel
        )
        self.lfp_heatmap_checkbox.setChecked(
            self.gui_config.report.render.include_lfp_heatmap_panel
        )
        self.bandpower_checkbox.setChecked(self.gui_config.report.render.include_bandpower_panel)
        self.unit_fr_checkbox.setChecked(self.gui_config.report.render.include_unit_fr_panel)
        self.unit_amp_checkbox.setChecked(
            self.gui_config.report.render.include_unit_amplitude_panel
        )
        self.summary_table_checkbox.setChecked(self.gui_config.report.render.include_summary_table)
        self.mua_rms_checkbox.setChecked(self.gui_config.report.render.include_mua_rms_panel)
        self._on_band_preset_changed(self.band_preset_combo.currentText())

    def _set_notch_controls(
        self,
        combo: QtWidgets.QComboBox,
        spin: QtWidgets.QDoubleSpinBox,
        value: float | None,
    ) -> None:
        if value is None:
            combo.setCurrentText("Off")
        elif abs(value - 50.0) < 1e-6:
            combo.setCurrentText("50 Hz")
        elif abs(value - 60.0) < 1e-6:
            combo.setCurrentText("60 Hz")
        else:
            combo.setCurrentText("Custom")
            spin.setValue(value)

    def _current_notch_value(
        self,
        combo: QtWidgets.QComboBox,
        spin: QtWidgets.QDoubleSpinBox,
    ) -> float | None:
        text = combo.currentText()
        if text == "Off":
            return None
        if text == "50 Hz":
            return 50.0
        if text == "60 Hz":
            return 60.0
        return float(spin.value())

    def _sync_gui_config_from_widgets(self) -> None:
        self.gui_config.report.render.group_by = self.group_by_combo.currentText()
        self.gui_config.mer.highpass_hz = float(self.mer_highpass_spin.value())
        lowpass = float(self.mer_lowpass_spin.value())
        self.gui_config.mer.lowpass_hz = lowpass if lowpass > 0 else None
        self.gui_config.mer.notch_hz = self._current_notch_value(
            self.mer_notch_combo,
            self.mer_notch_custom_spin,
        )
        self.gui_config.mer.filter_order = int(self.mer_filter_order_spin.value())
        self.gui_config.mer.preview_seconds = float(self.mer_preview_seconds_spin.value())
        self.gui_config.mer.threshold_mad = float(self.mer_threshold_spin.value())
        self.gui_config.mer.run_fast_sorting = self.mer_sort_checkbox.isChecked()

        self.gui_config.lfp_view.band_preset = self.band_preset_combo.currentText()
        self.gui_config.lfp_view.custom_band_low_hz = float(self.custom_band_low_spin.value())
        self.gui_config.lfp_view.custom_band_high_hz = float(self.custom_band_high_spin.value())
        self.gui_config.lfp_view.psd_method = self.psd_method_combo.currentText()
        self.gui_config.lfp_view.freq_min_hz = float(self.freq_min_spin.value())
        self.gui_config.lfp_view.freq_max_hz = float(self.freq_max_spin.value())
        self.gui_config.lfp_view.notch_hz = self._current_notch_value(
            self.lfp_notch_combo,
            self.lfp_notch_custom_spin,
        )
        self.gui_config.lfp_view.notch_width_hz = float(self.lfp_notch_width_spin.value())
        self._sync_report_panel_flags()
        self.gui_config.output_pdf = self.output_path_edit.text().strip() or None

    def _sync_report_panel_flags(self) -> None:
        self.gui_config.report.render.include_cover_page = self.cover_checkbox.isChecked()
        self.gui_config.report.render.include_spike_raster_panel = (
            self.spike_raster_checkbox.isChecked()
        )
        self.gui_config.report.render.include_lfp_heatmap_panel = (
            self.lfp_heatmap_checkbox.isChecked()
        )
        self.gui_config.report.render.include_bandpower_panel = self.bandpower_checkbox.isChecked()
        self.gui_config.report.render.include_unit_fr_panel = self.unit_fr_checkbox.isChecked()
        self.gui_config.report.render.include_unit_amplitude_panel = (
            self.unit_amp_checkbox.isChecked()
        )
        self.gui_config.report.render.include_summary_table = (
            self.summary_table_checkbox.isChecked()
        )
        self.gui_config.report.render.include_mua_rms_panel = self.mua_rms_checkbox.isChecked()

    def _set_busy(self, message: str) -> None:
        self.progress_bar.setRange(0, 0)
        self.statusBar().showMessage(message)

    def _set_idle_state(self) -> None:
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.statusBar().showMessage("Ready")

    def append_log(self, message: str) -> None:
        self.log_window.appendPlainText(message)

    def open_case_dialog(self) -> None:
        case_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Open AlphaOmega case")
        if not case_dir:
            return
        self._sync_gui_config_from_widgets()
        self._set_busy("Loading case...")
        self.append_log(f"Open case: {case_dir}")
        self.controller.load_case_async(
            case_dir,
            self.gui_config,
            on_result=self._on_case_loaded,
            on_error=self._on_task_error,
            on_finished=self._set_idle_state,
        )

    def _on_case_loaded(self, case_model: CaseModel) -> None:
        self.case_model = case_model
        self.case_summary.setPlainText("\n".join(case_model.summary_lines()))
        if not self.output_path_edit.text().strip():
            default_out = Path.cwd() / "output" / f"{case_model.case_name}_gui_report.pdf"
            self.output_path_edit.setText(str(default_out))
        self._populate_trajectory_list()
        self.append_log(f"Case loaded: {case_model.case_name}")

    def _populate_trajectory_list(self) -> None:
        self.trajectory_list.clear()
        if self.case_model is None:
            return
        for trajectory in self.case_model.sorted_trajectories(self.group_by_combo.currentText()):
            item = QtWidgets.QListWidgetItem(
                trajectory.display_label(self.group_by_combo.currentText())
            )
            item.setData(QtCore.Qt.UserRole, trajectory)
            self.trajectory_list.addItem(item)
        if self.trajectory_list.count():
            self.trajectory_list.setCurrentRow(0)

    def _on_group_by_changed(self, _value: str) -> None:
        self._sync_gui_config_from_widgets()
        self._populate_trajectory_list()

    def _on_trajectory_selected(self, row: int) -> None:
        if row < 0:
            return
        item = self.trajectory_list.item(row)
        if item is None:
            return
        trajectory = item.data(QtCore.Qt.UserRole)
        if not isinstance(trajectory, TrajectoryModel):
            return
        self.current_trajectory = trajectory
        self._populate_depth_table(trajectory)
        self._populate_channel_combos()
        self._schedule_preview_refresh()

    def _populate_depth_table(self, trajectory: TrajectoryModel) -> None:
        self._depth_table_loading = True
        self.depth_table.setRowCount(len(trajectory.depths))
        for row, depth_model in enumerate(trajectory.depths):
            depth_value = "" if depth_model.depth_mm is None else f"{depth_model.depth_mm:.3f}"
            depth_item = QtWidgets.QTableWidgetItem(depth_value)
            segment_item = QtWidgets.QTableWidgetItem(depth_model.segment_model.segment_id)
            depth_item.setData(QtCore.Qt.UserRole, depth_model)
            segment_item.setData(QtCore.Qt.UserRole, depth_model)
            self.depth_table.setItem(row, 0, depth_item)
            self.depth_table.setItem(row, 1, segment_item)
        self.depth_table.resizeColumnsToContents()
        self._depth_table_loading = False
        if trajectory.depths:
            self.depth_table.selectRow(0)

    def _populate_channel_combos(self) -> None:
        self.mer_channel_combo.blockSignals(True)
        self.lfp_channel_combo.blockSignals(True)
        self.mer_channel_combo.clear()
        self.lfp_channel_combo.clear()
        current = self._current_depth()
        if current is not None:
            self.mer_channel_combo.addItems(current.segment_model.available_mer_channels())
            self.lfp_channel_combo.addItems(current.segment_model.available_lfp_channels())
        self.mer_channel_combo.blockSignals(False)
        self.lfp_channel_combo.blockSignals(False)

    def _selected_depths(self) -> list[DepthModel]:
        rows = sorted({index.row() for index in self.depth_table.selectionModel().selectedRows()})
        out: list[DepthModel] = []
        for row in rows:
            item = self.depth_table.item(row, 0)
            if item is None:
                continue
            depth_model = item.data(QtCore.Qt.UserRole)
            if isinstance(depth_model, DepthModel):
                out.append(depth_model)
        if not out and self.current_trajectory is not None:
            return self.current_trajectory.depths
        return out

    def _current_depth(self) -> DepthModel | None:
        depths = self._selected_depths()
        if depths:
            return depths[0]
        if self.current_trajectory and self.current_trajectory.depths:
            return self.current_trajectory.depths[0]
        return None

    def _schedule_preview_refresh(self) -> None:
        if self.case_model is None or self.current_trajectory is None:
            return
        current_depth = self._current_depth()
        if current_depth is None:
            return
        self._sync_gui_config_from_widgets()
        selected_depths = self._selected_depths()
        self._mer_request_id += 1
        mer_request_id = self._mer_request_id
        self.controller.mer_preview_async(
            current_depth,
            selected_depths,
            max(self.mer_channel_combo.currentIndex(), 0),
            self.gui_config,
            on_result=lambda result, rid=mer_request_id: self._on_mer_preview_ready(rid, result),
            on_error=self._on_task_error,
        )
        self._lfp_request_id += 1
        lfp_request_id = self._lfp_request_id
        self.controller.lfp_preview_async(
            selected_depths,
            max(self.lfp_channel_combo.currentIndex(), 0),
            self.gui_config,
            on_result=lambda result, rid=lfp_request_id: self._on_lfp_preview_ready(rid, result),
            on_error=self._on_task_error,
        )

    def _on_mer_preview_ready(self, request_id: int, result) -> None:
        if request_id != self._mer_request_id:
            return
        report_config = self.gui_config.to_report_config()
        self.mer_signal_plot.draw_signal(
            result.times_s,
            result.values,
            title="MER Raw / Filtered Preview",
            ylabel=f"Amplitude ({result.units})",
            config=report_config,
        )
        self.mer_raster_plot.draw_raster(
            result.raster_depths_mm,
            result.raster_spike_times_s,
            report_config,
        )
        self.waveform_plot.draw_waveforms(
            result.waveform_time_ms, result.mean_waveforms, report_config
        )
        for warning in result.warnings:
            self.append_log(warning)

    def _on_lfp_preview_ready(self, request_id: int, result) -> None:
        if request_id != self._lfp_request_id:
            return
        report_config = self.gui_config.to_report_config()
        if self.lfp_heatmap_pg is not None:
            self.lfp_heatmap_pg.set_heatmap(
                result.depths_mm,
                result.freq_hz,
                result.psd_db_by_depth,
                result.band_limits_hz,
            )
        elif self.lfp_heatmap_plot is not None:
            self.lfp_heatmap_plot.draw_heatmap(
                result.depths_mm,
                result.freq_hz,
                result.psd_db_by_depth,
                result.band_limits_hz,
                result.band_name,
                report_config,
            )
        self.bandpower_plot.draw_bandpower(
            result.depths_mm,
            result.bandpower_db,
            result.band_name,
            result.band_limits_hz,
            report_config,
        )
        for warning in result.warnings:
            self.append_log(warning)

    def _on_task_error(self, message: str) -> None:
        self.append_log(message)
        self.statusBar().showMessage("Operation failed")
        QtWidgets.QMessageBox.critical(self, "AlphaOmega Reporter", message.splitlines()[-1])

    def _on_depth_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._depth_table_loading or item.column() != 0:
            return
        depth_model = item.data(QtCore.Qt.UserRole)
        if not isinstance(depth_model, DepthModel):
            return
        text = item.text().strip()
        try:
            new_depth = None if not text else float(text)
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid depth", "Depth must be numeric.")
            self._depth_table_loading = True
            item.setText("" if depth_model.depth_mm is None else f"{depth_model.depth_mm:.3f}")
            self._depth_table_loading = False
            return
        depth_model.segment_model.depth_mm = new_depth
        depth_model.segment_model.segment.meta["depth_mm"] = new_depth
        self._schedule_preview_refresh()

    def _on_band_preset_changed(self, value: str) -> None:
        is_custom = value == "custom"
        self.custom_band_low_spin.setEnabled(is_custom)
        self.custom_band_high_spin.setEnabled(is_custom)
        self._schedule_preview_refresh()

    def _choose_output_path(self) -> None:
        current = self.output_path_edit.text().strip() or str(Path.cwd() / "report.pdf")
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Choose report output",
            current,
            "PDF files (*.pdf)",
        )
        if path:
            self.output_path_edit.setText(path)

    def _generate_report(self) -> None:
        if self.case_model is None:
            QtWidgets.QMessageBox.warning(self, "No case loaded", "Open a case directory first.")
            return
        self._sync_gui_config_from_widgets()
        out_pdf = self.output_path_edit.text().strip()
        if not out_pdf:
            self._choose_output_path()
            out_pdf = self.output_path_edit.text().strip()
        if not out_pdf:
            return
        self._set_busy("Generating report...")
        self.append_log(f"Generate report: {out_pdf}")
        self.controller.generate_report_async(
            self.case_model,
            self.gui_config,
            out_pdf,
            on_result=self._on_report_generated,
            on_error=self._on_task_error,
            on_finished=self._set_idle_state,
        )

    def _on_report_generated(self, out_path: Path) -> None:
        self.append_log(f"Report written: {out_path}")
        QtWidgets.QMessageBox.information(self, "Report generated", str(out_path))

    def _save_config(self) -> None:
        self._sync_gui_config_from_widgets()
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save GUI config",
            str(Path.cwd() / "alphaomega_reporter_gui.yaml"),
            "YAML files (*.yaml *.yml);;JSON files (*.json)",
        )
        if not path:
            return
        saved = self.gui_config.save(path)
        self.append_log(f"Saved config: {saved}")

    def _load_config(self) -> None:
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load GUI config",
            str(Path.cwd()),
            "YAML files (*.yaml *.yml);;JSON files (*.json)",
        )
        if not path:
            return
        self.gui_config = GuiReportConfig.from_path(path)
        self._apply_gui_config_to_widgets()
        self.append_log(f"Loaded config: {path}")
        self._schedule_preview_refresh()
