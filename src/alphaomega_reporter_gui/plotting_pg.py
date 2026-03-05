from __future__ import annotations

import numpy as np

try:
    import pyqtgraph as pg
    from PySide6 import QtCore, QtWidgets

    PG_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    pg = None
    QtCore = None
    QtWidgets = None
    PG_AVAILABLE = False


if PG_AVAILABLE:

    class PyQtGraphHeatmapWidget(QtWidgets.QWidget):
        def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
            super().__init__(parent)
            self.graphics = pg.GraphicsLayoutWidget()
            self.plot = self.graphics.addPlot()
            self.plot.setLabel("bottom", "Depth (mm)")
            self.plot.setLabel("left", "Frequency (Hz)")
            self.image = pg.ImageItem()
            self.plot.addItem(self.image)
            self.band_region = pg.LinearRegionItem(orientation="horizontal", movable=False)
            self.band_region.setBrush(pg.mkBrush(244, 211, 94, 70))
            self.plot.addItem(self.band_region)
            self.colorbar = pg.ColorBarItem(values=(0, 1), colorMap="viridis")
            self.colorbar.setImageItem(self.image, insert_in=self.plot)
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.graphics)

        def clear(self, message: str = "No LFP stream found") -> None:
            self.plot.clear()
            self.plot.addItem(pg.TextItem(message, anchor=(0.5, 0.5)))

        def set_heatmap(
            self,
            depths_mm: list[float],
            freq_hz: np.ndarray,
            psd_db_by_depth: np.ndarray,
            band_limits_hz: tuple[float, float],
        ) -> None:
            if psd_db_by_depth.size == 0 or freq_hz.size == 0 or not depths_mm:
                self.clear()
                return
            self.plot.clear()
            self.plot.addItem(self.image)
            self.plot.addItem(self.band_region)
            data = np.asarray(psd_db_by_depth, dtype=np.float64).T
            rect = QtCore.QRectF(
                float(min(depths_mm)),
                float(freq_hz[0]),
                float(max(depths_mm) - min(depths_mm) or 1.0),
                float(freq_hz[-1] - freq_hz[0] or 1.0),
            )
            self.image.setImage(data, autoLevels=True)
            self.image.setRect(rect)
            self.band_region.setRegion(list(band_limits_hz))

else:

    class PyQtGraphHeatmapWidget:  # pragma: no cover - optional dependency fallback
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("pyqtgraph is not available")
