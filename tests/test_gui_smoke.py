from __future__ import annotations

from alphaomega_reporter_gui.app import create_application
from alphaomega_reporter_gui.main_window import MainWindow


def test_gui_main_window_smoke() -> None:
    app = create_application()
    window = MainWindow()
    try:
        assert app.applicationName() == "AlphaOmega Reporter"
        assert window.windowTitle() == "AlphaOmega Reporter"
        assert window.tabs.count() == 3
    finally:
        window.close()
