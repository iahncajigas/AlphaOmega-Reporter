from __future__ import annotations

import sys

try:
    from PySide6 import QtWidgets
except ImportError as exc:  # pragma: no cover - exercised only without GUI extra
    QtWidgets = None
    _QT_IMPORT_ERROR = exc
else:
    _QT_IMPORT_ERROR = None


def _require_qt() -> None:
    if QtWidgets is None:
        raise SystemExit(
            "PySide6 is not installed. Install the GUI extra with "
            "`python -m pip install -e .[gui]`."
        ) from _QT_IMPORT_ERROR


def create_application() -> QtWidgets.QApplication:
    _require_qt()
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
        app.setApplicationName("AlphaOmega Reporter")
        app.setOrganizationName("AlphaOmega Reporter")
    return app


def main() -> int:
    _require_qt()
    from alphaomega_reporter_gui.main_window import MainWindow

    app = create_application()
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
