from __future__ import annotations

import sys

from PySide6 import QtWidgets

from alphaomega_reporter_gui.main_window import MainWindow


def create_application() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
        app.setApplicationName("AlphaOmega Reporter")
        app.setOrganizationName("AlphaOmega Reporter")
    return app


def main() -> int:
    app = create_application()
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
