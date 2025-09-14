from __future__ import annotations

import sys

from PySide6.QtWidgets import QAction, QApplication, QMainWindow


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NCS Visualizer (MVP)")

        file_menu = self.menuBar().addMenu("Archivo")
        open_action = QAction("Abrir", self)
        close_action = QAction("Cerrar", self)
        file_menu.addAction(open_action)
        file_menu.addAction(close_action)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
