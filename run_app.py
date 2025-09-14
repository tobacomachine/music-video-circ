from PySide6.QtWidgets import QApplication
from app.ui.main_window import MainWindow
import sys


def main() -> None:
    """Entry point for the NCSViz application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
