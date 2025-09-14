from PySide6.QtWidgets import QMainWindow, QWidget


class MainWindow(QMainWindow):
    """Main window for NCSViz."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NCSViz")
        self.setCentralWidget(QWidget())
