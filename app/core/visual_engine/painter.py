"""Lightweight painter used for preview rendering.

The module tries to use Qt's :class:`QPainter` for drawing.  When the Qt
libraries are not available (e.g. in headless environments without ``libGL``)
it falls back to a pure Pillow/numpy implementation.  Only a small subset of
the drawing API required by the tests is implemented.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional Qt imports
try:  # pragma: no cover - exercised in environments with Qt available
    from PySide6.QtCore import QPointF, QRectF, Qt  # type: ignore
    from PySide6.QtGui import (  # type: ignore
        QBrush,
        QColor,
        QImage,
        QLinearGradient,
        QPainter,
        QPen,
        QRadialGradient,
    )

    _HAVE_QT = True
except Exception:  # pragma: no cover - fallback path
    _HAVE_QT = False


# ---------------------------------------------------------------------------
# Helper utilities
def _hex_to_rgb(c: str) -> Tuple[int, int, int]:
    c = c.lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    return tuple(int(c[i : i + 2], 16) for i in (0, 2, 4))


# ---------------------------------------------------------------------------
if _HAVE_QT:  # pragma: no cover - Qt path

    __all__ = ["Painter"]

    def _to_qcolor(color: Tuple[int, int, int] | str) -> QColor:
        if isinstance(color, tuple):
            r, g, b = color
            return QColor(int(r), int(g), int(b))
        qcol = QColor(color)
        if not qcol.isValid():
            raise ValueError(f"Invalid color: {color}")
        return qcol

    class Painter:
        """Qt based painter drawing onto a :class:`QImage`."""

        def __init__(self, width: int, height: int, headless: bool = False) -> None:
            self.width = int(width)
            self.height = int(height)
            self.headless = bool(headless)
            self._image: QImage | None = None
            self._painter: QPainter | None = None

        def begin_frame(self) -> None:
            fmt = QImage.Format_RGB888
            self._image = QImage(self.width, self.height, fmt)
            self._painter = QPainter(self._image)
            self._painter.setRenderHint(QPainter.Antialiasing)

        def end_frame(self) -> "QImage | np.ndarray":
            # Cierra el QPainter si sigue activo
            if self._painter is not None:
                self._painter.end()
                self._painter = None

            assert self._image is not None
            img = self._image

            if self.headless:
                # En PySide6, constBits() devuelve un memoryview (sin setsize).
                # Convertimos el buffer en un array 2D (height x bytesPerLine) y recortamos a width*channels.
                import numpy as np

                h = img.height()
                w = img.width()
                stride = img.bytesPerLine()  # puede ser > w*channels por alineación
                ptr = img.constBits()  # memoryview
                arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, stride))

                if img.format() == QImage.Format.Format_RGB888:
                    ch = 3
                    frame = arr[:, : w * ch].reshape((h, w, ch)).copy()
                elif img.format() in (
                    QImage.Format.Format_RGBA8888,
                    QImage.Format.Format_RGBA8888_Premultiplied,
                ):
                    ch = 4
                    rgba = arr[:, : w * ch].reshape((h, w, ch))
                    # Si tu pipeline espera RGB, quitamos alfa
                    frame = rgba[:, :, :3].copy()
                else:
                    # Fallback: convierte a RGB888 y repite el proceso (más lento)
                    rgb = img.convertToFormat(QImage.Format.Format_RGB888)
                    h2, w2, stride2 = rgb.height(), rgb.width(), rgb.bytesPerLine()
                    ptr2 = rgb.constBits()
                    arr2 = np.frombuffer(ptr2, dtype=np.uint8).reshape((h2, stride2))
                    frame = arr2[:, : w2 * 3].reshape((h2, w2, 3)).copy()

                return frame  # np.ndarray (H, W, 3) uint8

            # En modo no headless, devolvemos el QImage para que la UI lo muestre
            return img

        def clear(self, color: Tuple[int, int, int] | str) -> None:
            assert self._painter is not None
            self._painter.fillRect(0, 0, self.width, self.height, _to_qcolor(color))

        def fill_linear_gradient(self, colors: list[str], angle_deg: float) -> None:
            assert self._painter is not None
            cx, cy = self.width / 2.0, self.height / 2.0
            diag = math.hypot(self.width, self.height)
            ang = math.radians(angle_deg)
            dx = math.cos(ang) * diag / 2.0
            dy = math.sin(ang) * diag / 2.0
            grad = QLinearGradient(cx - dx, cy - dy, cx + dx, cy + dy)
            n = len(colors)
            if n == 1:
                grad.setColorAt(0.0, _to_qcolor(colors[0]))
                grad.setColorAt(1.0, _to_qcolor(colors[0]))
            else:
                for i, c in enumerate(colors):
                    grad.setColorAt(i / (n - 1), _to_qcolor(c))
            self._painter.fillRect(0, 0, self.width, self.height, QBrush(grad))

        def fill_radial_gradient(
            self,
            colors: list[str],
            center_ndc: Tuple[float, float] = (0.0, 0.0),
            radius_ndc: float = 1.0,
        ) -> None:
            assert self._painter is not None
            cx, cy = self._ndc_to_px(center_ndc[0], center_ndc[1])
            radius = radius_ndc * self._aspect_min() / 2.0
            grad = QRadialGradient(cx, cy, radius)
            n = len(colors)
            if n == 1:
                grad.setColorAt(0.0, _to_qcolor(colors[0]))
                grad.setColorAt(1.0, _to_qcolor(colors[0]))
            else:
                for i, c in enumerate(colors):
                    grad.setColorAt(i / (n - 1), _to_qcolor(c))
            self._painter.fillRect(0, 0, self.width, self.height, QBrush(grad))

        def stroke_circle_ndc(
            self,
            center: Tuple[float, float] = (0.0, 0.0),
            radius_ndc: float = 0.5,
            thickness_ndc: float = 0.02,
            color: str = "#FFFFFF",
            alpha: float = 1.0,
        ) -> None:
            assert self._painter is not None
            x, y = self._ndc_to_px(center[0], center[1])
            radius = radius_ndc * self._aspect_min() / 2.0
            thickness = max(1.0, thickness_ndc * self._aspect_min() / 2.0)
            qcol = _to_qcolor(color)
            qcol.setAlphaF(alpha)
            pen = QPen(qcol)
            pen.setWidthF(thickness)
            self._painter.setPen(pen)
            self._painter.setBrush(Qt.NoBrush)
            self._painter.drawEllipse(QPointF(x, y), radius, radius)

        def draw_radial_bars(
            self,
            base_radius_ndc: float,
            values: np.ndarray,
            thickness_ndc: float,
            roundness: float,
            color: str = "#FFFFFF",
            alpha: float = 1.0,
        ) -> None:
            assert self._painter is not None
            cx, cy = self.width / 2.0, self.height / 2.0
            base = base_radius_ndc * self._aspect_min() / 2.0
            max_r = self._aspect_min() / 2.0
            thickness = max(1.0, thickness_ndc * self._aspect_min() / 2.0)
            pen = QPen(_to_qcolor(color))
            pen.setWidthF(thickness)
            pen.setCapStyle(Qt.RoundCap if roundness > 0 else Qt.FlatCap)
            col = pen.color()
            col.setAlphaF(alpha)
            pen.setColor(col)
            self._painter.setPen(pen)
            n = len(values)
            for i, v in enumerate(values):
                ang = 2.0 * math.pi * i / n - math.pi / 2.0
                r = base + float(v) * (max_r - base)
                x1 = cx + base * math.cos(ang)
                y1 = cy + base * math.sin(ang)
                x2 = cx + r * math.cos(ang)
                y2 = cy + r * math.sin(ang)
                self._painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        def blit_center_image(
            self,
            pil_image: "Image.Image",
            scale: float = 1.0,
            rotation_deg: float = 0.0,
            alpha: float = 1.0,
        ) -> None:
            assert self._painter is not None
            from PIL import ImageQt  # lazy import

            qimg = QImage(ImageQt.ImageQt(pil_image.convert("RGBA")))
            w = qimg.width() * scale
            h = qimg.height() * scale
            self._painter.save()
            self._painter.translate(self.width / 2.0, self.height / 2.0)
            self._painter.rotate(rotation_deg)
            self._painter.setOpacity(alpha)
            rect = QRectF(-w / 2.0, -h / 2.0, w, h)
            self._painter.drawImage(rect, qimg)
            self._painter.restore()

        def _aspect_min(self) -> float:
            return float(min(self.width, self.height))

        def _ndc_to_px(self, x: float, y: float) -> Tuple[float, float]:
            half = self._aspect_min() / 2.0
            px = self.width / 2.0 + x * half
            py = self.height / 2.0 - y * half
            return px, py

else:  # Pillow based fallback -------------------------------------------------

    from PIL import Image, ImageDraw

    __all__ = ["Painter"]

    def _rgba(
        color: Tuple[int, int, int] | str, alpha: float
    ) -> Tuple[int, int, int, int]:
        if isinstance(color, str):
            r, g, b = _hex_to_rgb(color)
        else:
            r, g, b = color
        a = int(max(0.0, min(alpha, 1.0)) * 255)
        return r, g, b, a

    class Painter:
        """Fallback painter using Pillow and numpy."""

        def __init__(self, width: int, height: int, headless: bool = False) -> None:
            self.width = int(width)
            self.height = int(height)
            self.headless = bool(headless)
            self._img: Image.Image | None = None
            self._draw: ImageDraw.ImageDraw | None = None

        def begin_frame(self) -> None:
            self._img = Image.new("RGB", (self.width, self.height), (0, 0, 0))
            self._draw = ImageDraw.Draw(self._img, "RGBA")

        def end_frame(self) -> Image.Image | np.ndarray:
            assert self._img is not None
            if self.headless:
                return np.array(self._img)
            return self._img

        def clear(self, color: Tuple[int, int, int] | str) -> None:
            assert self._draw is not None
            if isinstance(color, str):
                fill = _hex_to_rgb(color)
            else:
                fill = color
            self._draw.rectangle([0, 0, self.width, self.height], fill=fill)

        def fill_linear_gradient(self, colors: list[str], angle_deg: float) -> None:
            arr = self._linear_gradient_array(colors, angle_deg)
            self._img = Image.fromarray(arr.astype(np.uint8), "RGB")
            self._draw = ImageDraw.Draw(self._img, "RGBA")

        def fill_radial_gradient(
            self,
            colors: list[str],
            center_ndc: Tuple[float, float] = (0.0, 0.0),
            radius_ndc: float = 1.0,
        ) -> None:
            # Simple radial gradient using numpy
            cx, cy = self._ndc_to_px(center_ndc[0], center_ndc[1])
            radius = radius_ndc * self._aspect_min() / 2.0
            y, x = np.ogrid[: self.height, : self.width]
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / radius
            dist = np.clip(dist, 0.0, 1.0)
            stops = np.linspace(0, 1, len(colors))
            cols = np.array([_hex_to_rgb(c) for c in colors], dtype=float)
            r = np.interp(dist, stops, cols[:, 0])
            g = np.interp(dist, stops, cols[:, 1])
            b = np.interp(dist, stops, cols[:, 2])
            arr = np.dstack([r, g, b]).astype(np.uint8)
            self._img = Image.fromarray(arr, "RGB")
            self._draw = ImageDraw.Draw(self._img, "RGBA")

        def stroke_circle_ndc(
            self,
            center: Tuple[float, float] = (0.0, 0.0),
            radius_ndc: float = 0.5,
            thickness_ndc: float = 0.02,
            color: str = "#FFFFFF",
            alpha: float = 1.0,
        ) -> None:
            assert self._draw is not None
            cx, cy = self._ndc_to_px(center[0], center[1])
            radius = radius_ndc * self._aspect_min() / 2.0
            thickness = max(1.0, thickness_ndc * self._aspect_min() / 2.0)
            bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
            self._draw.ellipse(bbox, outline=_rgba(color, alpha), width=int(thickness))

        def draw_radial_bars(
            self,
            base_radius_ndc: float,
            values: np.ndarray,
            thickness_ndc: float,
            roundness: float,
            color: str = "#FFFFFF",
            alpha: float = 1.0,
        ) -> None:
            assert self._draw is not None
            cx, cy = self.width / 2.0, self.height / 2.0
            base = base_radius_ndc * self._aspect_min() / 2.0
            max_r = self._aspect_min() / 2.0
            thickness = max(1.0, thickness_ndc * self._aspect_min() / 2.0)
            col = _rgba(color, alpha)
            n = len(values)
            for i, v in enumerate(values):
                ang = 2.0 * math.pi * i / n - math.pi / 2.0
                r = base + float(v) * (max_r - base)
                x1 = cx + base * math.cos(ang)
                y1 = cy + base * math.sin(ang)
                x2 = cx + r * math.cos(ang)
                y2 = cy + r * math.sin(ang)
                self._draw.line([(x1, y1), (x2, y2)], fill=col, width=int(thickness))

        def blit_center_image(
            self,
            pil_image: "Image.Image",
            scale: float = 1.0,
            rotation_deg: float = 0.0,
            alpha: float = 1.0,
        ) -> None:
            assert self._img is not None
            img = pil_image.convert("RGBA")
            if scale != 1.0:
                w = max(1, int(img.width * scale))
                h = max(1, int(img.height * scale))
                img = img.resize((w, h))
            if rotation_deg:
                img = img.rotate(rotation_deg, expand=True)
            if alpha < 1.0:
                a = img.split()[3]
                a = a.point(lambda p: int(p * alpha))
                img.putalpha(a)
            x = int(self.width / 2 - img.width / 2)
            y = int(self.height / 2 - img.height / 2)
            self._img.paste(img, (x, y), img)

        # ------------------------------------------------------------------
        def _aspect_min(self) -> float:
            return float(min(self.width, self.height))

        def _ndc_to_px(self, x: float, y: float) -> Tuple[float, float]:
            half = self._aspect_min() / 2.0
            px = self.width / 2.0 + x * half
            py = self.height / 2.0 - y * half
            return px, py

        def _linear_gradient_array(
            self, colors: list[str], angle_deg: float
        ) -> np.ndarray:
            w, h = self.width, self.height
            ang = math.radians(angle_deg)
            xx, yy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
            t = xx * math.cos(ang) + yy * math.sin(ang)
            t = (t - t.min()) / (t.max() - t.min())
            stops = np.linspace(0, 1, len(colors))
            cols = np.array([_hex_to_rgb(c) for c in colors], dtype=float)
            r = np.interp(t, stops, cols[:, 0])
            g = np.interp(t, stops, cols[:, 1])
            b = np.interp(t, stops, cols[:, 2])
            return np.dstack([r, g, b])
