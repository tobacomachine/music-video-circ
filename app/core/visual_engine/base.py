"""Layer interfaces and management for the visual engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from .painter import Painter
    from .types import RenderCtx


class Layer(Protocol):
    """Rendering layer protocol."""

    def update(self, ctx: "RenderCtx", dt: float) -> None:
        """Update internal state based on ``ctx`` and delta time ``dt``."""

    def render(self, ctx: "RenderCtx", painter: "Painter") -> None:
        """Render the layer using ``painter``."""


@dataclass
class LayerStack:
    """Manage and dispatch a sequence of layers."""

    layers: List[Layer]

    def __init__(self) -> None:  # noqa: D401 - brief dataclass init
        self.layers = []

    def add(self, layer: Layer) -> None:
        """Append ``layer`` to the stack."""

        self.layers.append(layer)

    def update(self, ctx: "RenderCtx", dt: float) -> None:
        """Update layers in order."""

        for layer in self.layers:
            layer.update(ctx, dt)

    def render(self, ctx: "RenderCtx", painter: "Painter") -> None:
        """Render layers in order."""

        for layer in self.layers:
            layer.render(ctx, painter)


__all__ = ["Layer", "LayerStack"]

