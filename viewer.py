"""Three-axis DICOM viewer widget."""

from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QLabel, QSlider,
    QPushButton, QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

_BG = "#111320"
_LABEL_COLOR = "#7fa8ff"
_CROSSHAIR_COLOR = "#ff5a5a"


class _SliceView(QWidget):
    """Single orthogonal view: matplotlib canvas + position slider."""

    # Emitted while the cursor moves over the image.
    #   dict(voxel=(Z, Y, X), ...)  when hovering a valid pixel
    #   None                        when the cursor leaves the image
    hovered      = pyqtSignal(object)
    # Emitted on a left-click over the image, with the same info dict as hovered.
    clicked      = pyqtSignal(object)
    # Emitted when this view's slice index changes (int slice index).
    slice_changed = pyqtSignal(int)

    def __init__(self, label: str, axis: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.axis   = axis
        self.volume: np.ndarray | None = None
        self.affine: np.ndarray | None = None
        self.spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)  # (X, Y, Z) mm
        self.wc     = 0.0
        self.ww     = 2000.0
        self._im    = None
        self._vline = None   # crosshair Line2D (vertical)
        self._hline = None   # crosshair Line2D (horizontal)
        self._setup_ui(label)

    def _setup_ui(self, label: str) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        lbl = QLabel(label)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(
            f"color:{_LABEL_COLOR};font-weight:700;font-size:10px;letter-spacing:0.5px;"
        )
        layout.addWidget(lbl)

        self.fig = Figure(facecolor=_BG)
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.ax  = self.fig.add_subplot(111)
        self.ax.set_facecolor(_BG)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.canvas.installEventFilter(self)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_press_event",  self._on_click)
        self.canvas.mpl_connect("axes_leave_event",    self._on_leave)
        self.canvas.mpl_connect("figure_leave_event",  self._on_leave)
        layout.addWidget(self.canvas, stretch=1)

        # Slider with single-step prev/next buttons on either side
        slider_row = QHBoxLayout()
        slider_row.setContentsMargins(0, 0, 0, 0)
        slider_row.setSpacing(4)

        self.prev_btn = QPushButton("◀")
        self.prev_btn.setObjectName("small")
        self.prev_btn.setFixedWidth(28)
        self.prev_btn.setToolTip("Previous slice")
        self.prev_btn.clicked.connect(lambda: self._step(-1))
        slider_row.addWidget(self.prev_btn)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.valueChanged.connect(self._on_slider)
        slider_row.addWidget(self.slider, stretch=1)

        self.next_btn = QPushButton("▶")
        self.next_btn.setObjectName("small")
        self.next_btn.setFixedWidth(28)
        self.next_btn.setToolTip("Next slice")
        self.next_btn.clicked.connect(lambda: self._step(1))
        slider_row.addWidget(self.next_btn)

        layout.addLayout(slider_row)

        self.pos_lbl = QLabel("—")
        self.pos_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pos_lbl.setStyleSheet("color:#44495e;font-size:10px;")
        layout.addWidget(self.pos_lbl)

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self, volume: np.ndarray, wc: float, ww: float,
             affine: np.ndarray | None = None,
             spacing: tuple[float, float, float] | None = None) -> None:
        self.volume = volume
        self.affine = affine
        self.spacing = spacing or (1.0, 1.0, 1.0)
        self.wc = wc
        self.ww = ww
        self._im = None  # force re-creation of imshow
        self._vline = None
        self._hline = None
        n = volume.shape[self.axis]
        self.slider.blockSignals(True)
        self.slider.setMaximum(n - 1)
        self.slider.setValue(n // 2)
        self.slider.blockSignals(False)
        self._refresh(n // 2)

    def set_wl(self, wc: float, ww: float) -> None:
        self.wc = wc
        self.ww = ww
        self._refresh(self.slider.value())

    def clear(self) -> None:
        self.volume = None
        self.affine = None
        self.spacing = (1.0, 1.0, 1.0)
        self._im    = None
        self._vline = None
        self._hline = None
        self.ax.clear()
        self.ax.set_facecolor(_BG)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.slider.setMaximum(0)
        self.slider.setValue(0)
        self.pos_lbl.setText("—")
        self.canvas.draw_idle()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_slice(self, idx: int) -> np.ndarray:
        v = self.volume
        if   self.axis == 0: return v[idx, :, :]
        elif self.axis == 1: return np.flipud(v[:, idx, :])
        else:                return np.flipud(v[:, :, idx])

    def _on_slider(self, value: int) -> None:
        self._refresh(value)
        self.slice_changed.emit(value)

    def _step(self, delta: int) -> None:
        """Move exactly one slice forward (+1) or backward (-1), clamped."""
        self.slider.setValue(
            max(0, min(self.slider.maximum(), self.slider.value() + delta))
        )

    def set_slice(self, idx: int) -> None:
        """Jump to slice *idx* (clamped). Used to sync views on click."""
        idx = max(0, min(self.slider.maximum(), int(idx)))
        if self.slider.value() != idx:
            self.slider.setValue(idx)

    def _refresh(self, idx: int) -> None:
        if self.volume is None:
            return
        s    = self._get_slice(idx)
        vmin = self.wc - self.ww / 2.0
        vmax = self.wc + self.ww / 2.0

        if self._im is None:
            self.ax.clear()
            self.ax.set_facecolor(_BG)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            for sp in self.ax.spines.values():
                sp.set_visible(False)
            self._im = self.ax.imshow(
                s, cmap="gray", vmin=vmin, vmax=vmax,
                aspect="equal", interpolation="nearest", origin="upper",
            )
            # Crosshair lines — created once, hidden until a hover positions them.
            self._vline = self.ax.axvline(
                x=0, color=_CROSSHAIR_COLOR, lw=0.8, alpha=0.9, visible=False,
            )
            self._hline = self.ax.axhline(
                y=0, color=_CROSSHAIR_COLOR, lw=0.8, alpha=0.9, visible=False,
            )
        else:
            self._im.set_data(s)
            self._im.set_clim(vmin, vmax)

        n = self.volume.shape[self.axis]
        self.pos_lbl.setText(f"{idx + 1} / {n}")
        self.canvas.draw_idle()

    # ── Coordinate mapping ─────────────────────────────────────────────────────

    def _cursor_to_voxel(self, xdata: float, ydata: float) -> tuple[int, int, int] | None:
        """Map cursor data coords in this view to a full volume index (Z, Y, X).

        The display volume has shape (Z, Y, X) = (n_slices, rows, cols).
        imshow uses origin='upper', so xdata=column, ydata=row of the shown 2D
        slice. Coronal/sagittal slices are shown flipud, so their vertical axis
        (Z) is inverted relative to the raw array.
        """
        if self.volume is None:
            return None
        nz, ny, nx = self.volume.shape
        col = int(round(xdata))
        row = int(round(ydata))
        idx = self.slider.value()

        if self.axis == 0:            # axial: shown array = v[idx, :, :] → (Y, X)
            z, y, x = idx, row, col
        elif self.axis == 1:          # coronal: flipud(v[:, idx, :]) → (Z, X)
            z, y, x = (nz - 1 - row), idx, col
        else:                          # sagittal: flipud(v[:, :, idx]) → (Z, Y)
            z, y, x = (nz - 1 - row), col, idx

        if not (0 <= z < nz and 0 <= y < ny and 0 <= x < nx):
            return None
        return z, y, x

    def update_crosshair(self, voxel: tuple[int, int, int] | None) -> None:
        """Show/position the crosshair for a target *voxel*, or hide it.

        The crosshair is only drawn if this view is currently displaying the
        slice that contains *voxel* (i.e. the matching slider index) — so it
        appears across all three views only when the correct slice is opened.
        """
        if self._vline is None or self._hline is None:
            return
        pos = self._voxel_to_cursor(voxel) if voxel is not None else None
        if pos is None:
            changed = self._vline.get_visible() or self._hline.get_visible()
            self._vline.set_visible(False)
            self._hline.set_visible(False)
            if changed:
                self.canvas.draw_idle()
            return
        xd, yd = pos
        self._vline.set_xdata([xd, xd])
        self._hline.set_ydata([yd, yd])
        self._vline.set_visible(True)
        self._hline.set_visible(True)
        self.canvas.draw_idle()

    def _voxel_to_cursor(self, voxel: tuple[int, int, int]) -> tuple[float, float] | None:
        """Inverse of _cursor_to_voxel: (Z, Y, X) → (xdata, ydata) for this view,
        or None if this view is not on the slice containing the voxel."""
        if self.volume is None:
            return None
        nz, ny, nx = self.volume.shape
        z, y, x = voxel
        idx = self.slider.value()
        if self.axis == 0:
            if idx != z:
                return None
            return float(x), float(y)
        elif self.axis == 1:
            if idx != y:
                return None
            return float(x), float(nz - 1 - z)
        else:
            if idx != x:
                return None
            return float(y), float(nz - 1 - z)

    # ── Mouse handlers ─────────────────────────────────────────────────────────

    def _build_info(self, voxel: tuple[int, int, int]) -> dict:
        """Assemble the coordinate info dict for a volume index (Z, Y, X)."""
        z, y, x = voxel
        value = float(self.volume[z, y, x])
        ras = None
        if self.affine is not None:
            r = self.affine @ np.array([z, y, x, 1.0])
            ras = (float(r[0]), float(r[1]), float(r[2]))
        sx, sy, sz = self.spacing
        world = (x * sx, y * sy, z * sz)   # image-coord mm from volume corner
        return {
            "voxel": (x, y, z), "index": voxel,
            "ras": ras, "world": world, "value": value,
        }

    def _on_motion(self, event) -> None:
        if self.volume is None or event.inaxes is not self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        voxel = self._cursor_to_voxel(event.xdata, event.ydata)
        if voxel is None:
            self.hovered.emit(None)
            return
        self.hovered.emit(self._build_info(voxel))

    def _on_click(self, event) -> None:
        if self.volume is None or event.inaxes is not self.ax:
            return
        if event.button != 1 or event.xdata is None or event.ydata is None:
            return
        voxel = self._cursor_to_voxel(event.xdata, event.ydata)
        if voxel is not None:
            self.clicked.emit(self._build_info(voxel))

    def _on_leave(self, event) -> None:
        self.hovered.emit(None)

    def eventFilter(self, obj, event):  # type: ignore[override]
        """Scroll wheel changes slice."""
        from PyQt6.QtCore import QEvent
        if obj is self.canvas and event.type() == QEvent.Type.Wheel:
            delta = event.angleDelta().y()
            self._step(-1 if delta > 0 else 1)
            return True
        return super().eventFilter(obj, event)


class DicomViewer(QWidget):
    """Three-axis DICOM viewer (axial / coronal / sagittal)."""

    # Emitted with hover info dict (voxel / ras / value) or None on leave.
    coords_changed = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._last_voxel: tuple[int, int, int] | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        vsplit = QSplitter(Qt.Orientation.Vertical)
        vsplit.setStyleSheet("QSplitter::handle{background:#1e2135;height:5px;}")

        # Top row: large axial view
        self.axial = _SliceView("AXIAL  (Z)", axis=0)
        vsplit.addWidget(self.axial)

        # Bottom row: coronal + sagittal side by side
        hsplit = QSplitter(Qt.Orientation.Horizontal)
        hsplit.setStyleSheet("QSplitter::handle{background:#1e2135;width:5px;}")
        self.coronal   = _SliceView("CORONAL  (Y)", axis=1)
        self.sagittal  = _SliceView("SAGITTAL  (X)", axis=2)
        hsplit.addWidget(self.coronal)
        hsplit.addWidget(self.sagittal)
        vsplit.addWidget(hsplit)

        vsplit.setStretchFactor(0, 3)
        vsplit.setStretchFactor(1, 2)
        layout.addWidget(vsplit)

        # Cross-view coordination
        for view in (self.axial, self.coronal, self.sagittal):
            view.hovered.connect(self._on_hover)
            view.clicked.connect(self._on_click)
            view.slice_changed.connect(self._on_slice_changed)

    @property
    def _views(self) -> tuple[_SliceView, _SliceView, _SliceView]:
        return (self.axial, self.coronal, self.sagittal)

    def load_volume(self, volume: np.ndarray, wc: float, ww: float,
                    affine: np.ndarray | None = None,
                    spacing: tuple[float, float, float] | None = None) -> None:
        self._last_voxel = None
        for view in self._views:
            view.load(volume, wc, ww, affine, spacing)

    # ── Cross-view crosshair coordination ──────────────────────────────────────

    def _on_hover(self, info) -> None:
        voxel = info["index"] if info else None
        self._last_voxel = voxel
        for view in self._views:
            view.update_crosshair(voxel)
        self.coords_changed.emit(info)

    def _on_click(self, info) -> None:
        """Sync all views to the clicked point so it becomes their intersection."""
        if not info:
            return
        voxel = info["index"]
        z, y, x = voxel
        self._last_voxel = voxel
        # Each view jumps to the slice containing the clicked voxel.
        self.axial.set_slice(z)     # axis 0 → Z
        self.coronal.set_slice(y)   # axis 1 → Y
        self.sagittal.set_slice(x)  # axis 2 → X
        for view in self._views:
            view.update_crosshair(voxel)
        self.coords_changed.emit(info)

    def _on_slice_changed(self, _value: int) -> None:
        # Re-evaluate crosshair visibility against the last hovered voxel so it
        # appears in a sibling view once that view is scrolled to the right slice.
        for view in self._views:
            view.update_crosshair(self._last_voxel)

    def set_wl(self, wc: float, ww: float) -> None:
        for view in (self.axial, self.coronal, self.sagittal):
            view.set_wl(wc, ww)

    def clear(self) -> None:
        self._last_voxel = None
        for view in self._views:
            view.clear()
        self.coords_changed.emit(None)
