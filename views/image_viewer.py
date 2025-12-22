"""
Image Viewer Module

Custom QGraphicsView for displaying large satellite images with zoom and pan support.
"""

from typing import Optional

from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QWheelEvent, QMouseEvent, QKeyEvent, QPainter


class ImageViewer(QGraphicsView):
    """
    Custom QGraphicsView for displaying satellite images.
    
    Supports:
    - Mouse wheel zoom (anchored at cursor position)
    - Middle/Left button drag for panning
    - Synchronized zoom/pan with another ImageViewer
    """
    
    # Signals for synchronization
    zoomChanged = pyqtSignal(float)  # Emitted when zoom level changes
    panChanged = pyqtSignal(QPointF)  # Emitted when pan position changes
    
    def __init__(self, parent=None):
        """Initialize the ImageViewer."""
        super().__init__(parent)
        
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._zoom_factor = 1.0
        self._min_zoom = 0.1
        self._max_zoom = 10.0
        
        # Enable drag mode for panning
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        
        # Enable smooth rendering
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        
        # Enable mouse tracking for better interaction
        self.setMouseTracking(True)
        
        # Set background color
        self.setBackgroundBrush(Qt.GlobalColor.darkGray)
        
        # Center on scene
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
    
    def load_image(self, image: QImage):
        """
        Load and display an image.
        
        Args:
            image: QImage to display
        """
        # Clear previous image
        if self._pixmap_item:
            self._scene.removeItem(self._pixmap_item)
        
        # Create pixmap from image
        pixmap = QPixmap.fromImage(image)
        
        # Add to scene
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self._pixmap_item)
        
        # Set scene rect to image size
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        
        # Fit image to view
        self.fit_to_view()
    
    def fit_to_view(self):
        """Fit the image to the current view size."""
        if self._pixmap_item:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom_factor = self.transform().m11()
            self.zoomChanged.emit(self._zoom_factor)
    
    def wheelEvent(self, event: QWheelEvent):
        """
        Handle mouse wheel events for zooming.
        
        Zoom is anchored at the cursor position.
        """
        if self._pixmap_item is None:
            return
        
        # Get zoom factor
        zoom_delta = 1.15
        if event.angleDelta().y() < 0:
            zoom_delta = 1.0 / zoom_delta
        
        # Calculate new zoom level
        new_zoom = self._zoom_factor * zoom_delta
        new_zoom = max(self._min_zoom, min(self._max_zoom, new_zoom))
        
        if new_zoom == self._zoom_factor:
            return
        
        # Get mouse position in scene coordinates
        old_pos = self.mapToScene(event.position().toPoint())
        
        # Apply zoom
        self._zoom_factor = new_zoom
        self.setTransform(self.transform().scale(zoom_delta, zoom_delta))
        
        # Get new mouse position
        new_pos = self.mapToScene(event.position().toPoint())
        
        # Adjust view to keep mouse position fixed
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())
        
        self.zoomChanged.emit(self._zoom_factor)
        event.accept()
    
    def set_zoom(self, zoom_factor: float, center_point: Optional[QPointF] = None):
        """
        Set zoom level programmatically.
        
        Args:
            zoom_factor: Zoom factor (1.0 = 100%)
            center_point: Point to center zoom on (in scene coordinates)
        """
        if self._pixmap_item is None:
            return
        
        zoom_factor = max(self._min_zoom, min(self._max_zoom, zoom_factor))
        
        if center_point is None:
            center_point = self.mapToScene(self.viewport().rect().center())
        
        # Calculate scale factor
        current_zoom = self.transform().m11()
        scale_factor = zoom_factor / current_zoom
        
        # Apply zoom
        self._zoom_factor = zoom_factor
        self.setTransform(self.transform().scale(scale_factor, scale_factor))
        
        # Adjust view to keep center point fixed
        new_center = self.mapToScene(self.viewport().rect().center())
        delta = center_point - new_center
        self.translate(delta.x(), delta.y())
        
        self.zoomChanged.emit(self._zoom_factor)
    
    def set_pan(self, scene_pos: QPointF):
        """
        Set pan position programmatically.
        
        Args:
            scene_pos: Position in scene coordinates to center on
        """
        if self._pixmap_item is None:
            return
        
        # Center on the scene position
        self.centerOn(scene_pos)
        self.panChanged.emit(scene_pos)
    
    def get_zoom(self) -> float:
        """Get current zoom factor."""
        return self._zoom_factor
    
    def get_scene_center(self) -> QPointF:
        """Get current center point in scene coordinates."""
        return self.mapToScene(self.viewport().rect().center())
    
    def clear(self):
        """Clear the displayed image."""
        if self._pixmap_item:
            self._scene.removeItem(self._pixmap_item)
            self._pixmap_item = None
        self._scene.setSceneRect(QRectF())
        self._zoom_factor = 1.0

