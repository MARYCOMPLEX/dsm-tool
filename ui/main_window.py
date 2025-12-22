"""
Main Window Module

Main application window with image viewers and processing controls.
"""

import os
from collections import deque
from pathlib import Path
from typing import Optional, Dict, List

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QPushButton,
    QFileDialog,
    QLineEdit,
    QComboBox,
    QLabel,
    QProgressBar,
    QMessageBox,
    QGroupBox,
    QCheckBox,
    QTabWidget,
    QListWidget,
    QListWidgetItem,
)
from PyQt6.QtCore import Qt, QSize, QPointF, QSignalBlocker, pyqtSlot

from views.image_viewer import ImageViewer
from loaders.image_loader import (
    find_image_files, load_rpc_metadata, load_image_as_qimage
)
from processors import AVAILABLE_PROCESSORS
from utils.threading import ProcessorThread
from tabs import AVAILABLE_TABS


class MainWindow(QMainWindow):
    """
    Main application window.
    
    Features:
    - Two synchronized image viewers
    - Image loading from directory
    - Processor selection and execution
    - Progress tracking
    """
    
    def __init__(self):
        """Initialize the main window."""
        super().__init__()

        # Language: 'en' or 'zh'
        self._language: str = "en"

        self._image1_path: Optional[str] = None
        self._image2_path: Optional[str] = None
        self._rpc1: Optional[Dict] = None
        self._rpc2: Optional[Dict] = None
        self._processor_thread: Optional[ProcessorThread] = None

        # For output preview: keep at most two checked items (FIFO queue of paths)
        self._output_selected = deque(maxlen=2)
        self._output_list_updating: bool = False
        
        self._init_ui()
        self._retranslate_ui()
        self._check_processors()
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Window title will be set in _retranslate_ui
        self.setMinimumSize(1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create two image viewers (inputs)
        self._viewer1 = ImageViewer()
        self._viewer2 = ImageViewer()
        self._viewer1_label = QLabel()
        self._viewer2_label = QLabel()

        # Container widgets to place label under each viewer
        self._viewer1_container = QWidget()
        v1_layout = QVBoxLayout(self._viewer1_container)
        v1_layout.setContentsMargins(0, 0, 0, 0)
        v1_layout.addWidget(self._viewer1)
        v1_layout.addWidget(self._viewer1_label)

        self._viewer2_container = QWidget()
        v2_layout = QVBoxLayout(self._viewer2_container)
        v2_layout.setContentsMargins(0, 0, 0, 0)
        v2_layout.addWidget(self._viewer2)
        v2_layout.addWidget(self._viewer2_label)
        
        # Connect viewers for synchronization
        self._viewer1.zoomChanged.connect(self._on_viewer1_zoom_changed)
        self._viewer1.panChanged.connect(self._on_viewer1_pan_changed)
        self._viewer2.zoomChanged.connect(self._on_viewer2_zoom_changed)
        self._viewer2.panChanged.connect(self._on_viewer2_pan_changed)
        
        # Create tab widget (tabs themselves are provided by plugins)
        self._tab_widget = QTabWidget()
        self._tabs = []

        # Output viewers (up to two images side-by-side), used by output tab plugin
        self._output_viewer1 = ImageViewer()
        self._output_viewer2 = ImageViewer()
        self._output_viewer1_label = QLabel()
        self._output_viewer2_label = QLabel()

        self._output_viewer1_container = QWidget()
        o1_layout = QVBoxLayout(self._output_viewer1_container)
        o1_layout.setContentsMargins(0, 0, 0, 0)
        o1_layout.addWidget(self._output_viewer1)
        o1_layout.addWidget(self._output_viewer1_label)

        self._output_viewer2_container = QWidget()
        o2_layout = QVBoxLayout(self._output_viewer2_container)
        o2_layout.setContentsMargins(0, 0, 0, 0)
        o2_layout.addWidget(self._output_viewer2)
        o2_layout.addWidget(self._output_viewer2_label)

        # Instantiate tab plugins and add their widgets
        for TabClass in AVAILABLE_TABS:
            try:
                tab_plugin = TabClass()
                widget = tab_plugin.create_widget(self)
                self._tabs.append(tab_plugin)
                self._tab_widget.addTab(widget, "")
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: failed to create tab '{TabClass}': {exc}")

        # Create control panel
        control_panel = self._create_control_panel()
        
        # Add to main layout
        main_layout.addWidget(self._tab_widget, stretch=3)
        main_layout.addWidget(control_panel, stretch=1)
    
    def _create_control_panel(self) -> QWidget:
        """Create the control panel widget."""
        panel = QWidget()
        panel.setMaximumWidth(350)
        panel.setMinimumWidth(300)
        layout = QVBoxLayout(panel)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # Language / Load section
        self._lang_group = QGroupBox()
        lang_layout = QVBoxLayout()

        lang_row = QHBoxLayout()
        self._lang_label = QLabel()
        self._lang_combo = QComboBox()
        self._lang_combo.addItem("English", "en")
        self._lang_combo.addItem("中文", "zh")
        self._lang_combo.currentIndexChanged.connect(self._on_language_changed)
        lang_row.addWidget(self._lang_label)
        lang_row.addWidget(self._lang_combo)
        lang_row.addStretch()
        lang_layout.addLayout(lang_row)

        # Load section
        self._load_group = QGroupBox()
        load_layout = QVBoxLayout()
        
        self._load_button = QPushButton()
        self._load_button.clicked.connect(self._on_load_folder)
        load_layout.addWidget(self._load_button)
        
        self._image_info_label = QLabel()
        self._image_info_label.setWordWrap(True)
        load_layout.addWidget(self._image_info_label)
        
        self._load_group.setLayout(load_layout)
        lang_layout.addWidget(self._load_group)

        self._lang_group.setLayout(lang_layout)
        layout.addWidget(self._lang_group)
        
        # Synchronization section
        self._sync_group = QGroupBox()
        sync_layout = QVBoxLayout()
        
        self._sync_checkbox = QCheckBox()
        self._sync_checkbox.setChecked(True)
        sync_layout.addWidget(self._sync_checkbox)
        
        self._sync_group.setLayout(sync_layout)
        layout.addWidget(self._sync_group)
        
        # Processor section
        self._processor_group = QGroupBox()
        processor_layout = QVBoxLayout()
        
        self._algorithm_label = QLabel()
        processor_layout.addWidget(self._algorithm_label)
        self._processor_combo = QComboBox()
        processor_layout.addWidget(self._processor_combo)
        
        self._output_dir_label = QLabel()
        processor_layout.addWidget(self._output_dir_label)
        output_layout = QHBoxLayout()
        self._output_edit = QLineEdit()
        self._output_edit.setPlaceholderText("")
        self._output_browse_button = QPushButton()
        self._output_browse_button.clicked.connect(self._on_browse_output)
        output_layout.addWidget(self._output_edit)
        output_layout.addWidget(self._output_browse_button)
        processor_layout.addLayout(output_layout)
        
        self._run_button = QPushButton()
        self._run_button.setEnabled(False)
        self._run_button.clicked.connect(self._on_run_processor)
        processor_layout.addWidget(self._run_button)
        
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        processor_layout.addWidget(self._progress_bar)
        
        self._cancel_button = QPushButton()
        self._cancel_button.setVisible(False)
        self._cancel_button.clicked.connect(self._on_cancel_processor)
        processor_layout.addWidget(self._cancel_button)
        
        self._processor_group.setLayout(processor_layout)
        layout.addWidget(self._processor_group)

        # Output preview control (file list + buttons)
        self._output_ctrl_group = QGroupBox()
        output_ctrl_layout = QVBoxLayout()
        self._output_hint_label = QLabel()
        output_ctrl_layout.addWidget(self._output_hint_label)

        list_row = QHBoxLayout()
        self._output_list = QListWidget()
        # We use checkboxes on each item instead of selection; see _on_output_item_changed
        self._output_list.itemChanged.connect(self._on_output_item_changed)
        list_row.addWidget(self._output_list, stretch=4)

        # Buttons stacked vertically to the right of the list
        btn_col = QVBoxLayout()
        self._output_refresh_button = QPushButton()
        self._output_refresh_button.clicked.connect(self._refresh_output_list)
        btn_col.addWidget(self._output_refresh_button)

        self._output_remove_button = QPushButton()
        self._output_remove_button.clicked.connect(self._remove_selected_outputs)
        btn_col.addWidget(self._output_remove_button)

        btn_col.addStretch()
        list_row.addLayout(btn_col, stretch=1)

        output_ctrl_layout.addLayout(list_row)
        self._output_ctrl_group.setLayout(output_ctrl_layout)
        layout.addWidget(self._output_ctrl_group)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        return panel

    def _retranslate_ui(self) -> None:
        """Apply language-dependent texts to all widgets."""
        zh = self._language == "zh"

        if zh:
            self.setWindowTitle("DSM 生成工具")

            self._lang_group.setTitle("语言 / 数据")
            self._lang_label.setText("语言:")

            self._load_group.setTitle("加载影像")
            self._load_button.setText("选择影像文件夹")
            if not self._image1_path:
                self._image_info_label.setText("尚未加载影像")

            self._sync_group.setTitle("视图联动")
            self._sync_checkbox.setText("同步缩放和平移")

            self._processor_group.setTitle("处理")
            self._algorithm_label.setText("算法:")
            self._output_dir_label.setText("输出目录:")
            self._output_edit.setPlaceholderText("选择输出文件夹...")
            self._output_browse_button.setText("浏览...")
            self._run_button.setText("运行")
            self._cancel_button.setText("取消")

            self._output_ctrl_group.setTitle("输出预览文件")
            self._output_hint_label.setText("输出目录中的 TIF/TIFF（最多选择 2 个进行预览）:")
            self._output_refresh_button.setText("刷新")
            self._output_remove_button.setText("移除")
        else:
            self.setWindowTitle("DSM Generation Tool")

            self._lang_group.setTitle("Language / Data")
            self._lang_label.setText("Language:")

            self._load_group.setTitle("Load Images")
            self._load_button.setText("Select Image Folder")
            if not self._image1_path:
                self._image_info_label.setText("No images loaded")

            self._sync_group.setTitle("View Synchronization")
            self._sync_checkbox.setText("Synchronize zoom and pan")

            self._processor_group.setTitle("Processing")
            self._algorithm_label.setText("Algorithm:")
            self._output_dir_label.setText("Output Directory:")
            self._output_edit.setPlaceholderText("Select output folder...")
            self._output_browse_button.setText("Browse...")
            self._run_button.setText("Run")
            self._cancel_button.setText("Cancel")

            self._output_ctrl_group.setTitle("Output Preview Files")
            self._output_hint_label.setText("TIF/TIFF in output folder (select up to 2):")
            self._output_refresh_button.setText("Refresh")
            self._output_remove_button.setText("Remove")

        # Update tab titles from plugins
        for index, tab_plugin in enumerate(getattr(self, "_tabs", [])):
            title = tab_plugin.title_zh if zh else tab_plugin.title_en
            self._tab_widget.setTabText(index, title)
            tab_plugin.on_language_changed(self._language)
    
    def _check_processors(self):
        """Check available processors and populate combo box."""
        if not AVAILABLE_PROCESSORS:
            title = "未找到处理算法" if self._language == "zh" else "No Processors Found"
            text = (
                "未找到任何处理算法，请在 processors 目录中添加算法实现。"
                if self._language == "zh"
                else "No processor implementations found. Please add processors to the processors directory."
            )
            QMessageBox.warning(self, title, text)
            return
        
        self._processor_combo.clear()
        for processor_class in AVAILABLE_PROCESSORS:
            # Create a temporary instance to get name
            try:
                temp_instance = processor_class()
                self._processor_combo.addItem(
                    temp_instance.name,
                    processor_class
                )
            except Exception as e:
                print(f"Warning: Failed to instantiate processor {processor_class}: {e}")
    
    @pyqtSlot()
    def _on_load_folder(self):
        """Handle folder selection for loading images."""
        caption = "选择影像文件夹" if self._language == "zh" else "Select Image Folder"
        folder = QFileDialog.getExistingDirectory(
            self,
            caption,
            "",
            QFileDialog.Option.ShowDirsOnly,
        )
        
        if not folder:
            return
        
        try:
            # Find image files
            image_paths, rpc_mapping = find_image_files(folder)
            self._image1_path, self._image2_path = image_paths
            
            # Load RPC metadata
            self._rpc1 = load_rpc_metadata(self._image1_path, rpc_mapping[self._image1_path])
            self._rpc2 = load_rpc_metadata(self._image2_path, rpc_mapping[self._image2_path])
            
            # Load and display images
            max_display_size = QSize(2000, 2000)  # Limit display size for performance
            
            image1_qimage = load_image_as_qimage(self._image1_path, max_display_size)
            image2_qimage = load_image_as_qimage(self._image2_path, max_display_size)
            
            self._viewer1.load_image(image1_qimage)
            self._viewer2.load_image(image2_qimage)
            
            # Update info label
            img1_name = Path(self._image1_path).name
            img2_name = Path(self._image2_path).name
            self._image_info_label.setText(
                f"Loaded:\n• {img1_name}\n• {img2_name}"
            )
            # Update input viewer filename labels
            self._viewer1_label.setText(img1_name)
            self._viewer2_label.setText(img2_name)
            
            # Enable run button
            self._run_button.setEnabled(True)

            if self._language == "zh":
                title = "影像加载成功"
                text = f"已成功加载 2 幅影像：\n{img1_name}\n{img2_name}"
            else:
                title = "Images Loaded"
                text = f"Successfully loaded 2 images:\n{img1_name}\n{img2_name}"

            QMessageBox.information(self, title, text)
        
        except Exception as e:
            title = "加载错误" if self._language == "zh" else "Load Error"
            text = f"加载影像失败：\n{str(e)}" if self._language == "zh" else f"Failed to load images:\n{str(e)}"
            QMessageBox.critical(self, title, text)
            self._image_info_label.setText("加载影像出错" if self._language == "zh" else "Error loading images")
            self._run_button.setEnabled(False)
    
    @pyqtSlot()
    def _on_browse_output(self):
        """Handle output directory selection."""
        caption = "选择输出文件夹" if self._language == "zh" else "Select Output Folder"
        folder = QFileDialog.getExistingDirectory(
            self,
            caption,
            self._output_edit.text() or "",
            QFileDialog.Option.ShowDirsOnly,
        )
        
        if folder:
            self._output_edit.setText(folder)
    
    @pyqtSlot()
    def _on_run_processor(self):
        """Handle processor execution."""
        # Validate inputs
        if not self._image1_path or not self._image2_path:
            title = "错误" if self._language == "zh" else "Error"
            text = "请先加载影像。" if self._language == "zh" else "Please load images first."
            QMessageBox.warning(self, title, text)
            return
        
        if not self._rpc1 or not self._rpc2:
            title = "错误" if self._language == "zh" else "Error"
            text = "RPC 元数据未加载。" if self._language == "zh" else "RPC metadata not loaded."
            QMessageBox.warning(self, title, text)
            return
        
        output_dir = self._output_edit.text().strip()
        if not output_dir:
            title = "错误" if self._language == "zh" else "Error"
            text = "请选择输出目录。" if self._language == "zh" else "Please select an output directory."
            QMessageBox.warning(self, title, text)
            return
        
        if not os.path.isdir(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except Exception as e:
                title = "错误" if self._language == "zh" else "Error"
                text = (
                    f"创建输出目录失败：\n{str(e)}"
                    if self._language == "zh"
                    else f"Failed to create output directory:\n{str(e)}"
                )
                QMessageBox.critical(self, title, text)
                return
        
        # Get selected processor
        processor_class = self._processor_combo.currentData()
        if not processor_class:
            title = "错误" if self._language == "zh" else "Error"
            text = "未选择处理算法。" if self._language == "zh" else "No processor selected."
            QMessageBox.warning(self, title, text)
            return
        
        try:
            processor = processor_class()
        except Exception as e:
            title = "错误" if self._language == "zh" else "Error"
            text = (
                f"创建处理器实例失败：\n{str(e)}"
                if self._language == "zh"
                else f"Failed to create processor instance:\n{str(e)}"
            )
            QMessageBox.critical(self, title, text)
            return
        
        # Disable controls
        self._run_button.setEnabled(False)
        self._load_button.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._cancel_button.setVisible(True)
        
        # Create and start thread
        self._processor_thread = ProcessorThread(
            processor,
            self._image1_path,
            self._image2_path,
            self._rpc1,
            self._rpc2,
            output_dir
        )
        
        self._processor_thread.finished.connect(self._on_processor_finished)
        self._processor_thread.error.connect(self._on_processor_error)
        self._processor_thread.progress.connect(self._on_processor_progress)
        
        self._processor_thread.start()
    
    @pyqtSlot()
    def _on_cancel_processor(self):
        """Handle processor cancellation."""
        if self._processor_thread and self._processor_thread.isRunning():
            self._processor_thread.cancel()
            self._processor_thread.terminate()
            self._processor_thread.wait()
            self._on_processor_error("Processing cancelled by user")
    
    @pyqtSlot(str)
    def _on_processor_finished(self, output_path: str):
        """Handle processor completion."""
        self._progress_bar.setValue(100)
        self._progress_bar.setVisible(False)
        self._cancel_button.setVisible(False)
        self._run_button.setEnabled(True)
        self._load_button.setEnabled(True)
        
        # Refresh output list and select the newly created file if possible
        self._refresh_output_list(select_path=output_path)

        if self._language == "zh":
            title = "处理完成"
            text = f"处理已成功完成！\n\n输出文件保存于：\n{output_path}"
        else:
            title = "Processing Complete"
            text = f"Processing completed successfully!\n\nOutput saved to:\n{output_path}"

        QMessageBox.information(self, title, text)
        
        self._processor_thread = None
    
    @pyqtSlot(str)
    def _on_processor_error(self, error_msg: str):
        """Handle processor error."""
        self._progress_bar.setVisible(False)
        self._cancel_button.setVisible(False)
        self._run_button.setEnabled(True)
        self._load_button.setEnabled(True)

        if self._language == "zh":
            title = "算法错误"
            text = f"算法执行失败：\n{error_msg}"
        else:
            title = "Algorithm Error"
            text = f"Algorithm failed:\n{error_msg}"

        QMessageBox.critical(self, title, text)
        
        self._processor_thread = None
    
    @pyqtSlot(int)
    def _on_processor_progress(self, value: int):
        """Update progress bar."""
        self._progress_bar.setValue(value)
    
    def _on_viewer1_zoom_changed(self, zoom: float):
        """Handle viewer 1 zoom change for synchronization."""
        if self._sync_checkbox.isChecked():
            center = self._viewer1.get_scene_center()
            # Block signals from viewer2 to avoid recursive updates
            blocker = QSignalBlocker(self._viewer2)
            try:
                self._viewer2.set_zoom(zoom, center)
            finally:
                del blocker
    
    def _on_viewer1_pan_changed(self, pos: QPointF):
        """Handle viewer 1 pan change for synchronization."""
        if self._sync_checkbox.isChecked():
            blocker = QSignalBlocker(self._viewer2)
            try:
                self._viewer2.set_pan(pos)
            finally:
                del blocker
    
    def _on_viewer2_zoom_changed(self, zoom: float):
        """Handle viewer 2 zoom change for synchronization."""
        if self._sync_checkbox.isChecked():
            center = self._viewer2.get_scene_center()
            blocker = QSignalBlocker(self._viewer1)
            try:
                self._viewer1.set_zoom(zoom, center)
            finally:
                del blocker
    
    def _on_viewer2_pan_changed(self, pos: QPointF):
        """Handle viewer 2 pan change for synchronization."""
        if self._sync_checkbox.isChecked():
            blocker = QSignalBlocker(self._viewer1)
            try:
                self._viewer1.set_pan(pos)
            finally:
                del blocker

    @pyqtSlot()
    def _refresh_output_list(self, select_path: Optional[str] = None):
        """Scan output directory and list available TIF/TIFF files."""
        output_dir = self._output_edit.text().strip()

        # Block itemChanged while we rebuild the list
        self._output_list_updating = True
        self._output_list.clear()
        self._output_selected.clear()
        # Clear preview viewers as well
        if hasattr(self, "_output_viewer1"):
            self._output_viewer1.clear()
        if hasattr(self, "_output_viewer2"):
            self._output_viewer2.clear()

        if not output_dir or not os.path.isdir(output_dir):
            return

        dir_path = Path(output_dir)
        exts = {".tif", ".tiff", ".TIF", ".TIFF"}
        files = sorted([p for p in dir_path.iterdir() if p.suffix in exts])

        for f in files:
            item = QListWidgetItem(f.name)
            # Enable user checkboxes
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            # Default unchecked
            item.setCheckState(Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, str(f))
            self._output_list.addItem(item)

            # Optionally pre-check the item that matches select_path
            if select_path and Path(select_path) == f:
                item.setCheckState(Qt.CheckState.Checked)
                self._output_selected.append(str(f))

        self._output_list_updating = False
        # After rebuilding, update viewers from current queue (at most 2)
        self._update_output_viewers_from_queue()

    def _update_output_viewers_from_queue(self) -> None:
        """Update the two output viewers based on the current queue."""
        if not hasattr(self, "_output_viewer1") or not hasattr(self, "_output_viewer2"):
            return

        # No selection: clear both viewers
        if not self._output_selected:
            self._output_viewer1.clear()
            self._output_viewer2.clear()
            if hasattr(self, "_output_viewer1_label"):
                self._output_viewer1_label.setText("")
            if hasattr(self, "_output_viewer2_label"):
                self._output_viewer2_label.setText("")
            return

        # Work on a snapshot of up to two paths
        paths = list(self._output_selected)[:2]
        max_display_size = QSize(2000, 2000)

        # Helper to load a single path into a viewer
        def load_into_viewer(path: str, viewer: ImageViewer, label: QLabel) -> None:
            if not path or not os.path.isfile(path):
                viewer.clear()
                label.setText("")
                return
            try:
                qimage = load_image_as_qimage(path, max_display_size)
                viewer.load_image(qimage)
                label.setText(Path(path).name)
            except Exception as exc:
                viewer.clear()
                label.setText("")
                if self._language == "zh":
                    title = "预览错误"
                    text = f"输出影像加载失败：\n{str(exc)}"
                else:
                    title = "Preview Error"
                    text = f"Failed to load output image:\n{str(exc)}"
                QMessageBox.critical(self, title, text)

        # Load first selection into viewer1
        load_into_viewer(paths[0], self._output_viewer1, self._output_viewer1_label)

        # Load second selection into viewer2 (if exists), otherwise clear viewer2
        if len(paths) > 1:
            load_into_viewer(paths[1], self._output_viewer2, self._output_viewer2_label)
        else:
            self._output_viewer2.clear()
            self._output_viewer2_label.setText("")

    @pyqtSlot(QListWidgetItem)
    def _on_output_item_changed(self, item: QListWidgetItem) -> None:
        """Handle checkbox state changes for output list items."""
        if self._output_list_updating:
            return

        path = item.data(Qt.ItemDataRole.UserRole)
        if not path:
            return

        checked = item.checkState() == Qt.CheckState.Checked

        if checked:
            # If already in queue, nothing to do
            if path in self._output_selected:
                self._update_output_viewers_from_queue()
                return

            # If queue is full, uncheck and remove the oldest item
            if len(self._output_selected) == self._output_selected.maxlen:
                oldest = self._output_selected.popleft()
                # Temporarily block updates while we uncheck the oldest
                self._output_list_updating = True
                for i in range(self._output_list.count()):
                    it = self._output_list.item(i)
                    if it.data(Qt.ItemDataRole.UserRole) == oldest:
                        it.setCheckState(Qt.CheckState.Unchecked)
                        break
                self._output_list_updating = False

            # Add new path to queue
            self._output_selected.append(path)
        else:
            # Unchecked: remove from queue if present
            try:
                self._output_selected.remove(path)
            except ValueError:
                pass

        # Finally update the viewers according to the queue
        self._update_output_viewers_from_queue()

    @pyqtSlot()
    def _remove_selected_outputs(self) -> None:
        """Clear all checkboxes and preview, keeping the file list unchanged."""
        self._output_list_updating = True
        self._output_selected.clear()
        for i in range(self._output_list.count()):
            item = self._output_list.item(i)
            item.setCheckState(Qt.CheckState.Unchecked)
        self._output_list_updating = False
        self._update_output_viewers_from_queue()

    @pyqtSlot()
    def _on_language_changed(self) -> None:
        """Handle language combo changes."""
        data = self._lang_combo.currentData()
        if data in ("en", "zh"):
            self._language = data
            self._retranslate_ui()

