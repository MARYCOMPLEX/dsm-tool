"""
Threading Utilities Module

Provides QThread-based worker for running processors in background.
Also exposes a simple hook system so other components can listen
to processor lifecycle events.
"""

from dataclasses import dataclass
from typing import Callable, Optional

from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot


@dataclass
class ProcessorHooks:
    """
    Optional hooks that can be attached to a ProcessorThread.

    All callbacks are optional and are invoked from the worker thread,
    so they must NOT touch Qt widgets directly.
    """

    on_start: Optional[Callable[[], None]] = None
    on_finish: Optional[Callable[[str], None]] = None
    on_error: Optional[Callable[[str], None]] = None
    on_progress: Optional[Callable[[int], None]] = None


class ProcessorWorker(QObject):
    """
    Worker object that runs a processor in a background thread.
    """

    # Signals
    finished = pyqtSignal(str)  # Emits output file path on success
    error = pyqtSignal(str)  # Emits error message on failure
    progress = pyqtSignal(int)  # Emits progress percentage (0-100)

    def __init__(
        self,
        processor,
        image1_path: str,
        image2_path: str,
        rpc1: dict,
        rpc2: dict,
        output_dir: str,
        hooks: Optional[ProcessorHooks] = None,
    ):
        """
        Initialize the worker.

        Args:
            processor: Processor instance to run
            image1_path: Path to first image
            image2_path: Path to second image
            rpc1: RPC metadata for first image
            rpc2: RPC metadata for second image
            output_dir: Output directory
            hooks: Optional hooks for lifecycle events
        """
        super().__init__()
        self._processor = processor
        self._image1_path = image1_path
        self._image2_path = image2_path
        self._rpc1 = rpc1
        self._rpc2 = rpc2
        self._output_dir = output_dir
        self._cancelled = False
        self._hooks = hooks

    @pyqtSlot()
    def run(self):
        """Execute the processor."""
        try:
            if self._hooks and self._hooks.on_start:
                self._hooks.on_start()

            def progress_cb(pct: int) -> None:
                value = max(0, min(100, int(pct)))
                if self._cancelled:
                    return
                self.progress.emit(value)
                if self._hooks and self._hooks.on_progress:
                    self._hooks.on_progress(value)

            # Initial progress
            progress_cb(1)

            if self._cancelled:
                self.error.emit("Processing cancelled")
                return

            # Run the processor
            output_path = self._processor.process(
                self._image1_path,
                self._image2_path,
                self._rpc1,
                self._rpc2,
                self._output_dir,
                progress_callback=progress_cb,
            )

            if self._cancelled:
                self.error.emit("Processing cancelled")
                return

            progress_cb(100)
            self.finished.emit(output_path)

        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if self._hooks and self._hooks.on_error:
                self._hooks.on_error(msg)
            self.error.emit(msg)

    def cancel(self):
        """Cancel the processing."""
        self._cancelled = True


class ProcessorThread(QThread):
    """
    Thread wrapper for running processors.

    Usage:
        thread = ProcessorThread(processor, ...)
        thread.finished.connect(on_finished)
        thread.error.connect(on_error)
        thread.start()

    To attach hooks for non-UI listeners:
        hooks = ProcessorHooks(on_progress=..., on_finish=...)
        thread = ProcessorThread(processor, ..., hooks=hooks)
    """

    finished = pyqtSignal(str)  # Emits output file path
    error = pyqtSignal(str)  # Emits error message
    progress = pyqtSignal(int)  # Emits progress percentage

    def __init__(
        self,
        processor,
        image1_path: str,
        image2_path: str,
        rpc1: dict,
        rpc2: dict,
        output_dir: str,
        hooks: Optional[ProcessorHooks] = None,
    ):
        """
        Initialize the thread.

        Args:
            processor: Processor instance to run
            image1_path: Path to first image
            image2_path: Path to second image
            rpc1: RPC metadata for first image
            rpc2: RPC metadata for second image
            output_dir: Output directory
            hooks: Optional hooks for lifecycle events
        """
        super().__init__()
        self._worker: Optional[ProcessorWorker] = None
        self._processor = processor
        self._image1_path = image1_path
        self._image2_path = image2_path
        self._rpc1 = rpc1
        self._rpc2 = rpc2
        self._output_dir = output_dir
        self._hooks = hooks

    def run(self):
        """Run the worker in this thread."""
        self._worker = ProcessorWorker(
            self._processor,
            self._image1_path,
            self._image2_path,
            self._rpc1,
            self._rpc2,
            self._output_dir,
            hooks=self._hooks,
        )

        # Move worker to this thread BEFORE connecting signals
        self._worker.moveToThread(self)

        # Connect worker signals to thread signals
        self._worker.finished.connect(self.finished.emit)
        self._worker.error.connect(self.error.emit)
        self._worker.progress.connect(self.progress.emit)

        # Execute directly in this thread
        self._worker.run()

    def cancel(self):
        """Cancel the processing."""
        if self._worker:
            self._worker.cancel()

