"""Project-specific exceptions so callers (CLI, UI) can show clear messages."""


class ShelfSightError(Exception):
    """Base class for expected, user-actionable errors."""


class InvalidImageError(ShelfSightError):
    """Image path missing, file undecodable, or array has the wrong shape/dtype."""


class ModelNotFoundError(ShelfSightError):
    """Model file does not exist (the project never downloads weights implicitly)."""


class DeviceUnavailableError(ShelfSightError):
    """A requested compute device (e.g. CUDA) is not available."""


class VideoError(ShelfSightError):
    """Video cannot be opened, decoded, or written."""
