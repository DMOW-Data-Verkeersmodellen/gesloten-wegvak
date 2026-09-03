"""
Auto-configuring logger for the corvia package.

This logger automatically configures itself on first use, but allows users
to customize the configuration if they want to.

Usage in package modules:
    from corvia.logger import get_logger
    logger = get_logger(__name__)
    logger.info("This just works!")

Advanced users can customize:
    from corvia.logger import configure_logging
    configure_logging(log_level=logging.DEBUG, log_to_file=True)
"""
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from typing import Optional
import atexit


# Global state to track if logging has been configured
_logging_configured = False

# Name of the package-level logger. Must match the top-level import name
# (e.g. "corvia" in "corvia.framework.network") for child loggers obtained
# via get_logger(__name__) to inherit this configuration.
_PACKAGE_LOGGER_NAME = "corvia"


def configure_logging(
    log_level: int = logging.INFO,
    console_level: Optional[int] = None,
    file_level: Optional[int] = None,
    log_to_file: bool = False,
    log_to_console: bool = True,
    log_dir: str = 'logs',
    log_format: str = 'detailed',
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    force_reconfigure: bool = False
) -> None:
    """
    Configure logging for the package.

    This function is optional - logging will auto-configure with defaults
    if you don't call it. Call this if you want to customize logging behavior.

    Parameters
    ----------
    log_level : int, default = logging.INFO
        Root logger level for the ``corvia`` logger.
    console_level : int, optional, default = None
        Console output level. Falls back to *log_level* when ``None``.
    file_level : int, optional, default = None
        File output level. Falls back to *log_level* when ``None``.
    log_to_file : bool, default = False
        Enable file logging.
    log_to_console : bool, default = True
        Enable console logging.
    log_dir : str, default = 'logs'
        Directory for log files.
    log_format : str, default = 'detailed'
        Format style - 'simple', 'detailed', or 'minimal'.
    max_bytes : int, default = 10*1024*1024
        Max log file size before rotation (default: 10MB).
    backup_count : int, default = 5
        Number of backup log files to keep.
    force_reconfigure : bool, default = False
        Force reconfiguration even if already configured.
    """
    global _logging_configured

    if _logging_configured and not force_reconfigure:
        logging.getLogger(__name__).warning(
            "Logging already configured. Use force_reconfigure=True to reconfigure."
        )
        return

    # Set defaults
    if console_level is None:
        console_level = log_level
    if file_level is None:
        file_level = log_level

    # Get root logger for the package
    # This affects all loggers in the package
    pkg_logger = logging.getLogger(_PACKAGE_LOGGER_NAME)
    pkg_logger.setLevel(log_level)

    # Clear existing handlers to avoid duplicates
    pkg_logger.handlers.clear()

    # Define format styles
    formats = {
        'minimal': '%(levelname)s - %(name)s - %(message)s',
        'simple': '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        'detailed': '%(asctime)s - %(levelname)s - %(name)s - [%(filename)s:%(lineno)d] - %(message)s',
    }

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_formatter = ShortNameFormatter(
            fmt='%(levelname)s - %(shortname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        pkg_logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)

        log_file = log_path / f"app_{datetime.now().strftime('%Y%m%d')}.log"

        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(file_level)
        log_format_str = formats.get(log_format, formats['detailed'])
        formatter = logging.Formatter(
            fmt=log_format_str,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_formatter = logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(name)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        pkg_logger.addHandler(file_handler)

    # If no handlers were added, add a NullHandler to prevent warnings
    if not pkg_logger.handlers:
        pkg_logger.addHandler(logging.NullHandler())

    # Don't propagate to the root logger - avoids duplicate lines if the
    # host application also configures the root logger.
    pkg_logger.propagate = False

    _logging_configured = True

    # Log configuration completion
    if log_to_console or log_to_file:
        logger = logging.getLogger(__name__)
        logger.debug(f"Logging configured: console={log_to_console}, file={log_to_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with automatic configuration.

    On first call, this automatically configures logging with sensible
    defaults. Users can customize by calling configure_logging() before
    importing modules that call get_logger().

    Parameters
    ----------
    name : str
        Logger name. Always pass ``__name__`` in package modules, so the
        logger name mirrors the module path (e.g. ``corvia.framework.network``).

    Returns
    -------
    logging.Logger
        A standard library logger, configured and ready to use.

    Examples
    --------
    In a module::

        from corvia.logger import get_logger
        logger = get_logger(__name__)

    In a class, to distinguish instances/classes in log lines::

        class NetworkBuilder:
            def __init__(self):
                self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
    """
    # Auto-configure on first use
    if not _logging_configured:
        configure_logging()  # default configuration

    return logging.getLogger(name)


def disable_logging() -> None:
    """
    Disable all logging output for the package.

    Useful for tests or when using the package as a library and you don't
    want any log output.
    """
    global _logging_configured
    pkg_logger = logging.getLogger(_PACKAGE_LOGGER_NAME)
    pkg_logger.handlers.clear()
    pkg_logger.addHandler(logging.NullHandler())
    pkg_logger.setLevel(logging.CRITICAL + 1)
    _logging_configured = True


def reset_logging() -> None:
    """
    Reset logging configuration to defaults.

    Useful for testing or if you need to reconfigure.
    """
    global _logging_configured
    _logging_configured = False
    logging.getLogger(_PACKAGE_LOGGER_NAME).handlers.clear()
    configure_logging()  # default configuration


# Ensure file handlers are closed on exit
@atexit.register
def _cleanup_handlers():
    """Close all file handlers on program exit."""
    for handler in logging.getLogger(_PACKAGE_LOGGER_NAME).handlers:
        if isinstance(handler, logging.FileHandler):
            handler.close()


class ShortNameFormatter(logging.Formatter):
    """Custom formatter that adds a 'shortname' attribute to log records."""
    def format(self, record: logging.LogRecord) -> str:
        # Extracts the last component (e.g., 'FlowResolver' from 'corvia.engine.flow_resolver.FlowResolver')
        record.shortname = record.name.split('.')[-1]
        return super().format(record)

__all__ = ['get_logger', 'configure_logging', 'disable_logging', 'reset_logging']