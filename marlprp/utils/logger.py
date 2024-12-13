import logging
from lightning.pytorch.utilities.rank_zero import rank_zero_only

def get_lightning_logger(name=__name__, rzo: bool = True) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    )
    if rzo:
        for level in logging_levels:
            setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger
