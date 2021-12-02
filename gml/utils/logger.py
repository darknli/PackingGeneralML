import logging
import os

_LEVEL_DICT = {
    "error": logging.ERROR,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "debug": logging.DEBUG,
}

FORMATTER = logging.Formatter(
    '%(asctime)s - %(levelname)s: - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')


def gat_loggar(level="info", **kwargs):
    name = kwargs.get("name", None)
    logger = logging.getLogger(name)  # 不加各算设置root logger

    ch = logging.StreamHandler()
    level = level.lower()
    assert level in _LEVEL_DICT
    ch.setLevel(level)
    formatter = kwargs.get("formatter", FORMATTER)
    ch.setFormatter(formatter)

    if "save_path" in kwargs:
        fh = logging.FileHandler(os.path.join(kwargs["kwargs"], 'log.txt'))
        save_level = kwargs.get("save_level", "info").lower()
        assert save_level in _LEVEL_DICT
        fh.setLevel(save_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    # 添加两个Handler
    logger.addHandler(ch)
    return logger


class LOG:
    def __init__(self, level="info", **kwargs):
        name = kwargs.get("name", None)
        self.logger = logging.getLogger(name)  # 不加名你设置root logger

        self.ch = logging.StreamHandler()
        self.set_print_level(level)
        self.formatter = kwargs.get("formatter", FORMATTER)
        self.ch.setFormatter(self.formatter)

        self.fh = None
        # 添加两个Handler
        self.logger.addHandler(self.ch)

    def set_filehandler(self, root, level=None):
        self.fh = logging.FileHandler(os.path.join(root))
        if level is None:
            level = "info"
        self.set_print_level(level)
        self.fh.setFormatter(self.formatter)
        self.logger.addHandler(self.fh)

    def set_print_level(self, level):
        level = level.lower()
        assert level in _LEVEL_DICT
        self.logger.setLevel(_LEVEL_DICT[level])
        self.ch.setLevel(_LEVEL_DICT[level])

    def set_save_level(self, level):
        if self.fh is None:
            self.logger.warning("没有设置FileHandler")
            return
        level = level.lower()
        assert level in _LEVEL_DICT
        self.fh.setLevel(_LEVEL_DICT[level])

    def error(self, *args, **kwargs):
        self.logger.error(*args, **kwargs)

    def info(self, *args, **kwargs):
        self.logger.info(*args, **kwargs)

    def warning(self, *args, **kwargs):
        self.logger.warning(*args, **kwargs)

    def debug(self, *args, **kwargs):
        self.logger.debug(*args, **kwargs)


logger = LOG()
