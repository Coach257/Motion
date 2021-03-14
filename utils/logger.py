import logging


class Logger(object):
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(fmt="%(asctime)s: %(message)s")
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)
        self.log_func = self.logger.info
        self.level2log = {logging.INFO: self.logger.info, logging.DEBUG: self.logger.debug}

    def set_level(self, level):
        self.log_func = self.level2log[level]

    def logging(self, msg):
        self.log_func(msg)