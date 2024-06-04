#!/usr/bin/env python
# -*- coding=utf-8 -*-
"""
    A Simple Logger Package
"""
import os
import sys
import json
import logging

import coloredlogs

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

version_info = json.load(open(os.path.join(BASE_DIR, 'version', 'version.json')))
__version__ = version_info['version']


class SimpleLogger(logging.Logger):
    """
        >>> from simple_loggers import SimpleLogger
        >>> logger = SimpleLogger(colored=True)
        >>> logger.info('info message ...')
        >>> logger = SimpleLogger(logfile='run.log', level=10)
        >>> logger.info('info message ...')
    """
    def __init__(self,
                 name=None,
                 level=logging.DEBUG,
                 fmt='[%(asctime)s %(name)s %(funcName)s %(levelname)s %(threadName)s:%(lineno)s] %(message)s',
                 datefmt='%Y-%m-%d %H:%M:%S',
                 logfile=None,
                 filemode='w',
                 stream=sys.stderr,
                 colored=True,
                 **kwargs):

        if type(level) == str:
            level = self.level_maps.get(level.lower(), 10)

        super(SimpleLogger, self).__init__(name, level)

        self.formatter = logging.Formatter(fmt, datefmt)

        if logfile:
            self._addFilehandler(logfile, filemode)
        else:
            self._addStreamHandler(stream)
            if colored:
                coloredlogs.install(fmt=fmt, level=level, logger=self)

    def _addFilehandler(self, filename, filemode):

        file_hdlr = logging.FileHandler(filename, filemode)
        file_hdlr.setFormatter(self.formatter)
        self.addHandler(file_hdlr)

    def _addStreamHandler(self, stream):

        stream_hdlr = logging.StreamHandler(stream)
        stream_hdlr.setFormatter(self.formatter)
        self.addHandler(stream_hdlr)

    @property
    def level_maps(self):
        return {
            'debug': 10,
            'info': 20,
            'warn': 30,
            'warning': 30,
            'error': 40,
            'fatal': 50,
            'critical': 50,
        }


if __name__ == '__main__':

    logger = SimpleLogger()
    logger.debug('debug message')
    logger.info('info message')
    logger.warning('warn message')
    logger.error('error message')

    logger2 = SimpleLogger(name='TEST2', logfile='out.log', filemode='w')
    logger2.debug('debug message')
    logger2.info('info message')
    logger2.warning('warn message')
    logger2.error('error message')

    logger3 = SimpleLogger(name='TEST3', level='info', fmt='[%(asctime)s %(levelname)s]: %(message)s')
    logger3.debug('debug message')
    logger3.info('info message')
    logger3.warning('warn message')
    logger3.error('error message')
