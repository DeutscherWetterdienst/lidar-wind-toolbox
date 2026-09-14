#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path


### used to generate configuration information ###
class config(object):
    content = {}

    def __init__(self, content):
        self.content = content

    @staticmethod
    def gen_confDict(arg=True, url=None):
        """reading the config into a dictionary for easy processing at later stages."""
        if url is None:
            raise ValueError("configuration path is required")

        confFile = [Path(url)]
        for file_name in confFile:
            with open(str(file_name)) as reader:
                # one step approach, fast but hard to understand
                return {
                    ((line.strip("\n")).split("="))[0].strip(): ((line.strip("\n")).split("="))[
                        1
                    ].strip()
                    for line in reader
                    if not line.startswith("#") and line != "\n"
                }
