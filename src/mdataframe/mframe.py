#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""mframe.py: Contains the main DataFrame wrapper."""

import pandas as pd
import pypipegraph as ppg
from pathlib import Path
from typing import Optional, Callable, List, Dict, Tuple, Any, Union
from pandas import DataFrame


__author__ = "Marco Mernberger"
__copyright__ = "Copyright (c) 2020 Marco Mernberger"
__license__ = "mit"


class MDF2(DataFrame):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    # def __getattr__(self, name):
    #     def method(*args):
    #         return self.transform(name, *args)

    #     return method
