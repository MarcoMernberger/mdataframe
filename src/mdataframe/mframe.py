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


"""
What we want:

- an easy way to encapsulate a series of commands, DF manipulations to be
converted to a job if neccessary
- we need to parse or encapsulate the code in a function
- we need the inputs
- we need the outputs
- if we have that, we can either
    - create a job
    - create a snakemake rule

- convert a juypter notebook into an workflow
"""
import ast


def main():
    with open("test.py", "r") as source:
        tree = ast.parse(source.read())
    print(dir(tree))
    print(tree._fields)
    print(tree.body)
    analyzer = Analyzer()
    analyzer.visit(tree)
    analyzer.report()


class Analyzer(ast.NodeVisitor):
    def __init__(self):
        self.stats = {"import": [], "from": []}

    def visit_Import(self, node):
        print(dir(node))
        print(node.names)
        for alias in node.names:
            print(dir(alias))
            print(alias.name, alias._fields, alias.lineno, alias.end_lineno, alias.names)
            self.stats["import"].append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.stats["from"].append(alias.name)
        self.generic_visit(node)

    def report(self):
        print(self.stats)
