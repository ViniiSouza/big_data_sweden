"""Pandas (non-parallel) baseline mirroring src/datasets/.

This package intentionally has no pyspark imports — it must be usable on
machines without a working Spark/JDK setup.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class DatasetInfo:
    """Metadata for each dataset used in the project.

    Mirrors src.datasets.base.DatasetInfo. Redefined here so the pandas
    package stays free of any pyspark import chain.
    """
    name: str
    path: str
    target_column: str
    size_category: str


class DatasetLoader(ABC):
    """Base class for pandas-based dataset loaders.

    Mirrors src.datasets.base.DatasetLoader but reads with pandas and
    operates on pandas.DataFrame.
    """

    @property
    @abstractmethod
    def info(self) -> DatasetInfo: ...

    def load(self) -> pd.DataFrame:
        """Raw CSV read. Default covers the common case; override if needed."""
        return pd.read_csv(self.info.path, low_memory=False)

    @abstractmethod
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame: ...

    def run(self) -> pd.DataFrame:
        """Shortcut: load and normalize."""
        return self.normalize(self.load())
