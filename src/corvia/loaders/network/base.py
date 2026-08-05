"""
base.py
=======
Common contract for source-format loaders.

Dependency chain::

    base.py  →  (no framework dependency)

A loader's only job is to translate one raw data source into the flat
record dictionaries expected by
:meth:`~corvia.framework.builder.NetworkBuilder.build_raw_network`::

    links_data, nodes_data, counts_data, crs = loader.load()
    network = NetworkBuilder.from_data(
        "MyNetwork", links_data, nodes_data, counts_data, crs=crs
    )

Add a new subclass of :class:`NetworkDataLoader` for every new source
format; :class:`~corvia.framework.builder.NetworkBuilder` and
:class:`~corvia.framework.network.Network` never need to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseNetworkLoader(ABC):
    """
    Abstract base class for format-specific network data loaders.

    Concrete subclasses read one specific raw source (a GeoPackage, a
    shapefile bundle, a CSV export, …) and translate it into the record
    shapes documented in the :mod:`~corvia.framework.builder` module
    docstring.

    Attributes
    ----------
    _source : str
        Stored source path.
    _verbose : bool
        Stored verbosity flag.
    _warnings : list of str
        Diagnostic messages collected during :meth:`load`.

    Examples
    --------
    >>> loader = VCGeoPackageLoader("R2_netwerk-VC.gpkg")
    >>> links_data, nodes_data, counts_data, crs = loader.load()
    >>> network = NetworkBuilder.from_data(
    ...     "Antwerp-R2", links_data, nodes_data, counts_data, crs=crs
    ... )
    """

    def __init__(self, source: str, verbose: bool = True) -> None:
        self._source: str = source
        self._verbose: bool = verbose
        self._warnings: List[str] = []

    # ------------------------------------------------------------------
    # Property getters
    # ------------------------------------------------------------------

    @property
    def source(self) -> str:
        """str : Path to the raw data source *(read-only)*."""
        return self._source

    @property
    def warnings(self) -> List[str]:
        """list of str : Diagnostic messages collected during :meth:`load` *(read-only snapshot)*."""
        return list(self._warnings)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _warn(self, message: str) -> None:
        """
        Record a diagnostic message produced while parsing.

        Parameters
        ----------
        message : str

        Notes
        -----
        Always appended to :attr:`warnings` for programmatic inspection;
        also printed immediately when :attr:`_verbose` is ``True``.
        """
        self._warnings.append(message)
        if self._verbose:
            print(message)

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    @abstractmethod
    def load(self) -> Tuple[List[dict], List[dict], List[dict], str]:
        """
        Parse the raw source and return builder-ready records.

        Returns
        -------
        links_data : list of dict
            See :mod:`~corvia.framework.builder` for the expected shape.
        nodes_data : list of dict
        counts_data : list of dict
        crs : str
            Coordinate reference system of the source, ready to pass to
            :meth:`~corvia.framework.builder.NetworkBuilder.from_data`.

        Notes
        -----
        Must be implemented by every concrete loader.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(source={self._source!r}, warnings={len(self._warnings)})"