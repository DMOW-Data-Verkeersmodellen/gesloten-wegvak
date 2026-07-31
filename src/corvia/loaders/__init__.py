"""
loaders
=======
Format-specific readers that translate raw GIS sources into the flat
record dictionaries consumed by
:class:`~corvia.framework.builder.NetworkBuilder`.

Every loader implements the :class:`~corvia.loaders.base.NetworkDataLoader`
interface, so a new source format only needs a new subclass — the rest of
the pipeline (:class:`~corvia.framework.builder.NetworkBuilder`,
:class:`~corvia.framework.network.Network`) never has to change.
"""

from corvia.loaders.base_network import BaseNetworkLoader
from corvia.loaders.vc_network import VCNetworkLoader

__all__ = ["NetworkDataLoader", "VCGeoPackageLoader"]