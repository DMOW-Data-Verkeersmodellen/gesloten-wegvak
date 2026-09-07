"""
loaders
=======
Format-specific readers that translate raw data sources
into the requested record shapes documented in the 
different modules.
"""

from corvia.loaders.network import BaseNetworkLoader, VCNetworkLoader
from corvia.loaders.flow_data_loader import VCFlowDataLoader

__all__ = [
    "VCFlowDataLoader", 
    "BaseNetworkLoader",
    "VCNetworkLoader", 
]