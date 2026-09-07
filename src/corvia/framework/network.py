"""
network.py
==========
Persistent, analysis-facing road network container.

:class:`Network` owns both the topology object graph (nodes, links, sections)
and the spatial GeoDataFrames.  It is the object you pass around to analysis
functions after the builder has finished.

Dependency chain::

    network.py  →  topology.py          (topology primitives)
                →  geopandas / shapely   (spatial layer)

The GeoDataFrames are indexed by the same IDs used on the Python objects,
so joining topology and geometry is a straightforward index look-up::

    geom = network.nodes_gdf.loc[node.node_id, "geometry"]
    gdf_row = network.links_gdf.loc[link.link_id]

Additional layers (zones, connectors, …) can be registered via
:meth:`Network.add_layer` and retrieved with :meth:`Network.get_layer`.
"""

from __future__ import annotations
from typing import Dict, List, Optional

import pandas as pd
import geopandas as gpd
from shapely.geometry.base import BaseGeometry

from corvia.framework.topology import SensorLocation, Link, Node, RoadSection
from corvia.logger import get_logger
logger = get_logger(__name__)


class Network:
    """
    The assembled, analysis-ready road network.

    Owns the topology object graph and the spatial GeoDataFrames.
    Normally created by :meth:`~builder.NetworkBuilder.build` rather
    than constructed directly.

    Parameters
    ----------
    name : str
        Name of the network.
    nodes : dict of {str: Node}
        Node registry keyed by ``node_id``.
    links : dict of {str: Link}
        Link registry keyed by ``link_id``.
    sections : dict of {str: RoadSection}
        Section registry keyed by ``section_id``.
    nodes_gdf : geopandas.GeoDataFrame
        Point geometries for all nodes.  Index must be ``node_id``.
    links_gdf : geopandas.GeoDataFrame
        Linestring geometries for all links.  Index must be ``link_id``.
    sensors_gdf : geopandas.GeoDataFrame
        Point geometries for all sensors.  Index must be ``location_id``.

    Attributes
    ----------
    _name : str
    _nodes : dict of {str: Node}
    _links : dict of {str: Link}
    _sections : dict of {str: RoadSection}
    _nodes_gdf : geopandas.GeoDataFrame
    _links_gdf : geopandas.GeoDataFrame
    _sensors_gdf : geopandas.GeoDataFrame
    _layers : dict of {str: geopandas.GeoDataFrame}
        Extra spatial layers (zones, connectors, …).

    Examples
    --------
    >>> network = NetworkBuilder.from_data(links_data, nodes_data, counts_data)
    >>> network.name
    'MyNetwork'
    >>> network.node_count
    4
    >>> network.nodes_gdf.crs
    <Geographic 2D CRS: EPSG:4326 ...>
    >>> network.get_node("N2").is_passthrough()
    True
    >>> network.node_geometry("N1")
    <POINT (4.35 50.85)>
    """

    def __init__(
        self,
        name: str,
        nodes: Dict[str, Node],
        links: Dict[str, Link],
        sections: Dict[str, RoadSection],
        nodes_gdf: gpd.GeoDataFrame,
        links_gdf: gpd.GeoDataFrame,
        sensors_gdf: gpd.GeoDataFrame,
    ) -> None:
        self._nodes: Dict[str, Node] = dict(nodes)
        self._links: Dict[str, Link] = dict(links)
        self._sections: Dict[str, RoadSection] = dict(sections)
        self._nodes_gdf: gpd.GeoDataFrame = nodes_gdf
        self._links_gdf: gpd.GeoDataFrame = links_gdf
        self._sensors_gdf: gpd.GeoDataFrame = sensors_gdf
        self._layers: Dict[str, gpd.GeoDataFrame] = {}
        self._name: str = name
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(
            f"Network '{self._name}' assembled: "
            f"{len(self._nodes)} node(s), "
            f"{len(self._links)} link(s), "
            f"{len(self._sections)} section(s)."
        )
    # ------------------------------------------------------------------
    # Object-graph registries (snapshots — internal dicts are protected)
    # ------------------------------------------------------------------

    @property
    def nodes(self) -> Dict[str, Node]:
        """dict of {str: Node} : Node registry *(read-only snapshot)*."""
        return dict(self._nodes)

    @property
    def links(self) -> Dict[str, Link]:
        """dict of {str: Link} : Link registry *(read-only snapshot)*."""
        return dict(self._links)

    @property
    def sections(self) -> Dict[str, RoadSection]:
        """dict of {str: RoadSection} : Section registry *(read-only snapshot)*."""
        return dict(self._sections)

    # ------------------------------------------------------------------
    # GeoDataFrames (returned directly — callers may read or extend them)
    # ------------------------------------------------------------------

    @property
    def nodes_gdf(self) -> gpd.GeoDataFrame:
        """
        geopandas.GeoDataFrame : Node geometries.

        Index is ``node_id``.  Geometry column contains Shapely ``Point``
        objects (or ``None`` where geometry was absent in source data).
        """
        return self._nodes_gdf

    @property
    def links_gdf(self) -> gpd.GeoDataFrame:
        """
        geopandas.GeoDataFrame : Link geometries.

        Index is ``link_id``.  Geometry column contains Shapely
        ``LineString`` objects (or ``None`` where absent).
        """
        return self._links_gdf

    @property
    def sensors_gdf(self) -> gpd.GeoDataFrame:
        """
        geopandas.GeoDataFrame : Sensor geometries.

        Index is ``location_id``.  Geometry column contains Shapely
        ``Point`` objects (or ``None`` where absent).
        """
        return self._sensors_gdf

    # ------------------------------------------------------------------
    # Counts
    # ------------------------------------------------------------------

    @property
    def node_count(self) -> int:
        """int : Total number of nodes."""
        return len(self._nodes)

    @property
    def link_count(self) -> int:
        """int : Total number of links."""
        return len(self._links)

    @property
    def section_count(self) -> int:
        """int : Total number of compiled road sections."""
        return len(self._sections)

    # ------------------------------------------------------------------
    # Object look-ups
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> Optional[Node]:
        """
        Look up a node by its identifier.

        Parameters
        ----------
        node_id : str

        Returns
        -------
        Node or None
        """
        return self._nodes.get(node_id)

    def get_link(self, link_id: str) -> Optional[Link]:
        """
        Look up a link by its identifier.

        Parameters
        ----------
        link_id : str

        Returns
        -------
        Link or None
        """
        return self._links.get(link_id)

    def get_section(self, section_id: str) -> Optional[RoadSection]:
        """
        Look up a road section by its identifier.

        Parameters
        ----------
        section_id : str

        Returns
        -------
        RoadSection or None
        """
        return self._sections.get(section_id)

    # ------------------------------------------------------------------
    # Geometry look-ups (bridge between object graph and GeoDataFrames)
    # ------------------------------------------------------------------

    def node_geometry(self, node_id: str) -> Optional[BaseGeometry]:
        """
        Return the Shapely geometry for a node.

        Parameters
        ----------
        node_id : str

        Returns
        -------
        shapely.geometry.base.BaseGeometry or None
            ``None`` when the node is absent from the GeoDataFrame or its
            geometry is null.
        """
        if node_id not in self._nodes_gdf.index:
            self.logger.warning(f"node_geometyr: '{node_id}' not found in nodes_gdf.")
            return None
        geom = self._nodes_gdf.loc[node_id, "geometry"]
        return None if pd.isna(geom) else geom

    def link_geometry(self, link_id: str) -> Optional[BaseGeometry]:
        """
        Return the Shapely geometry for a link.

        Parameters
        ----------
        link_id : str

        Returns
        -------
        shapely.geometry.base.BaseGeometry or None
        """
        if link_id not in self._links_gdf.index:
            self.logger.warning(f"link_geometyr: '{link_id}' not found in links_gdf.")
            return None
        geom = self._links_gdf.loc[link_id, "geometry"]
        return None if pd.isna(geom) else geom

    def sensor_geometry(self, location_id: str) -> Optional[BaseGeometry]:
            """
            Return the Shapely geometry for a link.
    
            Parameters
            ----------
            link_id : str
    
            Returns
            -------
            shapely.geometry.base.BaseGeometry or None
            """
            if location_id not in self._sensors_gdf.index:
                self.logger.warning(f"sensor_geometyr: '{location_id}' not found in sensor_gdf.")
                return None
            geom = self._sensors_gdf.loc[location_id, "geometry"]
            return None if pd.isna(geom) else geom

    # ------------------------------------------------------------------
    # Extra spatial layers (zones, connectors, …)
    # ------------------------------------------------------------------

    def add_layer(self, name: str, gdf: gpd.GeoDataFrame) -> None:
        """
        Register an additional new spatial layer on the network.
        Use :meth:`replace_layer` to overwrite.

        Parameters
        ----------
        name : str
            Layer identifier (e.g. ``"zones"``, ``"connectors"``).
        gdf : geopandas.GeoDataFrame
            The layer to register. 
        """
        if name in self._layers:
            error_msg = f"Cannot add layer '{name}' on network '{self._name}': already exists.  "
            logger.error(error_msg + "Use :meth:`replace_layer` to overwrite existing layers. ")
            raise ValueError(error_msg)
        self._layers[name] = gdf
        self.logger.debug(f"Layer '{name}' set on network '{self._name}' ({len(gdf)} rows).")

    def replace_layer(self, name: str, gdf: gpd.GeoDataFrame) -> None:
        """
        Add or overwrite a spatial layer.

        Parameters
        ----------
        name : str
        gdf : geopandas.GeoDataFrame
        """
        if name not in self._layers:
            logger.warning(f"Layer '{name}' does not exist on network '{self._name}': simply adding new layer. ")
        self._layers[name] = gdf
        self.logger.debug(f"Layer '{name}' set on network '{self._name}' ({len(gdf)} rows).")

    def get_layer(self, name: str) -> Optional[gpd.GeoDataFrame]:
        """
        Retrieve a registered spatial layer.

        Parameters
        ----------
        name : str

        Returns
        -------
        geopandas.GeoDataFrame or None
        """
        return self._layers.get(name)

    @property
    def layer_names(self) -> List[str]:
        """list of str : Names of all registered extra layers."""
        return list(self._layers.keys())

    # ------------------------------------------------------------------
    # Topology queries (convenience — delegate to object graph)
    # ------------------------------------------------------------------
    @property
    def junction_nodes(self) -> List[Node]:
        """
        Return all nodes that are junctions (merge or diverge points).

        Returns
        -------
        list of Node
        """
        return [n for n in self._nodes.values() if n.is_junction()]

    @property
    def source_nodes(self) -> List[Node]:
        """
        Return all source nodes (no incoming links).

        Returns
        -------
        list of Node
        """
        return [n for n in self._nodes.values() if n.is_source()]

    @property
    def sink_nodes(self) -> List[Node]:
        """
        Return all sink nodes (no outgoing links).

        Returns
        -------
        list of Node
        """
        return [n for n in self._nodes.values() if n.is_sink()]

    @property
    def monitored_sections(self) -> List[RoadSection]:
        """
        Return sections where no link carries a sensor.

        Returns
        -------
        list of RoadSection
        """
        return [s for s in self._sections.values() if s.is_monitored]

    @property
    def unmonitored_sections(self) -> List[RoadSection]:
        """
        Return sections where no link carries a sensor.

        Returns
        -------
        list of RoadSection
        """
        return [s for s in self._sections.values() if not s.is_monitored]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """
        Return a dictionary of network-level statistics.

        Returns
        -------
        dict
            Keys: ``node_count``, ``link_count``, ``section_count``,
            ``junction_count``, ``source_count``, ``sink_count``,
            ``monitored_links (%)``, ``monitored_sections (%)``,
            ``extra_layers``, ``crs``.
        """
        return {
            "node_count": self.node_count,
            "link_count": self.link_count,
            "section_count": self.section_count,
            "junction_count": len(self.junction_nodes),
            "source_count": len(self.source_nodes),
            "sink_count": len(self.sink_nodes),
            "monitored_links (%)": sum(
                1 for l in self._links.values() if l.has_sensor
            ) / self.link_count if self.link_count > 0 else 0.0,
            "monitored_sections (%)": sum(
                1 for s in self._sections.values() if s.is_monitored
            ) / self.section_count if self.section_count > 0 else 0.0,
            "extra_layers": self.layer_names,
            "crs": str(self._nodes_gdf.crs),
        }

    def __repr__(self) -> str:
        return (
            f"Network("
            f"nodes={self.node_count}, "
            f"links={self.link_count}, "
            f"sections={self.section_count}, "
            f"crs={self._nodes_gdf.crs})"
        )