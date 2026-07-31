"""
builder.py
==========
Construction tool that assembles a :class:`~network.Network` from raw GIS
record dictionaries.

The builder is a **temporary** object: run it, call :meth:`~NetworkBuilder.build`,
keep the returned :class:`~network.Network`, discard (or reset) the builder.

Dependency chain::

    builder.py  →  network.py   →  topology.py
               →  geopandas / shapely

Three explicit build steps
--------------------------
1. :meth:`NetworkBuilder.build_raw_network` — create :class:`~topology.Node`
   and :class:`~topology.Link` objects; store geometry strings for later GDF
   construction; attach :class:`~topology.SensorLocation` sensors.
2. :meth:`NetworkBuilder.compile_road_sections` — chain pass-through links
   into :class:`~topology.RoadSection` objects; wire direct section neighbours.
3. :meth:`NetworkBuilder.build` — convert stored geometry strings into Shapely
   objects, build GeoDataFrames, assemble and return the :class:`~network.Network`.

Or use the convenience classmethod :meth:`NetworkBuilder.from_data` to run all
three steps in one call.

Expected record shapes
----------------------
nodes_data  : list of dict — ``node_id`` (str), ``geometry`` (WKT str or shapely geometry, opt.)
links_data  : list of dict — ``link_id``, ``start_node``, ``end_node``,
              ``geometry`` (WKT str or shapely geometry, opt.)
counts_data : list of dict — ``location_id``, ``link_id``,
              ``name`` (str, opt.), ``lane_count`` (int, opt.)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Union

import geopandas as gpd
import pandas as pd
from shapely import wkt as shapely_wkt
from shapely.geometry.base import BaseGeometry

from corvia.framework.network import Network
from corvia.framework.topology import SensorLocation, Link, Node, RoadSection


class NetworkBuilder:
    """
    Assemble a :class:`~network.Network` from flat GIS record dictionaries.

    Parameters
    ----------
    (none)

    Attributes
    ----------
    _nodes : dict of {str: Node}
        Node registry built during :meth:`build_raw_network`.
    _links : dict of {str: Link}
        Link registry built during :meth:`build_raw_network`.
    _sections : dict of {str: RoadSection}
        Section registry built during :meth:`compile_road_sections`.
    _node_geometries : dict of {str: str, shapely geometry, or None}
        Node geometry (WKT string or already-built shapely geometry),
        consumed by :meth:`build`.
    _link_geometries : dict of {str: str, shapely geometry, or None}
        Link geometry (WKT string or already-built shapely geometry),
        consumed by :meth:`build`.

    Examples
    --------
    Four-step API::

        builder = NetworkBuilder()
        builder.begin(name_network="My Network", crs="EPSG:31370")
        builder.build_raw_network(links_data, nodes_data, counts_data)
        builder.compile_road_sections()
        network = builder.build()
        builder.end()  # optional, resets internal state for reuse

    One-liner::

        network = NetworkBuilder.from_data("My Network",
            links_data, nodes_data, counts_data, crs="EPSG:31370"
        )
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, Node] = {}
        self._links: Dict[str, Link] = {}
        self._sections: Dict[str, RoadSection] = {}
        self._node_geometries: Dict[str, Optional[Union[str, BaseGeometry]]] = {}
        self._link_geometries: Dict[str, Optional[Union[str, BaseGeometry]]] = {}
        self._crs: Optional[str] = None
        self._name_network: Optional[str] = None
        self._session_open: bool = False

    @property
    def node_count(self) -> int:
        """int : Nodes registered in the current session."""
        return len(self._nodes)
 
    @property
    def link_count(self) -> int:
        """int : Links registered in the current session."""
        return len(self._links)
 
    @property
    def name_network(self) -> str:
        """str : Name of the current network."""
        return self._name_network

    @property
    def crs(self) -> str:
        """str : Coordinate reference system for the current session."""
        return self._crs
    
    @property
    def is_session_open(self) -> bool:
        """bool : ``True`` when a session is active (between begin and end)."""
        return self._session_open

    def begin(self, name_network: str, crs: str = "EPSG:3812") -> None:
        """
        Start a new build session.
 
        Clears all internal state so the builder can be reused for a
        different network without creating a new instance.  Call this
        before :meth:`build_raw_network` on every reuse.
 
        Notes
        -----
        If a session is already open (i.e. :meth:`begin` was called
        without a matching :meth:`end`), the existing state is discarded
        and a fresh session starts.
 
        Examples
        --------
        >>> builder = NetworkBuilder()
        >>> builder.begin(name_network="Network_A", crs="EPSG:3812")
        >>> builder.build_raw_network(links_data_a, nodes_data_a, counts_data_a)
        >>> network_a = builder.end()
        >>> builder.begin(name_network="Network_B", crs="EPSG:31370")
        >>> builder.build_raw_network(links_data_b, nodes_data_b, counts_data_b)
        >>> network_b = builder.end()
        """
        self.reset()
        self._name_network = name_network
        self._crs = crs
        self._session_open = True

    def end(self) -> Network:
        """
        Finalise the current session and return the assembled network.
 
        Calls :meth:`compile_road_sections` automatically if it has not
        been called yet, then delegates to :meth:`build`.  Internal state
        is reset afterwards so the builder is ready for the next
        :meth:`begin` call.
 
        Returns
        -------
        Network

        Examples
        --------
        >>> builder.begin(crs="EPSG:3812")
        >>> builder.build_raw_network(links_data, nodes_data, counts_data)
        >>> network = builder.end()
        """
        if not self._session_open:
            raise RuntimeError(
                "No active session. Call begin() before end()."
            )
        if not self._links:
            raise RuntimeError(
                "No links found. Call build_raw_network() before end()."
            )
        if not self._sections:
            self.compile_road_sections()
        network = self.build()
        self.reset()
        return network

    # ------------------------------------------------------------------
    # Step 1 — raw network
    # ------------------------------------------------------------------

    def build_raw_network(
        self,
        links_data: List[dict],
        nodes_data: List[dict],
        counts_data: List[dict],
    ) -> None:
        """
        Create :class:`~topology.Node` and :class:`~topology.Link` objects
        from flat record dictionaries.

        Node–link connectivity is wired automatically by
        :class:`~topology.Link` on construction.  Geometry strings are stored
        separately and converted to Shapely objects only in :meth:`build`.

        Parameters
        ----------
        links_data : list of dict
            Each dict must contain ``link_id`` (str), ``start_node`` (str),
            ``end_node`` (str).  ``geometry`` (WKT str or shapely geometry) is optional.
        nodes_data : list of dict
            Each dict must contain ``node_id`` (str).
            ``geometry`` (WKT str or shapely geometry) is optional.
        counts_data : list of dict
            Each dict must contain ``location_id`` (str) and ``link_id``
            (str).  ``name`` (str) and ``lane_count`` (int) are optional.

        Notes
        -----
        Node objects are created for every ``node_id`` in *nodes_data* first.
        Any node referenced in *links_data* but absent from *nodes_data* is
        created on the fly with no geometry.  Call :meth:`reset` before
        re-running on a non-empty builder.
        """
        if not self._session_open:
            raise RuntimeError(
                "No active session. Call begin() before build_raw_network()."
            )
        
        # 1a. Instantiate nodes from nodes_data
        for n in nodes_data:
            node = Node(n["node_id"])
            self._nodes[node.node_id] = node
            self._node_geometries[node.node_id] = n.get("geometry")

        # 1b. Instantiate links; create any missing nodes on the fly
        for l in links_data:
            for nid in (l["start_node"], l["end_node"]):
                if nid not in self._nodes:
                    self._nodes[nid] = Node(nid)
                    self._node_geometries[nid] = None

            link = Link(
                link_id=l["link_id"],
                start_node=self._nodes[l["start_node"]],
                end_node=self._nodes[l["end_node"]],
            )
            self._links[link.link_id] = link
            self._link_geometries[link.link_id] = l.get("geometry")

        # 1c. Attach sensors
        for c in counts_data:
            link_id = c["link_id"]
            if link_id not in self._links:
                continue
            loc = SensorLocation(
                location_id=c["location_id"],
                link_id=link_id,
                name=c.get("name"),
                lane_count=c.get("lane_count", 0),
            )
            self._links[link_id].attach_sensor(loc)

    # ------------------------------------------------------------------
    # Step 2 — section compilation
    # ------------------------------------------------------------------

    def compile_road_sections(self) -> None:
        """
        Merge consecutive pass-through links into :class:`~topology.RoadSection`
        objects, then wire direct neighbour references between sections.

        A *pass-through node* is one where ``in_degree == out_degree == 1``.
        The algorithm seeds from each unvisited link, then traces forward and
        backward through pass-through nodes until hitting a junction, source,
        or sink.
        """
        if not self._session_open:
            raise RuntimeError(
                "No active session. Call begin() before compile_road_sections()."
            )
        
        if self.link_count == 0:
            raise RuntimeError(
                "No links found. Call build_raw_network() before compile_road_sections()."
            )
        
        if self._sections:
            raise RuntimeError("Sections already compiled. Call reset() or begin() before recompiling.")

        visited: set[str] = set()
        section_counter = 1

        for link_id, link in self._links.items():
            if link_id in visited:
                continue

            chain = [link]
            visited.add(link_id)

            # Trace forward through pass-through nodes
            current = link
            while current.end_node.is_passthrough():
                next_link = current.end_node.outgoing_links[0]
                if next_link.link_id in visited:
                    break
                chain.append(next_link)
                visited.add(next_link.link_id)
                current = next_link

            # Trace backward through pass-through nodes
            current = link
            while current.start_node.is_passthrough():
                prev_link = current.start_node.incoming_links[0]
                if prev_link.link_id in visited:
                    break
                chain.insert(0, prev_link)
                visited.add(prev_link.link_id)
                current = prev_link

            sec_id = f"SECTION_{section_counter:04d}"
            self._sections[sec_id] = RoadSection(sec_id, chain)
            section_counter += 1

    # ------------------------------------------------------------------
    # Step 3 — assemble Network
    # ------------------------------------------------------------------

    def build(self) -> Network:
        """
        Convert stored geometry strings into GeoDataFrames and return a
        fully assembled :class:`~network.Network`.

        Returns
        -------
        Network

        Notes
        -----
        Geometry strings that are ``None`` or fail WKT parsing result in
        null geometries in the GeoDataFrame — the corresponding rows are
        still present.
        """
        if not self._session_open:
            raise RuntimeError(
                "No active session. Call begin() and build_raw_network() first."
            )
        if self.node_count == 0:
            raise RuntimeError(
                "Nothing to build. Call build_raw_network() first."
            )

        return Network(
            name=self._name_network,
            nodes=self._nodes,
            links=self._links,
            sections=self._sections,
            nodes_gdf=self._build_nodes_gdf(self._crs),
            links_gdf=self._build_links_gdf(self._crs),
        )

    # ------------------------------------------------------------------
    # GeoDataFrame construction helpers
    # ------------------------------------------------------------------

    def _build_nodes_gdf(self, crs: str) -> gpd.GeoDataFrame:
        """
        Build the nodes GeoDataFrame from stored WKT geometry strings.

        Parameters
        ----------
        crs : str

        Returns
        -------
        geopandas.GeoDataFrame
            Index: ``node_id``.  Columns: ``geometry``.
        """
        ids = list(self._nodes.keys())
        geometries = [
            self._parse_geometry(self._node_geometries.get(nid)) for nid in ids
        ]
        return gpd.GeoDataFrame(
            {"geometry": geometries},
            index=pd.Index(ids, name="node_id"),
            crs=crs,
        )

    def _build_links_gdf(self, crs: str) -> gpd.GeoDataFrame:
        """
        Build the links GeoDataFrame from stored WKT geometry strings.

        Parameters
        ----------
        crs : str

        Returns
        -------
        geopandas.GeoDataFrame
            Index: ``link_id``.  Columns: ``geometry``.
        """
        ids = list(self._links.keys())
        geometries = [
            self._parse_geometry(self._link_geometries.get(lid)) for lid in ids
        ]
        return gpd.GeoDataFrame(
            {"geometry": geometries},
            index=pd.Index(ids, name="link_id"),
            crs=crs,
        )

    @staticmethod
    def _parse_geometry(geom: Optional[Union[str, BaseGeometry]]) -> Optional[BaseGeometry]:
        """
        Resolve a stored geometry value into a Shapely geometry.

        Parameters
        ----------
        geom : str, shapely.geometry.base.BaseGeometry, or None
            Either a WKT string (parsed via ``shapely.wkt.loads``) or an
            already-built Shapely geometry, returned unchanged.

        Returns
        -------
        shapely.geometry.base.BaseGeometry or None
            ``None`` when *geom* is ``None``/falsy or a WKT string 
            that fails to parse.
        """
        if isinstance(geom, BaseGeometry) or geom is None:
            return geom
        try:
            return shapely_wkt.loads(geom)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Convenience entry point
    # ------------------------------------------------------------------

    @classmethod
    def from_data(
        cls,
        name_network: str,
        links_data: List[dict],
        nodes_data: List[dict],
        counts_data: List[dict],
        crs: str = "EPSG:3812",
    ) -> Network:
        """
        Run all three build steps and return a :class:`~network.Network`.

        Equivalent to::

            builder = NetworkBuilder()
            builder.build_raw_network(links_data, nodes_data, counts_data)
            builder.compile_road_sections()
            return builder.build(crs=crs)

        Parameters
        ----------
        name_network : str
            Name of the network.
        links_data : list of dict
        nodes_data : list of dict
        counts_data : list of dict
        crs : str, optional
            Coordinate reference system.  Defaults to ``"EPSG:3812"``.

        Returns
        -------
        Network
        """
        builder = cls()
        builder.begin(name_network=name_network, crs=crs)
        builder.build_raw_network(links_data, nodes_data, counts_data)
        builder.compile_road_sections()
        return builder.end()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Clear all internal state, returning the builder to its initial state.

        Notes
        -----
        Back-references on discarded objects (e.g. ``link.parent_section``)
        are not cleaned up.  Discard those objects after calling ``reset()``.
        """
        self._nodes.clear()
        self._links.clear()
        self._sections.clear()
        self._node_geometries.clear()
        self._link_geometries.clear()
        self._crs = None
        self._name_network = None
        self._session_open = False

    def __repr__(self) -> str:
        return (
            f"NetworkBuilder("
            f"nodes={len(self._nodes)}, "
            f"links={len(self._links)}, "
            f"sections={len(self._sections)})"
        )