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
counts_data : list of dict — ``location_id``, ``link_id``, ``name`` (str, opt.), 
              ``lane_count`` (int, opt.), ``geometry`` (WKT str or shapely geometry, opt.)
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
from corvia.logger import get_logger
logger = get_logger(__name__)


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
    _sensor_geometries : dict of {str: str, shapely geometry, or None}
        Sensor geometry (WKT string or already-built shapely geometry),
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
        self._sensors: Dict[str, SensorLocation] = {}
        self._sections: Dict[str, RoadSection] = {}
        self._node_geometries: Dict[str, Optional[Union[str, BaseGeometry]]] = {}
        self._link_geometries: Dict[str, Optional[Union[str, BaseGeometry]]] = {}
        self._sensor_geometries: Dict[str, Optional[Union[str, BaseGeometry]]] = {}
        self._crs: Optional[str] = None
        self._name_network: Optional[str] = None
        self._session_open: bool = False
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

    @property
    def node_count(self) -> int:
        """int : Nodes registered in the current session."""
        return len(self._nodes)
 
    @property
    def link_count(self) -> int:
        """int : Links registered in the current session."""
        return len(self._links)

    @property
    def sensor_count(self) -> int:
        """int : Sensors registered in the current session."""
        return len(self._sensors)
 
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

        Parameters
        ----------
        name_network : str
            Name to assign to the network being built.
        crs : str, default = "EPSG:3812"
            Coordinate reference system for the session's GeoDataFrames.

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
        if self._session_open:
            self.logger.warning(
                f"begin() called with a session already open for '{self._name_network}'; "
                f"discarding it and starting a fresh session for '{name_network}'."
            )
        self.reset()
        self._name_network = name_network
        self._crs = crs
        self._session_open = True
        self.logger.info(f"Session started for network '{name_network}' (crs={crs}).")

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

        Raises
        ------
        RuntimeError
            If no session is open, or if no links have been registered yet.

        Examples
        --------
        >>> builder.begin(crs="EPSG:3812")
        >>> builder.build_raw_network(links_data, nodes_data, counts_data)
        >>> network = builder.end()
        """
        if not self._session_open:
            error_msg = f"end() called with no active session; call begin() first."
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        if not self._links:
            error_msg = f"end() called for network '{self._name_network}' with no links registred; call build_raw_network() first."
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        if not self._sections:
            self.compile_road_sections()
        network = self.build()
        self.logger.info(f"Session ended for network '{self._name_network}'.")
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
            (str).  ``name`` (str), ``lane_count`` (int) and
            ``geometry`` (WKT str or shapely geometry) are optional.

        Raises
        ------
        RuntimeError
            If no session is open (call :meth:`begin` first).

        Notes
        -----
        Node objects are created for every ``node_id`` in *nodes_data* first.
        Any node referenced in *links_data* but absent from *nodes_data* is
        created on the fly with no geometry. Any sensor in *counts_data*
        referencing an unknown ``link_id`` is skipped, not raised. Call
        :meth:`reset` before re-running on a non-empty builder.
        """
        if not self._session_open:
            error_msg = "build_raw_network() called with no active session; call begin() first."
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        self.logger.info(
            f"build_raw_network(): {len(nodes_data)} node record(s), "
            f"{len(links_data)} link record(s), {len(counts_data)} sensor record(s)."
        )
        
        # 1a. Instantiate nodes from nodes_data
        for n in nodes_data:
            nid = n["node_id"]
            if nid in self._nodes:
                error_msg = f"Duplicate node_id found in nodes_data: '{nid}'"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            node = Node(nid)
            self._nodes[nid] = node
            self._node_geometries[nid] = n.get("geometry")

        # 1b. Instantiate links; create any missing nodes on the fly
        implicit_nodes = 0
        for l in links_data:
            lid = l["link_id"]
            if lid in self._links:
                error_msg = f"Duplicate link_id found in links_data: '{lid}'"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            for nid in (l["start_node"], l["end_node"]):
                if nid not in self._nodes:
                    self._nodes[nid] = Node(nid)
                    self._node_geometries[nid] = None
                    implicit_nodes += 1

            link = Link(
                link_id=l["link_id"],
                start_node=self._nodes[l["start_node"]],
                end_node=self._nodes[l["end_node"]],
            )
            self._links[link.link_id] = link
            self._link_geometries[link.link_id] = l.get("geometry")

        # 1c. Attach sensors
        skipped_sensors = 0
        for c in counts_data:
            link_id = c["link_id"]
            if link_id not in self._links:
                skipped_sensors += 1
                self.logger.warning(
                    f"Sensor '{c['location_id']}' references unknown link '{link_id}'; skipped."
                )
                continue
            loc = SensorLocation(
                location_id=c["location_id"],
                link_id=link_id,
                name=c.get("name"),
                lane_count=c.get("lane_count", 0),
            )
            self._links[link_id].attach_sensor(loc)
            self._sensors[c["location_id"]] = loc
            self._sensor_geometries[c["location_id"]] = c.get("geometry")

        self.logger.info(
            f"build_raw_network() complete: {len(self._nodes)} node(s) ({implicit_nodes} implicit), "
            f"{len(self._links)} link(s), {len(self._sensors)} sensor(s) attached ({skipped_sensors} skipped)."
        )

    # ------------------------------------------------------------------
    # Step 2 — section compilation
    # ------------------------------------------------------------------

    def compile_road_sections(self, section_id_fn: Optional[Callable[[List[Link]], str]] = None) -> None:
        """
        Merge consecutive pass-through links into :class:`~topology.RoadSection`
        objects, then wire direct neighbour references between sections.

        A *pass-through node* is one where ``in_degree == out_degree == 1``.
        The algorithm seeds from each unvisited link, then traces forward and
        backward through pass-through nodes until hitting a junction, source,
        or sink.

        Parameters
        ----------
        section_id_fn : callable, optional
            Given the ordered list of links in a chain, return the
            ``section_id`` to assign to it. Defaults to ``None``, which
            keeps the built-in ``SECTION_0001``, ``SECTION_0002``, …
            numbering. Called once per chain, so the returned id only
            needs to be unique *within that call* — e.g. using the
            chain's first ``link_id`` is always safe, since every link
            belongs to exactly one chain.

        Raises
        ------
        RuntimeError
            If no session is open, if no links have been registered yet, or
            if sections have already been compiled for this session.

        Examples
        --------
        Key sections by their first link's id instead of the default
        numbering::

            builder.compile_road_sections(
                section_id_fn=lambda chain: chain[0].link_id
            )
        """
        if not self._session_open:
            error_msg = "compile_road_sections() called with no active session; call begin() first."
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        if self.link_count == 0:
            error_msg = (
                f"compile_road_sections() called for network '{self._name_network}' "
                "with no links registered; Call build_raw_network() first."
            )
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        if self._sections:
            error_msg = f"compile_road_sections() called again for network '{self._name_network}'; sections already compiled."
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

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

            sec_id = section_id_fn(chain) if section_id_fn else f"SECTION_{section_counter:04d}"
            self._sections[sec_id] = RoadSection(sec_id, chain)
            section_counter += 1

        self.logger.info(
            f"compile_road_section() complete: {len(self._sections)} section(s) compiled from {self.link_count} link(s)."
        )

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

        Raises
        ------
        RuntimeError
            If no session is open, or if no nodes have been registered yet.

        Notes
        -----
        Geometry strings that are ``None`` or fail WKT parsing result in
        null geometries in the GeoDataFrame — the corresponding rows are
        still present.
        """
        if not self._session_open:
            error_msg = f"No active session: call begin() and build_raw_netword() first."
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        if self.node_count == 0 or self.link_count == 0:
            error_msg = (
                f"Nothing to build: build() called for network '{self._name_network}' "
                f"with {self.node_count} nodes and {self.link_count} links registered."
            )
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        self.logger.info(f"Building GeoDataFrames and assembling Network '{self._name_network}'.")
        return Network(
            name=self._name_network,
            nodes=self._nodes,
            links=self._links,
            sections=self._sections,
            nodes_gdf=self._build_nodes_gdf(self._crs),
            links_gdf=self._build_links_gdf(self._crs),
            sensors_gdf=self._build_sensors_gdf(self._crs)
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
            Coordinate reference system to assign to the output GeoDataFrame.

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
            Coordinate reference system to assign to the output GeoDataFrame.

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

    def _build_sensors_gdf(self, crs: str) -> gpd.GeoDataFrame:
        """
        Build the sensors GeoDataFrame from stored WKT geometry strings.

        Parameters
        ----------
        crs : str
            Coordinate reference system to assign to the output GeoDataFrame.

        Returns
        -------
        geopandas.GeoDataFrame
            Index: ``location_id``.  Columns: ``geometry``.
        """
        ids = list(self._sensors.keys())
        geometries = [
            self._parse_geometry(self._sensor_geometries.get(cid)) for cid in ids
        ]
        return gpd.GeoDataFrame(
            {"geometry": geometries},
            index=pd.Index(ids, name="location_id"),
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
        except Exception as exc:
            logger.warning(f"Failed to parse WKT geometry {geom}: stored as null due to following error - {exc}")
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
        section_id_fn: Optional[Callable[[List[Link]], str]] = None,
    ) -> Network:
        """
        Run all three build steps and return a :class:`~network.Network`.

        Equivalent to::

            builder = NetworkBuilder()
            builder.build_raw_network(links_data, nodes_data, counts_data)
            builder.compile_road_sections(section_id_fn=section_id_fn)
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
        section_id_fn : callable, optional
            Forwarded to :meth:`compile_road_sections`.

        Returns
        -------
        Network

        Raises
        ------
        RuntimeError
            Propagated from :meth:`build_raw_network`, :meth:`compile_road_sections`,
            or :meth:`build` if the underlying build steps fail.
        """
        builder = cls()
        builder.begin(name_network=name_network, crs=crs)
        builder.build_raw_network(links_data, nodes_data, counts_data)
        builder.compile_road_sections(section_id_fn=section_id_fn)
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
        self._sensors.clear()
        self._sections.clear()
        self._node_geometries.clear()
        self._link_geometries.clear()
        self._sensor_geometries.clear()
        self._crs = None
        self._name_network = None
        self._session_open = False
        self.logger.debug("Builder state reset.")

    def __repr__(self) -> str:
        return (
            f"NetworkBuilder("
            f"nodes={len(self._nodes)}, "
            f"links={len(self._links)}, "
            f"sections={len(self._sections)})"
        )