"""
topology.py
===========
Road network primitives.  No external dependencies.
 
Object hierarchy after compilation::
 
    Node  ←──  Link  ──→  Node
                │
                ├── CountLocation   (optional sensor)
                └── RoadSection     (aggregate container)
 
    RoadSection  ◄──►  RoadSection  (direct neighbour references)
 
Geometry is intentionally absent from all classes here.  Spatial data
lives in GeoPandas GeoDataFrames managed by :class:`~network.Network`,
accessed via ``network.nodes_gdf.loc[node.node_id]``.
"""

from __future__ import annotations
from typing import List, Optional, Iterator

class Node:
    """
    Represents an begin or end point of a directed link.
    Can be a intersection or simple pass-through point in the network.
    
    Parameters
    ----------
    node_id : str
        Unique identifier (e.g. shapefile FID or custom label).

    Attributes
    ----------
    _node_id : str
        Stored node identifier.
    _incoming_links : list of Link
        Links whose ``end_node`` is this node.
    _outgoing_links : list of Link
        Links whose ``start_node`` is this node.
 
    Examples
    --------
    >>> n1, n2 = Node("N1"), Node("N2")
    >>> link = Link("L1", n1, n2)
    >>> n1.out_degree
    1
    >>> n1.is_source()
    True
    """

    def __init__(self, node_id: str) -> None:
        self._node_id = node_id
        self._incoming_links: List[Link] = []
        self._outgoing_links: List[Link] = []

    def summary(self) -> dict:
        """
        Return a dictionary of key metrics for this node.
 
        Returns
        -------
        dict
            Keys: ``node_id``, ``in_degree``, ``out_degree``.
        """
        return {
            "node_id": self._node_id,
            "in_degree": self.in_degree,
            "out_degree": self.out_degree
        }
 
    def __repr__(self) -> str:
        return f"Node({self._node_id} | in={self.in_degree}, out={self.out_degree})"

    # ------------------------------------------------------------------
    # Property getters and setters
    # ------------------------------------------------------------------

    @property
    def node_id(self) -> str:
        """str : Unique identifier of the node *(read-only)*."""
        return self._node_id

    @property
    def incoming_links(self) -> List[Link]:
        """list of Link : Links arriving at this node *(read-only snapshot)*."""
        return list(self._incoming_links)
 
    @property
    def outgoing_links(self) -> List[Link]:
        """list of Link : Links departing from this node *(read-only snapshot)*."""
        return list(self._outgoing_links)
    
    @property
    def in_degree(self) -> int:
        """int : Number of incoming links."""
        return len(self._incoming_links)
 
    @property
    def out_degree(self) -> int:
        """int : Number of outgoing links."""
        return len(self._outgoing_links)
 
    @property
    def degree(self) -> int:
        """int : Total number of connected links (in + out)."""
        return self.in_degree + self.out_degree
 
    def neighbours(self, direction: str = 'downstream') -> List[Node]:
        """
        List the nodes reachable from this node in one hop.

        Parameters
        ----------
        direction : str, default = 'downstream'
            Direction of traversal ('downstream' or 'upstream').
        Returns
        -------
        neighbours : list of Nodes
        """
        if direction == 'downstream':
            return [link.end_node for link in self._outgoing_links]
        elif direction == 'upstream':
            return [link.start_node for link in self._incoming_links]
        else:
            raise ValueError("Direction must be 'downstream' or 'upstream'.")

    # ------------------------------------------------------------------
    # Internal link registration (called by Link.__init__)
    # ------------------------------------------------------------------

    def _register_incoming(self, link: Link) -> None:
        """
        Register *link* as arriving at this node.
 
        Parameters
        ----------
        link : Link
            The directed link whose ``end_node`` is this node.
        """
        if link in self._incoming_links:
            raise ValueError(
                f"Link '{link.link_id}' already registered as incoming on {self}."
            )
        self._incoming_links.append(link)
 
    def _register_outgoing(self, link: Link) -> None:
        """
        Register *link* as departing from this node.
 
        Parameters
        ----------
        link : Link
            The directed link whose ``start_node`` is this node.
        """
        if link in self._outgoing_links:
            raise ValueError(
                f"Link '{link.link_id}' already registered as outgoing on {self}."
            )
        self._outgoing_links.append(link)
 
    def _deregister_incoming(self, link: Link) -> None:
        """Remove *link* from the incoming list (called on link deletion)."""
        try:
            self._incoming_links.remove(link)
        except ValueError:
            raise ValueError(
                f"Link '{link.link_id}' not found in incoming links of {self}."
            )
 
    def _deregister_outgoing(self, link: Link) -> None:
        """Remove *link* from the outgoing list (called on link deletion)."""
        try:
            self._outgoing_links.remove(link)
        except ValueError:
            raise ValueError(
                f"Link '{link.link_id}' not found in outgoing links of {self}."
            )
        
    # ------------------------------------------------------------------
    # Public query helpers
    # ------------------------------------------------------------------

    def is_junction(self) -> bool:
        """
        Determine whether this node acts as a real junction.
 
        A node is a junction if it has more than one incoming **or** more
        than one outgoing link (i.e. a merge or a diverge).
 
        Returns
        -------
        bool
            True when junction of more then two links.
        """
        return (self.in_degree > 1) or (self.out_degree > 1)
    
    def is_passthrough(self) -> bool:
        """
        Return ``True`` when the node has exactly one incoming and one
        outgoing link and carries no topological significance.
 
        Returns
        -------
        bool
        """
        return (self.in_degree == 1 and self.out_degree == 1)
 
    def is_source(self) -> bool:
        """
        Check whether the node is a source (no incoming links).
 
        Returns
        -------
        bool
            True when only outgoing links.
        """
        return (self.in_degree == 0) and (self.out_degree > 0)
 
    def is_sink(self) -> bool:
        """
        Check whether the node is a sink (no outgoing links).
 
        Returns
        -------
        bool
            True when only incoming links.
        """
        return (self.in_degree > 0) and (self.out_degree == 0)
 
    def get_link_to(self, other: Node) -> Optional[Link]:
        """
        Return the direct link from this node to *other*, if it exists.
 
        Parameters
        ----------
        other : Node
            The candidate node.
        direction : str, default = 'downstream'
            Direction of traversal ('downstream' or 'upstream').
 
        Returns
        -------
        Link or None
            The matching and outgoing link, or ``None`` if not adjacent.
        """
        for link in self._outgoing_links:
            if link.end_node is other:
                return link
        return None

    def get_link_from(self, other: Node) -> Optional[Link]:
        """
        Return the direct link to this node from *other*, if it exists.
 
        Parameters
        ----------
        other : Node
            The candidate downstream node.
 
        Returns
        -------
        Link or None
            The first matching incoming link, or ``None`` if not adjacent.
        """
        for link in self._incoming_links:
            if link.start_node is other:
                return link
        return None
    
    
class Link:
    """
    Represents a raw directed road segment (arc).

    A Link is a directed edge in the road-network graph.  On construction it
    automatically registers itself with both endpoint :class:`Node` objects.
    It may carry an optional :class:`CountLocation` sensor and belong to a
    :class:`RoadSection`.

    Parameters
    ----------
    link_id : str
        Unique identifier (e.g. shapefile FID or custom label).
    start_node : Node
        Upstream node where this link originates.
    end_node : Node
        Downstream node where this link terminates.

    Attributes
    ----------
    _link_id : str
        Stored link identifier.
    _start_node : Node
        Stored upstream node reference.
    _end_node : Node
        Stored downstream node reference.
    _sensor_location : SensorLocation or None
        Sensor attached to this link, if any.
    _parent_section : RoadSection or None
        Section that owns this link, if any.
 
    Examples
    --------
    >>> n1, n2 = Node("N1"), Node("N2")
    >>> link = Link("L1", n1, n2)
    >>> link
    Link(L1 | N1 -> N2)
    >>> link.has_sensor
    False
    >>> n1.out_degree   # auto-registered
    1
    """
    def __init__(self, link_id: str, start_node: Node, end_node: Node) -> None:
        self._link_id: str = link_id
        self._start_node: Node = start_node
        self._end_node: Node = end_node
        self._sensor_locations: List[SensorLocation] = []
        self._parent_section: Optional[RoadSection] = None
 
        # Auto-register with the endpoint nodes
        start_node._register_outgoing(self)
        end_node._register_incoming(self)

    def summary(self) -> dict:
        """
        Return a dictionary of key attributes for this link.
 
        Returns
        -------
        dict
            Keys: ``link_id``, ``start_node``, ``end_node``,
            ``has_sensor``, ``is_assigned``, ``geometry``.
        """
        return {
            "link_id": self._link_id,
            "start_node": str(self._start_node),
            "end_node": str(self._end_node),
            "sensor_count": len(self._sensor_locations),
            "is_assigned": self.is_assigned,
        }
 
    def __repr__(self) -> str:
        return (
            f"Link({self._link_id} | {self._start_node.node_id} -> {self._end_node.node_id})"
        )
    
    # ------------------------------------------------------------------
    # Property getters and setters
    # ------------------------------------------------------------------
 
    @property
    def link_id(self) -> str:
        """str : Unique identifier of the link *(read-only)*."""
        return self._link_id
 
    @property
    def start_node(self) -> Node:
        """Node : Upstream node *(read-only)*."""
        return self._start_node
 
    @property
    def end_node(self) -> Node:
        """Node : Downstream node *(read-only)*."""
        return self._end_node

    @property
    def sensor_locations(self) -> Optional[SensorLocation]:
        """list of SensorLocation : Sensors attached to this link *(read-only)*."""
        return list(self._sensor_locations)
 
    @property
    def parent_section(self) -> Optional[RoadSection]:
        """RoadSection or None : Section that owns this link *(read-only)*."""
        return self._parent_section
 
    @property
    def has_sensor(self) -> bool:
        """bool : ``True`` when at least one :class:`SensorLocation` is attached."""
        return len(self._sensor_locations) > 0
 
    @property
    def is_assigned(self) -> bool:
        """bool : ``True`` when this link belongs to a :class:`RoadSection`."""
        return self._parent_section is not None
    
    # ------------------------------------------------------------------
    # Sensor management
    # ------------------------------------------------------------------
 
    def attach_sensor(self, sensor_location: SensorLocation) -> None:
        """
        Attach a :class:`SensorLocation` sensor to this link.
 
        Parameters
        ----------
        sensor_location : SensorLocation
            The sensor to attach.
        """
        if any(s.location_id == sensor_location.location_id for s in self._sensor_locations):
            raise ValueError(
                f"Link '{self._link_id}' already carries this sensor "
                f"'{sensor_location.location_id}'."
            )
        self._sensor_locations.append(sensor_location)
 
    def detach_sensor(self, location_id: str) -> Optional[SensorLocation]:
        """
        Remove and return a specific sensor by its unique ID.
 
        Returns
        -------
        SensorLocation or None
            The detached sensor, or ``None`` if none was attached.
        """
        for idx, s in enumerate(self._sensor_locations):
            if s.location_id == location_id:
                return self._sensor_locations.pop(idx)
        return None
 
    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
 
    def create_reverse(self) -> Link:
        """
        Create a new :class:`Link` with start and end nodes swapped.
 
        The reversed link has ``"_rev"`` appended to the original ID,
        carries no sensor, and belongs to no section.  The geometry
        string is preserved as-is (caller is responsible for reversing
        coordinate order if needed).
 
        Returns
        -------
        Link
            A new link running in the opposite direction.
        """
        return Link(
            link_id=f"{self._link_id}_rev",
            start_node=self._end_node,
            end_node=self._start_node,
        )
 
    def is_adjacent_to(self, other: Link, direction: str = 'downstream') -> bool:
        """
        Check whether *other* can follow this link in a chain.
 
        Parameters
        ----------
        other : Link
            The candidate next link.
        direction : str, default = 'downstream'
            Direction of traversal ('downstream' or 'upstream').
 
        Returns
        -------
        bool
            ``True`` when ``self.end_node is other.start_node``.
        """
        if direction == 'downstream':
            return self._end_node is other.start_node
        elif direction == 'upstream':
            return self._start_node is other.end_node


class SensorLocation:
    """
    A physical sensor station attached to a specific :class:`Link`.
 
    A SensorLocation represents a loop detector, radar, or any other traffic
    counting device mounted at a fixed point on the network.  It is
    associated with exactly one link (by ID) and may cover one or more lanes.
 
    Parameters
    ----------
    location_id : str
        Unique identifier for this sensor station.
    link_id : str
        Identifier of the :class:`Link` this sensor is mounted on.
    name : str, optional
        Human-readable label (e.g. ``"A10 km 23.4"``).
    lane_count : int, optional
        Number of lanes covered by this sensor.  Must be >= 1 or 0 (unknown).
        Defaults to ``1``.
    """
 
    def __init__(self, location_id: str, link_id: str, name: Optional[str] = None, lane_count: int = 0) -> None:
        if lane_count < 0:
            raise ValueError(f"lane_count must be >= 1 or 0 (unknown), got {lane_count}.")
        self._location_id: str = location_id
        self._link_id: str = link_id
        self._name: Optional[str] = name
        self._lane_count: int = lane_count

    def summary(self) -> dict:
        """
        Return a dictionary summarising this sensor's metadata.
 
        Returns
        -------
        dict
            Keys: ``location_id``, ``link_id``, ``name``, ``lane_count``.
        """
        return {
            "location_id": self._location_id,
            "link_id": self._link_id,
            "name": self._name,
            "lane_count": self._lane_count,
        }
 
    def __repr__(self) -> str:
        return f"SensorLocation({self._location_id} on Link {self._link_id})"
 
    # ------------------------------------------------------------------
    # Property getters and setters
    # ------------------------------------------------------------------
 
    @property
    def location_id(self) -> str:
        """str : Unique identifier of the sensor *(read-only)*."""
        return self._location_id
 
    @property
    def link_id(self) -> str:
        """str : Identifier of the host link *(read-only)*."""
        return self._link_id
 
    @property
    def name(self) -> Optional[str]:
        """str or None : Human-readable station label."""
        return self._name
 
    @name.setter
    def name(self, value: Optional[str]) -> None:
        self._name = value
 
    @property
    def lane_count(self) -> int:
        """int : Number of lanes covered by this sensor."""
        return self._lane_count
 
    @lane_count.setter
    def lane_count(self, value: int) -> None:
        if value < 0:
            raise ValueError(f"lane_count must be >= 1 or 0 (unknown), got {value}.")
        self._lane_count = value

    @property
    def is_lane_count_known(self) -> bool:
        """bool : ``True`` if lane count is known, ``False`` otherwise."""
        return self._lane_count > 0
 
    @property
    def display_name(self) -> str:
        """str : ``name`` if set, otherwise falls back to ``location_id``."""
        return self._name if self._name else self._location_id


class RoadSection:
    """
    An ordered, aggregated sequence of :class:`Link` objects with no
    intermediate exits or entries.
 
    A RoadSection groups a topologically consistent chain of links that form
    a continuous, uninterrupted stretch of highway (e.g. between two ramps).
    It exposes sensor coverage metrics and provides convenient iteration and
    look-up helpers.
 
    Parameters
    ----------
    section_id : str
        Unique identifier for this road section.
    links : list of Link
        Ordered list of directed links forming the section.  Must be
        non-empty and topologically consistent (each link's ``end_node``
        must equal the next link's ``start_node``).

    Attributes
    ----------
    _section_id : str
        Stored section identifier.
    _links : list of Link
        Stored links.
    _upstream_sections : list of RoadSection
        Sections that lead into this one.
    _downstream_sections : list of RoadSection
        Sections that lead out of this one.
 
    Examples
    --------
    >>> n1, n2, n3 = Node("N1"), Node("N2"), Node("N3")
    >>> l1, l2 = Link("L1", n1, n2), Link("L2", n2, n3)
    >>> sec = RoadSection("S1", [l1, l2])
    >>> sec.entry_node
    Node(N1, in=0, out=1)
    >>> sec.exit_node
    Node(N3, in=1, out=0)
    """

    def __init__(self, section_id: str, links: List[Link]) -> None:
        if not links:
            raise ValueError("A RoadSection must contain at least one Link.")
        self._validate_chain(links)
 
        self._section_id: str = section_id
        self._links: List[Link] = list(links)
 
        # Back-reference each link to this section
        for link in self._links:
            link._parent_section = self
            
    def summary(self) -> dict:
        """
        Return a dictionary summarising the section's key metrics.
 
        Returns
        -------
        dict
            Keys: ``section_id``, ``link_count``, ``sensor_count``,
            ``coverage_ratio``, ``is_fully_monitored``,
            ``entry_node``, ``exit_node``.
        """
        return {
            "section_id": self._section_id,
            "entry_node": str(self.entry_node),
            "exit_node": str(self.exit_node),
            "link_count": self.link_count,
            "sensor_count": self.sensor_count,
            "upstream_sections_count": len(self.upstream_in)+len(self.upstream_out),
            "downstream_sections_count": len(self.downstream_in)+len(self.downstream_out),
        }

    def __repr__(self):
        return (f"RoadSection({self.section_id} | {self.entry_node.node_id} -> {self.exit_node.node_id})"
        )

    # ------------------------------------------------------------------
    # Property getters and setters
    # ------------------------------------------------------------------
    
    @property
    def section_id(self) -> str:
        """str : Unique identifier of the section *(read-only)*."""
        return self._section_id
    
    @property
    def links(self) -> List[Link]:
        """list of Link : Ordered links of the section *(read-only)*."""
        return list(self._links)
    
    @property
    def link_count(self) -> int:
        """int : Total number of links in this section."""
        return len(self._links)

    @property
    def sensor_locations(self) -> List[SensorLocation]:
        """Dynamically fetch all physical sensors embedded inside this section."""
        locations = []
        for link in self.links:
            locations.extend(link.sensor_locations)
        return locations
    
    @property
    def sensor_count(self) -> int:
        """int : Number of links that carry a sensor."""
        return len(self.sensor_locations)

    @property
    def is_monitored(self) -> bool:
        """bool : ``True`` when a link in the section has a sensor."""
        return any(link.has_sensor for link in self._links)

    @property
    def entry_node(self) -> Node:
        """Node : Upstream boundary node of the section."""
        return self._links[0].start_node
 
    @property
    def exit_node(self) -> Node:
        """Node : Downstream boundary node of the section."""
        return self._links[-1].end_node
    
    @property
    def upstream_in(self) -> List[RoadSection]:
        """list of RoadSection : Sections that feed directly into this one *(read-only)*."""
        return list({
            link.parent_section
            for link in self.entry_node.incoming_links
            if link.parent_section is not None
        })
    
    @property
    def upstream_out(self) -> List[RoadSection]:
        """list of RoadSection : Sections leaving the entry node besides this one *(read-only)*."""
        return list({
            link.parent_section
            for link in self.entry_node.outgoing_links
            if (link.parent_section is not None) and (link.parent_section is not self)
        })
 
    @property
    def downstream_in(self) -> List[RoadSection]:
        """list of RoadSection : Sections entering the exit node besides this one *(read-only)*."""
        return list({
            link.parent_section
            for link in self.exit_node.incoming_links
            if (link.parent_section is not None) and (link.parent_section is not self)
        })
    
    @property
    def downstream_out(self) -> List[RoadSection]:
        """list of RoadSection : Sections leaving the exit node *(read-only)*."""
        return list({
            link.parent_section
            for link in self.exit_node.outgoing_links
            if link.parent_section is not None
        })
    
    # ------------------------------------------------------------------
    # Topology management
    # ------------------------------------------------------------------
 
    @staticmethod
    def _validate_chain(links: List[Link]) -> None:
        """
        Verify that consecutive links share boundary nodes.
 
        Parameters
        ----------
        links : list of Link
            The ordered link sequence to validate.
        """
        for i, (a, b) in enumerate(zip(links, links[1:])):
            if a.end_node is not b.start_node:
                raise ValueError(
                    f"Broken chain at position {i}: Link '{a.link_id}' ends at "
                    f"{a.end_node} but Link '{b.link_id}' starts at {b.start_node}."
                )
    
    def append_link(self, link: Link) -> None:
        """
        Append a new link to the *end* of the section.
 
        Parameters
        ----------
        link : Link
            The link to append.  Its ``start_node`` must equal the current
            :attr:`exit_node` of the section.
        """
        if link.start_node is not self.exit_node:
            raise ValueError(
                f"Cannot append Link '{link.link_id}': its start_node "
                f"({link.start_node}) does not match the current exit_node "
                f"({self.exit_node})."
            )
        self._links.append(link)
        link._parent_section = self
 
    def get_link_by_id(self, link_id: str) -> Optional[Link]:
        """
        Look up a link inside this section by its identifier.
 
        Parameters
        ----------
        link_id : str
            The link identifier to search for.
 
        Returns
        -------
        Link or None
            The matching link, or ``None`` if not found.
        """
        for link in self._links:
            if link.link_id == link_id:
                return link
        return None
 
    def iter_links(self) -> Iterator[Link]:
        """
        Yield links in topological order from entry to exit.
 
        Yields
        ------
        Link
            Each link in the section, in order.
        """
        yield from self._links
 
    def iter_sensors(self) -> Iterator[SensorLocation]:
        """
        Yield all sensors in the section in link order.
 
        Yields
        ------
        SensorLocation
            Each sensor found on a link, in topological order.
        """
        for link in self._links:
            yield from link.sensor_locations