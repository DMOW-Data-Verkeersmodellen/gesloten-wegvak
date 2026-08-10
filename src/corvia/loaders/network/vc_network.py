"""
vc_gpkg_loader.py
==================
Loader for the "VC" segment-based GeoPackage format.

Source layers expected in the ``.gpkg``::

    segmenten               — one row per directed road segment (link)
    segmentdelen_aangepast  — segment parts; maps SD_ID (part) -> SG_ID (segment)
    locposten                — sensor / count locations, keyed on SD_ID

Dependency chain::

    vc_gpkg_loader.py  →  base.py
                        →  geopandas / pandas

.. note::
   "VC" reflects the naming used in the sample source file
   (``R2_netwerk-VC.gpkg``) — most likely short for *Verkeerscentrum*.
   Rename the class if your team uses a different name for this format.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from corvia.loaders.network.base import BaseNetworkLoader


class VCNetworkLoader(BaseNetworkLoader):
    """
    Load a "VC" segment GeoPackage into builder-ready records.

    Reads three layers from the source GeoPackage — segments, segment
    parts and sensor locations — and assembles ``links_data``,
    ``nodes_data`` and ``counts_data`` in the shape expected by
    :meth:`~corvia.framework.builder.NetworkBuilder.build_raw_network`.
    Sensors are mapped onto links through the segment-parts layer
    (``SD_ID`` → ``SG_ID``); segments' declared neighbours (``Vorige`` /
    ``Volgende``) are cross-checked against actual node connectivity, and
    any mismatch is recorded as a warning rather than raised.

    Parameters
    ----------
    source : str
        Path to the ``.gpkg`` file.
    links_layer : str, optional
        Name of the segments layer. Defaults to ``"segmenten"``.
    parts_layer : str, optional
        Name of the segment-parts layer. Defaults to
        ``"segmentdelen_aangepast"``.
    sensors_layer : str, optional
        Name of the sensor-locations layer. Defaults to ``"locposten"``.
    include_geometry : bool, optional
        Whether to carry link geometries through as WKT strings.
        Defaults to ``True``.
    verbose : bool, optional
        See :class:`~corvia.loaders.base.NetworkDataLoader`. Defaults to
        ``True``.

    Attributes
    ----------
    _links_layer : str
    _parts_layer : str
    _sensors_layer : str
    _include_geometry : bool

    Examples
    --------
    >>> loader = VCGeoPackageLoader("R2_netwerk-VC.gpkg")
    >>> links_data, nodes_data, counts_data, crs = loader.load()
    >>> network = NetworkBuilder.from_data(
    ...     "Antwerp-R2", links_data, nodes_data, counts_data, crs=crs
    ... )
    """

    #: Column rename maps, kept as class attributes so a source with
    #: slightly different column names can be supported by overriding
    #: them on a subclass instead of editing the parsing logic.
    MAP_COLUMNS_LINK = {"SG_ID": "link_id", "MP_begin": "start_node", "MP_einde": "end_node"}
    MAP_COLUMNS_SENSOR = {"LOCPOST": "location_id", "SD_ID": "part_id"}

    def __init__(
        self,
        source: str,
        links_layer: str = "segmenten",
        parts_layer: str = "segmentdelen",
        sensors_layer: str = "locposten",
        include_geometry: bool = True,
        verbose: bool = True,
    ) -> None:
        super().__init__(source, verbose=verbose)
        self._links_layer: str = links_layer
        self._parts_layer: str = parts_layer
        self._sensors_layer: str = sensors_layer
        self._include_geometry: bool = include_geometry

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def load(self) -> Tuple[List[dict], List[dict], List[dict], str]:
        """
        Read the GeoPackage and return builder-ready records.

        Returns
        -------
        links_data : list of dict
        nodes_data : list of dict
        counts_data : list of dict
        crs : str

        Notes
        -----
        Equivalent to calling :meth:`_build_nodes`, :meth:`_build_links`,
        :meth:`_map_sensors` and :meth:`_validate_topology` in sequence.
        Parsing issues are recorded in :attr:`warnings`, not raised —
        inspect it after calling :meth:`load`.
        """
        segments_df = gpd.read_file(self._source, layer=self._links_layer)
        parts_df = gpd.read_file(self._source, layer=self._parts_layer)
        sensors_df = gpd.read_file(self._source, layer=self._sensors_layer)
        sensors_df = sensors_df[sensors_df["LOCPOST"]<1000000]  # exclude virtual locposten

        nodes_data = self._build_nodes(segments_df)
        links_data = self._build_links(segments_df)
        sensors_data = self._map_sensors(sensors_df, parts_df, links_data)
        self._validate_topology(segments_df)

        return links_data, nodes_data, sensors_data, segments_df.crs.to_string()

    # ------------------------------------------------------------------
    # Parsing steps
    # ------------------------------------------------------------------

    def _build_nodes(self, segments_df: gpd.GeoDataFrame) -> List[dict]:
        """
        Derive node records from every unique segment endpoint.

        Parameters
        ----------
        segments_df : geopandas.GeoDataFrame
            Raw segments layer.

        Returns
        -------
        list of dict
            One record per unique node, with ``node_id`` (str) — cast
            the same way as ``start_node`` / ``end_node`` in
            :meth:`_build_links`, so both refer to the same identifiers.
        """
        begin = segments_df["MP_begin"].dropna().astype(int).astype(str)
        end = segments_df["MP_einde"].dropna().astype(int).astype(str)
        unique_node_ids = set(begin) | set(end)
        return [{"node_id": node_id} for node_id in unique_node_ids]

    def _build_links(self, segments_df: gpd.GeoDataFrame) -> List[dict]:
        """
        Derive link records from the segments layer.

        Parameters
        ----------
        segments_df : geopandas.GeoDataFrame
            Raw segments layer.

        Returns
        -------
        list of dict
            Records with ``link_id``, ``start_node``, ``end_node`` and,
            when ``include_geometry`` is set, a ``geometry`` Shapely object.
        """
        links_df = segments_df.rename(columns=self.MAP_COLUMNS_LINK)
        cols = list(self.MAP_COLUMNS_LINK.values())
        links_df[cols] = links_df[cols].astype(int).astype(str)

        if self._include_geometry:
            cols = cols + ["geometry"]

        return links_df[cols].to_dict(orient="records")

    def _map_sensors(
        self,
        sensors_df: gpd.GeoDataFrame,
        parts_df: gpd.GeoDataFrame,
        links_data: List[dict],
    ) -> List[dict]:
        """
        Map sensor locations onto links via the segment-parts layer.

        Parameters
        ----------
        sensors_df : geopandas.GeoDataFrame
            Raw sensor-locations layer.
        parts_df : geopandas.GeoDataFrame
            Raw segment-parts layer (``SD_ID`` -> ``SG_ID``).
        links_data : list of dict
            Already-built link records, used to check that a mapped
            segment actually exists as a link.

        Returns
        -------
        list of dict
            Records with ``location_id`` and ``link_id``. Sensors whose
            part cannot be mapped to a known link are dropped.
        """
        sensors = sensors_df.rename(columns=self.MAP_COLUMNS_SENSOR)

        # Identify rows with missing part_id
        loc_missing_parts = sensors[sensors["part_id"].isna()]["location_id"]
        if len(loc_missing_parts) > 0:
            self._warn(
                f"[Warning] Locposts with id's '{loc_missing_parts}' skipped because it has a NULL part_id (SD_ID)."
            )

        # Filter out missing part_id rows before casting
        sensors = sensors.dropna(subset=["part_id"]).copy()
        sensors = sensors[["location_id", "part_id"]].astype(int).astype(str)

        if self._include_geometry:
            sensors["geometry"] = sensors_df["geometry"]

        parts = parts_df[["SD_ID", "SG_ID"]].astype(int).astype(str)
        part_to_link = dict(zip(parts["SD_ID"], parts["SG_ID"]))

        valid_link_ids = {link["link_id"] for link in links_data}

        sensors["link_id"] = None
        unmapped = 0
        for idx, row in sensors.iterrows():
            parent_link_id = part_to_link.get(row["part_id"])
            if parent_link_id is None:
                self._warn(
                    f"[Warning] Locpost '{row['location_id']}' has SD_id '{row['part_id']}' "
                    f"which is missing from the segmentdelen lookup layer."
                )
                unmapped += 1
                continue
            if parent_link_id not in valid_link_ids:
                self._warn(
                    f"[Warning] Parent segment '{parent_link_id}' for locpost '{row['location_id']}' "
                    f"not found in the segmenten layer."
                )
                unmapped += 1
                continue
            sensors.loc[idx, "link_id"] = parent_link_id

        sensors = sensors.dropna(subset=["link_id"])
        self._warn(
            f"Successfully mapped {len(sensors)} sensors to network links."
            + (f" Failed to map {unmapped} sensors." if unmapped else "")
        )
        cols = ["location_id", "link_id"]
        return sensors[cols + ["geometry"] if self._include_geometry else cols].to_dict(orient="records")

    def _validate_topology(self, segments_df: gpd.GeoDataFrame) -> None:
        """
        Cross-check declared neighbours against actual node connectivity.

        Parameters
        ----------
        segments_df : geopandas.GeoDataFrame
            Raw segments layer. Uses the ``Vorige`` (previous) and
            ``Volgende`` (next) columns when present.

        Notes
        -----
        Purely diagnostic — mismatches are recorded via
        :meth:`~corvia.loaders.base.BaseNetworkLoader._warn` and never
        stop parsing.
        """
        by_id = {str(row["SG_ID"]): row for _, row in segments_df.iterrows()}

        for _, row in segments_df.iterrows():
            sg_id = str(row["SG_ID"])
            start_node = str(row["MP_begin"])
            end_node = str(row["MP_einde"])

            prev_link_ids = [v.strip() for v in str(row.get("Vorige", "")).split(",") if v.strip()]
            next_link_ids = [v.strip() for v in str(row.get("Volgende", "")).split(",") if v.strip()]

            for prev_id in prev_link_ids:
                prev = by_id.get(prev_id)
                if prev is not None and str(prev["MP_einde"]) != start_node:
                    self._warn(
                        f"[Warning] Topology mismatch: upstream segment '{prev_id}' "
                        f"ends at node '{prev['MP_einde']}', but segment '{sg_id}' "
                        f"starts at '{start_node}'."
                    )

            for next_id in next_link_ids:
                nxt = by_id.get(next_id)
                if nxt is not None and str(nxt["MP_begin"]) != end_node:
                    self._warn(
                        f"[Warning] Topology mismatch: downstream segment '{next_id}' "
                        f"starts at node '{nxt['MP_begin']}', but segment '{sg_id}' "
                        f"ends at '{end_node}'."
                    )