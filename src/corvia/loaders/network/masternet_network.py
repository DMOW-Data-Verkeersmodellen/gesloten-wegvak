"""
masternet_network.py
====================
Loader for the "masternet" shapefile bundle.

Source files (all share one path prefix)::

    <prefix>_link.shp            — required, one row per directed link
    <prefix>_countlocation.shp   — required, one row per count location
    <prefix>_node.shp            — optional, one row per node (with geometry)

Dependency chain::

    masternet_network.py  →  base.py
                          →  geopandas / pandas
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Set, Tuple

import geopandas as gpd
import pandas as pd
from shapely.geometry.base import BaseGeometry

from corvia.loaders.network.base import BaseNetworkLoader


class MasterNetLoader(BaseNetworkLoader):
    """
    Load a masternet shapefile bundle into builder-ready records.

    Incoming / outgoing links are *not* read from the source: the builder
    derives them from ``start_node`` / ``end_node`` on each link.

    Filtering and fix-up rules
    --------------------------
    * **IDs** (``NO``, ``FROMNODENO``, ``TONODENO``, ``LINKNO``) are cast
      ``float -> int -> str``. Missing or ``0`` IDs are invalid and the row
      is dropped.
    * **Transport systems**: a link is only loaded when ``TSYSSET`` is
      non-null and contains at least one of :attr:`ALLOWED_TSYS`
      (``PW``, ``VR``, ``BUS``).
    * **Reverse links**: the same link ``NO`` may appear twice when the
      second row runs in the opposite direction (its start node is the
      first row's end node and vice versa). The suffix
      :attr:`REVERSE_SUFFIX` (``"_rev"``) is only added when *both*
      directions are loaded, i.e. survive the ``TSYSSET`` filter: if only
      one direction is loaded it keeps the plain ``NO`` as its id. Any
      other repeated ``NO`` among loaded links is dropped as a duplicate.
      Ids therefore depend on the ``TSYSSET`` filter.
    * **Count locations** are attached through ``LINKNO``. When a link
      ``NO`` exists in both directions, ``FROMNODENO`` / ``TONODENO``
      select the direction. A count location whose link was not loaded
      (e.g. excluded by ``TSYSSET``) is dropped.
    * **Count location selection and id**: only count locations whose
      ``NO`` lies in ``count_no_range`` (inclusive) are loaded; in the
      current masternet these are exactly the rows with ``MODE`` null,
      which makes ``CODE`` unique. ``location_id`` is the ``CODE`` (as a
      string), *not* the ``NO``. Rows without a ``CODE`` (null, blank or
      ``0``) and repeated ``CODE`` values are dropped. Virtual sensors,
      i.e. numeric ``CODE`` values above ``max_code``, are dropped too.
    * **Nodes** (when the node file is supplied) are used as given,
      including geometry. Nodes not referenced by any loaded link are
      dropped. Link endpoints missing from the node file are logged and
      created without geometry by the builder.
    * **CRS**: the link layer's CRS is the main CRS; other layers with a
      different CRS are reprojected to it, and layers without a CRS are
      assumed to already use it.

    Every dropped object is recorded in :attr:`dropped` (and written to
    CSV when ``dropped_report_path`` is given).

    Parameters
    ----------
    source : str
        Path prefix of the bundle, e.g. ``"../data/mynet"`` for
        ``../data/mynet_link.shp``.
    include_geometry : bool, optional
        Whether to carry geometries through as Shapely objects.
        Defaults to ``True``.
    dropped_report_path : str, optional
        If given, a CSV with columns ``object_type``, ``id``, ``row``,
        ``reason``, ``source_no`` is written there at the end of
        :meth:`load`. ``row`` is the 0-based row position in the source
        layer; ``id`` is the id the object would have in the network
        (``CODE`` for count locations); ``source_no`` is the count
        location's ``NO`` (empty for other objects).
    count_no_range : tuple of (int, int) or None, optional
        Inclusive ``(min, max)`` range of count-location ``NO`` values to
        load. Defaults to ``(124685, 128536)``. ``None`` loads every count
        location (then ``CODE`` is no longer guaranteed unique and repeated
        codes are dropped).
    max_code : int or None, optional
        Highest numeric ``CODE`` that is loaded; larger codes are virtual
        sensors and are dropped. Defaults to ``999999``. ``None`` disables
        the check. Non-numeric codes (e.g. ``"31001H"``) are never
        treated as virtual.

    Attributes
    ----------
    _include_geometry : bool
    _dropped_report_path : str or None
    _count_no_range : tuple of (int, int) or None
    _max_code : int or None
    _dropped : list of dict

    Examples
    --------
    >>> loader = MasterNetLoader("../data/mynet", dropped_report_path="dropped.csv")
    >>> links_data, nodes_data, counts_data, crs = loader.load()
    >>> network = NetworkBuilder.from_data(
    ...     "MyNet", links_data, nodes_data, counts_data, crs=crs
    ... )
    """

    #: Separator between the entries of the ``TSYSSET`` field.
    TSYS_DELIMITER: str = ","
    #: Suffix appended to the link id of a reverse-direction duplicate.
    REVERSE_SUFFIX: str = "_rev"

    LINK_SUFFIX = "_link.shp"
    COUNT_SUFFIX = "_countlocation.shp"
    NODE_SUFFIX = "_node.shp"

    def __init__(
        self,
        source: str,
        include_geometry: bool = True,
        dropped_report_path: Optional[str] = None,
        count_no_range: Optional[Tuple[int, int]] = (124685, 128536),
        max_code: Optional[int] = 999_999,
        allowed_tsys: Set[str] = frozenset({"PW", "VR", "BUS"}),
    ) -> None:
        super().__init__(source)
        self._include_geometry: bool = include_geometry
        self._dropped_report_path: Optional[str] = dropped_report_path
        self._count_no_range: Optional[Tuple[int, int]] = count_no_range
        self._max_code: Optional[int] = max_code
        self._dropped: List[dict] = []
        self.ALLOWED_TSYS = allowed_tsys

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dropped(self) -> pd.DataFrame:
        """pandas.DataFrame : Objects dropped by the last :meth:`load` call."""
        return pd.DataFrame(self._dropped, columns=["object_type", "id", "row", "reason", "source_no"])

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def load(self) -> Tuple[List[dict], List[dict], List[dict], str]:
        """
        Read the shapefiles and return builder-ready records.

        Returns
        -------
        links_data : list of dict
        nodes_data : list of dict
        counts_data : list of dict
        crs : str

        Raises
        ------
        FileNotFoundError
            If the link or count-location shapefile is missing.
        KeyError
            If a required column is missing.
        ValueError
            If the link layer has no CRS.

        Notes
        -----
        Dropped objects are logged (one warning per object type and
        reason, with counts) and kept in :attr:`dropped`.
        """
        self._dropped = []
        link_path = f"{self._source}{self.LINK_SUFFIX}"
        count_path = f"{self._source}{self.COUNT_SUFFIX}"
        node_path = f"{self._source}{self.NODE_SUFFIX}"

        for path in (link_path, count_path):
            if not os.path.isfile(path):
                error_msg = f"Required masternet file not found: '{path}'."
                self.logger.error(error_msg)
                raise FileNotFoundError(error_msg)
        has_nodes = os.path.isfile(node_path)
        self.logger.info(
            f"Loading masternet '{self._source}' "
            f"({'with' if has_nodes else 'without'} node file)."
        )

        links_df = gpd.read_file(link_path)
        self._require_columns(links_df, ["NO", "FROMNODENO", "TONODENO", "TSYSSET"], link_path)
        main_crs = links_df.crs
        if main_crs is None:
            error_msg = f"Link layer '{link_path}' has no CRS; cannot determine the main CRS."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        counts_df = self._read_layer(count_path, main_crs)
        self._require_columns(counts_df, ["NO", "CODE", "FROMNODENO", "TONODENO", "LINKNO"], count_path)
        self._check_count_range(counts_df)
        nodes_df = None
        if has_nodes:
            nodes_df = self._read_layer(node_path, main_crs)
            self._require_columns(nodes_df, ["NO"], node_path)

        links_data, links_by_no = self._build_links(links_df)
        if nodes_df is not None:
            nodes_data = self._build_nodes_from_file(nodes_df, links_data)
        else:
            nodes_data = self._derive_nodes(links_data)
        counts_data = self._build_counts(counts_df, links_by_no)

        self._report_dropped()
        self.logger.info(
            f"load() complete: {len(nodes_data)} node(s), {len(links_data)} link(s), "
            f"{len(counts_data)} count location(s)."
        )
        return links_data, nodes_data, counts_data, main_crs.to_string()

    # ------------------------------------------------------------------
    # Parsing steps
    # ------------------------------------------------------------------

    def _build_links(
        self, links_df: gpd.GeoDataFrame
    ) -> Tuple[List[dict], Dict[str, List[Tuple[str, str, Optional[str]]]]]:
        """
        Derive link records, applying ID, reverse-link and TSYSSET rules.

        Parameters
        ----------
        links_df : geopandas.GeoDataFrame
            Raw link layer.

        Returns
        -------
        links_data : list of dict
            Records with ``link_id``, ``start_node``, ``end_node`` and,
            when ``include_geometry`` is set, ``geometry``.
        links_by_no : dict
            Maps each raw link ``NO`` to a list of
            ``(start_node, end_node, link_id_or_None)`` tuples for every
            ID-valid row; ``None`` marks rows dropped by ``TSYSSET``.
            Used to attach count locations.
        """
        links_data: List[dict] = []
        links_by_no: Dict[str, List[Tuple[str, str, Optional[str]]]] = {}
        first_seen: Dict[str, Tuple[str, str]] = {}
        used_ids: Set[str] = set()

        rows = zip(links_df["NO"], links_df["FROMNODENO"], links_df["TONODENO"],
                   links_df["TSYSSET"], links_df.geometry)
        for pos, (no, frm, to, tsys, geom) in enumerate(rows):
            raw_id, start, end = self._to_id(no), self._to_id(frm), self._to_id(to)
            if raw_id is None or start is None or end is None:
                self._drop("link", raw_id, pos, "missing or zero NO / FROMNODENO / TONODENO")
                continue

            # 1. TSYSSET filter (before id assignment, so "_rev" only appears
            #    when both directions are actually loaded)
            if not self._allowed_modes(tsys):
                reason = ("TSYSSET is NULL: no transport system allowed"
                          if self._is_blank(tsys)
                          else f"TSYSSET '{tsys}' allows none of {sorted(self.ALLOWED_TSYS)}")
                self._drop("link", raw_id, pos, reason)
                links_by_no.setdefault(raw_id, []).append((start, end, None))
                continue

            # 2. Id assignment among loaded links only
            link_id = raw_id
            if raw_id in first_seen:
                first_start, first_end = first_seen[raw_id]
                if (start, end) == (first_end, first_start):
                    link_id = f"{raw_id}{self.REVERSE_SUFFIX}"
                else:
                    self._drop("link", raw_id, pos,
                               "duplicate NO that is not the reverse of the first loaded occurrence")
                    continue
            else:
                first_seen[raw_id] = (start, end)
            if link_id in used_ids:
                self._drop("link", link_id, pos, "duplicate link id (reverse link already present)")
                continue
            used_ids.add(link_id)
            links_by_no.setdefault(raw_id, []).append((start, end, link_id))

            record = {"link_id": link_id, "start_node": start, "end_node": end}
            if self._include_geometry:
                record["geometry"] = self._clean_geometry(geom)
            links_data.append(record)

        self.logger.debug(f"Derived {len(links_data)} link(s) from link layer.")
        return links_data, links_by_no

    def _derive_nodes(self, links_data: List[dict]) -> List[dict]:
        """
        Derive node records from the endpoints of the loaded links.

        Parameters
        ----------
        links_data : list of dict
            Loaded link records.

        Returns
        -------
        list of dict
            One ``{"node_id": ...}`` record per unique endpoint, sorted.
        """
        node_ids = {l["start_node"] for l in links_data} | {l["end_node"] for l in links_data}
        self.logger.debug(f"Derived {len(node_ids)} node(s) from link endpoints (no node file).")
        return [{"node_id": nid} for nid in sorted(node_ids)]

    def _build_nodes_from_file(
        self, nodes_df: gpd.GeoDataFrame, links_data: List[dict]
    ) -> List[dict]:
        """
        Build node records (with geometry) from the node layer.

        Parameters
        ----------
        nodes_df : geopandas.GeoDataFrame
            Raw node layer, already in the main CRS.
        links_data : list of dict
            Loaded link records, used to drop unreferenced nodes and to
            report link endpoints absent from the node layer.

        Returns
        -------
        list of dict
            Records with ``node_id`` and, when ``include_geometry`` is
            set, ``geometry``.
        """
        used = {l["start_node"] for l in links_data} | {l["end_node"] for l in links_data}
        seen: Set[str] = set()
        nodes_data: List[dict] = []

        for pos, (no, geom) in enumerate(zip(nodes_df["NO"], nodes_df.geometry)):
            node_id = self._to_id(no)
            if node_id is None:
                self._drop("node", None, pos, "missing or zero NO")
                continue
            if node_id in seen:
                self._drop("node", node_id, pos, "duplicate node NO")
                continue
            seen.add(node_id)
            if node_id not in used:
                self._drop("node", node_id, pos, "not referenced by any loaded link")
                continue
            record = {"node_id": node_id}
            if self._include_geometry:
                record["geometry"] = self._clean_geometry(geom)
            nodes_data.append(record)

        missing = used - seen
        if missing:
            self.logger.warning(
                f"[Warning] {len(missing)} link endpoint(s) are missing from the node file "
                f"and will have no geometry (e.g. {sorted(missing)[:10]})."
            )
        return nodes_data

    def _build_counts(
        self,
        counts_df: gpd.GeoDataFrame,
        links_by_no: Dict[str, List[Tuple[str, str, Optional[str]]]],
    ) -> List[dict]:
        """
        Select count locations and map them onto loaded links.

        Parameters
        ----------
        counts_df : geopandas.GeoDataFrame
            Raw count-location layer, already in the main CRS.
        links_by_no : dict
            Link lookup returned by :meth:`_build_links`.

        Returns
        -------
        list of dict
            Records with ``location_id``, ``link_id`` and, when
            ``include_geometry`` is set, ``geometry``.
        """
        counts_data: List[dict] = []
        seen_codes: Set[str] = set()
        mismatched: List[str] = []

        rows = zip(counts_df["NO"], counts_df["CODE"], counts_df["LINKNO"],
                   counts_df["FROMNODENO"], counts_df["TONODENO"], counts_df.geometry)
        for pos, (no, code, linkno, frm, to, geom) in enumerate(rows):
            src_no = self._to_id(no)
            loc_id = self._code_to_id(code)
            raw_link, start, end = self._to_id(linkno), self._to_id(frm), self._to_id(to)

            def drop(reason: str) -> None:
                self._drop("countlocation", loc_id, pos, reason, source_no=src_no)

            if src_no is None:
                drop("missing or zero NO")
                continue
            if self._count_no_range is not None:
                lo, hi = self._count_no_range
                if not lo <= int(src_no) <= hi:
                    drop(f"NO outside count_no_range [{lo}, {hi}]")
                    continue
            if loc_id is None:
                drop("missing CODE (null, blank or 0)")
                continue
            if self._max_code is not None and loc_id.isdigit() and int(loc_id) > self._max_code:
                drop(f"virtual sensor: CODE > {self._max_code}")
                continue
            if raw_link is None:
                drop("missing or zero LINKNO")
                continue
            if loc_id in seen_codes:
                drop("duplicate CODE")
                continue
            seen_codes.add(loc_id)

            candidates = list(dict.fromkeys(links_by_no.get(raw_link, [])))
            if not candidates:
                drop(f"LINKNO {raw_link} not found in link layer (or link row invalid)")
                continue

            exact = [c for c in candidates if (c[0], c[1]) == (start, end)]
            if len(exact) == 1:
                chosen = exact[0]
            elif len(candidates) == 1:
                chosen = candidates[0]
                if start is not None and end is not None:
                    mismatched.append(loc_id)
            else:
                drop(f"LINKNO {raw_link} exists in both directions and FROMNODENO/TONODENO "
                     f"match neither")
                continue

            if chosen[2] is None:
                drop(f"link {raw_link} was not loaded (TSYSSET excludes PW/VR/BUS)")
                continue

            record = {"location_id": loc_id, "link_id": chosen[2]}
            if self._include_geometry:
                record["geometry"] = self._clean_geometry(geom)
            counts_data.append(record)

        if mismatched:
            self.logger.warning(
                f"[Warning] {len(mismatched)} count location(s) have FROMNODENO/TONODENO that "
                f"disagree with their link's nodes; attached via LINKNO anyway "
                f"(e.g. {mismatched[:10]})."
            )
        return counts_data

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_layer(self, path: str, main_crs) -> gpd.GeoDataFrame:
        """Read a shapefile and bring it into ``main_crs``."""
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            self.logger.warning(f"[Warning] '{path}' has no CRS; assuming it is in {main_crs}.")
            gdf = gdf.set_crs(main_crs)
        elif gdf.crs != main_crs:
            self.logger.warning(f"[Warning] '{path}' is in {gdf.crs}; reprojecting to {main_crs}.")
            gdf = gdf.to_crs(main_crs)
        return gdf

    def _require_columns(self, df: pd.DataFrame, columns: List[str], path: str) -> None:
        """Raise a clear ``KeyError`` when required columns are absent."""
        missing = [c for c in columns if c not in df.columns]
        if missing:
            error_msg = f"'{path}' is missing required column(s) {missing}; found {list(df.columns)}."
            self.logger.error(error_msg)
            raise KeyError(error_msg)

    def _allowed_modes(self, tsysset) -> Set[str]:
        """Return the allowed transport systems contained in a ``TSYSSET`` value."""
        if self._is_blank(tsysset):
            return set()
        modes = {m.strip().upper() for m in str(tsysset).split(self.TSYS_DELIMITER)}
        return modes & self.ALLOWED_TSYS

    @classmethod
    def _has_code(cls, value) -> bool:
        """``True`` unless a ``CODE`` value is null, blank or ``0``."""
        if cls._is_blank(value):
            return False
        try:
            return float(value) != 0
        except (TypeError, ValueError):
            return True  # non-numeric text code

    @classmethod
    def _code_to_id(cls, value) -> Optional[str]:
        """Normalise a ``CODE`` to a string id; ``None`` when null, blank or 0."""
        if not cls._has_code(value):
            return None
        text = str(value).strip()
        if re.fullmatch(r"\d+\.0+", text):  # "123.0" -> "123"
            text = text.split(".")[0]
        return text

    def _check_count_range(self, counts_df: pd.DataFrame) -> None:
        """Warn when ``count_no_range`` and a blank ``MODE`` no longer coincide."""
        if self._count_no_range is None or "MODE" not in counts_df.columns:
            return
        lo, hi = self._count_no_range
        no = pd.to_numeric(counts_df["NO"], errors="coerce")
        in_range = no.between(lo, hi)
        blank_mode = counts_df["MODE"].isna() | (counts_df["MODE"].astype(str).str.strip() == "")
        in_range_with_mode = int((in_range & ~blank_mode).sum())
        out_range_blank = int((~in_range & blank_mode).sum())
        if in_range_with_mode or out_range_blank:
            self.logger.warning(
                f"[Warning] count_no_range {self._count_no_range} no longer matches MODE-null rows: "
                f"{in_range_with_mode} in-range row(s) have a MODE, "
                f"{out_range_blank} out-of-range row(s) have a null MODE."
            )

    @staticmethod
    def _is_blank(value) -> bool:
        return value is None or bool(pd.isna(value)) or not str(value).strip()

    @staticmethod
    def _to_id(value) -> Optional[str]:
        """Cast a numeric ID to ``str``; ``None`` for missing, non-numeric or zero."""
        if value is None or pd.isna(value):
            return None
        try:
            int_value = int(value)
        except (TypeError, ValueError):
            return None
        return str(int_value) if int_value != 0 else None

    @staticmethod
    def _clean_geometry(geom) -> Optional[BaseGeometry]:
        """Return *geom* unless it is null or empty."""
        return geom if isinstance(geom, BaseGeometry) and not geom.is_empty else None

    def _drop(
        self,
        object_type: str,
        object_id: Optional[str],
        row: int,
        reason: str,
        source_no: Optional[str] = None,
    ) -> None:
        """Record a dropped object."""
        self._dropped.append(
            {"object_type": object_type, "id": object_id, "row": row,
             "reason": reason, "source_no": source_no}
        )

    def _report_dropped(self) -> None:
        """Log a summary of dropped objects and optionally write the CSV report."""
        df = self.dropped
        if df.empty:
            self.logger.info("No objects were dropped.")
            return
        summary = df.groupby(["object_type", "reason"]).size()
        for (object_type, reason), count in summary.items():
            self.logger.warning(f"[Warning] Dropped {count} {object_type}(s): {reason}.")
        if self._dropped_report_path:
            df.to_csv(self._dropped_report_path, index=False)
            self.logger.info(
                f"Wrote {len(df)} dropped object(s) to '{self._dropped_report_path}'."
            )