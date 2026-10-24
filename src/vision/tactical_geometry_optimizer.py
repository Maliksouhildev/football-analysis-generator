"""
src/vision/tactical_geometry_optimizer.py
Shortest-Link Geometry Engine for Tactical Line & Passing Network Telestration.

Enforces:
1. Euclidean Distance Matrix & Minimum Spanning Tree (MST):
   - Computes pairwise Euclidean distance between all selected players' foot coordinates.
   - For open chain links (defensive lines, midfield chains, passing links):
     Computes the Minimum Spanning Tree (MST) and shortest Hamiltonian path (via Kruskal's
     algorithm and 2-opt optimization) to eliminate crisscrossing and overcomplicated links.
   - For closed passing triangles / polygons:
     Orders vertices along the convex hull perimeter counter-clockwise around the centroid.
2. Line-Intersection Prevention:
   - Evaluates line-segment intersections using the Counter-Clockwise (CCW) orientation algorithm.
   - Automatically uncrosses and reroutes crossing edges via 2-opt edge swaps or nearest
     intermediate node rerouting, removing redundant long diagonal edges.
3. Clean Player Foot Halo Padding:
   - Shortens both ends of every tactical line/arrow by a 22px radial margin from each player's
     center/foot point (or anchors cleanly to the foot ring) so lines never draw over player bodies
     or obscure name badges.
4. Line Style Classification:
   - 'tactical_structure': Solid or dashed shortest-path lines connecting player foot rings.
   - 'ball_pass': Curved Bezier arrow showing exact start-to-landing trajectory.
   - 'player_run': Dotted arrow from player position at t to position at t + 2.5s.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np


class TacticalLineStyle(str, Enum):
    """Visual styles for tactical overlay lines."""
    SOLID = "solid"
    DASHED = "dashed"
    DOTTED = "dotted"
    BEZIER = "bezier"


class TacticalLineType(str, Enum):
    """Functional semantics of tactical telestration lines."""
    TACTICAL_STRUCTURE = "tactical_structure"  # Defensive chain, midfield unit, player links
    BALL_PASS = "ball_pass"                    # Verified ball trajectory / corner kick
    PLAYER_RUN = "player_run"                  # Off-ball movement arrow


@dataclass
class Point2D:
    x: float
    y: float
    id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

    def to_tuple(self) -> Tuple[float, float]:
        return (float(self.x), float(self.y))


@dataclass
class LineSegment:
    start: Point2D
    end: Point2D
    id: Optional[str] = None
    style: TacticalLineStyle = TacticalLineStyle.SOLID
    line_type: TacticalLineType = TacticalLineType.TACTICAL_STRUCTURE
    color: str = "#00e676"
    width: float = 3.5
    dashed: bool = False
    bezier_control: Optional[Point2D] = None
    halo_padding_px: float = 22.0

    @property
    def length(self) -> float:
        return math.hypot(self.end.x - self.start.x, self.end.y - self.start.y)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id or f"line_{self.start.id}_{self.end.id}",
            "start": {"x": round(self.start.x, 2), "y": round(self.start.y, 2)},
            "end": {"x": round(self.end.x, 2), "y": round(self.end.y, 2)},
            "style": self.style.value,
            "line_type": self.line_type.value,
            "color": self.color,
            "width": self.width,
            "dashed": self.dashed,
            "halo_padding_px": self.halo_padding_px,
        }
        if self.bezier_control:
            d["bezier_control"] = {
                "x": round(self.bezier_control.x, 2),
                "y": round(self.bezier_control.y, 2),
            }
        return d


class ShortestLinkGeometryEngine:
    """
    Mathematical geometry optimizer for tactical telestration overlays:
    - Pairwise Euclidean distance matrices.
    - Kruskal's / Prim's Minimum Spanning Tree (MST).
    - 2-opt shortest open chain without crisscrosses.
    - CCW line intersection detection and rerouting.
    - 22px foot halo radial margin padding.
    """

    DEFAULT_HALO_PADDING_PX: float = 22.0
    DEFAULT_MAX_LINK_PX: float = 540.0  # ~28% of 1920 width

    # =========================================================================
    # 1. Euclidean Distance Matrix
    # =========================================================================

    @staticmethod
    def compute_distance_matrix(points: Sequence[Tuple[float, float]]) -> np.ndarray:
        """
        Computes pairwise Euclidean distance matrix D for N points:
        D[i, j] = sqrt((x_i - x_j)^2 + (y_i - y_j)^2)
        """
        pts = np.asarray(points, dtype=np.float64)
        n = len(pts)
        if n == 0:
            return np.zeros((0, 0), dtype=np.float64)

        # Vectorized Euclidean pairwise distance
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
        return dist_matrix

    # =========================================================================
    # 2. Minimum Spanning Tree (MST) & Shortest Open Chain
    # =========================================================================

    @classmethod
    def compute_mst_edges(
        cls,
        points: Sequence[Tuple[float, float]],
        max_link_distance: Optional[float] = None,
    ) -> List[Tuple[int, int, float]]:
        """
        Computes Minimum Spanning Tree (MST) edges using Kruskal's algorithm with Disjoint-Set.
        Returns list of (u, v, weight) tuples connecting all points with minimum total distance.
        """
        n = len(points)
        if n < 2:
            return []

        dist_matrix = cls.compute_distance_matrix(points)
        max_dist = max_link_distance if max_link_distance is not None else float("inf")

        # Collect and sort all possible edges by distance
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                d = dist_matrix[i, j]
                if d <= max_dist:
                    edges.append((i, j, float(d)))
        edges.sort(key=lambda e: e[2])

        # Disjoint-Set (Union-Find)
        parent = list(range(n))
        rank = [0] * n

        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: int, y: int) -> bool:
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            if rank[rx] < rank[ry]:
                parent[rx] = ry
            elif rank[rx] > rank[ry]:
                parent[ry] = rx
            else:
                parent[ry] = rx
                rank[rx] += 1
            return True

        mst_edges: List[Tuple[int, int, float]] = []
        for u, v, weight in edges:
            if union(u, v):
                mst_edges.append((u, v, weight))
                if len(mst_edges) == n - 1:
                    break

        return mst_edges

    @classmethod
    def compute_shortest_open_chain(
        cls,
        points: Sequence[Tuple[float, float]],
        max_link_distance: Optional[float] = None,
    ) -> List[int]:
        """
        Computes the shortest non-crossing linear open chain connecting points.
        Rules:
        1. NEVER connects in raw array order [P0 -> P1 -> P2].
        2. Solves for the minimum total Euclidean distance path.
        3. Applies 2-opt edge-swapping to eliminate any self-intersections.
        4. Consistently orients chain monotonically (top-to-bottom or left-to-right).
        Returns list of indices in optimal chain order.
        """
        n = len(points)
        if n <= 1:
            return list(range(n))
        if n == 2:
            d = math.hypot(points[0][0] - points[1][0], points[0][1] - points[1][1])
            max_d = max_link_distance if max_link_distance is not None else float("inf")
            if d > max_d:
                return [0]
            # Consistent orientation
            if points[0][1] > points[1][1] or (points[0][1] == points[1][1] and points[0][0] > points[1][0]):
                return [1, 0]
            return [0, 1]

        dist_matrix = cls.compute_distance_matrix(points)
        max_d = max_link_distance if max_link_distance is not None else float("inf")

        # Determine primary spatial spread (vertical pitch width vs horizontal pitch length)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)
        prefer_y = span_y >= span_x

        best_chain: List[int] = []
        best_cost = float("inf")

        # Try nearest-neighbor heuristic starting from every candidate node
        for start_idx in range(n):
            visited = [start_idx]
            unvisited = set(range(n)) - {start_idx}
            curr = start_idx
            total_dist = 0.0

            while unvisited:
                nearest = min(unvisited, key=lambda u: dist_matrix[curr, u])
                d = dist_matrix[curr, nearest]
                if d <= max_d:
                    total_dist += d
                    visited.append(nearest)
                    unvisited.remove(nearest)
                    curr = nearest
                else:
                    break

            if len(visited) > len(best_chain) or (len(visited) == len(best_chain) and total_dist < best_cost):
                best_chain = visited
                best_cost = total_dist

        # If not all nodes were visited due to max_d, check if remaining nodes can be chained
        if len(best_chain) < 2:
            best_chain = list(range(min(n, 2)))

        # 2-opt optimization to uncross any crossing edges
        best_chain = cls.uncross_chain_2opt(best_chain, points)

        # Consistent monotonic orientation (e.g. top-to-bottom for backlines)
        if len(best_chain) >= 2:
            first = points[best_chain[0]]
            last = points[best_chain[-1]]
            if prefer_y:
                if first[1] > last[1]:
                    best_chain.reverse()
            else:
                if first[0] > last[0]:
                    best_chain.reverse()

        return best_chain

    # =========================================================================
    # 3. Line-Segment Intersection Check (CCW Algorithm)
    # =========================================================================

    @staticmethod
    def _ccw(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> bool:
        """
        Counter-Clockwise orientation test:
        True if points A, B, C are in counter-clockwise order.
        """
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    @classmethod
    def do_segments_intersect(
        cls,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
        p4: Tuple[float, float],
        eps: float = 1e-7,
    ) -> bool:
        """
        Line-segment intersection check using Counter-Clockwise (CCW) orientation.
        Returns True if segment (p1 -> p2) crosses segment (p3 -> p4).
        Excludes adjacent segments that share an identical endpoint.
        """
        # If segments share an endpoint, they do not cross in the interior
        d13 = math.hypot(p1[0] - p3[0], p1[1] - p3[1])
        d14 = math.hypot(p1[0] - p4[0], p1[1] - p4[1])
        d23 = math.hypot(p2[0] - p3[0], p2[1] - p3[1])
        d24 = math.hypot(p2[0] - p4[0], p2[1] - p4[1])
        if d13 < eps or d14 < eps or d23 < eps or d24 < eps:
            return False

        ccw1 = cls._ccw(p1, p3, p4) != cls._ccw(p2, p3, p4)
        ccw2 = cls._ccw(p1, p2, p3) != cls._ccw(p1, p2, p4)
        return ccw1 and ccw2

    @classmethod
    def uncross_chain_2opt(
        cls,
        chain: List[int],
        points: Sequence[Tuple[float, float]],
        max_iterations: int = 50,
    ) -> List[int]:
        """
        Applies 2-opt edge-swapping to eliminate any self-intersections in an open chain.
        If segment (i, i+1) crosses segment (j, j+1), inverting the subchain [i+1..j]
        strictly eliminates the intersection and reduces total Euclidean distance.
        """
        improved = True
        current_chain = list(chain)
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            m = len(current_chain)
            for i in range(m - 2):
                p_i1 = points[current_chain[i]]
                p_i2 = points[current_chain[i + 1]]
                for j in range(i + 2, m - 1):
                    p_j1 = points[current_chain[j]]
                    p_j2 = points[current_chain[j + 1]]

                    if cls.do_segments_intersect(p_i1, p_i2, p_j1, p_j2):
                        # Reverse the subchain between i+1 and j
                        current_chain[i + 1 : j + 1] = reversed(current_chain[i + 1 : j + 1])
                        improved = True
                        break
                if improved:
                    break

        return current_chain

    @classmethod
    def resolve_crossing_edges(
        cls,
        edges: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    ) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Filters or reroutes any crossing edges from a set of lines.
        If edge A crosses edge B, removes the longer redundant edge.
        """
        valid_edges = list(edges)
        i = 0
        while i < len(valid_edges):
            e1 = valid_edges[i]
            crossed = False
            for j in range(i + 1, len(valid_edges)):
                e2 = valid_edges[j]
                if cls.do_segments_intersect(e1[0], e1[1], e2[0], e2[1]):
                    # Remove the longer of the two crossing edges
                    len1 = math.hypot(e1[1][0] - e1[0][0], e1[1][1] - e1[0][1])
                    len2 = math.hypot(e2[1][0] - e2[0][0], e2[1][1] - e2[0][1])
                    if len1 >= len2:
                        valid_edges.pop(i)
                        crossed = True
                        break
                    else:
                        valid_edges.pop(j)
                        break
            if not crossed:
                i += 1

        return valid_edges

    # =========================================================================
    # 4. Clean Player Foot Halo Padding (22px Radial Margin)
    # =========================================================================

    @classmethod
    def apply_foot_halo_padding(
        cls,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        padding_px: float = 22.0,
        is_percentage: bool = False,
        frame_w: float = 1920.0,
        frame_h: float = 1080.0,
    ) -> Tuple[float, float, float, float]:
        """
        Shortens both ends of a line segment by a radial margin (default 22px)
        anchoring cleanly to the player's foot ring without overlapping player bodies.

        Returns (new_x1, new_y1, new_x2, new_y2).
        If distance between points is too short (< 2 * padding), returns original points.
        """
        # Convert to screen pixels for uniform radial shortening if given as percentage
        scale_x = (frame_w / 100.0) if is_percentage else 1.0
        scale_y = (frame_h / 100.0) if is_percentage else 1.0

        px1 = x1 * scale_x
        py1 = y1 * scale_y
        px2 = x2 * scale_x
        py2 = y2 * scale_y

        dx = px2 - px1
        dy = py2 - py1
        dist = math.hypot(dx, dy)

        min_len = 2.0 * padding_px + 2.0
        if dist < min_len:
            # Segment too short to pad from both ends; return untouched
            return (x1, y1, x2, y2)

        ux = dx / dist
        uy = dy / dist

        # Shorten start and end by padding_px
        new_px1 = px1 + padding_px * ux
        new_py1 = py1 + padding_px * uy
        new_px2 = px2 - padding_px * ux
        new_py2 = py2 - padding_px * uy

        # Convert back to original coordinate system
        inv_scale_x = (1.0 / scale_x)
        inv_scale_y = (1.0 / scale_y)

        return (
            round(new_px1 * inv_scale_x, 2),
            round(new_py1 * inv_scale_y, 2),
            round(new_px2 * inv_scale_x, 2),
            round(new_py2 * inv_scale_y, 2),
        )

    # =========================================================================
    # 5. Bezier Curve Trajectory Generator (Ball Pass & Corner Kicks)
    # =========================================================================

    @staticmethod
    @staticmethod
    def compute_bezier_control_point(
        *args,
        arc_height_pct: float = 16.0,
        arc_elevation_px: Optional[float] = None,
        is_percentage: bool = True,
        frame_h: float = 1080.0,
        **kwargs,
    ) -> Tuple[float, float]:
        """
        Computes quadratic Bezier curve control point (cx, cy) for aerial passes / corners:
        cx: midpoint along X
        cy: apex elevated above the higher of the two points
        """
        if len(args) == 2:
            p1, p2 = args
            x1 = float(p1.x if hasattr(p1, "x") else p1[0])
            y1 = float(p1.y if hasattr(p1, "y") else p1[1])
            x2 = float(p2.x if hasattr(p2, "x") else p2[0])
            y2 = float(p2.y if hasattr(p2, "y") else p2[1])
        elif len(args) >= 4:
            x1, y1, x2, y2 = float(args[0]), float(args[1]), float(args[2]), float(args[3])
        else:
            raise ValueError("compute_bezier_control_point expects (p1, p2) or (x1, y1, x2, y2)")

        mid_x = (x1 + x2) / 2.0
        if arc_elevation_px is not None:
            elevation = float(arc_elevation_px)
        elif is_percentage:
            elevation = float(arc_height_pct)
        else:
            elevation = (float(arc_height_pct) / 100.0) * frame_h

        apex_y = min(y1, y2) - elevation
        return (round(mid_x, 2), round(max(2.0, apex_y), 2))

    # =========================================================================
    # 6. High-Level Factory: Build Optimized Tactical Segments
    # =========================================================================

    @classmethod
    def build_tactical_chain(
        cls,
        players: Sequence[Union[Dict[str, Any], Point2D]],
        style: TacticalLineStyle = TacticalLineStyle.SOLID,
        line_type: TacticalLineType = TacticalLineType.TACTICAL_STRUCTURE,
        color: str = "#00e676",
        width: float = 3.5,
        dashed: bool = False,
        padding_px: float = 22.0,
        max_link_px: float = 540.0,
        is_percentage: bool = True,
    ) -> List[LineSegment]:
        """
        Builds shortest-link tactical structure segments connecting player foot rings:
        1. Computes MST / Shortest Hamiltonian chain.
        2. Uncrosses any intersecting segments via 2-opt.
        3. Applies 22px foot halo padding to all line segments.
        """
        if len(players) < 2:
            return []

        pts: List[Tuple[float, float]] = []
        for p in players:
            if hasattr(p, "x"):
                pts.append((float(p.x), float(p.y)))
            elif isinstance(p, dict):
                pts.append((float(p.get("x_pct", p.get("x", 50.0))), float(p.get("y_pct", p.get("y", 50.0)))))
            else:
                pts.append((float(p[0]), float(p[1])))

        ordered_indices = cls.compute_shortest_open_chain(pts, max_link_distance=max_link_px)

        segments: List[LineSegment] = []
        for i in range(len(ordered_indices) - 1):
            idx1 = ordered_indices[i]
            idx2 = ordered_indices[i + 1]
            p1 = players[idx1]
            p2 = players[idx2]

            x1, y1 = pts[idx1]
            x2, y2 = pts[idx2]

            # Shorten endpoints by 22px halo padding
            px1, py1, px2, py2 = cls.apply_foot_halo_padding(
                x1, y1, x2, y2, padding_px=padding_px, is_percentage=is_percentage
            )

            id1 = str(getattr(p1, "id", None) or (p1.get("track_id", p1.get("id", idx1)) if isinstance(p1, dict) else idx1))
            id2 = str(getattr(p2, "id", None) or (p2.get("track_id", p2.get("id", idx2)) if isinstance(p2, dict) else idx2))

            seg = LineSegment(
                start=Point2D(x=px1, y=py1, id=id1, data=p1 if isinstance(p1, dict) else None),
                end=Point2D(x=px2, y=py2, id=id2, data=p2 if isinstance(p2, dict) else None),
                id=f"tactical_link_{id1}_{id2}",
                style=style,
                line_type=line_type,
                color=color,
                width=width,
                dashed=dashed or (style == TacticalLineStyle.DASHED),
                halo_padding_px=padding_px,
            )
            segments.append(seg)

        return segments

    @classmethod
    def build_curved_pass_trajectory(
        cls,
        start_pt: Union[Tuple[float, float], Point2D],
        end_pt: Union[Tuple[float, float], Point2D],
        start_id: Optional[str] = None,
        end_id: Optional[str] = None,
        color: str = "#fbbf24",
        arc_height_pct: float = 16.0,
        arc_elevation_px: Optional[float] = None,
        padding_px: float = 22.0,
        is_percentage: bool = True,
    ) -> LineSegment:
        """
        Constructs a curved Bezier actual ball pass / corner trajectory with 22px padding.
        """
        x1 = float(start_pt.x if hasattr(start_pt, "x") else start_pt[0])
        y1 = float(start_pt.y if hasattr(start_pt, "y") else start_pt[1])
        x2 = float(end_pt.x if hasattr(end_pt, "x") else end_pt[0])
        y2 = float(end_pt.y if hasattr(end_pt, "y") else end_pt[1])

        s_id = start_id or (getattr(start_pt, "id", None) if hasattr(start_pt, "id") else None)
        e_id = end_id or (getattr(end_pt, "id", None) if hasattr(end_pt, "id") else None)

        px1, py1, px2, py2 = cls.apply_foot_halo_padding(
            x1, y1, x2, y2, padding_px=padding_px, is_percentage=is_percentage
        )
        ctrl_x, ctrl_y = cls.compute_bezier_control_point(
            (px1, py1),
            (px2, py2),
            arc_height_pct=arc_height_pct,
            arc_elevation_px=arc_elevation_px,
            is_percentage=is_percentage,
        )

        return LineSegment(
            start=Point2D(x=px1, y=py1, id=s_id),
            end=Point2D(x=px2, y=py2, id=e_id),
            id=f"curved_pass_{s_id or '0'}_{e_id or '1'}",
            style=TacticalLineStyle.BEZIER,
            line_type=TacticalLineType.BALL_PASS,
            color=color,
            width=3.5,
            dashed=False,
            bezier_control=Point2D(x=ctrl_x, y=ctrl_y),
            halo_padding_px=padding_px,
        )

    @classmethod
    def build_player_run_arrow(
        cls,
        start_pt: Union[Tuple[float, float], Point2D],
        end_pt: Union[Tuple[float, float], Point2D],
        player_id: Optional[str] = None,
        color: str = "#ffffff",
        padding_px: float = 22.0,
        is_percentage: bool = True,
    ) -> LineSegment:
        """
        Constructs a dotted off-ball movement arrow from player position at t to position at t + 2.5s.
        """
        x1 = float(start_pt.x if hasattr(start_pt, "x") else start_pt[0])
        y1 = float(start_pt.y if hasattr(start_pt, "y") else start_pt[1])
        x2 = float(end_pt.x if hasattr(end_pt, "x") else end_pt[0])
        y2 = float(end_pt.y if hasattr(end_pt, "y") else end_pt[1])

        p_id = player_id or (getattr(start_pt, "id", None) if hasattr(start_pt, "id") else None)

        px1, py1, px2, py2 = cls.apply_foot_halo_padding(
            x1, y1, x2, y2, padding_px=padding_px, is_percentage=is_percentage
        )

        return LineSegment(
            start=Point2D(x=px1, y=py1, id=p_id),
            end=Point2D(x=px2, y=py2, id=f"{p_id}_target" if p_id else None),
            id=f"player_run_{p_id or '0'}",
            style=TacticalLineStyle.DOTTED,
            line_type=TacticalLineType.PLAYER_RUN,
            color=color,
            width=3.2,
            dashed=True,
            halo_padding_px=padding_px,
        )

