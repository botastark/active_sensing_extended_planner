import math
import numpy as np
from typing import Dict, Tuple, List, Optional

# Directions as unit vectors
# self.actions = {"up", "down", "front", "back", "left", "right", "hover"}
# Front - north, Back - south
# right - east, left - west


DIRECTIONS = {
    "front": (0, 1, 0),
    "back": (0, -1, 0),
    "right": (1, 0, 0),
    "left": (-1, 0, 0),
    "up": (0, 0, 1),
    "down": (0, 0, -1),
}


# def coord2idp(
#     coord: Tuple[float, float, float],
#     steps: Tuple[float, float, float],
#     origin: Tuple[float, float, float],
# ) -> Tuple[int, int, int]:
#     """Convert continuous coordinates to discrete indices based on step sizes and origin."""
#     return (
#         int(round((coord[0] - origin[0]) / steps[0])),
#         int(round((coord[1] - origin[1]) / steps[1])),
#         int(round((coord[2] - origin[2]) / steps[2])),
#     )


def coord_to_index(
    coord_m: Tuple[float, float, float],
    step_m_per_index: Tuple[float, float, float],
    origin_m: Tuple[float, float, float],
) -> Tuple[int, int, int]:
    """Map world coordinates (meters) to integer indices (cells/levels) given per-axis step sizes and origin."""
    return (
        int(round((coord_m[0] - origin_m[0]) / step_m_per_index[0])),
        int(round((coord_m[1] - origin_m[1]) / step_m_per_index[1])),
        int(round((coord_m[2] - origin_m[2]) / step_m_per_index[2])),
    )


# #### CLASS POINT
# class Point:
#     def __init__(
#         self,
#         idp=Tuple[int, int, int],
#         steps=Tuple[float, float, float],
#         origin=Tuple[float, float, float],
#         given_coord: bool = False,
#     ):
#         if given_coord:
#             idp = coord2idp(idp, steps, origin)

#         self.idp = idp
#         self.coord = tuple(i * s for i, s in zip(idp, steps))
#         self.coord = tuple(i + s for i, s in zip(self.coord, origin))
#         # Store which movement directions are possible from here
#         self.neighbors: Dict[str, Tuple[int, int, int]] = {}

#     def add_neighbor(self, direction: str, idx: Tuple[int, int, int]):
#         self.neighbors[direction] = idx

#     def __repr__(self):
#         return (
#             f"Point({self.idp}, {self.coord}, neighbors={list(self.neighbors.keys())})"
#         )

#     def get_coord(self) -> Tuple[float, float, float]:
#         return self.coord


# #### CLASS PYRAMID_3D over a rectangual field
# class Pyramid3D:
#     def __init__(
#         self,
#         field_size: Tuple[float, float],
#         step_length_in_cells: int,
#         cell_dimension_in_meters: float,
#         fov: float,
#         center_on_origin: bool = True,
#     ):
#         self.field_size = field_size
#         self.cell_dimension_in_meters = cell_dimension_in_meters
#         self.step_length_in_cells = step_length_in_cells
#         self.step_length_in_meters = step_length_in_cells * cell_dimension_in_meters
#         self.fov = fov

#         # Origin of the grid on the ground
#         self.origin = (0, 0, 0)

#         # self.steps is a tuple that represents the dimension in meters of a step along the three directions (x, y and z)
#         self.steps = None

#         # the size is a tuple representing the number of steps to cover the field along the 3 directions (x, y and z)
#         self.size = None

#         # compute the grid parameters according to the preferred method, symmetric or asymmetric
#         # self.init_symmetric_grid()
#         self.init_asymmetric_grid()
#         if center_on_origin:
#             self.origin = (-self.field_size[0] / 2.0, -self.field_size[1] / 2.0, 0.0)


#         # create and build the grid
#         self.grid: Dict[Tuple[int, int, int], Point] = {}
#         self._build_grid()
#         print(
#             f"Pyramid: size={self.size}, steps={self.steps}, cell_size={self.asym_cell_dimension_in_meters}m"
#         )
class GridPoint:
    """
    A node in the configuration space grid.

    Attributes
    ----------
    idp : (ix, iy, iz) integer index in grid coordinates
          ix, iy are in cells; iz is in altitude levels (starting from 1).
    coord : (x, y, z) in meters
    neighbors : dict from direction -> neighbor (ix, iy, iz)
    """

    def __init__(
        self,
        idp: Tuple[int, int, int],
        step_m_per_index: Tuple[float, float, float],
        origin_m: Tuple[float, float, float],
        given_coord: bool = False,
    ):
        if given_coord:
            idp = coord_to_index(idp, step_m_per_index, origin_m)

        self.idp: Tuple[int, int, int] = idp
        # convert index -> meters: origin + idp * step
        self.coord: Tuple[float, float, float] = tuple(
            i * s for i, s in zip(idp, step_m_per_index)
        )
        self.coord = tuple(c + o for c, o in zip(self.coord, origin_m))
        self.neighbors: Dict[str, Tuple[int, int, int]] = {}

    def add_neighbor(self, direction: str, idx: Tuple[int, int, int]) -> None:
        self.neighbors[direction] = idx

    def __repr__(self):
        return f"GridPoint({self.idp}, {self.coord}, neighbors={list(self.neighbors.keys())})"

    def get_coord(self) -> Tuple[float, float, float]:
        return self.coord


# ------------------------------
# Pyramidal Configuration Space
# ------------------------------
class Pyramid3D:
    """
    3D configuration space over a rectangular field with a pyramidal (shrinking) footprint.

    Public constructor parameters
    -----------------------------
    field_size : (width_x_m, height_y_m)
    step_length_in_cells : k  (number of cells per lateral "move")
    cell_dimension_in_meters : nominal cell size (m) used to define target lateral stride (k * cell)
    fov : full field-of-view in radians
    center_on_origin : if True, center field at (0,0)

    Key internal conventions
    ------------------------
    - Index space: x,y in CELLS; z in LEVELS (z starts at 1).
    - step_m_per_index = (dx_m_per_cell, dy_m_per_cell, dz_m_per_level)
    - size_idx = (nx_cells, ny_cells, nz_levels)
    """

    def __init__(
        self,
        field_size: Tuple[float, float],
        step_length_in_cells: int,
        cell_dimension_in_meters: float,
        fov: float,
        center_on_origin: bool = True,
    ):
        # Field geometry (meters)
        self.field_size_m: Tuple[float, float] = field_size
        self.origin_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)

        # Lateral stride target and camera
        self.cell_size_m: float = cell_dimension_in_meters
        self.move_stride_cells: int = step_length_in_cells  # k
        self.move_stride_m: float = step_length_in_cells * cell_dimension_in_meters
        self.fov: float = fov

        # Derived grid parameters (set by init_*_grid)
        self.step_m_per_index: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.size_idx: Tuple[int, int, int] = (0, 0, 0)
        self.cell_xy_m: Tuple[float, float] = (0.0, 0.0)  # (cell_x_m, cell_y_m)
        self.stride_xy_m: Tuple[float, float] = (0.0, 0.0)  # (stride_x_m, stride_y_m)

        # Choose grid layout (asymmetric by default)
        # self._init_symmetric_grid()
        self._init_asymmetric_grid()
        if center_on_origin:
            self.origin_m = (
                -self.field_size_m[0] / 2.0,
                -self.field_size_m[1] / 2.0,
                0.0,
            )

        # Build graph
        self.grid: Dict[Tuple[int, int, int], GridPoint] = {}
        self._build_nodes()

        print(
            f"Pyramid: size_idx={self.size_idx}, step_m_per_index={self.step_m_per_index}, "
            f"cell_xy_m={self.cell_xy_m}"
        )

    # def init_asymmetric_grid(self):
    #     Nx_steps = round(self.field_size[0] / self.step_length_in_meters)
    #     Ny_steps = round(self.field_size[1] / self.step_length_in_meters)

    #     # actual lateral steps (meters) that tile the field exactly
    #     self.asym_step_length_in_meters = (
    #         self.field_size[0] / Nx_steps,
    #         self.field_size[1] / Ny_steps,
    #     )

    #     self.asym_cell_dimension_in_meters = (
    #         self.asym_step_length_in_meters[0] / self.step_length_in_cells,
    #         self.asym_step_length_in_meters[1] / self.step_length_in_cells,
    #     )

    #     # grid size in cells
    #     size_x = round(self.field_size[0] / self.asym_cell_dimension_in_meters[0])
    #     size_y = round(self.field_size[1] / self.asym_cell_dimension_in_meters[1])

    #     # vertical extent (limit by the tighter side)
    #     if size_x < size_y:
    #         min_size = size_x
    #         step_length = self.asym_step_length_in_meters[0]
    #     else:
    #         min_size = size_y
    #         step_length = self.asym_step_length_in_meters[1]

    #     self.size = (size_x, size_y, int(min_size / (2 * self.step_length_in_cells)))
    #     self.steps = (
    #         self.asym_cell_dimension_in_meters[0],
    #         self.asym_cell_dimension_in_meters[1],
    #         step_length / math.tan(self.fov / 2),
    #     )

    def _init_asymmetric_grid(self) -> None:
        """
        Rectangular cells so an integer number of lateral moves tiles the field exactly per axis.
        """
        Nx_moves = round(self.field_size_m[0] / self.move_stride_m)
        Ny_moves = round(self.field_size_m[1] / self.move_stride_m)

        # Actual lateral stride per axis (meters)
        self.stride_xy_m = (
            self.field_size_m[0] / Nx_moves,
            self.field_size_m[1] / Ny_moves,
        )
        # Cell size per axis (meters per cell)
        self.cell_xy_m = (
            self.stride_xy_m[0] / self.move_stride_cells,
            self.stride_xy_m[1] / self.move_stride_cells,
        )

        # Grid size in cells
        nx = round(self.field_size_m[0] / self.cell_xy_m[0])
        ny = round(self.field_size_m[1] / self.cell_xy_m[1])

        # Vertical step defined by the limiting axis' stride
        if nx < ny:
            min_n = nx
            lateral_stride_m = self.stride_xy_m[0]
        else:
            min_n = ny
            lateral_stride_m = self.stride_xy_m[1]

        nz = int(min_n / (2 * self.move_stride_cells))
        self.size_idx = (nx, ny, nz)
        self.step_m_per_index = (
            self.cell_xy_m[0],
            self.cell_xy_m[1],
            lateral_stride_m / math.tan(self.fov / 2.0),
        )

        print(f"Asymmetric lateral move counts (Nx, Ny): ({Nx_moves}, {Ny_moves})")
        print(f"Asymmetric lateral stride (m): {self.stride_xy_m}")

    # ------------------------------------------------------------------
    # Node Generation
    # ------------------------------------------------------------------
    def _build_nodes(self) -> None:
        """Create all nodes and link valid neighbors within the pyramid."""
        self.idp_min = (math.inf, math.inf, math.inf)
        self.idp_max = (-math.inf, -math.inf, -math.inf)

        k = self.move_stride_cells
        nx, ny, nz = self.size_idx

        for z0 in range(nz):  # store as z = level starting at 1
            z = z0 + 1
            for ix in np.arange(0, nx + 1, k):
                for iy in np.arange(0, ny + 1, k):
                    node = GridPoint((ix, iy, z), self.step_m_per_index, self.origin_m)

                    if self._footprint_fits_field(node):
                        # neighbors in 6 directions (index space)
                        for direction, (dx, dy, dz) in DIRECTIONS.items():
                            nix, niy, niz = ix + dx * k, iy + dy * k, z + dz
                            neighbor = GridPoint(
                                (nix, niy, niz), self.step_m_per_index, self.origin_m
                            )
                            if self._footprint_fits_field(neighbor):
                                node.add_neighbor(direction, (nix, niy, niz))

                        self.grid[(ix, iy, z)] = node
                        # track min/max index range at z=1 (for starts)
                        if ix < self.idp_min[0]:
                            self.idp_min = (ix, self.idp_min[1], 1)
                        if iy < self.idp_min[1]:
                            self.idp_min = (self.idp_min[0], iy, 1)
                        if ix > self.idp_max[0]:
                            self.idp_max = (ix, self.idp_max[1], 1)
                        if iy > self.idp_max[1]:
                            self.idp_max = (self.idp_max[0], iy, 1)

    # # iterate on all points of a rectangular grid and retain only those that are within the pyramid
    # # note that the z points start from the minimum altitude, that is 1 z-step high. This is manually set
    # def _build_grid(self):
    #     self.idp_min = (math.inf, math.inf, math.inf)
    #     self.idp_max = (-math.inf, -math.inf, -math.inf)
    #     for z in range(0, self.size[2]):
    #         for x in np.arange(0, self.size[0] + 1, self.step_length_in_cells):
    #             for y in np.arange(0, self.size[1] + 1, self.step_length_in_cells):
    #                 point = Point((x, y, z + 1), self.steps, self.origin)

    #                 if self.point_inside_pyramid(point):
    #                     # check all 6 possible directions
    #                     for direction, (dx, dy, dz) in DIRECTIONS.items():
    #                         dx = dx * self.step_length_in_cells
    #                         dy = dy * self.step_length_in_cells
    #                         nx, ny, nz = x + dx, y + dy, z + 1 + dz
    #                         neighbour = Point((nx, ny, nz), self.steps, self.origin)
    #                         if self.point_inside_pyramid(neighbour):
    #                             point.add_neighbor(direction, (nx, ny, nz))
    #                     self.grid[(x, y, z + 1)] = point
    #                     if x < self.idp_min[0]:
    #                         self.idp_min = (x, self.idp_min[1], 1)
    #                     if y < self.idp_min[1]:
    #                         self.idp_min = (self.idp_min[0], y, 1)

    #                     if x > self.idp_max[0]:
    #                         self.idp_max = (x, self.idp_max[1], 1)
    #                     if y > self.idp_max[1]:
    #                         self.idp_max = (self.idp_max[0], y, 1)
    #                     # print("Adding point", x, y, z + 1)

    # ------------------------------------------------------------------
    # Geometry checks
    # ------------------------------------------------------------------
    def _footprint_fits_field(self, node: GridPoint) -> bool:
        """
        True if the square footprint centered at node.coord fits completely inside the field.
        Uses meter-space bounds with tiny tolerance to avoid excluding edge nodes via FP drift.
        """
        z = node.idp[2]
        if not (1 <= z <= self.size_idx[2]):
            return False

        # half-width of footprint at this altitude (meters)
        half_w = (z * self.step_m_per_index[2]) * math.tan(self.fov / 2.0)

        x, y, _ = node.coord
        min_x, max_x = self.origin_m[0], self.origin_m[0] + self.field_size_m[0]
        min_y, max_y = self.origin_m[1], self.origin_m[1] + self.field_size_m[1]

        eps = 1e-9 * max(self.field_size_m[0], self.field_size_m[1])
        return (min_x + half_w - eps <= x <= max_x - half_w + eps) and (
            min_y + half_w - eps <= y <= max_y - half_w + eps
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_point(
        self, ix: int, iy: int, iz: int, coord: bool = False
    ) -> Optional[GridPoint]:
        """Fetch a grid node by index triplet; if coord=True, interpret (ix,iy,iz) as meters and map to index."""
        if coord:
            idp = coord_to_index((ix, iy, iz), self.step_m_per_index, self.origin_m)
            return self.grid.get(idp)
        return self.grid.get((ix, iy, iz))

    # def get_future_point(
    #     self, coord: Tuple[float, float, float], direction: str
    # ) -> Point:
    #     current_p = Point((coord), self.steps, self.origin, given_coord=True)
    #     point = self.get_point(*current_p.idp)
    #     if direction in point.neighbors:
    #         neighbor_idx = point.neighbors.get(direction)
    #         future = self.get_point(*neighbor_idx)
    #         return future.coord
    #     return None

    # def overlap_percentage(self) -> Dict[float, float]:
    #     """
    #     returns a dictionary with altitude levels as keys and overlap percentage as values
    #     """
    #     overlap_dict = {}
    #     alts = self.get_altitude_levels()
    #     for alt in alts:
    #         footprint_size = 2 * alt * math.tan(self.fov / 2)
    #         overlap = 1 - (self.step_length_in_meters / footprint_size)
    #         overlap_dict[alt] = overlap * 100  # percentage
    #     return overlap_dict

    def get_future_point(
        self, coord: Tuple[float, float, float], direction: str
    ) -> Optional[Tuple[float, float, float]]:
        """
        Given a world coordinate on a valid node, step one move in 'direction' and return the future coord (meters).
        """
        current = GridPoint(
            coord, self.step_m_per_index, self.origin_m, given_coord=True
        )
        node = self.get_point(*current.idp)
        if node is None:  # outside grid
            return None
        if direction in node.neighbors:
            neighbor_idx = node.neighbors[direction]
            future = self.get_point(*neighbor_idx)
            return future.get_coord() if future is not None else None
        return None

    def overlap_percentage(self) -> Dict[float, float]:
        """
        returns a dictionary with altitude levels as keys and overlap percentage as values.
        NOTE: uses the *effective* lateral stride that defines the vertical step:
              effective_stride = step_m_per_index[2] * tan(fov/2)
        """
        out: Dict[float, float] = {}
        effective_stride = self.step_m_per_index[2] * math.tan(self.fov / 2.0)
        for alt in self.get_altitude_levels():
            footprint_w = 2.0 * alt * math.tan(self.fov / 2.0)  # meters
            if footprint_w <= 0.0:
                overlap = 0.0
            else:
                overlap = 1.0 - (effective_stride / footprint_w)
                overlap = max(0.0, min(1.0, overlap))
            out[alt] = overlap * 100.0
        return out

    # def get_altitude_levels(self) -> List[float]:
    #     """
    #     returns altitude levels in meters
    #     """
    #     n_h = self.size[2]

    #     altitude_levels = [
    #         (z + 1) * self.steps[2] for z in range(n_h)
    #     ]  # z starts from 1
    #     return altitude_levels

    # def __repr__(self):
    #     return f"Pyramid3D({self.size}, {self.steps})"

    def get_altitude_levels(self) -> List[float]:
        """Return altitude levels (meters) for iz = 1..nz."""
        _, _, nz = self.size_idx
        return [(z + 1) * self.step_m_per_index[2] for z in range(nz)]

    def __repr__(self):
        return f"Pyramid3D(size_idx={self.size_idx}, step_m_per_index={self.step_m_per_index})"

    # def sample_start_position(self, position: str = "corner", rng=None) -> Point:
    #     """
    #     Get random VALID start at the lowest altitude (z=1)..
    #     - position="corner": random (min|max x, min|max y, 1)
    #     - position="edge": random edge at z=1, random lattice point along that edge
    #     """

    #     ix_min, ix_max = self.idp_min[0], self.idp_max[0]
    #     iy_min, iy_max = self.idp_min[1], self.idp_max[1]
    #     z_level = 1  # lowest altitude
    #     if rng is None:
    #         seed = 123
    #         rng = np.random.default_rng(seed)

    #     if position == "corner":
    #         return self.get_point(
    #             rng.choice([ix_min, ix_max]), rng.choice([iy_min, iy_max]), z_level
    #         )
    #     elif position == "edge":
    #         # choose one edge uniformly
    #         step = self.step_length_in_cells
    #         xs = list(range(ix_min, ix_max + 1, step))
    #         ys = list(range(iy_min, iy_max + 1, step))
    #         edge = rng.choice(["south", "north", "west", "east"])
    #         if edge == "south":  # y fixed to min
    #             x, y = rng.choice(xs), iy_min
    #         elif edge == "north":  # y fixed to max
    #             x, y = rng.choice(xs), iy_max
    #         elif edge == "west":  # x fixed to min
    #             x, y = ix_min, rng.choice(ys)
    #         else:  # "east": x fixed to max
    #             x, y = ix_max, rng.choice(ys)
    #         return self.get_point(x, y, z_level)
    #     else:
    #         raise ValueError("position must be 'corner' or 'edge'.")

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample_start_position(
        self, position: str = "corner", rng=None
    ) -> Optional[GridPoint]:
        """
        Random valid start at the lowest altitude (iz=1).
        - position="corner": random among {(min_x,min_y), (min_x,max_y), (max_x,min_y), (max_x,max_y)}
        - position="edge": random index along a random field edge at iz=1
        - position="center": center of the field at highest altitude
        """
        ix_min, ix_max = self.idp_min[0], self.idp_max[0]
        iy_min, iy_max = self.idp_min[1], self.idp_max[1]
        iz = 1

        if rng is None:
            rng = np.random.default_rng(123)
        # TODO: start with fixed point for testing
        if position == "corner":
            # return self.get_point(
            #     rng.choice([ix_min, ix_max]), rng.choice([iy_min, iy_max]), iz
            # )
            return self.get_point(ix_min, iy_min, iz)
        # TODO: returns min alt center
        if position == "center":
            # === 1. Continuous center of the field (meters) ===
            cx_m = self.origin_m[0] + self.field_size_m[0] / 2
            cy_m = self.origin_m[1] + self.field_size_m[1] / 2

            # === 2. Convert to nearest grid indices ===
            center_ix, center_iy, _ = coord_to_index(
                (cx_m, cy_m, 0.0),
                self.step_m_per_index,
                self.origin_m,
            )

            # === 3. Try to find this center at highest altitude ===
            # for iz in reversed(range(1, self.size_idx[2] + 1)):
            # min alt
            for iz in range(1, self.size_idx[2] + 1):
                node = self.get_point(center_ix, center_iy, iz)

                if node is not None:
                    return node

            raise ValueError(
                f"No valid center point in the pyramid for indices ({center_ix}, {center_iy}, *)."
            )

        if position == "edge":
            k = self.move_stride_cells
            xs = list(range(ix_min, ix_max + 1, k))
            ys = list(range(iy_min, iy_max + 1, k))
            edge = rng.choice(["south", "north", "west", "east"])
            if edge == "south":
                x, y = rng.choice(xs), iy_min  # y fixed min
            elif edge == "north":
                x, y = rng.choice(xs), iy_max  # y fixed max
            elif edge == "west":
                x, y = ix_min, rng.choice(ys)  # x fixed min
            else:
                x, y = ix_max, rng.choice(ys)  # x fixed max
            return self.get_point(x, y, iz)

        raise ValueError("position must be 'corner' or 'edge'.")
