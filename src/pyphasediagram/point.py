import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Point:
    def __init__(self, idx: int, phi1: float, phi2: float):
        """A point in phase space with coordinates (phi1, phi2) and an index for graph node identification."""
        self.idx = int(idx)
        self.phi1 = float(phi1)
        self.phi2 = float(phi2)

    def dist(self, pt) -> float:
        """Euclidean distance between this point and another Point."""
        if not isinstance(pt, Point):
            raise NotImplementedError(
                "Distance can only be computed between Point instances."
            )
        return np.sqrt((self.phi1 - pt.phi1) ** 2 + (self.phi2 - pt.phi2) ** 2)

    def __repr__(self):
        return f"Point(phi1={self.phi1:.3f}, phi2={self.phi2:.3f})"

    def __add__(self, pt):
        if not isinstance(pt, Point):
            raise NotImplementedError(
                "Addition can only be performed between Point instances."
            )
        return np.array([self.phi1 + pt.phi1, self.phi2 + pt.phi2])

    def __sub__(self, pt):
        if not isinstance(pt, Point):
            raise NotImplementedError(
                "Subtraction can only be performed between Point instances."
            )
        return np.array([self.phi1 - pt.phi1, self.phi2 - pt.phi2])

    def plot(self, s=10, **kwargs):
        """Convenience method to plot this point."""
        plt.scatter(self.phi1, self.phi2, s=s, **kwargs)

    def is_between(self, pt1, pt2, tol=np.pi / 8):
        """
        Determine if this point is approximately between pt1 and pt2 by checking if the angle between the vectors (self->pt1) and (self->pt2) is close to 180 degrees within a tolerance in units of radians
        """
        if not (isinstance(pt1, Point) and isinstance(pt2, Point)):
            raise NotImplementedError(
                "is_between can only be computed between Point instances."
            )
        self_to_1 = (pt1 - self) / np.linalg.norm(pt1 - self)
        self_to_2 = (pt2 - self) / np.linalg.norm(pt2 - self)
        cos_angle = np.dot(self_to_1, self_to_2)
        return cos_angle < np.cos(np.pi - tol)

    def is_close_to(self, pt, tol=1e-3):
        """Determine if this point is close to another point by checking if the distance between them is less than a specified tolerance."""
        if not isinstance(pt, Point):
            raise NotImplementedError(
                "is_close_to can only be computed between Point instances."
            )
        return self.dist(pt) < tol

    def __iter__(self):
        yield self.phi1
        yield self.phi2


class SpinodalPoint(Point):
    def __repr__(self):
        return f"SpinodalPoint(phi1={self.phi1:.3f}, phi2={self.phi2:.3f})"


class CriticalPoint(Point):
    def __init__(self, idx: int, phi1: float, phi2: float, dphi1: float, dphi2: float):
        """A critical point on the spinodal curve with coordinates (phi1, phi2) and a normalized direction vector (dphi1, dphi2) indicating the direction of the spinodal curve at this point. The direction vector is normalized to have unit length."""
        super().__init__(idx, phi1, phi2)
        dphi_norm = np.sqrt(dphi1**2 + dphi2**2)
        self.dphi1 = float(dphi1) / dphi_norm
        self.dphi2 = float(dphi2) / dphi_norm

    @classmethod
    def from_points(cls, idx: int, pt1: SpinodalPoint, pt2: SpinodalPoint):
        """Create a CriticalPoint from two SpinodalPoints by taking the midpoint of their coordinates as the critical point's coordinates and the vector from pt2 to pt1 as the direction vector. The direction vector is normalized to have unit length."""
        phi1 = (pt1.phi1 + pt2.phi1) / 2
        phi2 = (pt1.phi2 + pt2.phi2) / 2

        dphi1 = pt1.phi1 - pt2.phi1
        dphi2 = pt1.phi2 - pt2.phi2
        dphi_norm = np.sqrt(dphi1**2 + dphi2**2)
        dphi1 = float(dphi1 / dphi_norm)
        dphi2 = float(dphi2 / dphi_norm)

        return cls(idx, phi1, phi2, dphi1, dphi2)

    def get_phi_and_v_init(self):

        phi_init = np.array(
            [
                self.phi1 + self.dphi1 * 1e-3,
                self.phi2 + self.dphi2 * 1e-3,
                self.phi1 - self.dphi1 * 1e-3,
                self.phi2 - self.dphi2 * 1e-3,
            ]
        )
        v_init = np.array([self.dphi1, self.dphi2, -self.dphi1, -self.dphi2])
        return phi_init, v_init

    def plot(self, s=10, **kwargs):
        """Convenience method to plot this critical point with a different marker and color than regular SpinodalPoints. By default, the marker is a yellow star."""
        if "marker" not in kwargs:
            kwargs["marker"] = "*"
        if "color" not in kwargs:
            kwargs["color"] = "yellow"
        if "edgecolor" not in kwargs:
            kwargs["edgecolor"] = "black"
        super().plot(s=s, **kwargs)

    def __repr__(self):
        return f"CriticalPoint(phi1={self.phi1:.3f}, phi2={self.phi2:.3f}, dphi1={self.dphi1:.3f}, dphi2={self.dphi2:.3f})"


class BinodalPoint:
    def __init__(
        self, phi1a: float, phi2a: float, phi1b: float, phi2b: float, sv: float
    ):
        self.pta = Point(0, phi1a, phi2a)
        self.ptb = Point(1, phi1b, phi2b)
        self.sv = sv

    def from_points(pt1: Point, pt2: Point, sv: float):
        """Create a BinodalPoint from two Points representing the compositions of the two coexisting phases and a scalar value sv representing the spinodal value at this binodal point. The coordinates of the two points are used to initialize the two phases of the binodal point."""
        return BinodalPoint(pt1.phi1, pt1.phi2, pt2.phi1, pt2.phi2, sv)

    def __repr__(self):
        return f"BinodalPoint(phi_a=({self.pta.phi1:.4f}, {self.pta.phi2:.4f}), phi_b=({self.ptb.phi1:.4f}, {self.ptb.phi2:.4f}, sv={self.sv:.2e}))"

    def __iter__(self):
        yield from self.pta
        yield from self.ptb

    def to_numpy(self):
        return np.array([self.pta.phi1, self.pta.phi2, self.ptb.phi1, self.ptb.phi2])

    def plot(self, s=20, **kwargs):
        """Convenience method to plot this binodal point by plotting the two coexisting phases with different markers and colors. By default, the first phase is plotted as a blue circle and the second phase is plotted as an orange square."""
        color = kwargs.get("color", "purple")
        marker = kwargs.get("marker", "o")
        edgecolor = kwargs.get("edgecolor", "black")
        self.pta.plot(s=s, color=color, marker=marker, edgecolor=edgecolor)
        self.ptb.plot(s=s, color=color, marker=marker, edgecolor=edgecolor)

    def is_similar_to(self, bpt, tol=1e-3):
        """Determine if this binodal point is similar to another binodal point by checking if the compositions of the two coexisting phases are close to each other within a specified tolerance."""
        if not isinstance(bpt, BinodalPoint):
            raise NotImplementedError(
                "is_similar_to can only be computed between BinodalPoint instances."
            )
        pta, ptb = self.pta, self.ptb
        for permutation in range(2):
            pta, ptb = ptb, pta
            if pta.is_close_to(bpt.pta, tol) and ptb.is_close_to(bpt.ptb, tol):
                return True

        return False

