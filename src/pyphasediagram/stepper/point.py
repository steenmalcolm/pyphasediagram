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
        """Euclidean distance between this point and another SpinodalPoint."""
        if not isinstance(pt, SpinodalPoint):
            raise NotImplementedError(
                "Distance can only be computed between SpinodalPoint instances."
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

    def __repr__(self):
        return f"BinodalPoint(phi_a=({self.pta.phi1:.4f}, {self.pta.phi2:.4f}), phi_b=({self.ptb.phi1:.4f}, {self.ptb.phi2:.4f}, sv={self.sv:.2e}))"

    def __iter__(self):
        yield from self.pta
        yield from self.ptb

    def to_numpy(self):
        return np.array([self.pta.phi1, self.pta.phi2, self.ptb.phi1, self.ptb.phi2])


if __name__ == "__main__":
    # Test if I can add two spinodal point instances and get a numpy array back
    pt1 = SpinodalPoint(0, 0.2, 0.3)
    pt2 = SpinodalPoint(1, 0.4, 0.5)
    pt_sum = pt1 + pt2
    print(pt_sum)  # Should print a numpy array with the sum of the coordinates
    # Test if I can get the distance between two spinodal point instances
    dist = pt1.dist(pt2)
    print(dist)  # Should print the Euclidean distance between pt1 and pt2
