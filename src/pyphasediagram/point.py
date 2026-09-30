"""Point-like data structures used by phase-diagram calculations.

The classes in this module represent compositions in the two independent
coordinates ``(phi1, phi2)``.  They also carry the direction and phase-pair
information needed to initialize spinodal and binodal calculations.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class Point:
    """Represent an indexed point in two-dimensional composition space.

    Parameters
    ----------
    idx : int
        Identifier used when the point is stored as a graph node.
    phi1 : float
        First independent composition coordinate.
    phi2 : float
        Second independent composition coordinate.

    Attributes
    ----------
    idx : int
        Integer point identifier.
    phi1 : float
        First independent composition coordinate.
    phi2 : float
        Second independent composition coordinate.

    Notes
    -----
    The constructor converts its arguments to built-in ``int`` and ``float``
    values but does not validate whether the coordinates lie in the physical
    composition domain.
    """

    def __init__(self, idx: int, phi1: float, phi2: float):
        self.idx = int(idx)
        self.phi1 = float(phi1)
        self.phi2 = float(phi2)

    def dist(self, pt) -> float:
        """Calculate the Euclidean distance to another point.

        Parameters
        ----------
        pt : Point
            Point to which the distance is calculated.

        Returns
        -------
        float
            Euclidean distance in composition space.

        Raises
        ------
        NotImplementedError
            If ``pt`` is not a :class:`Point` instance.
        """
        if not isinstance(pt, Point):
            raise NotImplementedError(
                "Distance can only be computed between Point instances."
            )
        return np.sqrt((self.phi1 - pt.phi1) ** 2 + (self.phi2 - pt.phi2) ** 2)

    def __repr__(self):
        """Return a representation containing the composition coordinates."""
        return f"Point(phi1={self.phi1:.3f}, phi2={self.phi2:.3f})"

    def __add__(self, pt):
        """Add the coordinates of two points.

        Parameters
        ----------
        pt : Point
            Point whose coordinates are added to this point.

        Returns
        -------
        numpy.ndarray
            Coordinate sum ``[phi1, phi2]`` with shape ``(2,)``.

        Raises
        ------
        NotImplementedError
            If ``pt`` is not a :class:`Point` instance.
        """
        if not isinstance(pt, Point):
            raise NotImplementedError(
                "Addition can only be performed between Point instances."
            )
        return np.array([self.phi1 + pt.phi1, self.phi2 + pt.phi2])

    def __sub__(self, pt):
        """Subtract another point's coordinates from this point.

        Parameters
        ----------
        pt : Point
            Point whose coordinates are subtracted from this point.

        Returns
        -------
        numpy.ndarray
            Coordinate difference ``[phi1, phi2]`` with shape ``(2,)``.

        Raises
        ------
        NotImplementedError
            If ``pt`` is not a :class:`Point` instance.
        """
        if not isinstance(pt, Point):
            raise NotImplementedError(
                "Subtraction can only be performed between Point instances."
            )
        return np.array([self.phi1 - pt.phi1, self.phi2 - pt.phi2])

    def plot(self, s=10, **kwargs):
        """Plot the point on the current Matplotlib axes.

        Parameters
        ----------
        s : float, optional
            Marker size passed to :func:`matplotlib.pyplot.scatter`.
        **kwargs
            Additional keyword arguments passed to
            :func:`matplotlib.pyplot.scatter`.

        Returns
        -------
        None
        """
        plt.scatter(self.phi1, self.phi2, s=s, **kwargs)

    def is_between(self, pt1, pt2, tol=np.pi / 8):
        """Check whether this point lies approximately between two points.

        The check compares the angle between the vectors from this point to
        ``pt1`` and ``pt2`` with an angle of pi radians.

        Parameters
        ----------
        pt1 : Point
            First endpoint of the comparison.
        pt2 : Point
            Second endpoint of the comparison.
        tol : float, optional
            Maximum angular deviation from pi, in radians.

        Returns
        -------
        bool
            ``True`` when the two direction vectors are opposite within
            ``tol``.

        Raises
        ------
        NotImplementedError
            If either endpoint is not a :class:`Point` instance.

        Notes
        -----
        Both endpoints are assumed to differ from this point so that their
        direction vectors can be normalized.
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
        """Check whether another point is within a distance tolerance.

        Parameters
        ----------
        pt : Point
            Point to compare with this point.
        tol : float, optional
            Exclusive upper bound for the Euclidean distance.

        Returns
        -------
        bool
            ``True`` if the distance to ``pt`` is less than ``tol``.

        Raises
        ------
        NotImplementedError
            If ``pt`` is not a :class:`Point` instance.
        """
        if not isinstance(pt, Point):
            raise NotImplementedError(
                "is_close_to can only be computed between Point instances."
            )
        return self.dist(pt) < tol

    def __iter__(self):
        """Iterate over the two composition coordinates.

        Yields
        ------
        float
            ``phi1`` followed by ``phi2``.
        """
        yield self.phi1
        yield self.phi2


class SpinodalPoint(Point):
    """Represent a sampled point on a spinodal curve.

    Parameters
    ----------
    idx : int
        Identifier used when the point is stored in the spinodal graph.
    phi1 : float
        First independent composition coordinate.
    phi2 : float
        Second independent composition coordinate.
    """

    def __repr__(self):
        """Return a representation identifying the point as spinodal."""
        return f"SpinodalPoint(phi1={self.phi1:.3f}, phi2={self.phi2:.3f})"


class CriticalPoint(Point):
    """Represent a critical point and its local spinodal direction.

    Parameters
    ----------
    idx : int
        Identifier used when the point is stored as a graph node.
    phi1 : float
        First independent composition coordinate.
    phi2 : float
        Second independent composition coordinate.
    dphi1 : float
        First component of the local direction vector.
    dphi2 : float
        Second component of the local direction vector.

    Attributes
    ----------
    idx : int
        Integer point identifier.
    phi1 : float
        First independent composition coordinate.
    phi2 : float
        Second independent composition coordinate.
    dphi1 : float
        First component of the normalized direction vector.
    dphi2 : float
        Second component of the normalized direction vector.

    Notes
    -----
    The direction ``(dphi1, dphi2)`` is normalized during initialization and
    must therefore have nonzero length.
    """

    def __init__(self, idx: int, phi1: float, phi2: float, dphi1: float, dphi2: float):
        super().__init__(idx, phi1, phi2)
        dphi_norm = np.sqrt(dphi1**2 + dphi2**2)
        self.dphi1 = float(dphi1) / dphi_norm
        self.dphi2 = float(dphi2) / dphi_norm

    @classmethod
    def from_points(cls, idx: int, pt1: SpinodalPoint, pt2: SpinodalPoint):
        """Construct a critical point from two spinodal points.

        Parameters
        ----------
        idx : int
            Identifier assigned to the critical point.
        pt1 : SpinodalPoint
            First spinodal point.
        pt2 : SpinodalPoint
            Second spinodal point.

        Returns
        -------
        CriticalPoint
            Point at the midpoint of ``pt1`` and ``pt2``, directed from
            ``pt2`` toward ``pt1``.

        Notes
        -----
        The two input points must have distinct coordinates so that their
        difference can be normalized.
        """
        phi1 = (pt1.phi1 + pt2.phi1) / 2
        phi2 = (pt1.phi2 + pt2.phi2) / 2

        dphi1 = pt1.phi1 - pt2.phi1
        dphi2 = pt1.phi2 - pt2.phi2
        dphi_norm = np.sqrt(dphi1**2 + dphi2**2)
        dphi1 = float(dphi1 / dphi_norm)
        dphi2 = float(dphi2 / dphi_norm)

        return cls(idx, phi1, phi2, dphi1, dphi2)

    def get_phi_and_v_init(self):
        """Create initial phase coordinates and a tracing direction.

        The two phases are placed on opposite sides of the critical point at
        offsets of ``1e-3`` along the stored direction vector.

        Returns
        -------
        phi_init : numpy.ndarray
            Initial phase compositions with shape ``(4,)`` and ordering
            ``[phi1a, phi2a, phi1b, phi2b]``.
        v_init : numpy.ndarray
            Initial tracing direction with shape ``(4,)`` and ordering
            ``[dphi1a, dphi2a, dphi1b, dphi2b]``.
        """

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
        """Plot the critical point on the current Matplotlib axes.

        Parameters
        ----------
        s : float, optional
            Marker size passed to :meth:`Point.plot`.
        **kwargs
            Additional scatter-plot options. The defaults are a yellow star
            with a black edge.

        Returns
        -------
        None
        """
        if "marker" not in kwargs:
            kwargs["marker"] = "*"
        if "color" not in kwargs:
            kwargs["color"] = "yellow"
        if "edgecolor" not in kwargs:
            kwargs["edgecolor"] = "black"
        super().plot(s=s, **kwargs)

    def __repr__(self):
        """Return a representation containing coordinates and direction."""
        return f"CriticalPoint(phi1={self.phi1:.3f}, phi2={self.phi2:.3f}, dphi1={self.dphi1:.3f}, dphi2={self.dphi2:.3f})"


class BinodalPoint:
    """Represent the compositions of two coexisting phases.

    Parameters
    ----------
    phi1a : float
        First composition coordinate of phase A.
    phi2a : float
        Second composition coordinate of phase A.
    phi1b : float
        First composition coordinate of phase B.
    phi2b : float
        Second composition coordinate of phase B.

    Attributes
    ----------
    pta : Point
        Composition of phase A, assigned point index 0.
    ptb : Point
        Composition of phase B, assigned point index 1.
    """

    def __init__(self, phi1a: float, phi2a: float, phi1b: float, phi2b: float):
        self.pta = Point(0, phi1a, phi2a)
        self.ptb = Point(1, phi1b, phi2b)

    @classmethod
    def from_points(cls, pt1: Point, pt2: Point):
        """Construct a binodal point from two phase-composition points.

        Parameters
        ----------
        pt1 : Point
            Composition of phase A.
        pt2 : Point
            Composition of phase B.

        Returns
        -------
        BinodalPoint
            Binodal point containing the coordinates of ``pt1`` and ``pt2``.

        """
        return cls(pt1.phi1, pt1.phi2, pt2.phi1, pt2.phi2, sv)

    def __repr__(self):
        """Return a representation containing both phase compositions."""
        return f"BinodalPoint(phi_a=({self.pta.phi1:.4f}, {self.pta.phi2:.4f}), phi_b=({self.ptb.phi1:.4f}, {self.ptb.phi2:.4f}))"

    def to_numpy(self):
        """Return both phase compositions as a flat array.

        Returns
        -------
        numpy.ndarray
            Phase coordinates with shape ``(4,)`` and ordering
            ``[phi1a, phi2a, phi1b, phi2b]``.
        """
        return np.array([self.pta.phi1, self.pta.phi2, self.ptb.phi1, self.ptb.phi2])

    def plot(self, s=20, **kwargs):
        """Plot both coexisting phase compositions.

        Parameters
        ----------
        s : float, optional
            Marker size passed to :meth:`Point.plot`.
        **kwargs
            Plot style options. ``color``, ``marker``, and ``edgecolor`` are
            used, with defaults of purple, ``"o"``, and black, respectively.

        Returns
        -------
        None
        """
        color = kwargs.get("color", "purple")
        marker = kwargs.get("marker", "o")
        edgecolor = kwargs.get("edgecolor", "black")
        self.pta.plot(s=s, color=color, marker=marker, edgecolor=edgecolor)
        self.ptb.plot(s=s, color=color, marker=marker, edgecolor=edgecolor)


class BinodalInitialPoint(BinodalPoint):
    """Represent a binodal initial state and its tracing direction.

    Parameters
    ----------
    phi_init : numpy.ndarray
        Initial phase coordinates with shape ``(4,)`` and ordering
        ``[phi1a, phi2a, phi1b, phi2b]``.
    v_init : numpy.ndarray
        Initial tracing direction with shape ``(4,)`` and matching coordinate
        ordering.

    Attributes
    ----------
    pta : Point
        Initial composition of phase A.
    ptb : Point
        Initial composition of phase B.
    phi_init : numpy.ndarray
        Initial phase coordinates supplied to the constructor.
    v_init : numpy.ndarray
        Initial tracing direction normalized to unit length.

    Notes
    -----
    ``v_init`` must have nonzero length so that it can be normalized.
    """

    def __init__(self, phi_init: np.ndarray, v_init: np.ndarray):
        super().__init__(*phi_init)
        self.phi_init = phi_init
        self.v_init = v_init / np.linalg.norm(v_init)

    def is_similar_to(self, bipt, dist_tol=1e-3, angle_tol=np.pi / 8):
        """Check whether another binodal initial point is equivalent.

        Phase labels may be exchanged, and tracing directions may have
        opposite signs. Both phase compositions and the direction must agree
        within their respective tolerances.

        Parameters
        ----------
        bipt : BinodalInitialPoint
            Initial point to compare with this one.
        dist_tol : float, optional
            Exclusive distance tolerance for each phase composition.
        angle_tol : float, optional
            Maximum angular deviation between tracing directions, in radians.

        Returns
        -------
        bool
            ``True`` if the initial points describe the same phase pair and
            tracing direction up to phase exchange and direction reversal.
        """

        is_similar = False
        pta, ptb = self.pta, self.ptb
        v_init = self.v_init.copy()

        for permutation in range(2):
            pta, ptb = ptb, pta
            v_init = v_init[[2, 3, 0, 1]]
            # Distance proximity check
            is_similar = pta.is_close_to(bipt.pta, dist_tol) and ptb.is_close_to(
                bipt.ptb, dist_tol
            )
            # Angle proximity check
            is_similar = is_similar and abs(np.dot(v_init, bipt.v_init)) > np.cos(
                angle_tol
            )
            if is_similar:
                return True

        return False

    def plot(self, **kwargs):
        """Plot both phases and their initial tracing directions.

        Parameters
        ----------
        **kwargs
            Plot style options passed to :meth:`BinodalPoint.plot` for the
            phase markers. Direction arrows are drawn in red.

        Returns
        -------
        None
        """
        super().plot(**kwargs)
        v = self.v_init * 1e-3
        plt.arrow(
            self.pta.phi1,
            self.pta.phi2,
            v[0],
            v[1],
            head_width=0.002,
            head_length=0.02,
            fc="red",
            ec="red",
        )
        plt.arrow(
            self.ptb.phi1,
            self.ptb.phi2,
            v[2],
            v[3],
            head_width=0.002,
            head_length=0.02,
            fc="red",
            ec="red",
        )

    def __repr__(self):
        """Return a representation of the phases and tracing direction."""
        return f"BinodalInitialPoint(phi_a=({self.pta.phi1:.4f}, {self.pta.phi2:.4f}), phi_b=({self.ptb.phi1:.4f}, {self.ptb.phi2:.4f}), v_init={self.v_init})"

    def __iter__(self):
        """Iterate over the initial coordinates and normalized direction.

        Yields
        ------
        numpy.ndarray
            ``phi_init`` followed by ``v_init``; both have shape ``(4,)``.
        """
        yield self.phi_init
        yield self.v_init
