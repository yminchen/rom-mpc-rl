#!/usr/bin/env python
"""
Vector operations
"""
import math
import numpy as np

# Local modules
import common.math_utils as tfu


def unit(vector):
    """
    Return vector divided by Euclidean (L2) norm

    Parameters
    ----------
    vector: array_like
      The input vector

    Returns
    -------
    unit : array_like
      Vector divided by L2 norm

    Examples
    --------
    >>> import numpy as np
    >>> import trifinger_mujoco.utils as tfu
    >>> v0 = np.random.random(3)
    >>> v1 = tfu.vector.unit(v0)
    >>> np.allclose(v1, v0 / np.linalg.norm(v0))
    True
    """
    v = np.asarray(vector).squeeze()
    return v / math.sqrt((v**2).sum())


def norm(vector):
    """
    Return vector Euclidaan (L2) norm

    Parameters
    ----------
    vector: array_like
      The input vector

    Returns
    -------
    norm: float
      The computed norm

    Examples
    --------
    >>> import numpy as np
    >>> import trifinger_mujoco.utils as tfu
    >>> v = np.random.random(3)
    >>> n = tfu.vector.norm(v)
    >>> numpy.allclose(n, np.linalg.norm(v))
    True
    """
    return math.sqrt((np.asarray(vector) ** 2).sum())


def perpendicular(vector):
    """
    Find an arbitrary perpendicular vector

    Parameters
    ----------
    vector: array_like
      The input vector

    Returns
    -------
    result: array_like
      The perpendicular vector
    """
    if np.allclose(vector, np.zeros(3)):
        # vector is [0, 0, 0]
        raise ValueError("Input vector cannot be a zero vector")
    u = unit(vector)
    if np.allclose(u[:2], np.zeros(2)):
        return tfu.Y_AXIS
    result = np.array([-u[1], u[0], 0], dtype=np.float64)
    return result


def skew(vector):
    """
    Returns the 3x3 skew matrix of the input vector.

    The skew matrix is a square matrix `R` whose transpose is also its negative;
    that is, it satisfies the condition :math:`-R = R^T`.

    Parameters
    ----------
    vector: array_like
      The input array

    Returns
    -------
    R: array_like
      The resulting 3x3 skew matrix
    """
    skv = np.roll(np.roll(np.diag(np.asarray(vector).flatten()), 1, 1), -1, 0)
    return skv - skv.T


def transform_between_vectors(vector_a, vector_b):
    """
    Compute the transformation that aligns two vectors

    Parameters
    ----------
    vector_a: array_like
      The initial vector
    vector_b: array_like
      The goal vector

    Returns
    -------
    transform: array_like
      The transformation between `vector_a` a `vector_b`
    """
    newaxis = unit(vector_b)
    oldaxis = unit(vector_a)
    # Limits the value of `c` to be within the range C{[-1, 1]}
    c = np.clip(np.dot(oldaxis, newaxis), -1.0, 1.0)
    angle = np.arccos(c)
    if np.isclose(c, -1.0) or np.allclose(newaxis, oldaxis):
        axis = perpendicular(newaxis)
    else:
        axis = unit(np.cross(oldaxis, newaxis))
    transform = tfu.axis_angle.to_transform(axis, angle)
    return transform
