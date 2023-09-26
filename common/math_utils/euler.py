#!/usr/bin/env python
"""
Generic Euler rotations
"""
import math
import numpy as np

# Local modules
import common.math_utils as tfu


def to_axis_angle(ai, aj, ak, axes="sxyz"):
    """
    Return axis-angle rotation from Euler angles and axes sequence

    Parameters
    ----------
    ai: float
      First rotation angle (according to axes).
    aj: float
      Second rotation angle (according to axes).
    ak: float
      Third rotation angle (according to axes).
    axes: str, optional
      Axis specification; one of 24 axis sequences as string or encoded tuple

    Returns
    -------
    axis: array_like
      axis around which rotation occurs
    angle: float
      angle of rotation

    Examples
    --------
    >>> import numpy as np
    >>> import trifinger_mujoco.utils as tfu
    >>> axis, angle = tfu.euler.to_axis_angle(0, 1.5, 0, 'szyx')
    >>> np.allclose(axis, [0, 1, 0])
    True
    >>> angle
    1.5
    """
    T = to_transform(ai, aj, ak, axes)
    axis, angle, _ = tfu.transform.to_axis_angle(T)
    return axis, angle


def to_quaternion(ai, aj, ak, axes="sxyz"):
    """
    Returns a quaternion from Euler angles and axes sequence

    Parameters
    ----------
    ai: float
      First rotation angle (according to axes).
    aj: float
      Second rotation angle (according to axes).
    ak: float
      Third rotation angle (according to axes).
    axes: str, optional
      Axis specification; one of 24 axis sequences as string or encoded tuple

    Returns
    -------
    q: array_like
      Quaternion in w, x, y z (real, then vector) format

    Notes
    -----
    Quaternions :math:`w + ix + jy + kz` are represented as :math:`[w, x, y, z]`.

    Examples
    --------
    >>> import numpy as np
    >>> import trifinger_mujoco.utils as tfu
    >>> q = tfu.euler.to_quaternion(1, 2, 3, 'ryxz')
    >>> np.allclose(q, [0.435953, 0.310622, -0.718287, 0.444435])
    True
    """
    try:
        firstaxis, parity, repetition, frame = tfu._AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        tfu._TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis + 1
    j = tfu._NEXT_AXIS[i + parity - 1] + 1
    k = tfu._NEXT_AXIS[i - parity] + 1

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk

    q = np.empty((4,))
    if repetition:
        q[0] = cj * (cc - ss)
        q[i] = cj * (cs + sc)
        q[j] = sj * (cc + ss)
        q[k] = sj * (cs - sc)
    else:
        q[0] = cj * cc + sj * ss
        q[i] = cj * sc - sj * cs
        q[j] = cj * ss + sj * cc
        q[k] = cj * cs - sj * sc
    if parity:
        q[j] *= -1.0
    return q


def to_transform(ai, aj, ak, axes="sxyz"):
    """
    Return homogeneous transformation matrix from Euler angles and axes sequence.

    Parameters
    ----------
    ai: float
      First rotation angle (according to axes).
    aj: float
      Second rotation angle (according to axes).
    ak: float
      Third rotation angle (according to axes).
    axes: str, optional
      Axis specification; one of 24 axis sequences as string or encoded tuple

    Returns
    -------
    T: array_like
      Homogeneous transformation (4x4)

    Examples
    --------
    >>> import numpy as np
    >>> import trifinger_mujoco.utils as tfu
    >>> T = tfu.euler.to_transform(1, 2, 3, 'syxz')
    >>> np.allclose(np.sum(T[0]), -1.34786452)
    True
    >>> T = tfu.euler.to_transform(1, 2, 3, (0, 1, 0, 1))
    >>> np.allclose(np.sum(T[0]), -0.383436184)
    True
    """
    try:
        firstaxis, parity, repetition, frame = tfu._AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        tfu._TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = tfu._NEXT_AXIS[i + parity]
    k = tfu._NEXT_AXIS[i - parity + 1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    T = np.identity(4)
    if repetition:
        T[i, i] = cj
        T[i, j] = sj * si
        T[i, k] = sj * ci
        T[j, i] = sj * sk
        T[j, j] = -cj * ss + cc
        T[j, k] = -cj * cs - sc
        T[k, i] = -sj * ck
        T[k, j] = cj * sc + cs
        T[k, k] = cj * cc - ss
    else:
        T[i, i] = cj * ck
        T[i, j] = sj * sc - cs
        T[i, k] = sj * cc + ss
        T[j, i] = cj * sk
        T[j, j] = sj * ss + cc
        T[j, k] = sj * cs - sc
        T[k, i] = -sj
        T[k, j] = cj * si
        T[k, k] = cj * ci
    return T
