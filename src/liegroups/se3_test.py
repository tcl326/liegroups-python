from numpy.core.numeric import identity
from numpy.testing import assert_almost_equal
import pytest

import math

import numpy as np

from liegroups.se3 import SE3
from liegroups.util import normalize_range, norm


def test_identity():
    identity = SE3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert identity.coeff[0] == 0
    assert identity.coeff[1] == 0
    assert identity.coeff[2] == 0
    assert identity.coeff[3] == 0
    assert identity.coeff[4] == 0
    assert identity.coeff[5] == 0
    assert identity == SE3.identity()


def test_random():
    random = SE3.random()
    assert isinstance(random, SE3)
    assert random != SE3.random()

    assert random.coeff[0] <= 1
    assert random.coeff[0] >= -1
    assert random.coeff[1] <= 1
    assert random.coeff[1] >= -1
    assert norm(random.coeff[3:]) < math.pi
    assert norm(random.coeff[3:]) >= 0


def test_matrix():
    identity = SE3.identity()

    np.testing.assert_array_equal(identity.matrix, np.identity(4))

    random = SE3.random()
    random_matrix = random.matrix

    assert_almost_equal(
        random_matrix[0:3, 0:3] @ random_matrix[0:3, 0:3].T, identity.matrix[0:3, 0:3]
    )
    assert random.matrix[0][3] == random.coeff[0]
    assert random.matrix[1][3] == random.coeff[1]
    assert random.matrix[2][3] == random.coeff[2]


def test_from_matrix():
    identity = SE3.identity()

    assert SE3.from_matrix(identity.matrix) == identity

    # Known 3D rigid transformation matrix with rotation around z-axis
    # See https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
    angle = math.pi / 3
    x = 1
    y = 2
    z = 3
    matrix = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0, x],
            [math.sin(angle), math.cos(angle), 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ]
    )
    transformation = SE3.from_matrix(matrix)

    # Make sure that the coefficient corresponds to rotation of angle around z-axis
    assert_almost_equal(transformation.coeff[0], 1)
    assert_almost_equal(transformation.coeff[1], 2)
    assert_almost_equal(transformation.coeff[2], 3)
    assert_almost_equal(transformation.coeff[3], 0)
    assert_almost_equal(transformation.coeff[4], 0)
    assert_almost_equal(transformation.coeff[5], angle)


def test_inverse():
    random = SE3.random()
    random_inverse = random.inverse()
    identity = SE3.identity()

    # Ensures that the property R @ R ^ -1 = I holds

    np.testing.assert_almost_equal(
        random.matrix @ random_inverse.matrix, identity.matrix
    )

    j_minv_m = np.zeros((SE3.dof, SE3.dof))
    random_inverse = random.inverse(j_minv_m)

    np.testing.assert_almost_equal(j_minv_m[0:3, 0:3], -random.rotation)
    np.testing.assert_almost_equal(j_minv_m[3:6, 3:6], -random.rotation)
    np.testing.assert_almost_equal(j_minv_m[3:6, 0:3], np.zeros((3, 3,)))


def test_compose():
    I = SE3.identity()

    x = SE3.random()
    y = SE3.random()
    z = SE3.random()

    x_inverse = x.inverse()

    # Test Identity
    assert x.almost_equal(I.compose(x))
    assert x.almost_equal(x.compose(I))

    # Test Inverse
    assert I.almost_equal(x.compose(x_inverse))
    assert I.almost_equal(x_inverse.compose(x))

    # Test Associativity
    xy_z = (x.compose(y)).compose(z)
    x_yz = x.compose(y.compose(z))

    assert xy_z.almost_equal(x_yz)

    # Test jacobians

    j_mc_ma = np.zeros((SE3.dof, SE3.dof))
    j_mc_mb = np.zeros((SE3.dof, SE3.dof))

    c = x.compose(y, j_mc_ma, j_mc_mb)

    np.testing.assert_almost_equal(np.identity(SE3.dof), j_mc_ma @ y.adjoint())
    np.testing.assert_almost_equal(j_mc_mb, np.identity(SE3.dof))


def test_act():
    vec = np.array([1, 0, 0])

    zero_rotation = SE3(0, 0, 0, 0, 0, 0)
    pi_2_rotation = SE3(0, 0, 0, 0, 0, math.pi / 2)
    random = SE3.random()

    # Test zero/identity rotation
    vout = zero_rotation.act(vec)
    np.testing.assert_almost_equal(vec, vout)

    # Test pi / 2 rotation
    vout = pi_2_rotation.act(vec)
    np.testing.assert_almost_equal(vout, np.array([0, 1, 0]))

    # Test random rotation
    vout = random.act(vec)
    # Ensures that inverse of random obtains the same vec
    assert_almost_equal(vec, random.inverse().act(vout))

    # Test jacobians
    j_vout_m = np.zeros((SE3.dim, SE3.dof))
    j_vout_v = np.zeros((SE3.dim, SE3.dim))

    _ = pi_2_rotation.act(vec, j_vout_m, j_vout_v)

    np.testing.assert_almost_equal(
        j_vout_m,
        np.array([[0, -1, 0, 0, 0, -1], [1, 0, 0, 0, 0, 0], [0, 0, 1, 0, -1, 0]]),
    )
    np.testing.assert_almost_equal(j_vout_v, pi_2_rotation.rotation)


def test_rjac():
    random = SE3.random()
    rjac = random.rjac()
    assert rjac.shape == (SE3.dof, SE3.dof)


def test_rjacinv():
    random = SE3.random()
    rjac = random.rjac()
    rjacinv = random.rjacinv()
    np.testing.assert_almost_equal(rjac @ rjacinv, np.identity(SE3.dof))


def test_exp():
    tangent = np.array([1, 1, 1, 0, 0, math.pi / 2])
    exp_map = SE3.exp(tangent)
    assert_almost_equal(exp_map.log()[0], tangent[0])
    assert_almost_equal(exp_map.log()[1], tangent[1])
    assert_almost_equal(exp_map.log()[2], tangent[2])

    j_m_t = np.zeros((SE3.dof, SE3.dof))
    m = SE3.exp(tangent, j_m_t)

    np.testing.assert_almost_equal(m.rjac(), j_m_t)


def test_log():
    random = SE3.random()
    tangent = random.log()
    assert random.almost_equal(SE3.exp(tangent))

    j_t_m = np.zeros((SE3.dof, SE3.dof))
    _ = random.log(j_t_m)
    np.testing.assert_almost_equal(random.rjacinv(), j_t_m)


def test_adjoint():
    random = SE3.random()
    adj = random.adjoint()
    np.testing.assert_almost_equal(adj[0:3, 0:3], random.rotation)


def test_plus_minus_operator():
    tau = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    random = SE3.random()
    random_plus = random.plus(tau)
    diff = random_plus.minus(random)

    np.testing.assert_almost_equal(tau, diff)
