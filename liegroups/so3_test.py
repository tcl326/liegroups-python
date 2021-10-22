from numpy.core.numeric import identity
from numpy.testing import assert_almost_equal
import pytest

import math

import numpy as np

from liegroups.so3 import SO3
from liegroups.util import norm


def test_identity():
    identity = SO3(0.0, 0.0, 0.0)
    assert identity.coeff[0] == 0
    assert identity.coeff[1] == 0
    assert identity.coeff[2] == 0
    assert identity == SO3.identity()


def test_random():
    random = SO3.random()
    assert isinstance(random, SO3)
    assert random != SO3.random()

    assert norm(random.coeff) < math.pi
    assert norm(random.coeff) >= 0


def test_matrix():
    identity = SO3.identity()

    np.testing.assert_array_equal(identity.matrix, np.identity(3))

    random = SO3.random()
    random_matrix = random.matrix

    # Ensure that R @ R.T = I
    assert_almost_equal(random_matrix @ random_matrix.T, identity.matrix)


def test_from_matrix():
    identity = SO3.identity()

    assert SO3.from_matrix(identity.matrix) == identity

    # Known 3D rotation matrix around z-axis
    # See https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
    angle = math.pi / 3
    matrix = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    rotation = SO3.from_matrix(matrix)

    # Make sure that the coefficient corresponds to rotation of angle around z-axis
    assert_almost_equal(rotation.coeff[0], 0)
    assert_almost_equal(rotation.coeff[1], 0)
    assert_almost_equal(rotation.coeff[2], angle)


def test_inverse():
    random = SO3.random()
    random_inverse = random.inverse()
    identity = SO3.identity()

    # Ensures that the property R @ R ^ -1 = I holds

    np.testing.assert_almost_equal(
        random.matrix @ random_inverse.matrix, identity.matrix
    )

    j_minv_m = np.zeros((SO3.dof, SO3.dof))
    random_inverse = random.inverse(j_minv_m)

    np.testing.assert_almost_equal(j_minv_m, -random.matrix)


def test_compose():
    I = SO3.identity()

    x = SO3.random()
    y = SO3.random()
    z = SO3.random()

    x_inverse = x.inverse()

    # Test Identity
    assert x.almost_equal(I.compose(x))
    assert x.almost_equal(x.compose(I))

    # Test Inverse
    assert I == x.compose(x_inverse)
    assert I == x_inverse.compose(x)

    # Test Associativity
    xy_z = (x.compose(y)).compose(z)
    x_yz = x.compose(y.compose(z))

    # assert xy_z.almost_equal(x_yz)
    np.testing.assert_almost_equal(xy_z.matrix, x_yz.matrix)

    # Test jacobians

    j_mc_ma = np.zeros((SO3.dof, SO3.dof))
    j_mc_mb = np.zeros((SO3.dof, SO3.dof))

    c = x.compose(y, j_mc_ma, j_mc_mb)

    np.testing.assert_almost_equal(j_mc_ma, y.matrix.T)
    np.testing.assert_almost_equal(j_mc_mb, np.identity(SO3.dof))


def test_act():
    vec = np.array([0, 0, 1])

    zero_rotation = SO3(0, 0, 0)
    pi_2_rotation_x = SO3(math.pi / 2, 0, 0)
    random = SO3.random()

    # Test zero/identity rotation
    vout = zero_rotation.act(vec)
    np.testing.assert_almost_equal(vec, vout)

    # Test pi / 2 rotation around x axis
    vout = pi_2_rotation_x.act(vec)
    np.testing.assert_almost_equal(vout, np.array([0, -1, 0]))

    # Test random rotation
    vout = random.act(vec)
    # Ensures that vout is still on the unit circle
    assert_almost_equal(np.linalg.norm(vec), np.linalg.norm(vout))

    # Test jacobians
    j_vout_m = np.zeros((SO3.dim, SO3.dof))
    j_vout_v = np.zeros((SO3.dim, SO3.dim))

    _ = pi_2_rotation_x.act(vec, j_vout_m, j_vout_v)

    expected_j_vout_m = np.array([[0, 0, 0], [0, math.pi / 2, 0], [0, 0, math.pi / 2]])

    np.testing.assert_almost_equal(j_vout_m, expected_j_vout_m)
    np.testing.assert_almost_equal(j_vout_v, pi_2_rotation_x.matrix)


def test_rjac():
    random = SO3.random()
    assert random.rjac().shape == (SO3.dof, SO3.dof)


def test_exp():
    tangent = np.array([math.pi / 2, 0, 0])
    exp_map = SO3.exp(tangent)
    assert_almost_equal(list(exp_map.coeff), tangent.tolist())

    j_m_t = np.zeros((SO3.dof, SO3.dof))
    m = SO3.exp(tangent, j_m_t)

    np.testing.assert_almost_equal(j_m_t, m.rjac())


def test_log():
    random = SO3.random()
    tangent = random.log()
    assert_almost_equal(tangent.tolist(), list(random.coeff))

    j_t_m = np.zeros((SO3.dof, SO3.dof))
    _ = random.log(j_t_m)
    np.testing.assert_almost_equal(j_t_m, random.rjacinv())


def test_adjoint():
    m = SO3.random()
    adj = m.adjoint()
    np.testing.assert_almost_equal(adj, m.matrix)


def test_plus_minus_operator():
    tau = np.array([0.1, 0, 0])
    random = SO3.random()
    random_plus = random.plus(tau)
    diff = random_plus.minus(random)

    np.testing.assert_almost_equal(tau, diff)
