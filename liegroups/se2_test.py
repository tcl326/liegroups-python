from numpy.core.numeric import identity
from numpy.testing import assert_almost_equal
import pytest

import math

import numpy as np

from liegroups.se2 import SE2


def test_identity():
    identity = SE2(0.0, 0.0, 0.0)
    assert identity.coeff[0] == 0
    assert identity.coeff[1] == 0
    assert identity.coeff[2] == 0
    assert identity == SE2.identity()


def test_random():
    random = SE2.random()
    assert isinstance(random, SE2)
    assert random != SE2.random()

    assert random.coeff[2] <= math.pi
    assert random.coeff[2] >= -math.pi
    assert random.coeff[0] <= 1
    assert random.coeff[0] >= -1
    assert random.coeff[1] <= 1
    assert random.coeff[1] >= -1


def test_matrix():
    identity = SE2.identity()

    np.testing.assert_array_equal(identity.matrix, np.identity(3))

    random = SE2.random()
    random_matrix = random.matrix

    assert_almost_equal(random_matrix[0][0], math.cos(random.coeff[2]))
    assert_almost_equal(random_matrix[0][1], -math.sin(random.coeff[2]))
    assert_almost_equal(random_matrix[1][0], math.sin(random.coeff[2]))
    assert_almost_equal(random_matrix[0][2], random.coeff[0])
    assert_almost_equal(random_matrix[1][2], random.coeff[1])
    assert_almost_equal(random_matrix[2][2], 1)


def test_from_matrix():
    identity = SE2.identity()

    assert SE2.from_matrix(identity.matrix) == identity

    # Test to see that the angle is normalized between [-PI, PI]
    angle = math.pi / 2 + 2 * math.pi
    rotation = SE2(0, 0, angle)

    assert_almost_equal(SE2.from_matrix(rotation.matrix).coeff[2], angle - 2 * math.pi)


def test_inverse():
    random = SE2.random()
    random_inverse = random.inverse()
    identity = SE2.identity()

    # Ensures that the property R @ R ^ -1 = I holds

    np.testing.assert_almost_equal(
        random.matrix @ random_inverse.matrix, identity.matrix
    )

    j_minv_m = np.zeros((SE2.dof, SE2.dof))
    random_inverse = random.inverse(j_minv_m)

    assert j_minv_m[2][2] == -1
    assert j_minv_m[0][2] == -random.coeff[1]
    assert j_minv_m[1][2] == random.coeff[0]
    np.testing.assert_almost_equal(j_minv_m[0:2, 0:2], -random.rotation)


def test_compose():
    I = SE2.identity()

    x = SE2.random()
    y = SE2.random()
    z = SE2.random()

    x_inverse = x.inverse()

    # Test Identity
    assert x == I.compose(x)
    assert x == x.compose(I)

    # Test Inverse
    assert I.almost_equal(x.compose(x_inverse))
    assert I.almost_equal(x_inverse.compose(x))

    # Test Associativity
    xy_z = (x.compose(y)).compose(z)
    x_yz = x.compose(y.compose(z))

    assert xy_z.almost_equal(x_yz)

    # Test jacobians

    j_mc_ma = np.zeros((SE2.dof, SE2.dof))
    j_mc_mb = np.zeros((SE2.dof, SE2.dof))

    c = x.compose(y, j_mc_ma, j_mc_mb)

    np.testing.assert_almost_equal(np.identity(SE2.dof), j_mc_ma @ y.adjoint())
    np.testing.assert_almost_equal(j_mc_mb, np.identity(SE2.dof))


def test_act():
    vec = np.array([1, 0])

    zero_rotation = SE2(0, 0, 0)
    pi_2_rotation = SE2(0, 0, math.pi / 2)
    random = SE2.random()

    # Test zero/identity rotation
    vout = zero_rotation.act(vec)
    np.testing.assert_almost_equal(vec, vout)

    # Test pi / 2 rotation
    vout = pi_2_rotation.act(vec)
    np.testing.assert_almost_equal(vout, np.array([0, 1]))

    # Test random rotation
    vout = random.act(vec)
    # Ensures that inverse of random obtains the same vec
    assert_almost_equal(vec, random.inverse().act(vout))

    # Test jacobians
    j_vout_m = np.zeros((SE2.dim, SE2.dof))
    j_vout_v = np.zeros((SE2.dim, SE2.dim))

    _ = pi_2_rotation.act(vec, j_vout_m, j_vout_v)

    np.testing.assert_almost_equal(j_vout_m, np.array([[0, -1, -1], [1, 0, 0]]))
    np.testing.assert_almost_equal(j_vout_v, np.array([[0, -1], [1, 0]]))


# def test_rjac():
#     random = SO2.random()
#     assert random.rjac()[0][0] == 1


# def test_exp():
#     tangent = np.array([math.pi / 2])
#     exp_map = SO2.exp(tangent)
#     assert_almost_equal(exp_map.coeff[0], tangent[0])

#     j_m_t = np.zeros((SO2.dof, SO2.dof))
#     _ = SO2.exp(tangent, j_m_t)

#     assert j_m_t[0][0] == 1


# def test_log():
#     random = SO2.random()
#     tangent = random.log()
#     assert_almost_equal(tangent[0], random.coeff[0])

#     j_t_m = np.zeros((SO2.dof, SO2.dof))
#     _ = random.log(j_t_m)
#     assert j_t_m[0][0] == 1


# def test_adjoint():
#     adj = SO2.random().adjoint()
#     assert adj[0][0] == 1


# def test_plus_minus_operator():
#     tau = np.array([0.1])
#     random = SO2.random()
#     random_plus = random.plus(tau)
#     diff = random_plus.minus(random)

#     np.testing.assert_almost_equal(tau, diff)
