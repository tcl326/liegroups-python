from numpy.core.numeric import identity
from numpy.testing import assert_almost_equal
import pytest

import math

import numpy as np

from liegroups.so2 import SO2


def test_identity():
    identity = SO2(0.0)
    assert identity.coeff[0] == 0
    assert identity == SO2.identity()


def test_random():
    random = SO2.random()
    assert isinstance(random, SO2)
    assert random != SO2.random()

    assert random.coeff[0] <= math.pi
    assert random.coeff[0] >= -math.pi


def test_matrix():
    identity = SO2.identity()

    np.testing.assert_array_equal(identity.matrix, np.identity(2))

    random = SO2.random()
    random_matrix = random.matrix

    assert_almost_equal(random_matrix[0][0], math.cos(random.coeff[0]))
    assert_almost_equal(random_matrix[0][1], -math.sin(random.coeff[0]))
    assert_almost_equal(random_matrix[1][0], math.sin(random.coeff[0]))


def test_from_matrix():
    identity = SO2.identity()

    assert SO2.from_matrix(identity.matrix) == identity

    # Test to see that the angle is normalized between [-PI, PI]
    angle = math.pi / 2 + 2 * math.pi
    rotation = SO2(angle)

    assert_almost_equal(SO2.from_matrix(rotation.matrix).coeff[0], angle - 2 * math.pi)


def test_inverse():
    random = SO2.random()
    random_inverse = random.inverse()
    identity = SO2.identity()

    # Ensures that the property R @ R ^ -1 = I holds

    np.testing.assert_almost_equal(
        random.matrix @ random_inverse.matrix, identity.matrix
    )

    j_minv_m = np.zeros((SO2.dof, SO2.dof))
    random_inverse = random.inverse(j_minv_m)

    assert j_minv_m[0][0] == -1


def test_compose():
    I = SO2.identity()

    x = SO2.random()
    y = SO2.random()
    z = SO2.random()

    x_inverse = x.inverse()

    # Test Identity
    assert x == I.compose(x)
    assert x == x.compose(I)

    # Test Inverse
    assert I == x.compose(x_inverse)
    assert I == x_inverse.compose(x)

    # Test Associativity
    xy_z = (x.compose(y)).compose(z)
    x_yz = x.compose(y.compose(z))

    assert xy_z.almost_equal(x_yz)

    # Test jacobians

    j_mc_ma = np.zeros((SO2.dof, SO2.dof))
    j_mc_mb = np.zeros((SO2.dof, SO2.dof))

    c = x.compose(y, j_mc_ma, j_mc_mb)

    assert j_mc_ma[0][0] == 1
    assert j_mc_mb[0][0] == 1


def test_act():
    vec = np.array([1, 0])

    zero_rotation = SO2(0)
    pi_2_rotation = SO2(math.pi / 2)
    random = SO2.random()

    # Test zero/identity rotation
    vout = zero_rotation.act(vec)
    np.testing.assert_almost_equal(vec, vout)

    # Test pi / 2 rotation
    vout = pi_2_rotation.act(vec)
    np.testing.assert_almost_equal(vout, np.array([0, 1]))

    # Test random rotation
    vout = random.act(vec)
    # Ensures that vout is still on the unit circle
    assert_almost_equal(np.linalg.norm(vec), np.linalg.norm(vout))

    # Test jacobians
    j_vout_m = np.zeros((SO2.dim, SO2.dof))
    j_vout_v = np.zeros((SO2.dim, SO2.dim))

    _ = pi_2_rotation.act(vec, j_vout_m, j_vout_v)

    np.testing.assert_almost_equal(j_vout_m, np.array([[-1, 0]]).T)
    np.testing.assert_almost_equal(j_vout_v, np.array([[0, -1], [1, 0]]))


def test_rjac():
    random = SO2.random()
    assert random.rjac()[0][0] == 1


def test_exp():
    tangent = np.array([math.pi / 2])
    exp_map = SO2.exp(tangent)
    assert_almost_equal(exp_map.coeff[0], tangent[0])

    j_m_t = np.zeros((SO2.dof, SO2.dof))
    _ = SO2.exp(tangent, j_m_t)

    assert j_m_t[0][0] == 1


def test_log():
    random = SO2.random()
    tangent = random.log()
    assert_almost_equal(tangent[0], random.coeff[0])

    j_t_m = np.zeros((SO2.dof, SO2.dof))
    _ = random.log(j_t_m)
    assert j_t_m[0][0] == 1


def test_adjoint():
    adj = SO2.random().adjoint()
    assert adj[0][0] == 1


def test_plus_minus_operator():
    tau = np.array([0.1])
    random = SO2.random()
    random_plus = random.plus(tau)
    diff = random_plus.minus(random)

    np.testing.assert_almost_equal(tau, diff)
