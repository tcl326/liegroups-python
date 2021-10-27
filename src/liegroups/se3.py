"""
Implementation of the Special Euclidean Group 3
"""

import random
import math
from functools import cached_property

import numpy as np
import numpy.typing as npt
from liegroups.base import (
    LieGroupBase,
    Adjoint,
    Jacobian,
    OptionalJacobian,
    Tangent,
    Vector,
    eps,
)
from liegroups.so3 import SO3
from liegroups.util import normalize_range, norm, uniform_sampling_n_ball_muller, clip


class SE3(LieGroupBase):
    dof = 6
    dim = 3

    def __init__(
        self, x: float, y: float, z: float, theta1: float, theta2: float, theta3: float
    ):
        """
        Initialize the SE2 group element using the angle of rotation in Radians and translation in x and y

        Args:
            x: the translation distance from the origin on the X axis
            y: the translation distance from the origin on the Y axis
            z: the translation distance from the origin on the Z axis
            theta1: first element of the theta vector 
            theta2: second element of the theta vector
            theta3: third element of theta vector
        """
        # Normalize the theta vector so ||theta|| is between (0, PI]
        theta = (
            theta1,
            theta2,
            theta3,
        )
        theta_norm = norm(theta)
        normalized_theta_norm = normalize_range(theta_norm, 0, math.pi)
        if theta_norm != normalized_theta_norm:
            theta = tuple(normalized_theta_norm * t / theta_norm for t in theta)

        super().__init__(x, y, z, *theta)
        self.so3 = SO3(*theta)

    @classmethod
    def identity(cls) -> LieGroupBase:
        """
        Return the identity of the group
        """
        return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    @classmethod
    def random(cls) -> LieGroupBase:
        """
        Return the a random element of the group
        """
        theta = SO3.random().coeff
        return cls(
            random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), *theta,
        )

    @cached_property
    def rotation(self) -> np.ndarray:
        """
        Return the matrix representation of the rotation
        """
        return self.so3.matrix

    @cached_property
    def translation(self) -> np.ndarray:
        """
        Return the matrix representation of the translation
        """
        x = self.coeff[0]
        y = self.coeff[1]
        z = self.coeff[2]
        return np.array([x, y, z])

    @cached_property
    def matrix(self) -> np.ndarray:
        """
        Return the matrix representation of the Lie group element

        See Eqs. (152)
        """
        matrix = np.identity(self.dim + 1)
        matrix[0:3, 0:3] = self.rotation
        matrix[0:3, 3] = self.translation
        return matrix

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> LieGroupBase:
        """
        Construct the Lie group element from its matrix representation

        This method does not validate whether or not the matrix is well formed.

        See Eqs. (152)

        Args:
            matrix: matrix representation of the SO2 group element.
        
        Return:
            The equivalent SO2 group
        """
        theta = SO3.from_matrix(matrix[0:3, 0:3]).coeff
        return cls(matrix[0][3], matrix[1][3], matrix[2][3], *theta)

    def inverse(self, J_minv_m: OptionalJacobian = None) -> LieGroupBase:
        """Returns the inverse of the this Lie Group Object instance
        
        See Eqs. (3) for general inverse
        See Eqs. (154) for inverse specific to the SE2 group
        See Eqs. (160) for the Jacobian of the inverse

        Args:
            J_minv_m: The Jacobian of the inverse with respect to self
        
        Returns:
            The inverese of self
        """
        if J_minv_m is not None:
            assert J_minv_m.shape == (self.dof, self.dof)
            J_minv_m[...] = -self.adjoint()

        trans_inv = (-self.rotation.T @ self.translation).tolist()
        return self.__class__(
            trans_inv[0],
            trans_inv[1],
            trans_inv[2],
            -self.coeff[3],
            -self.coeff[4],
            -self.coeff[5],
        )

    def _compose(
        self,
        other: "SE3",
        J_mc_ma: OptionalJacobian = None,
        J_mc_mb: OptionalJacobian = None,
    ) -> LieGroupBase:
        """Returns the composition of self and another element of the same Lie group.

        See Eqs. (1,2,3,4)
        See Eqs. (155) for implementation specific to SE2
        See Eqs. (161, 162) for Jacobian implementation

        Args:
            other: Another element of the same Lie group
            J_mc_ma: The Jacobian of the composition wrt self
            J_mc_mb: The Jacobian of the composition wrt other
        
        Returns:
            The composition of self and other (self @ Other)
        """
        Ra = self.rotation
        ta = self.translation

        Rb = other.rotation
        tb = other.translation

        if J_mc_ma is not None:
            x, y, z = tb.tolist()
            skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            J_mc_ma[3][3] = 1
            J_mc_ma[0:3, 0:3] = Rb.T
            J_mc_ma[3:6, 3:6] = Rb.T
            J_mc_ma[0:3, 3:6] = -Rb.T @ skew

        if J_mc_mb is not None:
            J_mc_mb[...] = np.identity(self.dof)

        m = np.identity(self.dim + 1)
        m[0:3, 0:3] = Ra @ Rb
        m[0:3, 3] = ta + Ra @ tb

        return self.__class__.from_matrix(m)

    def act(
        self,
        vec: Vector,
        J_vout_m: OptionalJacobian = None,
        J_vout_v: OptionalJacobian = None,
    ) -> Vector:
        """Perform the action of the group on a point in the vector space

        See Eqs. (165, 166, 167)

        Args:
            vec: A point in the vector space
            J_vout_m: Jacobian of the output vector wrt to self
            J_vout_v: Jacobian of the output vector wrt to vec
        
        Returns:
            A point acted on by the group
        """
        if J_vout_m is not None:
            x, y, z = vec.tolist()
            skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            J_vout_m[0:3, 0:3] = self.rotation
            J_vout_m[0:3, 3:6] = -self.rotation @ skew

        if J_vout_v is not None:
            J_vout_v[...] = self.rotation

        return self.translation + self.rotation @ vec

    @staticmethod
    def q_matrix(p1, p2, p3, theta1, theta2, theta3) -> npt.NDArray:

        theta = norm((theta1, theta2, theta3))
        theta_sq = theta * theta

        if math.isclose(theta, 0, abs_tol=eps):
            # See https://ethaneade.com/lie_groups.pdf Eqs. (158, 160)
            a = 0.5
            b = 1 / 6 * (1 - theta_sq / 20 * (1 - theta_sq / 42 * (1 - theta_sq / 72)))
            c = 1 / 24 * (1 - theta_sq / 30 * (1 - theta_sq / 56 * (1 - theta_sq / 90)))
            d = (
                1
                / 120
                * (1 - theta_sq / 42 * (1 - theta_sq / 72 * (1 - theta_sq / 110)))
            )
            d = 1 / 2 * (c - 3 * d)
        else:
            a = 0.5
            b = (theta - math.sin(theta)) / (theta_sq * theta)
            c = (1 - theta_sq / 2 - math.cos(theta)) / (theta_sq * theta_sq)
            d = (
                1
                / 2
                * (
                    c
                    - 3
                    * (theta - math.sin(theta) - theta_sq * theta / 6)
                    / (theta_sq * theta_sq * theta)
                )
            )

        p_x = np.array([[0, -p3, p2], [p3, 0, -p1], [-p2, p1, 0]])
        theta_x = np.array(
            [[0, -theta3, theta2], [theta3, 0, -theta1], [-theta2, theta1, 0]]
        )
        theta_x_sq = theta_x @ theta_x
        return (
            a * p_x
            + b * (theta_x @ p_x + p_x @ theta_x + theta_x @ p_x @ theta_x)
            - c * (theta_x_sq @ p_x + p_x @ theta_x_sq - 3 * theta_x @ p_x @ theta_x)
            - d * (theta_x @ p_x @ theta_x_sq + theta_x_sq @ p_x @ theta_x)
        )

    def rjac(self) -> Jacobian:
        """Compute the right jacobian of self

        See Eqs. (41) for general computation
        See Eqs. (179a) for SE3 specific

        Remember J_r(theta) = J_l(-theta) where J_r(theta) and J_l(theta) are the left and right jacobian of the SO3 group
        """
        jacobian = np.identity(self.dof)

        jacobian[0:3, 0:3] = self.so3.rjac()
        jacobian[3:6, 3:6] = self.so3.rjac()

        jacobian[0:3, 3:6] = SE3.q_matrix(*(-self.log()).tolist())

        return jacobian

    def rjacinv(self) -> Jacobian:
        """Compute the inverse of right jacobian of self

        See Eqs. (179b)
        """
        jacobian = np.identity(self.dof)

        jacobian[0:3, 0:3] = self.so3.rjacinv()
        jacobian[3:6, 3:6] = self.so3.rjacinv()

        jacobian[0:3, 3:6] = (
            -self.so3.rjacinv()
            @ SE3.q_matrix(*(-self.log()).tolist())
            @ self.so3.rjacinv()
        )

        return jacobian

    @classmethod
    def exp(cls, tangent: Tangent, J_m_t: OptionalJacobian = None) -> LieGroupBase:
        """Compute the exponential map of the given tagent vector. The dimension of the vector should match the LieGroupBase.dof value

        See Eqs. (23)
        See Eqs. (173) for conversion between rho and t

        Args:
            J_m_t: Jacobian of the Lie group element wrt to the given tangent

        Returns:
            Exponential map of the tagent vector
        """
        p1, p2, p3, theta1, theta2, theta3 = tangent.tolist()
        so3 = SO3(theta1, theta2, theta3)

        x, y, z = (so3.rjac().T @ np.array([p1, p2, p3])).tolist()

        m = cls(x, y, z, theta1, theta2, theta3)

        if J_m_t is not None:
            J_m_t[...] = m.rjac()

        return m

    def log(self, J_t_m: OptionalJacobian = None) -> Tangent:
        """Compute the tagent vector of the transformation, it is equivalent to the inverse of exponential map

        See Eqs. (24)
        See Eqs. (173) for SE3 specific implementation

        Args:
            J_t_m: Jacobian of the tagent wrt to self
        
        Returns:
            The log() map of self in vector form
        """

        if J_t_m is not None:
            J_t_m[...] = self.rjacinv()

        p1, p2, p3 = (self.so3.rjacinv().T @ self.translation).tolist()

        return np.array([p1, p2, p3, self.coeff[3], self.coeff[4], self.coeff[5]])

    def adjoint(self) -> Adjoint:
        """Compute the adjoint of the transformation

        See Eqs. (29)
        See Eqs. (123) for SO2 specifics
        """
        adj = np.identity(self.dof)
        x, y, z = self.translation.tolist()
        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        adj[0:3, 0:3] = self.rotation
        adj[0:3, 3:6] = skew @ self.rotation
        adj[3:6, 3:6] = self.rotation
        return adj
