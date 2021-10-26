"""
Implementation of the Special Euclidean Group 2
"""

import random
import math
from functools import cached_property

import numpy as np
from liegroups.base import (
    LieGroupBase,
    Adjoint,
    Jacobian,
    OptionalJacobian,
    Tangent,
    Vector,
    eps,
)
from liegroups.util import normalize_range


class SE2(LieGroupBase):
    dof = 3
    dim = 2

    def __init__(self, x: float, y: float, theta: float):
        """
        Initialize the SE2 group element using the angle of rotation in Radians and translation in x and y

        Args:
            x: the translation distance from the origin on the X axis
            y: the translation distance from the origin on the Y axis
            theta: the angle of rotation in radians, the action of the group rotates a point counter clockwise by thetea angle about the origin
        """
        # Normalize the angle between (-PI, PI]
        theta = normalize_range(theta, -math.pi, math.pi)
        super().__init__(x, y, theta)

    @classmethod
    def identity(cls) -> LieGroupBase:
        """
        Return the identity of the group
        """
        return cls(0.0, 0.0, 0.0)

    @classmethod
    def random(cls) -> LieGroupBase:
        """
        Return the a random element of the group
        """
        return cls(
            random.uniform(-1, 1),
            random.uniform(-1, 1),
            random.uniform(-math.pi, math.pi),
        )

    @cached_property
    def rotation(self) -> np.ndarray:
        """
        Return the matrix representation of the rotation
        """
        theta = self.coeff[2]
        return np.array(
            [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)],]
        )

    @cached_property
    def translation(self) -> np.ndarray:
        """
        Return the matrix representation of the translation
        """
        x = self.coeff[0]
        y = self.coeff[1]
        return np.array([x, y])

    @cached_property
    def matrix(self) -> np.ndarray:
        """
        Return the matrix representation of the Lie group element

        See Eqs. (152)
        """
        matrix = np.identity(self.dof)
        matrix[0:2, 0:2] = self.rotation
        matrix[0:2, 2] = self.translation
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
        return cls(matrix[0][2], matrix[1][2], math.atan2(matrix[1][0], matrix[0][0]))

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

        trans_inv = -self.rotation.T @ self.translation
        return self.__class__(trans_inv[0], trans_inv[1], -self.coeff[2])

    def _compose(
        self,
        other: "SE2",
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
            skew = np.array([[0, -1], [1, 0]])
            J_mc_ma[2][2] = 1
            J_mc_ma[0:2, 0:2] = Rb.T
            J_mc_ma[0:2, 2] = Rb.T @ skew @ tb

        if J_mc_mb is not None:
            J_mc_mb[...] = np.identity(self.dof)

        m = np.identity(self.dof)
        m[0:2, 0:2] = Ra @ Rb
        m[0:2, 2] = ta + Ra @ tb

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
            skew = np.array([[0, -1], [1, 0]])
            J_vout_m[0:2, 0:2] = self.rotation
            J_vout_m[0:2, 2] = self.rotation @ skew @ vec

        if J_vout_v is not None:
            J_vout_v[...] = self.rotation

        return self.translation + self.rotation @ vec

    def rjac(self) -> Jacobian:
        """Compute the right jacobian of self

        See Eqs. (41) for general computation
        See Eqs. (126) for SO2 specific
        """
        p1, p2, theta = self.log().tolist()

        if math.isclose(theta, 0, abs_tol=eps):
            theta_sq = theta ** 2
            # Use taylor series expansion when theta is close to 0
            # See section 11 of https://ethaneade.com/lie_groups.pdf
            a = 1 - theta_sq / 6 * (1 - theta_sq / 20 * (1 - theta_sq / 42))
            b = (
                theta
                / 2
                * (1 - theta_sq / 12 * (1 - theta_sq / 30 * (1 - theta_sq / 56)))
            )
        else:
            a = math.sin(theta) / theta
            b = (1 - math.cos(theta)) / theta

        theta_sq = theta ** 2
        if math.isclose(theta_sq, 0, abs_tol=eps):
            # Use taylor series expansion when theta is close to 0
            # See section 11 of https://ethaneade.com/lie_groups.pdf
            c = (
                theta
                * 1
                / 6
                * (1 - theta_sq / 20 * (1 - theta_sq / 42 * (1 - theta_sq / 72)))
            )
            d = 1 / 2 * (1 - theta_sq / 12 * (1 - theta_sq / 30 * (1 - theta_sq / 56)))
        else:
            c = (theta - math.sin(theta)) / (theta_sq)
            d = (1 - math.cos(theta)) / (theta_sq)

        return np.array([[a, b, p1 * c - p2 * d], [-b, a, p1 * d + p2 * c], [0, 0, 1]])

    def rjacinv(self) -> Jacobian:
        """Compute the inverse of right jacobian of self
        """
        p1, p2, theta = self.log().tolist()

        theta_sq = theta * theta

        print(theta_sq)

        if math.isclose(theta_sq, 0, abs_tol=eps):
            a = 1 - theta_sq / 12 * (
                1 + theta_sq / 60 * (1 + theta_sq / 42 * (1 + theta_sq / 40))
            )
            b = 1 / 2
            c = (
                theta
                / 12
                * (1 + theta_sq / 60 * (1 + theta_sq / 42 * (1 + theta_sq / 40)))
            )
        else:
            a = -theta * math.sin(theta) / (2 * math.cos(theta) - 2)
            b = 1 / 2
            c = (theta * math.sin(theta) + 2 * math.cos(theta) - 2) / (
                2 * theta * (math.cos(theta) - 1)
            )

        return np.array(
            [
                [a, -theta / 2, p2 * b + p1 * c],
                [theta / 2, a, -p1 * b + p2 * c],
                [0, 0, 1],
            ]
        )

    @classmethod
    def exp(cls, tangent: Tangent, J_m_t: OptionalJacobian = None) -> LieGroupBase:
        """Compute the exponential map of the given tagent vector. The dimension of the vector should match the LieGroupBase.dof value

        See Eqs. (23)
        See Eqs. (156) for conversion between rho and t

        Args:
            J_m_t: Jacobian of the Lie group element wrt to the given tangent

        Returns:
            Exponential map of the tagent vector
        """
        theta = tangent[2]

        theta_sq = theta ** 2

        if math.isclose(theta, 0, abs_tol=eps):
            theta_sq = theta ** 2
            # Use taylor series expansion when theta is close to 0
            # See section 11 of https://ethaneade.com/lie_groups.pdf
            a = 1 - theta_sq / 6 * (1 - theta_sq / 20 * (1 - theta_sq / 42))
            b = (
                theta
                / 2
                * (1 - theta_sq / 12 * (1 - theta_sq / 30 * (1 - theta_sq / 56)))
            )
        else:
            a = math.sin(theta) / theta
            b = (1 - math.cos(theta)) / theta

        x = a * tangent[0] - b * tangent[1]
        y = b * tangent[0] + a * tangent[1]

        m = cls(x, y, theta)

        if J_m_t is not None:
            J_m_t[...] = m.rjac()

        return m

    def log(self, J_t_m: OptionalJacobian = None) -> Tangent:
        """Compute the tagent vector of the transformation, it is equivalent to the inverse of exponential map

        See Eqs. (24)

        The derivation here is obtained by setting [t_x, t_y] = V(theta) * [rho_1, rho_2] as shown in Eqs. (156)

        Args:
            J_t_m: Jacobian of the tagent wrt to self
        
        Returns:
            The log() map of self in vector form
        """
        theta = self.coeff[2]

        theta_sq = theta ** 2

        if math.isclose(theta, 0, abs_tol=eps):
            theta_sq = theta ** 2
            # Use taylor series expansion when theta is close to 0
            # See section 11 of https://ethaneade.com/lie_groups.pdf
            a = 1 - theta_sq / 6 * (1 - theta_sq / 20 * (1 - theta_sq / 42))
            b = (
                theta
                / 2
                * (1 - theta_sq / 12 * (1 - theta_sq / 30 * (1 - theta_sq / 56)))
            )
        else:
            a = math.sin(theta) / theta
            b = (1 - math.cos(theta)) / theta

        den = a * a + b * b

        a = a / den
        b = b / den

        if J_t_m is not None:
            J_t_m[...] = self.rjacinv()

        return np.array(
            [
                a * self.coeff[0] + b * self.coeff[1],
                -b * self.coeff[0] + a * self.coeff[1],
                self.coeff[2],
            ]
        )

    def adjoint(self) -> Adjoint:
        """Compute the adjoint of the transformation

        See Eqs. (29)
        See Eqs. (123) for SO2 specifics
        """
        adj = np.identity(self.dof)
        skew = np.array([[0, -1], [1, 0]])
        adj[0:2, 0:2] = self.rotation
        adj[0:2, 2] = -skew @ self.translation
        return adj
