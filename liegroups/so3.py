"""
Implementation of the Special Orthogonal Group 2
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
)
from liegroups.util import normalize_range, norm, uniform_sampling_n_ball_muller, clip

eps = 10 ** -8


class SO3(LieGroupBase):
    dof = 3
    dim = 3

    def __init__(self, theta1: float, theta2: float, theta3: float):
        """
        Initialize the SO3 group element theta = [theta1, theta2, theta3] which represents the integrated rotation in angle-axis form with
        angle ||theta|| and unit axis theta / ||theta||

        See Example 4 for more information

        Args:
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
            theta = (normalized_theta_norm * t / theta_norm for t in theta)

        super().__init__(*theta)

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
        theta = tuple(a * math.pi for a in uniform_sampling_n_ball_muller(3))
        return cls(*theta)

    @cached_property
    def matrix(self) -> np.ndarray:
        """
        Return the matrix representation of the Lie group element

        See Eqs. (134)
        """
        coeff = self.coeff
        w = np.array(coeff)
        theta = math.sqrt(w.dot(w))

        skew = np.array(
            [
                [0, -coeff[2], coeff[1]],
                [coeff[2], 0, -coeff[0]],
                [-coeff[1], coeff[0], 0],
            ]
        )

        if math.isclose(theta, 0, abs_tol=eps):
            theta_sq = theta ** 2
            # Use taylor series expansion when theta is close to 0
            # See section 11 of https://ethaneade.com/lie_groups.pdf
            a = 1 - theta_sq / 6 * (1 - theta_sq / 20 * (1 - theta_sq / 42))
            b = 1 / 2 * (1 - theta_sq / 12 * (1 - theta_sq / 30 * (1 - theta_sq / 56)))
        else:
            a = math.sin(theta) / theta
            b = (1 - math.cos(theta)) / theta ** 2

        return np.identity(3) + a * skew + b * (skew @ skew)

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> LieGroupBase:
        """
        Construct the Lie group element from its matrix representation

        This method does not validate whether or not the matrix is well formed.

        See Eqs. (135)

        Args:
            matrix: matrix representation of the SO3 group element.
        
        Return:
            The equivalent SO3 group
        """
        theta = math.acos(clip((np.trace(matrix) - 1) / 2, -1, 1))
        if math.isclose(theta, 0, abs_tol=eps):
            theta_sq = theta ** 2
            a_inv = 1 - theta_sq / 6 * (1 - theta_sq / 20 * (1 - theta_sq / 42))
            a = 1 / a_inv
        else:
            a = theta / math.sin(theta)
        params = (matrix - matrix.T) / 2 * a
        omega = tuple((params[2][1], params[0][2], params[1][0]))
        return cls(*omega)

    def inverse(self, J_minv_m: OptionalJacobian = None) -> LieGroupBase:
        """Returns the inverse of the this Lie Group Object instance
        
        See Eqs. (3) for general inverse
        See Eqs. (118) for inverse specific to the SO2 group
        See Eqs. (140) for the Jacobian of the inverse

        Args:
            J_minv_m: The Jacobian of the inverse with respect to self
        
        Returns:
            The inverese of self
        """
        if J_minv_m is not None:
            assert J_minv_m.shape == (self.dof, self.dof)
            J_minv_m[...] = -self.matrix

        return self.__class__(*tuple(-c for c in self.coeff))

    def _compose(
        self,
        other: LieGroupBase,
        J_mc_ma: OptionalJacobian = None,
        J_mc_mb: OptionalJacobian = None,
    ) -> LieGroupBase:
        """Returns the composition of self and another element of the same Lie group.

        See Eqs. (1,2,3,4)
        See Eqs. (141, 142) for implementation specific to SO3

        Args:
            other: Another element of the same Lie group
            J_mc_ma: The Jacobian of the composition wrt self
            J_mc_mb: The Jacobian of the composition wrt other
        
        Returns:
            The composition of self and other (self @ Other)
        """
        if J_mc_ma is not None:
            J_mc_ma[...] = other.matrix.T

        if J_mc_mb is not None:
            J_mc_mb[...] = np.identity(self.dof)

        return self.from_matrix(self.matrix @ other.matrix)

    def act(
        self,
        vec: Vector,
        J_vout_m: OptionalJacobian = None,
        J_vout_v: OptionalJacobian = None,
    ) -> Vector:
        """Perform the action of the group on a point in the vector space

        See Eqs. (150, 151)
    
        Args:
            vec: A point in the vector space
            J_vout_m: Jacobian of the output vector wrt to self
            J_vout_v: Jacobian of the output vector wrt to vec
        
        Returns:
            A point acted on by the group
        """
        if J_vout_m is not None:
            skew = np.array(
                [
                    [0, -self.coeff[2], self.coeff[1]],
                    [self.coeff[2], 0, -self.coeff[0]],
                    [-self.coeff[1], self.coeff[0], 0],
                ]
            )
            J_vout_m[...] = -self.matrix @ skew

        if J_vout_v is not None:
            J_vout_v[...] = self.matrix

        return self.matrix @ vec.T

    def rjac(self) -> Jacobian:
        """Compute the right jacobian of self

        See Eqs. (41) for general computation
        See Eqs. (143) for SO3 specific
        """
        skew = np.array(
            [
                [0, -self.coeff[2], self.coeff[1]],
                [self.coeff[2], 0, -self.coeff[0]],
                [-self.coeff[1], self.coeff[0], 0],
            ]
        )
        theta = norm(self.coeff)

        if math.isclose(theta, 0, abs_tol=eps):
            theta_sq = theta ** 2
            # Use taylor series expansion when theta is close to 0
            # See section 11 of https://ethaneade.com/lie_groups.pdf
            a = 1 / 2 * (1 - theta_sq / 12 * (1 - theta_sq / 30 * (1 - theta_sq / 56)))
            b = 1 / 6 * (1 - theta_sq / 20 * (1 - theta_sq / 42 * (1 - theta_sq / 72)))
        else:
            a = (1 - math.cos(theta)) / theta ** 2
            b = (theta - math.sin(theta)) / theta ** 3
        return np.identity(self.dof) - a * skew + b * skew @ skew

    def rjacinv(self) -> Jacobian:
        """Compute the right jacobian of self

        See Eqs. (41) for general computation
        See Eqs. (144) for SO3 specific
        """
        skew = np.array(
            [
                [0, -self.coeff[2], self.coeff[1]],
                [self.coeff[2], 0, -self.coeff[0]],
                [-self.coeff[1], self.coeff[0], 0],
            ]
        )
        theta = norm(self.coeff)

        if math.isclose(theta, 0, abs_tol=eps):
            a = 0
        else:
            a = 1 / theta ** 2 - (1 + math.cos(theta)) / (2 * theta * math.sin(theta))
        return np.identity(self.dof) + 0.5 * skew + a * skew @ skew

    @classmethod
    def exp(cls, tangent: Tangent, J_m_t: OptionalJacobian = None) -> LieGroupBase:
        """Compute the exponential map of the given tagent vector. The dimension of the vector should match the LieGroupBase.dof value

        See Eqs. (23)

        Args:
            J_m_t: Jacobian of the Lie group element wrt to the given tangent

        Returns:
            Exponential map of the tagent vector
        """
        m = cls(*tangent.tolist())

        if J_m_t is not None:
            J_m_t[...] = m.rjac()

        return m

    def log(self, J_t_m: OptionalJacobian = None) -> Tangent:
        """Compute the tagent vector of the transformation, it is equivalent to the inverse of exponential map

        See Eqs. (24)

        Args:
            J_t_m: Jacobian of the tagent wrt to self
        
        Returns:
            The log() map of self in vector form
        """
        if J_t_m is not None:
            J_t_m[...] = self.rjacinv()

        return np.array(self.coeff)

    def adjoint(self) -> Adjoint:
        """Compute the adjoint of the transformation

        See Eqs. (29)
        See Eqs. (139) for SO3 specifics
        """
        return self.matrix
