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


class SO2(LieGroupBase):
    dof = 1
    dim = 2

    def __init__(self, theta: float):
        """
        Initialize the SO2 group element using the angle of rotation in Radians

        Args:
            theta: the angle of rotation in radians, the action of the group rotates a point counter clockwise by thetea angle about the origin
        """
        # Normalize the angle between (-PI, PI]
        # See: https://stackoverflow.com/questions/24234609/standard-way-to-normalize-an-angle-to-%CF%80-radians-in-java/24234924

        theta = theta - 2 * math.pi * math.floor((theta + math.pi) / (2 * math.pi))
        super().__init__(theta)

    @classmethod
    def identity(cls) -> LieGroupBase:
        """
        Return the identity of the group
        """
        return cls(0.0)

    @classmethod
    def random(cls) -> LieGroupBase:
        """
        Return the a random element of the group
        """
        return cls(random.uniform(-math.pi, math.pi))

    @cached_property
    def matrix(self) -> np.ndarray:
        """
        Return the matrix representation of the Lie group element

        See Eqs. (116)
        """
        theta = self.coeff[0]
        return np.array(
            [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)],]
        )

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> LieGroupBase:
        """
        Construct the Lie group element from its matrix representation

        This method does not validate whether or not the matrix is well formed.

        See Eqs. (113)

        Args:
            matrix: matrix representation of the SO2 group element.
        
        Return:
            The equivalent SO2 group
        """
        return cls(math.atan2(matrix[1][0], matrix[0][0]))

    def inverse(self, J_minv_m: OptionalJacobian = None) -> LieGroupBase:
        """Returns the inverse of the this Lie Group Object instance
        
        See Eqs. (3) for general inverse
        See Eqs. (118) for inverse specific to the SO2 group
        See Eqs. (124) for the Jacobian of the inverse

        Args:
            J_minv_m: The Jacobian of the inverse with respect to self
        
        Returns:
            The inverese of self
        """
        if J_minv_m is not None:
            assert J_minv_m.shape == (self.dof, self.dof)
            J_minv_m[0][0] = -1

        return self.__class__(-self.coeff[0])

    def _compose(
        self,
        other: LieGroupBase,
        J_mc_ma: OptionalJacobian = None,
        J_mc_mb: OptionalJacobian = None,
    ) -> LieGroupBase:
        """Returns the composition of self and another element of the same Lie group.

        See Eqs. (1,2,3,4)
        See Eqs. (118) for implementation specific to SO2

        Args:
            other: Another element of the same Lie group
            J_mc_ma: The Jacobian of the composition wrt self
            J_mc_mb: The Jacobian of the composition wrt other
        
        Returns:
            The composition of self and other (self @ Other)
        """
        if J_mc_ma is not None:
            J_mc_ma[0][0] = 1

        if J_mc_mb is not None:
            J_mc_mb[0][0] = 1

        return self.__class__(self.coeff[0] + other.coeff[0])

    def act(
        self,
        vec: Vector,
        J_vout_m: OptionalJacobian = None,
        J_vout_v: OptionalJacobian = None,
    ) -> Vector:
        """Perform the action of the group on a point in the vector space

        Args:
            vec: A point in the vector space
            J_vout_m: Jacobian of the output vector wrt to self
            J_vout_v: Jacobian of the output vector wrt to vec
        
        Returns:
            A point acted on by the group
        """
        if J_vout_m is not None:
            skew = np.array([[0, 1], [1, 0]])
            J_vout_m[...] = self.matrix @ skew @ vec.reshape(2, 1)

        if J_vout_v is not None:
            J_vout_v[...] = self.matrix

        return self.matrix @ vec.T

    def rjac(self) -> Jacobian:
        """Compute the right jacobian of self

        See Eqs. (41) for general computation
        See Eqs. (126) for SO2 specific
        """
        return np.array([[1]])

    def rjacinv(self) -> Jacobian:
        """Compute the inverse of right jacobian of self
        """
        return self.rjac()

    @classmethod
    def exp(cls, tangent: Tangent, J_m_t: OptionalJacobian = None) -> LieGroupBase:
        """Compute the exponential map of the given tagent vector. The dimension of the vector should match the LieGroupBase.dof value

        See Eqs. (23)

        Args:
            J_m_t: Jacobian of the Lie group element wrt to the given tangent

        Returns:
            Exponential map of the tagent vector
        """
        m = cls(tangent[0])

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
            J_t_m[0][0] = 1

        return np.array(self.coeff)

    def adjoint(self) -> Adjoint:
        """Compute the adjoint of the transformation

        See Eqs. (29)
        See Eqs. (123) for SO2 specifics
        """
        return np.array([[1]])
