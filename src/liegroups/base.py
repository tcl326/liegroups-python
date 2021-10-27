"""
Defines the basic interface of Lie group elements
"""


from __future__ import annotations  # For forward declaration of type hints

import math
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import numpy.typing as npt

from liegroups.exceptions import LieGroupMismatch

Adjoint = npt.NDArray
Vector = npt.ArrayLike
Tangent = npt.ArrayLike
Jacobian = npt.NDArray
OptionalJacobian = Optional[Jacobian]

eps = 10 ** -9


class LieGroupBase(ABC):
    """
    The Base class that defines the basic interface to Lie groups

    The Eqs. here refers to the Eqs. listed in the PDF A Micro Lie Theory for State Estimation in Robotics by Sola et al.
    https://arxiv.org/pdf/1812.01537.pdf
    """

    def __init__(self, *params):
        """
        Initialize a Lie group element using easy to understand parameters
        """
        self._p = params

    @property
    def coeff(self):
        return self._p

    @property
    @classmethod
    @abstractmethod
    def dof(cls) -> int:
        """
        The degrees of freedom of the Lie group. This is equivalent to the dimension of the vector representation of the associated Lie algebra
        """
        pass

    @property
    @classmethod
    @abstractmethod
    def dim(cls) -> int:
        """
        The dimension of the underlying square matrix representation
        """
        pass

    @classmethod
    @abstractmethod
    def identity(cls) -> LieGroupBase:
        """
        Return the identity of the group
        """
        pass

    @classmethod
    @abstractmethod
    def random(cls) -> LieGroupBase:
        """
        Return the a random element of the group
        """
        pass

    @property
    @abstractmethod
    def matrix(self) -> np.ndarray:
        """
        Return the matrix representation of the Lie group element
        """
        pass

    @classmethod
    @abstractmethod
    def from_matrix(cls, matrix: np.ndarray) -> LieGroupBase:
        """
        Construct the Lie group element from its matrix representation
        """
        pass

    @abstractmethod
    def inverse(self, J_minv_m: OptionalJacobian = None) -> LieGroupBase:
        """Returns the inverse of the this Lie Group Object instance
        
        See Eqs. (3)

        Args:
            J_minv_m: The Jacobian of the inverse with respect to self
        
        Returns:
            The inverese of self
        """
        pass

    def compose(
        self,
        other: LieGroupBase,
        J_mc_ma: OptionalJacobian = None,
        J_mc_mb: OptionalJacobian = None,
    ) -> LieGroupBase:
        """Returns the composition of self and another element of the same Lie group.

        See Eqs. (1,2,3,4)

        Args:
            other: Another element of the same Lie group
            J_mc_ma: The Jacobian of the composition wrt self
            J_mc_mb: The Jacobian of the composition wrt other
        
        Returns:
            The composition of self and other (self @ Other)
        """
        if self.__class__ != other.__class__:
            raise LieGroupMismatch(
                f"The compose method operates on two elements of the same group, got {self.__class__.__name__} and {other.__class__.__name__}"
            )

        return self._compose(other, J_mc_ma, J_mc_mb)

    @abstractmethod
    def _compose(
        self,
        other: LieGroupBase,
        J_mc_ma: OptionalJacobian = None,
        J_mc_mb: OptionalJacobian = None,
    ) -> LieGroupBase:
        """Returns the composition of self and another element of the same Lie group."""
        pass

    @abstractmethod
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
        pass

    @classmethod
    @abstractmethod
    def exp(cls, tangent: Tangent, J_m_t: OptionalJacobian = None) -> LieGroupBase:
        """Compute the exponential map of the given tagent vector. The dimension of the vector should match the LieGroupBase.dof value

        See Eqs. (23)

        Args:
            J_m_t: Jacobian of the Lie group element wrt to the given tangent

        Returns:
            Exponential map of the tagent vector
        """
        pass

    @abstractmethod
    def log(self, J_t_m: OptionalJacobian = None) -> Tangent:
        """Compute the tagent vector of the transformation, it is equivalent to the inverse of exponential map

        See Eqs. (24)

        Args:
            J_t_m: Jacobian of the tagent wrt to self
        
        Returns:
            The log() map of self in vector form
        """
        pass

    @abstractmethod
    def adjoint(self) -> Adjoint:
        """Compute the adjoint of the transformation

        See Eqs. (29)
        """
        pass

    @abstractmethod
    def rjac(self) -> Jacobian:
        """Compute the right jacobian of self

        See Eqs. (41)
        """
        pass

    @abstractmethod
    def rjacinv(self) -> Jacobian:
        """Compute the inverse of right jacobian of self
        """
        pass

    def plus(
        self,
        tangent: Tangent,
        J_mout_t: OptionalJacobian = None,
        J_mout_m: OptionalJacobian = None,
    ) -> LieGroupBase:
        """Compute the right plus operation of the Lie group.

        See Eqs. (25)

        Args:
            J_mout_t: Jacobian of the ouput wrt to the tangent vector
            J_mout_m: Jacobian of the output wrt to self
        
        Returns:
            The resulting Lie group element
        """
        if J_mout_t is not None:
            J_mout_t[...] = self.rjac()
        return self.compose(self.__class__.exp(tangent), J_mout_m, None)

    def minus(
        self,
        other: LieGroupBase,
        J_t_ma: OptionalJacobian = None,
        J_t_mb: OptionalJacobian = None,
    ) -> Tangent:
        """Compute the right minus operation of the Lie group.

        See Eqs. (26)

        Args:
            J_t_ma: Jacobian of the ouput wrt to self
            J_t_mb: Jacobian of the output wrt to other
        
        Returns:
            The resulting tangent in vector form
        """
        diff = other.inverse().compose(self)
        if J_t_ma is not None:
            J_t_ma[...] = diff.rjacinv()

        if J_t_mb is not None:
            J_t_mb[...] = -(diff.inverse().rjacinv())

        return diff.log()

    def almost_equal(self, other: LieGroupBase, abs_tol=eps) -> bool:
        for c1, c2 in zip(self.coeff, other.coeff):
            if not math.isclose(c1, c2, abs_tol=abs_tol):
                return False
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{repr(self.coeff)}"

    def __eq__(self, other: LieGroupBase) -> bool:
        return self.__class__ == other.__class__ and self.coeff == other.coeff

    def __hash__(self) -> int:
        return hash(self.coeff)

    def __matmul__(self, other: LieGroupBase) -> LieGroupBase:
        """
        Convenient function to compose Lie group elements together, e.g G_c = G_a @ G_b
        """
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.compose(other)

    def __mul__(self, vec: Vector) -> Vector:
        """
        Convenient function to apply the Lie group action on a point in vector space, e.g v_out = G * v_in
        """
        if not isinstance(vec, np.ndarray) and len(vec) != self.dim:
            return NotImplemented
        return self.act(vec)

    def __add__(self, tangent: Tangent) -> LieGroupBase:
        """
        Convenient function to apply the right plus operation of Lie group
        """
        if not isinstance(tangent, np.ndarray) and len(tangent) != self.dof:
            return NotImplemented

        return self.plus(tangent)

    def __sub__(self, other: LieGroupBase) -> Tangent:
        """
        Convenient function to apply the minus operation of Lie group
        """
        if not isinstance(other, type(self)):
            return NotImplemented

        return self.minus(other)
