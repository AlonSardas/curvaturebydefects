from typing import Union

import numpy as np


class SpringLattice(object):
    def __init__(self, dots: np.ndarray, springs: np.ndarray,
                 spring_constant: Union[float, np.ndarray],
                 springs_rest_length: Union[float, np.ndarray], mass: float = 1):
        """
        Creates a springs lattice containing N nodes.
        Each node is connected to m other nodes by springs. The springs may differ
        by their spring constant.

        :param dots: Array of shape (N, 3) containing the coordinates of the nodes
        :param springs: Array of shape (N, m) telling to which other nodes (by index)
            the nth node is connected
        :param spring_constant: Array of shape (N, m) containing the K constant
            value of the springs, by Hook's law.
        :param springs_rest_length: Array of shape (N, m) containing the rest length
            of the springs
        :param mass: The mass of the nodes, used for relaxation
        """
        shape = dots.shape
        if not shape[1] == 3:
            raise ValueError(f'The expected shape of the dots should be (N, 3). Got: {shape}')
        self.N = shape[0]
        shape = springs.shape
        self.m = shape[1]
        if not shape[0] == self.N:
            raise ValueError(f'The expected shape of the springs should be (N, m), '
                             f'N={self.N}. Got: {shape}')
        self.dots = dots
        self.springs = springs
        self.spring_constant = spring_constant
        self.spring_rest_length = springs_rest_length
        self.mass = mass

    def calculate_forces(self) -> np.ndarray:
        """
        :return: Array of shape (3, N) of the forces act on each node
        """
        relative_vectors: np.ndarray = self.dots[self.springs]
        assert relative_vectors.shape == (self.N, self.m, 3)
        relative_vectors -= self.dots[:, np.newaxis, :]
        a = np.sqrt(np.sum(relative_vectors ** 2, axis=2))
        forces = (self.spring_constant * (a - self.spring_rest_length) / a)[:, :, np.newaxis] * relative_vectors
        forces = np.sum(forces, axis=1)
        assert forces.shape == (self.N, 3)
        return forces
