from itertools import product
from collections import defaultdict
import numpy as np


class KmerDistanceCalculator:
    def __init__(self, k, alphabet="ARNDCEQGHILKMFPSTWYV"):
        self.k = k
        self.alphabet = sorted(alphabet)
        self.omega = ["".join(t) for t in product(self.alphabet, repeat=k)]
        self.index = {}
        for i, letter in enumerate(self.alphabet):
            self.index[letter] = i

    def count_kmers(self, sequence):
        counts = defaultdict(int)
        for start in range(1 + len(sequence) - self.k):
            kmer = sequence[start : start + self.k]
            counts[kmer] += 1
        return counts

    def kmer_vector(self, sequence, normalize=True):
        counts = self.count_kmers(sequence)
        v = [counts[kmer] for kmer in self.omega]
        if normalize:
            L2 = np.sqrt(np.sum([num * num for kmer, num in counts.items()]))
            v = [count / L2 for count in v]
        return v

    def omega_offset(self, kmer):
        total = 0
        for i, letter in enumerate(kmer):
            total = total * len(self.alphabet)
            total += self.index[letter]
        return total

    def kmer_vector_sparse(self, sequence, normalize=True):
        """
        Return counts and indices of only those kmers that are
        actually in the sequence. Suitable for creating a sparse
        matrix of type coo.
        """
        counts = self.count_kmers(sequence)
        if normalize:
            L2 = np.sqrt(np.sum([num * num for kmer, num in counts.items()]))
            dtype = float
        else:
            L2 = 1
            dtype = int
        V = np.empty((len(counts),), dtype=dtype)
        J = np.empty((len(counts),), dtype=dtype)
        for i, kmer in enumerate(sorted(counts.keys())):
            V[i] = counts[kmer] / L2
            J[i] = self.omega_offset(kmer)
        return V, J
