#!/usr/bin/env python3
"""
chiasmus.py  (Python translation of Boyd F. Edwards' chiasmus.f)

This program estimates the likelihood that high-level chiastic structure
appears by chance in a multiset of elements, using either:

- Exact L for *simple* chiasms (each chiastic element appears exactly twice,
  and there are no nonchiastic elements), or
- Monte Carlo (random rearrangements) for general/complex cases.

It also computes P = probability that at least M chiastic opportunities
(out of N) would be observed, given per-opportunity likelihood L
(binomial tail).

Translated from the Fortran code included in the prompt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple


# ----------------------------
# Numerical Recipes ran1 RNG
# ----------------------------

@dataclass
class Ran1:
    """
    Re-implementation of the Numerical Recipes ran1() generator used
    in the Fortran code, for close behavioral equivalence.
    """
    idum: int = -1

    # Constants (from Fortran)
    IA: int = 16807
    IM: int = 2147483647
    IQ: int = 127773
    IR: int = 2836
    NTAB: int = 32
    EPS: float = 1.2e-7

    def __post_init__(self) -> None:
        self.AM = 1.0 / self.IM
        self.NDIV = 1 + (self.IM - 1) // self.NTAB
        self.RNMX = 1.0 - self.EPS
        self.iv = [0] * (self.NTAB + 1)  # 1-based
        self.iy = 0

    def random(self) -> float:
        """Return uniform random float in (0,1), like ran1()."""
        if self.idum <= 0 or self.iy == 0:
            self.idum = max(-self.idum, 1)
            for j in range(self.NTAB + 8, 0, -1):
                k = self.idum // self.IQ
                self.idum = self.IA * (self.idum - k * self.IQ) - self.IR * k
                if self.idum < 0:
                    self.idum += self.IM
                if j <= self.NTAB:
                    self.iv[j] = self.idum
            self.iy = self.iv[1]

        k = self.idum // self.IQ
        self.idum = self.IA * (self.idum - k * self.IQ) - self.IR * k
        if self.idum < 0:
            self.idum += self.IM

        j = 1 + self.iy // self.NDIV
        self.iy = self.iv[j]
        self.iv[j] = self.idum

        temp = self.AM * self.iy
        if temp > self.RNMX:
            return self.RNMX
        return temp


# ----------------------------
# Core math routines
# ----------------------------

def findp(nopp: int, nchi: int, L: float) -> float:
    """
    Fortran findp:
      ppp = P(X >= nchi) where X ~ Binomial(nopp, L)
    computed using a recurrence.
    """
    pii = (1.0 - L) ** nopp
    ppp = 1.0 - pii
    if nchi > 1:
        rat = L / (1.0 - L)
        for i in range(1, nchi):
            pii = pii * (nopp + 1 - i) * rat / i
            ppp = ppp - pii
    return ppp


def permute(
    l: List[int],
    kk: List[int],
    nn: int,
    m: int,
    rng: Ran1
) -> Tuple[List[int], List[List[int]]]:
    """
    Fortran permute(l,p,q,kk,nn,m,idum)
    Returns:
      p[1..nn] permuted multiset
      q[0..nn][1..m] remaining counts AFTER position i
    """
    # Make a working copy ll[1..nn]
    ll = l[:]  # already 1-based with dummy at index 0

    # p is 1-based
    p = [0] * (nn + 1)

    # Randomly build p from the multiset list ll
    # Fortran loop: do i = nn, 1, -1
    #   j = ran1(idum)*i + 1
    #   p(i) = ll(j)
    #   collapse
    for i in range(nn, 0, -1):
        j = int(rng.random() * i) + 1  # 1..i inclusive
        p[i] = ll.pop(j)

    # q is (nn+1) x (m+1), 1-based for element index
    q = [[0] * (m + 1) for _ in range(nn + 1)]

    # q(0,j) = kk(j)
    for j in range(1, m + 1):
        q[0][j] = kk[j]

    # q(i,*) copies q(i-1,*), then decrements q(i,p(i))
    for i in range(1, nn + 1):
        prev = q[i - 1]
        row = prev[:]          # copy counts
        row[p[i]] -= 1         # remove current item
        q[i] = row

    return p, q


def max_chiastic_order_for_permutation(
    p: List[int],
    q: List[List[int]],
    kk: List[int],
    nn: int,
    m: int,
    mm: int,
    mu: int
) -> int:
    """
    This is the big backtracking search in the Fortran main loop.

    Returns n = deepest (maximum) chiastic order found in this permutation.
    """
    # u(j)=0/1 for used elements (1..m)
    u = [0] * (m + 1)

    # ii indices are 1..(2*mm+1)
    ii = [0] * (2 * mm + 2)
    ii[2 * mm + 1] = nn + 1  # Fortran: ii(2n+1) = nn by definition; here nn+1 matches code use

    k = 1
    ii[k] = 1
    finished = False
    n = 0  # deepest order found

    while not finished:
        j = p[ii[k]]              # element at level k
        k2 = 2 * mm - k + 2       # partner index for (k-1)'th element
        i = ii[k2] - 1            # max possible index of second appearance

        if i <= ii[k]:
            nmax = -1
        else:
            nmax = k - 1
            # Count elements that could still participate within bounds
            for jj in range(1, m + 1):
                njj = q[ii[k] - 1][jj] - q[i][jj]
                if njj > 1 and u[jj] == 0:
                    nmax += 1
            nj = q[ii[k] - 1][j] - q[i][j]

        if nmax <= n:
            # Abandon this level
            if k == 1:
                finished = True
            else:
                k -= 1
                u[p[ii[k]]] = 0
                ii[k] += 1

        elif u[j] == 1 or nj < 2:
            # Can't use this element here; try next position
            ii[k] += 1

        else:
            # Find a matching second occurrence of element j by scanning backwards
            while p[i] != j:
                i -= 1

            # Record deepest structure encountered
            if k > n:
                n = k

            # Store second occurrence index
            ii[2 * mm - k + 1] = i

            # Mark used element unless duplicates allowed
            if mu == 0:
                u[j] = 1

            if k == mm:
                finished = True
            else:
                k += 1
                ii[k] = ii[k - 1] + 1

    return n


# ----------------------------
# Main interactive program
# ----------------------------

def prompt_int(msg: str) -> int:
    while True:
        try:
            return int(input(msg))
        except ValueError:
            print("Please enter an integer.")


def prompt_yes_no(msg: str) -> bool:
    ans = input(msg).strip().lower()
    return ans.startswith("y")


def main() -> None:
    MMAX = 100
    NNMAX = 200

    rng = Ran1(idum=-1)

    print("Program to calculate chiastic likelihood")
    print("This program is free for noncommercial use.")
    print("See readme file for instructions.")
    print("Boyd F. Edwards, bedwards@wvu.edu, 24 April 2010 (translated to Python)")
    print()

    while True:
        _ = input("Chiasm: ")  # prompt is for user reference only

        # mc = number of chiastic elements
        while True:
            mc = prompt_int("Number n of chiastic elements: ")
            if mc > MMAX:
                print("Your value of n exceeds 100. Please enter a smaller value.")
                continue
            if mc < 1:
                print("Your value of n is less than 1. Please enter a larger value.")
                continue
            break

        # kk[1..m] but we don't yet know mn; start with chiastic appearances
        while True:
            parts = input("Number of appearances of each chiastic element: ").strip().split(",")
            if len(parts) == 1:
                # allow space-separated too
                parts = input("  (Try space-separated) ").strip().split()
            try:
                chi_counts = [int(x.strip()) for x in parts if x.strip() != ""]
            except ValueError:
                print("Please enter integers, e.g. 2,2,2 or 2 2 2")
                continue

            if len(chi_counts) != mc:
                print(f"Expected {mc} numbers. Please reenter.")
                continue

            if any(x < 2 for x in chi_counts):
                print("One of your numbers is less than 2. Please reenter these numbers.")
                continue
            break

        mn = prompt_int("Number m of nonchiastic elements: ")
        if mn < 0:
            print("Your value of m is negative. Using 0.")
            mn = 0

        m = mc + mn
        if m > MMAX:
            print("Your value of n+m exceeds 100. Please enter a smaller value of m.")
            continue

        non_counts: List[int] = []
        if mn > 0:
            while True:
                parts = input("Number of appearances of each nonchiastic element: ").strip().split(",")
                if len(parts) == 1:
                    parts = input("  (Try space-separated) ").strip().split()
                try:
                    non_counts = [int(x.strip()) for x in parts if x.strip() != ""]
                except ValueError:
                    print("Please enter integers.")
                    continue

                if len(non_counts) != mn:
                    print(f"Expected {mn} numbers. Please reenter.")
                    continue
                if any(x < 2 for x in non_counts):
                    print("One of your numbers is less than 2. Please reenter these numbers.")
                    continue
                break

        # Build kk[1..m] (1-based)
        kk = [0] + chi_counts + non_counts

        # Build l[1..nn]
        nn = 0
        nlev = 0
        l = [0]  # 1-based dummy
        for j in range(1, m + 1):
            nlev += kk[j] // 2
            for _ in range(kk[j]):
                nn += 1
                if nn > NNMAX:
                    print("Your total number of element appearances exceeds 200. Please try again.")
                    nn = 0
                    break
                l.append(j)
            if nn == 0:
                break
        if nn == 0:
            continue

        # Simple exact calculation? (mn==0 and nn==2*mc)
        did_exact = False
        if mn == 0 and nn == mc * 2:
            if prompt_yes_no("This chiasm is simple. Calculate L exactly? (yes/no): "):
                L = 1.0
                for i in range(1, mc + 1):
                    L /= (2 * i - 1)
                err = 0.0
                did_exact = True

        if not did_exact:
            np = prompt_int("Number r of rearrangements: ")
            if np < 1:
                print("Your value of r is less than 1. Please enter a larger value.")
                continue

            mu = 0
            mm = m
            # Duplicate levels logic
            if nlev > mc:
                while True:
                    ndup = prompt_int("Number of duplicate levels (normally 0): ")
                    if ndup < 0 or ndup + mc > nlev:
                        print("Your value is too large, or is negative. Please enter a new value.")
                        continue
                    break
                if ndup > 0:
                    mu = 1
                    mm = nn // 2
                    mc = mc + ndup

            # Statistics arrays: npn[0..mm], but only 1..mm used
            npn = [0] * (mm + 1)

            inc = max(np // 40, 1)
            print('Calculating... Type Ctrl-C to quit.')
            print("|---------|---------|---------|---------|")

            try:
                for ip in range(1, np + 1):
                    if ip % inc == 0:
                        print("x", end="", flush=True)

                    p, q = permute(l, kk, nn, m, rng)
                    n_found = max_chiastic_order_for_permutation(p, q, kk, nn, m, mm, mu)
                    if 0 <= n_found <= mm:
                        npn[n_found] += 1
            except KeyboardInterrupt:
                print("\nInterrupted early. Results are based on partial runs.")
                np = ip

            # Cumulative distribution npnh[n] = permutations with order >= n
            npnh = [0] * (mm + 1)
            npnh[mm] = npn[mm]
            for n in range(mm - 1, 0, -1):
                npnh[n] = npnh[n + 1] + npn[n]

            print("x")
            # L = npnh(mc)/np
            if mc > mm:
                # Should not happen, but avoid index error.
                L = 0.0
                err = 0.0
            else:
                L = npnh[mc] / np
                err = math.sqrt(npnh[mc]) / np

        print(f"Reordering likelihood L  ={L:18.16f}")
        print(f"Margin of error (+ or -) ={err:18.16f}")

        # Calculate P?
        if prompt_yes_no("Calculate P? (yes/no): "):
            while True:
                nopp = prompt_int("Number N of chiastic opportunities: ")
                if nopp < 1:
                    print("Your value of N is less than 1. Please enter a larger value.")
                    continue
                break
            while True:
                nchi = prompt_int("Number M of these that are chiastic: ")
                if nchi < 1:
                    print("Your value of M is less than 1. Please enter a larger value.")
                    continue
                break

            P = findp(nopp, nchi, L)

            # Error propagation like the Fortran code
            if L + err < 1.0:
                P_alt = findp(nopp, nchi, L + err)
            elif L - err > 0.0:
                P_alt = findp(nopp, nchi, L - err)
            else:
                P_alt = 100.0

            Perr = abs(P - P_alt)

            print(f"Chiastic likelihood P    ={P:18.16f}")
            print(f"Margin of error (+ or -) ={Perr:18.16f}")

        if not prompt_yes_no("Perform another calculation? (yes/no): "):
            print("Copy results and press return to quit.")
            input()
            break


if __name__ == "__main__":
    main()
