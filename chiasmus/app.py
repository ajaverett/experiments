from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import math
import time
import streamlit as st


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



def parse_counts(text: str):
    """
    Accepts '2,2,2' or '2 2 2' and returns [2,2,2]
    """
    s = (text or "").strip()
    if not s:
        return []
    s = s.replace(",", " ")
    parts = [p for p in s.split() if p]
    return [int(x) for x in parts]


st.set_page_config(
    page_title="Chiasmus Likelihood",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Minimal embedded look (good for iframes)
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.1rem; padding-bottom: 1.2rem; max-width: 860px; }
      header { visibility: hidden; height: 0px; }
      footer { visibility: hidden; height: 0px; }
      /* make metrics tighter */
      [data-testid="stMetricValue"] { font-size: 1.35rem; }
      [data-testid="stMetricLabel"] { font-size: 0.95rem; opacity: 0.9; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("## Chiasmus Likelihood")
st.caption("Monte Carlo + exact simple-case likelihood (L), with optional P tail probability.")


# -----------------------------
# Inputs
# -----------------------------
with st.container(border=True):
    st.markdown("### Inputs")

    row1 = st.columns(2)
    with row1[0]:
        mc = st.number_input("Chiastic elements (mc)", min_value=1, value=5, step=1)
    with row1[1]:
        mn = st.number_input("Nonchiastic elements (mn)", min_value=0, value=0, step=1)

    chi_counts_text = st.text_input(
        "Chiastic appearances (comma/space separated)",
        value="2,2,2,2,2",
        help="Example: 6,3,2,2,2,2,2",
    )

    non_counts_text = st.text_input(
        "Nonchiastic appearances (comma/space separated)",
        value="",
        help="Only used if mn > 0",
        disabled=(int(mn) == 0),
    )

    row2 = st.columns(2)
    with row2[0]:
        r = st.number_input("Rearrangements r", min_value=1, value=10000, step=1000)
    with row2[1]:
        ndup = st.number_input("Duplicate levels (ndup)", min_value=0, value=0, step=1)

    st.divider()

    row3 = st.columns([1, 1, 1.2])
    with row3[0]:
        calc_p = st.checkbox("Calculate P", value=False)
    with row3[1]:
        N = st.number_input("N opportunities", min_value=1, value=100, step=1, disabled=not calc_p)
    with row3[2]:
        M = st.number_input("M chiastic", min_value=1, value=1, step=1, disabled=not calc_p)

    run = st.button("Run", type="primary", use_container_width=True)


# -----------------------------
# Run computation + loading UI
# -----------------------------
if run:
    try:
        chi_counts = parse_counts(chi_counts_text)
        non_counts = parse_counts(non_counts_text) if int(mn) > 0 else []

        # Pretty "loading" experience:
        # - spinner
        # - progress bar that animates (fake progress, but helpful UX)
        progress = st.progress(0, text="Preparing calculation…")

        with st.spinner("Calculating chiastic likelihood…"):
            # Fake progress while compute runs (Streamlit doesn't support true progress
            # from a plain function unless compute_chiasmus reports it)
            for pct in (10, 25, 40):
                time.sleep(0.08)
                progress.progress(pct, text="Running permutations…")

            res = compute_chiasmus(
                mc_in=int(mc),
                chi_counts=chi_counts,
                mn=int(mn),
                non_counts=non_counts,
                r=int(r),
                ndup=int(ndup),
                calc_p=bool(calc_p),
                nopp=int(N) if calc_p else None,
                nchi=int(M) if calc_p else None,
                seed=-1,
            )

            progress.progress(90, text="Finalizing…")
            time.sleep(0.05)

        progress.progress(100, text="Complete ✓")
        time.sleep(0.05)
        progress.empty()

        # -----------------------------
        # Results (prettier layout)
        # -----------------------------
        st.markdown("### Results")

        method = res.get("method", "unknown")
        nn = res.get("nn")
        m_total = res.get("m")
        mc_used = res.get("mc_used")

        # Summary cards
        top = st.columns(4)
        top[0].metric("Method", method)
        top[1].metric("nn", nn if nn is not None else "—")
        top[2].metric("m", m_total if m_total is not None else "—")
        top[3].metric("Order threshold", mc_used if mc_used is not None else "—")

        st.divider()

        # Big metrics
        L = res["L"]
        L_err = res["L_err"]

        main = st.columns(2)
        main[0].metric("Reordering likelihood L", f"{L:.16f}")
        main[1].metric("L margin of error (+/−)", f"{L_err:.16f}")

        # Optional P section
        if "P" in res:
            st.divider()
            st.markdown("#### P (probability of observing ≥ M chiastic cases out of N)")

            pcols = st.columns(4)
            pcols[0].metric("N", res.get("N", "—"))
            pcols[1].metric("M", res.get("M", "—"))
            pcols[2].metric("Chiastic likelihood P", f"{res['P']:.16f}")
            pcols[3].metric("P margin of error (+/−)", f"{res['P_err']:.16f}")

        # Raw details (collapsed)
        with st.expander("Show raw output"):
            st.json(res)

    except Exception as e:
        st.error(f"Error: {e}")


