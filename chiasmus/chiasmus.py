from __future__ import annotations

import math
import asyncio
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

from shiny import App, ui, render, reactive


# ============================================================
#  RNG: Numerical Recipes ran1 (same structure as Fortran code)
# ============================================================

@dataclass
class Ran1:
    idum: int = -1

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
        return min(temp, self.RNMX)


# ==============================
#  Core math (Fortran findp)
# ==============================

def findp(nopp: int, nchi: int, L: float) -> float:
    # ppp = P(X >= nchi), X~Binomial(nopp, L)
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
    ll = l[:]  # 1-based list, index 0 is dummy

    p = [0] * (nn + 1)
    for i in range(nn, 0, -1):
        j = int(rng.random() * i) + 1  # 1..i
        p[i] = ll.pop(j)

    q = [[0] * (m + 1) for _ in range(nn + 1)]
    for j in range(1, m + 1):
        q[0][j] = kk[j]

    for i in range(1, nn + 1):
        row = q[i - 1][:]
        row[p[i]] -= 1
        q[i] = row

    return p, q


def max_chiastic_order_for_permutation(
    p: List[int],
    q: List[List[int]],
    nn: int,
    m: int,
    mm: int,
    mu: int
) -> int:
    # Backtracking search translated from Fortran main loop
    u = [0] * (m + 1)
    ii = [0] * (2 * mm + 2)
    ii[2 * mm + 1] = nn + 1

    k = 1
    ii[k] = 1
    finished = False
    n = 0

    while not finished:
        j = p[ii[k]]
        k2 = 2 * mm - k + 2
        i = ii[k2] - 1

        if i <= ii[k]:
            nmax = -1
        else:
            nmax = k - 1
            # count candidates that have >=2 instances within bounds and are unused
            for jj in range(1, m + 1):
                njj = q[ii[k] - 1][jj] - q[i][jj]
                if njj > 1 and u[jj] == 0:
                    nmax += 1
            nj = q[ii[k] - 1][j] - q[i][j]

        if nmax <= n:
            if k == 1:
                finished = True
            else:
                k -= 1
                u[p[ii[k]]] = 0
                ii[k] += 1

        elif u[j] == 1 or nj < 2:
            ii[k] += 1

        else:
            while p[i] != j:
                i -= 1

            if k > n:
                n = k

            ii[2 * mm - k + 1] = i
            if mu == 0:
                u[j] = 1

            if k == mm:
                finished = True
            else:
                k += 1
                ii[k] = ii[k - 1] + 1

    return n


# ==========================================
#  Compute wrapper (what PyShiny calls)
# ==========================================

def parse_counts(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    # allow "2,2,3" or "2 2 3"
    s = s.replace(",", " ")
    parts = [p for p in s.split() if p]
    return [int(x) for x in parts]


def compute_chiasmus(
    mc_in: int,
    chi_counts: List[int],
    mn: int,
    non_counts: List[int],
    r: int,
    ndup: int,
    calc_p: bool,
    nopp: Optional[int],
    nchi: Optional[int],
    seed: int = -1,
    NNMAX: int = 200,
    MMAX: int = 100,
) -> Dict[str, Any]:
    """
    Returns dict with:
      L, L_err, (optional) P, P_err, plus some metadata.
    """

    # basic validation
    mc = mc_in
    if mc < 1 or mc > MMAX:
        raise ValueError(f"mc must be 1..{MMAX}")

    if len(chi_counts) != mc:
        raise ValueError(f"Need exactly {mc} chiastic counts")

    if any(x < 2 for x in chi_counts):
        raise ValueError("All chiastic counts must be >= 2")

    if mn < 0:
        raise ValueError("mn must be >= 0")

    if len(non_counts) != mn:
        raise ValueError(f"Need exactly {mn} nonchiastic counts")

    if any(x < 2 for x in non_counts):
        raise ValueError("All nonchiastic counts must be >= 2")

    m = mc + mn
    if m > MMAX:
        raise ValueError(f"mc+mn must be <= {MMAX}")

    # 1-based kk
    kk = [0] + chi_counts + non_counts

    # Build l[1..nn], nlev
    nn = 0
    nlev = 0
    l = [0]  # dummy
    for j in range(1, m + 1):
        nlev += kk[j] // 2
        for _ in range(kk[j]):
            nn += 1
            if nn > NNMAX:
                raise ValueError(f"Total appearances nn exceeds {NNMAX}")
            l.append(j)

    # ---- exact simple case ----
    if mn == 0 and nn == mc * 2:
        # L = product_{i=1..mc} 1/(2i-1)
        L = 1.0
        for i in range(1, mc + 1):
            L /= (2 * i - 1)
        L_err = 0.0
        out = {"L": L, "L_err": L_err, "method": "exact", "nn": nn, "m": m, "mc_used": mc}
    else:
        # Monte Carlo
        if r < 1:
            raise ValueError("r must be >= 1")

        mu = 0
        mm = m

        if nlev > mc:
            if ndup < 0 or (ndup + mc) > nlev:
                raise ValueError("ndup is invalid (too large or negative).")
            if ndup > 0:
                mu = 1
                mm = nn // 2
                mc = mc + ndup  # matches Fortran

        rng = Ran1(idum=seed)

        npn = [0] * (mm + 1)

        for _ip in range(1, r + 1):
            p, q = permute(l, kk, nn, m, rng)
            n_found = max_chiastic_order_for_permutation(p, q, nn, m, mm, mu)
            if 0 <= n_found <= mm:
                npn[n_found] += 1

        npnh = [0] * (mm + 1)
        npnh[mm] = npn[mm]
        for n in range(mm - 1, 0, -1):
            npnh[n] = npnh[n + 1] + npn[n]

        if mc > mm:
            L = 0.0
            L_err = 0.0
        else:
            L = npnh[mc] / r
            L_err = math.sqrt(npnh[mc]) / r

        out = {"L": L, "L_err": L_err, "method": "monte_carlo", "nn": nn, "m": m, "mc_used": mc}

    # ---- P calculation (binomial tail) ----
    if calc_p:
        if nopp is None or nchi is None:
            raise ValueError("Need N and M to compute P.")
        if nopp < 1 or nchi < 1:
            raise ValueError("N and M must be >= 1.")

        L = out["L"]
        L_err = out["L_err"]

        P = findp(nopp, nchi, L)

        if L + L_err < 1.0:
            P_alt = findp(nopp, nchi, L + L_err)
        elif L - L_err > 0.0:
            P_alt = findp(nopp, nchi, L - L_err)
        else:
            P_alt = 100.0

        P_err = abs(P - P_alt)
        out.update({"P": P, "P_err": P_err, "N": nopp, "M": nchi})

    return out


# ============================================================
#  PyShiny UI
# ============================================================

app_ui = ui.page_fluid(
    ui.h2("Chiastic Likelihood (Fortran → Python → PyShiny)"),

    ui.layout_sidebar(
        ui.sidebar(
            ui.input_numeric("mc", "Number n of chiastic elements (mc)", value=5, min=1, step=1),
            ui.input_text_area(
                "chi_counts",
                "Appearances of each chiastic element (comma/space separated)",
                "2,2,2,2,2",
                rows=2,
            ),

            ui.hr(),

            ui.input_numeric("mn", "Number m of nonchiastic elements (mn)", value=0, min=0, step=1),
            ui.panel_conditional(
                "input.mn > 0",
                ui.input_text_area(
                    "non_counts",
                    "Appearances of each nonchiastic element (comma/space separated)",
                    "2",
                    rows=2,
                ),
            ),

            ui.hr(),

            ui.input_numeric("r", "Rearrangements (Monte Carlo r)", value=10000, min=1, step=1000),
            ui.input_numeric("ndup", "Duplicate levels (ndup, usually 0)", value=0, min=0, step=1),

            ui.hr(),

            ui.input_checkbox("calc_p", "Calculate P", value=False),
            ui.panel_conditional(
                "input.calc_p",
                ui.input_numeric("N", "N opportunities", value=100, min=1, step=1),
                ui.input_numeric("M", "M chiastic", value=1, min=1, step=1),
            ),

            ui.hr(),

            ui.input_action_button("run", "Run calculation", class_="btn-primary"),
            ui.input_action_button("cancel", "Cancel (if running)", class_="btn-secondary"),
        ),

        ui.card(
            ui.card_header("Results"),
            ui.output_text_verbatim("results"),
        ),
    ),
)


# ============================================================
#  Server logic (ExtendedTask for long computations)
# ============================================================

@reactive.extended_task
async def run_calc(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs in the background so the UI doesn't freeze during long r.
    (ExtendedTask behavior documented by Posit.) :contentReference[oaicite:1]{index=1}
    """
    # Yield once to keep event loop happy
    await asyncio.sleep(0)

    return compute_chiasmus(**params)


def server(input, output, session):

    @reactive.Effect
    @reactive.event(input.run)
    def _():
        # Parse counts from text areas
        try:
            mc = int(input.mc())
            mn = int(input.mn())

            chi_counts = parse_counts(input.chi_counts())
            non_counts = parse_counts(input.non_counts()) if mn > 0 else []

            params = dict(
                mc_in=mc,
                chi_counts=chi_counts,
                mn=mn,
                non_counts=non_counts,
                r=int(input.r()),
                ndup=int(input.ndup()),
                calc_p=bool(input.calc_p()),
                nopp=int(input.N()) if input.calc_p() else None,
                nchi=int(input.M()) if input.calc_p() else None,
                seed=-1,
            )

            run_calc.invoke(params)

        except Exception as e:
            # Put error into a fake “completed” result by invoking a tiny run
            # (simpler than building separate error plumbing).
            run_calc.invoke({"mc_in": 1, "chi_counts": [2], "mn": 0, "non_counts": [],
                            "r": 1, "ndup": 0, "calc_p": False, "nopp": None, "nchi": None,
                            "seed": -1})
            session.send_notification(f"Input error: {e}", type="error")

    @reactive.Effect
    @reactive.event(input.cancel)
    def _():
        run_calc.cancel()
        session.send_notification("Cancelled.", type="message")

    @output
    @render.text
    def results():
        # This will automatically show a “busy” state while running
        # due to ExtendedTask semantics. :contentReference[oaicite:2]{index=2}
        res = run_calc.result()

        lines = []
        lines.append(f"Method: {res.get('method')}")
        lines.append(f"nn (total appearances): {res.get('nn')}")
        lines.append(f"m (total elements): {res.get('m')}")
        lines.append(f"mc_used (order threshold): {res.get('mc_used')}")
        lines.append("")
        lines.append(f"Reordering likelihood L  = {res['L']:.16f}")
        lines.append(f"Margin of error (+/-)    = {res['L_err']:.16f}")

        if "P" in res:
            lines.append("")
            lines.append(f"N opportunities          = {res['N']}")
            lines.append(f"M chiastic               = {res['M']}")
            lines.append(f"Chiastic likelihood P    = {res['P']:.16f}")
            lines.append(f"Margin of error (+/-)    = {res['P_err']:.16f}")

        return "\n".join(lines)


app = App(app_ui, server)
