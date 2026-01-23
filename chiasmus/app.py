# app.py
# Streamlit UI for chiastic likelihood calculator (iframe-friendly)
# Assumes you have: from chiasmus import compute_chiasmus

import time
import streamlit as st
from chiasmus import compute_chiasmus


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


