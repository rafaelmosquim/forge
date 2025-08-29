# -*- coding: utf-8 -*-
"""
Minimal clickable-picks demo for a Streamlit UI.
- 3 materials ("choices") on the left
- Each has 3 producer options on the right
- Click a producer node in the diagram to pick it (1-of-N per material)
- Uses plotly Sankey + (optional) streamlit-plotly-events for click capture
- Falls back to radios if the plugin isn't available

Run:
  pip install streamlit plotly streamlit-plotly-events
  streamlit run streamlit_demo.py
"""

from __future__ import annotations
import streamlit as st
import json

st.set_page_config(page_title="Clickable Picks Demo", layout="wide", initial_sidebar_state="collapsed")

# ---------------------------
# Demo data (3 x 3 with fun names)
# ---------------------------
AMBIGUOUS = [
    ("Meltixium", ["Route Omega", "Route Sigma", "Route Kappa"]),
    ("Ferrospark", ["Onsite Nova", "Market Quasar", "Hybrid Pulsar"]),
    ("Steeladine", ["Process Lynx", "Process Manta", "Process Orion"]) ,
]

# Persistent state: material -> chosen producer
if "picks" not in st.session_state:
    st.session_state.picks = {}

# ---------------------------
# Diagram builder + click handler
# ---------------------------

def render_clickable_sankey(ambiguous):
    """Try to render a clickable Sankey. If plugin missing, return False."""
    try:
        from streamlit_plotly_events import plotly_events
        import plotly.graph_objects as go
    except Exception:
        st.info(
            "Install **streamlit-plotly-events** to click on the diagram "
            "(e.g. `pip install streamlit-plotly-events`). Fallback controls are shown below."
        )
        return False

    # Build nodes
    node_labels = []
    node_types = []   # 'mat' or 'prod'
    prod_to_mat = {}
    xs, ys = [], []
    src, tgt, val = [], [], []

    n_groups = len(ambiguous)
    def _norm(i, n):
        return 0.05 + 0.9 * (i / max(1, n - 1)) if n > 1 else 0.5

    for gi, (mat, options) in enumerate(ambiguous):
        # material node
        m_idx = len(node_labels)
        node_labels.append(mat)
        node_types.append("mat")
        xs.append(0.05)
        ys.append(_norm(gi, n_groups))

        # producer nodes
        base_y = ys[-1]
        step = 0.08 if len(options) > 1 else 0.0
        start = base_y - step * (len(options) - 1) / 2
        for oi, prod in enumerate(options):
            p_idx = len(node_labels)
            node_labels.append(prod)
            node_types.append("prod")
            prod_to_mat[p_idx] = mat
            xs.append(0.60)
            ys.append(min(0.95, max(0.05, start + oi * step)))
            src.append(m_idx)
            tgt.append(p_idx)
            val.append(1)

    # Color the nodes based on selection state
    colors = []
    for i, lab in enumerate(node_labels):
        if node_types[i] == "mat":
            colors.append("rgba(37,99,235,0.25)")  # blue-ish
        else:
            mat = prod_to_mat[i]
            picked = st.session_state.picks.get(mat)
            colors.append("rgba(22,163,74,0.60)" if picked == lab else "rgba(0,0,0,0.18)")

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(label=node_labels, pad=12, thickness=18, color=colors, x=xs, y=ys),
        link=dict(source=src, target=tgt, value=val, color="rgba(0,0,0,0.08)")
    )])
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))

    evs = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key="pick_sankey")
    if evs:
        ev = evs[0]
        # Plotly Sankey returns clicked node index in 'pointNumber' for node layer
        node_idx = ev.get("pointNumber") if isinstance(ev, dict) else None
        if node_idx is None:
            node_idx = ev.get("pointIndex") if isinstance(ev, dict) else None
        if node_idx is not None and 0 <= node_idx < len(node_labels):
            if node_types[node_idx] == "prod":
                mat = prod_to_mat[node_idx]
                prod = node_labels[node_idx]
                st.session_state.picks[mat] = prod
                st.toast(f"Selected '{prod}' for {mat}")
    return True

# ---------------------------
# Layout
# ---------------------------
left, right = st.columns([2.3, 1.2], gap="large")

with left:
    st.markdown("### Click a producer node to choose a treatment")
    sankey_ok = render_clickable_sankey(AMBIGUOUS)

    # Fallback controls if the plugin isn't installed
    if not sankey_ok:
        st.markdown("#### Fallback controls")
        for mat, options in AMBIGUOUS:
            cur = st.session_state.picks.get(mat, options[0])
            choice = st.radio(mat, options, horizontal=True, index=options.index(cur))
            st.session_state.picks[mat] = choice

    st.markdown("""
    <small>Legend: <span style='color:#2563eb'>blue</span> = material, <span style='color:#16a34a'>green</span> = selected producer, grey = available option</small>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    if c1.button("Reset"):
        st.session_state.picks = {}
        st.experimental_rerun()
    if c2.button("Auto-pick demo"):
        # Arbitrary autopick: choose the last option for each
        st.session_state.picks = {m: opts[-1] for m, opts in AMBIGUOUS}
        st.experimental_rerun()

with right:
    st.markdown("### Your picks")
    done = sum(1 for m, _ in AMBIGUOUS if st.session_state.picks.get(m))
    st.write(f"{done} / {len(AMBIGUOUS)} selected")

    # Compact summary table
    import pandas as pd
    rows = []
    for m, opts in AMBIGUOUS:
        rows.append({"Material": m, "Selected Producer": st.session_state.picks.get(m, "—")})
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, use_container_width=True)

    st.download_button(
        "Download picks (JSON)",
        data=json.dumps(st.session_state.picks, indent=2).encode("utf-8"),
        file_name="picks_demo.json",
        mime="application/json",
    )

st.caption("© Demo — Clickable Sankey pickboard (3×3)")
