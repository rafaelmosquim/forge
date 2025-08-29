# -*- coding: utf-8 -*-
"""
Three alternative wizard/stepper layouts for Streamlit (no sidebar).
Pick one pattern to migrate your app UI. Each demo is self-contained and
works without extra deps. Keep `from __future__ import annotations` at the
very top to avoid SyntaxError.

Patterns:
  A) Simple Stepper (pills header + Back/Next)
  B) Locked Tabs Wizard (tabs that refuse skipping ahead)
  C) URL-Driven Wizard (step in query params for shareable URLs)

Run:
  streamlit run wizard_demos.py

Switch demos at the top radio (or set DEMO_DEFAULT below).
Replace placeholder steps with your real ones.
"""

from __future__ import annotations
import streamlit as st
from dataclasses import dataclass

# -------------------- shared styles --------------------
st.set_page_config(page_title="Wizard Demos", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
[data-testid="stSidebar"], [data-testid="collapsedControl"]{display:none;}
.stepbar{position:sticky; top:0; z-index:100; background:#fff; padding:.25rem .25rem .5rem .25rem;}
.step-chip{display:inline-block; padding:.25rem .6rem; border-radius:999px; border:1px solid #d1d5db; margin-right:.25rem; font-size:.85rem;}
.step-chip.active{border-color:#2563eb; background:rgba(37,99,235,.08);} 
.step-chip.done{border-color:#16a34a; background:rgba(22,163,74,.08);} 
.hr { height:1px; background:linear-gradient(to right,#e5e7eb,#cbd5e1,#e5e7eb); margin:.25rem 0 .75rem 0; border:0; }
button[kind="primary"], [data-testid="baseButton-primary"]{background:#16a34a !important; border-color:#16a34a !important; color:#fff !important;}
button[kind="primary"]:hover, [data-testid="baseButton-primary"]:hover{background:#15803d !important; border-color:#15803d !important;}
</style>
""", unsafe_allow_html=True)

# -------------------- small helpers --------------------

def step_pills(cur:int, labels:list[str]):
    html = ["<div class='stepbar'>"]
    for i, lab in enumerate(labels, start=1):
        cls = "step-chip"
        if i < cur: cls += " done"
        elif i == cur: cls += " active"
        html.append(f"<span class='{cls}'>{lab}</span>")
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)

@dataclass
class DemoState:
    step: int = 1
    pick_a: str | None = None
    pick_b: str | None = None
    pick_c: str | None = None

if "demo_state" not in st.session_state:
    st.session_state.demo_state = DemoState()

# -------------------- demo selector --------------------
DEMO_DEFAULT = "A) Simple Stepper"
demo_choice = st.radio(
    "Choose a demo layout:",
    ["A) Simple Stepper", "B) Locked Tabs", "C) URL-Driven"],
    index=["A) Simple Stepper", "B) Locked Tabs", "C) URL-Driven"].index(DEMO_DEFAULT),
    horizontal=True
)

# ======================================================
# A) SIMPLE STEPPER (PILLS + BACK/NEXT)
# ======================================================

def demo_simple_stepper():
    S = st.session_state.demo_state
    labels = ["1. Pick A", "2. Pick B", "3. Pick C", "4. Review"]
    step_pills(S.step, labels)

    left, right = st.columns([1.6, 2.4], gap="large")

    with left:
        if S.step == 1:
            st.header("Step 1 — Pick A")
            S.pick_a = st.radio("Choose option A:", ["Alpha", "Beta", "Gamma"],
                                index=["Alpha", "Beta", "Gamma"].index(S.pick_a) if S.pick_a in ["Alpha", "Beta", "Gamma"] else 0)
        elif S.step == 2:
            st.header("Step 2 — Pick B")
            S.pick_b = st.selectbox("Choose option B:", ["Basil", "Bay", "Boron"],
                                    index=["Basil", "Bay", "Boron"].index(S.pick_b) if S.pick_b in ["Basil", "Bay", "Boron"] else 0)
        elif S.step == 3:
            st.header("Step 3 — Pick C")
            S.pick_c = st.radio("Choose option C:", ["Cobalt", "Cedar", "Cygnus"],
                                index=["Cobalt", "Cedar", "Cygnus"].index(S.pick_c) if S.pick_c in ["Cobalt", "Cedar", "Cygnus"] else 0,
                                horizontal=True)
        elif S.step == 4:
            st.header("Step 4 — Review")
            st.write({"A": S.pick_a, "B": S.pick_b, "C": S.pick_c})
            st.success("Looks good! Replace this page with your run + results.")

        b1, b2 = st.columns(2)
        if b1.button("◀ Back", disabled=S.step == 1):
            S.step = max(1, S.step - 1)
        if b2.button("Next ▶", type="primary", disabled=S.step == len(labels)):
            S.step = min(len(labels), S.step + 1)

    with right:
        st.subheader("Live Preview")
        st.write("Selected so far:")
        st.write({"A": S.pick_a, "B": S.pick_b, "C": S.pick_c})
        st.caption("Drop your Mermaid/Sankey preview here if desired.")

# ======================================================
# B) LOCKED TABS WIZARD (prevent skipping ahead)
# ======================================================

def demo_locked_tabs():
    S = st.session_state.demo_state
    labels = ["Step 1", "Step 2", "Step 3", "Review"]
    step_pills(S.step, ["1. A", "2. B", "3. C", "4. Review"])  # reuse pills for consistency

    tabs = st.tabs(labels)

    with tabs[0]:
        st.header("Step 1 — A")
        S.pick_a = st.radio("A:", ["Alpha", "Beta", "Gamma"],
                            index=["Alpha", "Beta", "Gamma"].index(S.pick_a) if S.pick_a in ["Alpha", "Beta", "Gamma"] else 0)
        if st.button("Continue to Step 2", type="primary"):
            S.step = max(S.step, 2)

    with tabs[1]:
        if S.step < 2:
            st.warning("Finish Step 1 first."); st.stop()
        st.header("Step 2 — B")
        S.pick_b = st.selectbox("B:", ["Basil", "Bay", "Boron"],
                                index=["Basil", "Bay", "Boron"].index(S.pick_b) if S.pick_b in ["Basil", "Bay", "Boron"] else 0)
        if st.button("Continue to Step 3", type="primary"):
            S.step = max(S.step, 3)

    with tabs[2]:
        if S.step < 3:
            st.warning("Finish Step 2 first."); st.stop()
        st.header("Step 3 — C")
        S.pick_c = st.radio("C:", ["Cobalt", "Cedar", "Cygnus"],
                            index=["Cobalt", "Cedar", "Cygnus"].index(S.pick_c) if S.pick_c in ["Cobalt", "Cedar", "Cygnus"] else 0)
        if st.button("Continue to Review", type="primary"):
            S.step = max(S.step, 4)

    with tabs[3]:
        if S.step < 4:
            st.warning("Finish Step 3 first."); st.stop()
        st.header("Review")
        st.write({"A": S.pick_a, "B": S.pick_b, "C": S.pick_c})
        st.success("Replace this with run_scenario(...) and your results.")

# ======================================================
# C) URL-DRIVEN WIZARD (shareable step links)
# ======================================================

def demo_url_driven():
    S = st.session_state.demo_state

    # read step from URL ?step=N (defaults to current state)
    qp = st.experimental_get_query_params()
    if "step" in qp:
        try: S.step = int(qp["step"][0])
        except Exception: pass

    def set_step(n:int):
        S.step = max(1, min(4, n))
        st.experimental_set_query_params(step=S.step)
        st.experimental_rerun()

    labels = ["1. A", "2. B", "3. C", "4. Review"]
    step_pills(S.step, labels)

    if S.step == 1:
        st.header("Step 1 — A")
        S.pick_a = st.radio("A:", ["Alpha", "Beta", "Gamma"],
                            index=["Alpha", "Beta", "Gamma"].index(S.pick_a) if S.pick_a in ["Alpha", "Beta", "Gamma"] else 0)
    elif S.step == 2:
        st.header("Step 2 — B")
        S.pick_b = st.selectbox("B:", ["Basil", "Bay", "Boron"],
                                index=["Basil", "Bay", "Boron"].index(S.pick_b) if S.pick_b in ["Basil", "Bay", "Boron"] else 0)
    elif S.step == 3:
        st.header("Step 3 — C")
        S.pick_c = st.radio("C:", ["Cobalt", "Cedar", "Cygnus"],
                            index=["Cobalt", "Cedar", "Cygnus"].index(S.pick_c) if S.pick_c in ["Cobalt", "Cedar", "Cygnus"] else 0,
                            horizontal=True)
    else:
        st.header("Review")
        st.write({"A": S.pick_a, "B": S.pick_b, "C": S.pick_c})

    b1, b2 = st.columns(2)
    b1.button("◀ Back", disabled=S.step==1, on_click=lambda: set_step(S.step-1))
    b2.button("Next ▶", type="primary", disabled=S.step==4, on_click=lambda: set_step(S.step+1))

# -------------------- run selected demo --------------------
if demo_choice.startswith("A"):
    demo_simple_stepper()
elif demo_choice.startswith("B"):
    demo_locked_tabs()
else:
    demo_url_driven()
