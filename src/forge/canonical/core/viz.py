"""Plotting helpers (Sankey builders), duplicated from the monolith."""
from __future__ import annotations

import plotly.graph_objects as go
import numpy as np


def make_mass_sankey(prod_lvl, recipes_dict, min_flow=0.5, title="Mass Flow Sankey"):
    """3-layer Sankey (Material_in → Process → Material_out).

    Units: recipe units × runs. Filters links < min_flow.
    """
    mats_in, mats_out, procs = set(), set(), set()
    links = []  # (source_label, target_label, value)

    for p, runs in prod_lvl.items():
        if runs <= 1e-12 or p not in recipes_dict:
            continue
        proc = recipes_dict[p]
        procs.add(p)

        # inputs: Material_in -> Process
        for m, amt in proc.inputs.items():
            val = runs * float(amt)
            if val >= min_flow:
                mats_in.add(m)
                links.append((f"[IN] {m}", f"[P] {p}", val))

        # outputs: Process -> Material_out
        for m, amt in proc.outputs.items():
            val = runs * float(amt)
            if val >= min_flow:
                mats_out.add(m)
                links.append((f"[P] {p}", f"[OUT] {m}", val))

    mat_in_labels  = sorted(f"[IN] {m}"  for m in mats_in)
    proc_labels    = sorted(f"[P] {p}"   for p in procs)
    mat_out_labels = sorted(f"[OUT] {m}" for m in mats_out)

    labels = mat_in_labels + proc_labels + mat_out_labels
    index  = {lab: i for i, lab in enumerate(labels)}

    sources, targets, values, link_labels = [], [], [], []
    for s, t, v in links:
        sources.append(index[s])
        targets.append(index[t])
        values.append(v)
        link_labels.append(f"{s} → {t}: {v:,.3f}")

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(label=labels, pad=15, thickness=18),
        link=dict(source=sources, target=targets, value=values, label=link_labels)
    )])
    fig.update_layout(title_text=title, font_size=12)
    return fig


def make_energy_sankey(energy_balance_df, min_MJ=10.0, title="Energy Flow Sankey"):
    df = energy_balance_df.copy().drop(index=[r for r in ["TOTAL"] if r in energy_balance_df.index], errors="ignore")
    carriers = [c for c in df.columns if (df[c].abs() > min_MJ).any()]
    procs = [p for p in df.index if (df.loc[p, carriers].abs() > min_MJ).any()]
    carrier_labels = [f"[E] {c}" for c in carriers]
    proc_labels    = [f"[P] {p}" for p in procs]
    labels = carrier_labels + proc_labels
    index  = {lab: i for i, lab in enumerate(labels)}
    sources, targets, values, link_labels = [], [], [], []
    for p in procs:
        for c in carriers:
            val = float(df.at[p, c])
            if val > min_MJ:
                sources.append(index[f"[E] {c}"])
                targets.append(index[f"[P] {p}"])
                values.append(val)
                link_labels.append(f"{c} → {p}: {val:,.1f} MJ")
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(label=labels, pad=15, thickness=18),
        link=dict(source=sources, target=targets, value=values, label=link_labels)
    )])
    fig.update_layout(title_text=title, font_size=12)
    return fig


def make_energy_to_process_sankey(
    energy_balance_df,
    emissions_df=None,
    title="Energy → Processes (ranked)",
    min_MJ=10.0,
    sort_by="emissions",
    exclude_carriers=("TOTAL", "Utility Plant"),
):
    """Simple Sankey: energy carriers (MJ) → processes, ranked by CO₂ or MJ."""
    eb = energy_balance_df.copy()
    carriers = [c for c in eb.columns if c not in exclude_carriers]
    procs = [r for r in eb.index if r not in ("TOTAL", "Utility Plant")]

    if sort_by == "emissions" and emissions_df is not None and "TOTAL CO2e" in emissions_df.columns:
        order = (
            emissions_df.loc[[p for p in procs if p in emissions_df.index], "TOTAL CO2e"]
            .fillna(0)
            .sort_values(ascending=False)
            .index.tolist()
        )
    else:
        order = (
            eb.loc[procs, carriers]
            .sum(axis=1)
            .sort_values(ascending=False)
            .index.tolist()
        )

    nodes = carriers + order
    idx = {n: i for i, n in enumerate(nodes)}

    src, tgt, val, hover = [], [], [], []
    for p in order:
        for c in carriers:
            mj = float(eb.at[p, c]) if (p in eb.index and c in eb.columns) else 0.0
            if mj > min_MJ:
                src.append(idx[c]); tgt.append(idx[p]); val.append(mj)
                if emissions_df is not None and p in emissions_df.index:
                    co2 = float(emissions_df.at[p, "TOTAL CO2e"]) if "TOTAL CO2e" in emissions_df.columns else 0.0
                    hover.append(f"{c} → {p}<br>{mj:,.1f} MJ<br>{co2:,.1f} kg CO₂e")
                else:
                    hover.append(f"{c} → {p}<br>{mj:,.1f} MJ")

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=nodes, pad=18, thickness=16),
        link=dict(source=src, target=tgt, value=val,
                  hovertemplate="%{customdata}<extra></extra>",
                  customdata=hover)
    ))
    fig.update_layout(title=title, font=dict(size=12), height=700)
    return fig


def make_hybrid_sankey(
    energy_balance_df,
    emissions_df,
    title="Hybrid Sankey: Energy → Processes → CO₂",
    min_MJ=10.0,
    min_kg=0.1,
    co2_scale=None,
    include_direct_and_energy_sinks=True,
):
    """Carriers (MJ) → processes → CO₂ sinks (kg). CO₂ link widths scaled to match MJ."""
    eb = energy_balance_df.copy()
    if emissions_df is None or emissions_df.empty:
        raise ValueError("emissions_df is empty, cannot build hybrid sankey.")
    em = emissions_df.copy()

    proc_rows = [r for r in eb.index if r not in ("TOTAL",)]
    proc_rows = [r for r in proc_rows if r in em.index and r != "Utility Plant"]
    carriers = [c for c in eb.columns if c != "TOTAL"]

    total_energy_MJ = float(eb.loc[proc_rows, carriers].clip(lower=0).sum().sum())
    total_co2_kg = float(em.loc[proc_rows, "TOTAL CO2e"].clip(lower=0).sum())
    if co2_scale is None:
        co2_scale = (total_energy_MJ / total_co2_kg) if total_co2_kg > 0 else 1.0

    carrier_nodes = carriers[:]
    process_nodes = proc_rows[:]
    sink_nodes = ["CO₂ (energy)", "CO₂ (direct)"] if include_direct_and_energy_sinks else ["CO₂"]
    nodes = carrier_nodes + process_nodes + sink_nodes
    node_index = {n: i for i, n in enumerate(nodes)}

    src, tgt, val, hover = [], [], [], []

    # carriers → processes
    for p in process_nodes:
        for c in carriers:
            mj = float(eb.at[p, c]) if (p in eb.index and c in eb.columns) else 0.0
            if mj > min_MJ:
                src.append(node_index[c]); tgt.append(node_index[p]); val.append(mj)
                hover.append(f"{c} → {p}<br>{mj:,.1f} MJ")

    # processes → CO₂ sinks
    for p in process_nodes:
        e_kg = float(em.at[p, "Energy Emissions"]) if "Energy Emissions" in em.columns and p in em.index else 0.0
        d_kg = float(em.at[p, "Direct Emissions"]) if "Direct Emissions" in em.columns and p in em.index else 0.0
        if include_direct_and_energy_sinks:
            if e_kg > min_kg:
                src.append(node_index[p]); tgt.append(node_index["CO₂ (energy)"]); val.append(e_kg * co2_scale)
                hover.append(f"{p} → CO₂ (energy)<br>{e_kg:,.2f} kg (×{co2_scale:,.3g})")
            if d_kg > min_kg:
                src.append(node_index[p]); tgt.append(node_index["CO₂ (direct)"]); val.append(d_kg * co2_scale)
                hover.append(f"{p} → CO₂ (direct)<br>{d_kg:,.2f} kg (×{co2_scale:,.3g})")
        else:
            co2_kg = e_kg + d_kg
            if co2_kg > min_kg:
                src.append(node_index[p]); tgt.append(node_index["CO₂"]); val.append(co2_kg * co2_scale)
                hover.append(f"{p} → CO₂<br>{co2_kg:,.2f} kg (×{co2_scale:,.3g})")

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=nodes, pad=18, thickness=16),
        link=dict(source=src, target=tgt, value=val,
                  hovertemplate="%{customdata}<extra></extra>", customdata=hover)
    ))
    fig.update_layout(
        title=dict(text=f"{title}<br><sup>Scale: 1 kg × {co2_scale:,.3g} = 1 width unit</sup>"),
        font=dict(size=12), height=700
    )
    return fig


__all__ = [
    "make_mass_sankey",
    "make_energy_sankey",
    "make_energy_to_process_sankey",
    "make_hybrid_sankey",
]
