from __future__ import annotations

from io import BytesIO
from pathlib import Path
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from capital_lineage import (
    add_edge_performance,
    build_lineage,
    build_return_store,
    read_workbook,
    root_selector_table,
    selected_root_edges,
)


st.set_page_config(page_title="Portfolio Capital Lineage", layout="wide")


@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes):
    trades, return_data, source_labels = read_workbook(file_bytes)
    return trades, return_data, source_labels


@st.cache_data(show_spinner=False)
def build_cached(file_bytes: bytes, max_gap_days: int):
    trades, return_data, _ = load_data(file_bytes)
    store = build_return_store(trades, return_data)
    lineage = build_lineage(trades, max_gap_days=max_gap_days)
    edges_perf = add_edge_performance(lineage, store)
    return lineage, edges_perf, store.diagnostics, store.latest_date


def fmt_pct(x):
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x:+.1%}"


def bezier_points(x0, y0, x1, y1, bend, n=36):
    dx = x1 - x0
    c1x = x0 + 0.34 * dx
    c2x = x0 + 0.66 * dx
    c1y = y0 + bend
    c2y = y1 + bend
    t = np.linspace(0, 1, n)
    x = (1-t)**3*x0 + 3*(1-t)**2*t*c1x + 3*(1-t)*t**2*c2x + t**3*x1
    y = (1-t)**3*y0 + 3*(1-t)**2*t*c1y + 3*(1-t)*t**2*c2y + t**3*y1
    return x, y




def prune_to_reachable(root_node: str, edges: pd.DataFrame) -> pd.DataFrame:
    """After visual filters, keep only edges still connected to the selected root."""
    if edges.empty:
        return edges
    active = {root_node}
    kept_ids = set()
    for _ in range(len(edges) + 1):
        newly = edges[edges["source_node"].isin(active) & ~edges["edge_id"].isin(kept_ids)]
        if newly.empty:
            break
        kept_ids.update(newly["edge_id"].tolist())
        before = len(active)
        active.update(newly["target_node"].tolist())
        if len(active) == before and len(newly) == 0:
            break
    return edges[edges["edge_id"].isin(kept_ids)].copy()

def layout_nodes(root_node: str, edges: pd.DataFrame, nodes: pd.DataFrame, mode: str):
    # Longest-path generation in this chronological DAG.
    gen = {root_node: 0}
    ordered = edges.sort_values(["buy_date", "sale_date", "edge_id"])
    for _ in range(max(2, len(ordered) + 1)):
        changed = False
        for _, e in ordered.iterrows():
            s, t = e["source_node"], e["target_node"]
            if s in gen:
                new = gen[s] + 1
                if new > gen.get(t, -1):
                    gen[t] = new
                    changed = True
        if not changed:
            break

    active_nodes = set(edges["source_node"]) | set(edges["target_node"]) | {root_node}
    node_info = nodes[nodes["node_id"].isin(active_nodes)].set_index("node_id")

    by_gen = {}
    for node in active_nodes:
        by_gen.setdefault(gen.get(node, 0), []).append(node)

    y = {root_node: 0.0}
    for g in sorted(by_gen):
        if g == 0:
            continue
        candidates = by_gen[g]
        parent_scores = []
        for node in candidates:
            incoming = edges[edges["target_node"] == node]
            parents = [(r["source_node"], r["root_flow"]) for _, r in incoming.iterrows() if r["source_node"] in y]
            if parents:
                denom = sum(w for _, w in parents) or 1.0
                score = sum(y[p] * w for p, w in parents) / denom
            else:
                score = 0.0
            date = node_info.loc[node, "date"] if node in node_info.index else pd.Timestamp.min
            parent_scores.append((score, date, node))
        parent_scores.sort(key=lambda z: (z[0], z[1], z[2]))
        n = len(parent_scores)
        spacing = max(1.2, 6.0 / max(1, n))
        positions = (np.arange(n) - (n - 1) / 2.0) * spacing
        for pos, (_, _, node) in zip(positions, parent_scores):
            y[node] = float(pos)

    if mode == "Calendar time":
        root_date = pd.Timestamp(node_info.loc[root_node, "date"]) if root_node in node_info.index else pd.Timestamp(edges["sale_date"].min())
        x = {}
        for node in active_nodes:
            d = pd.Timestamp(node_info.loc[node, "date"]) if node in node_info.index else root_date
            x[node] = max(0.0, (d - root_date).days / 365.25)
    else:
        x = {node: float(gen.get(node, 0)) for node in active_nodes}

    return x, y, node_info


def make_lineage_figure(root_row, root_edges, nodes, metric, curve_strength):
    if root_edges.empty:
        return go.Figure()
    root_node = root_row["node_id"]
    x, y, node_info = layout_nodes(root_node, root_edges, nodes, st.session_state.get("layout_mode", "Generation"))

    max_flow = max(float(root_edges["root_flow"].max()), 1e-6)
    fig = go.Figure()

    # Draw the thinnest links first so larger flows remain legible.
    for _, e in root_edges.sort_values("root_flow").iterrows():
        s, t = e["source_node"], e["target_node"]
        if s not in x or t not in x:
            continue
        perf = e[metric]
        if pd.isna(perf):
            direction = 0.0
            color = "rgba(125,125,125,0.55)"
        elif perf > 0:
            direction = 1.0
            color = "rgba(34,139,34,0.68)"
        elif perf < 0:
            direction = -1.0
            color = "rgba(190,45,45,0.68)"
        else:
            direction = 0.0
            color = "rgba(125,125,125,0.55)"

        dx = max(0.6, abs(x[t] - x[s]))
        magnitude = 0.55 if pd.isna(perf) else min(1.6, 0.55 + abs(float(perf)) * 1.2)
        bend = direction * curve_strength * magnitude * max(0.7, math.sqrt(dx))
        bx, by = bezier_points(x[s], y[s], x[t], y[t], bend)
        width = 1.3 + 11.0 * math.sqrt(max(float(e["root_flow"]), 0.0) / max_flow)

        hover = (
            f"<b>{e['source_company']} → {e['target_company']}</b><br>"
            f"Selected-root flow: {e['root_flow']:.3f}% of portfolio<br>"
            f"Total allocation on edge: {e['allocation_amount']:.3f}%<br>"
            f"Sale: {pd.Timestamp(e['sale_date']).date()}<br>"
            f"Purchase: {pd.Timestamp(e['buy_date']).date()}<br>"
            f"Funding gap: {e['gap_days'] if pd.notna(e['gap_days']) else 'n/a'} days<br>"
            f"Match confidence: {e['confidence']}<br>"
            f"Actual target return: {fmt_pct(e['actual_return'])}<br>"
            f"Sell-stock counterfactual: {fmt_pct(e['counterfactual_return'])}<br>"
            f"Total reallocation benefit: {fmt_pct(e['relative_total'])}<br>"
            f"Selection-only benefit: {fmt_pct(e['relative_selection'])}<br>"
            f"Performance coverage: {e['performance_coverage']:.0%}"
        )
        if bool(e.get("performance_truncated", False)):
            hover += "<br>Counterfactual truncated by return-series availability"

        fig.add_trace(go.Scatter(
            x=bx, y=by, mode="lines",
            line=dict(width=width, color=color),
            hovertemplate=hover + "<extra></extra>",
            showlegend=False,
        ))

    # Node flow determines marker size. Labels are company names; dates sit in hover to reduce clutter.
    node_flows = {}
    for _, e in root_edges.iterrows():
        node_flows[e["source_node"]] = max(node_flows.get(e["source_node"], 0.0), float(e["root_flow"]))
        node_flows[e["target_node"]] = node_flows.get(e["target_node"], 0.0) + float(e["root_flow"])

    node_ids = sorted(set(root_edges["source_node"]) | set(root_edges["target_node"]), key=lambda n: (x.get(n, 0), y.get(n, 0)))
    nx, ny, texts, hovers, sizes = [], [], [], [], []
    for node in node_ids:
        info = node_info.loc[node]
        nx.append(x[node]); ny.append(y[node])
        company = info["company"]
        texts.append(company)
        hovers.append(
            f"<b>{company}</b><br>"
            f"Node date: {pd.Timestamp(info['date']).date()}<br>"
            f"Transaction: {info['transaction_type']}<br>"
            f"Trade size: {float(info['amount']):.3f}%"
        )
        sizes.append(12 + 14 * math.sqrt(max(node_flows.get(node, 0.0), 0.0) / max_flow))

    fig.add_trace(go.Scatter(
        x=nx, y=ny, mode="markers+text",
        marker=dict(size=sizes, color="rgba(32,45,64,0.92)", line=dict(width=1, color="white")),
        text=texts, textposition="middle right",
        textfont=dict(size=10),
        hovertext=hovers, hovertemplate="%{hovertext}<extra></extra>",
        showlegend=False,
    ))

    fig.update_layout(
        height=min(1600, max(760, 30 * int(root_edges["target_node"].nunique()))),
        margin=dict(l=30, r=220, t=45, b=45),
        xaxis=dict(title="Generation" if st.session_state.get("layout_mode", "Generation") == "Generation" else "Years since original sale", showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hoverlabel=dict(align="left"),
        title=dict(text="Capital lineage: upward arcs = relative benefit, downward arcs = opportunity cost", x=0.01),
    )
    return fig


st.title("Portfolio Capital Lineage")
st.caption("Trace each original sale through subsequent reallocations and score every switch against the stock that was sold.")

with st.sidebar:
    st.header("Data")
    upload = st.file_uploader("Trade-history workbook", type=["xlsx"])
    local_default = Path("LTGG Full Trade History for New Sankey.xlsx")
    if upload is not None:
        file_bytes = upload.getvalue()
    elif local_default.exists():
        file_bytes = local_default.read_bytes()
        st.caption(f"Using local file: {local_default.name}")
    else:
        st.info("Upload the LTGG workbook to begin.")
        st.stop()

    st.header("Matching")
    max_gap = st.slider("Maximum sale-to-purchase funding gap (days)", 0, 365, 90, 5)
    st.caption("Closest prior sale dates are used first. Multiple sales on the same date are prorated.")

with st.spinner("Building capital lineage and trade-decision attribution..."):
    lineage, edges_perf, diagnostics, latest_return_date = build_cached(file_bytes, max_gap)

roots = root_selector_table(lineage, include_external=False)
if roots.empty:
    st.error("No sale roots were found in the workbook.")
    st.stop()

with st.sidebar:
    st.header("Lineage")
    labels = roots["label"].tolist()
    selected_label = st.selectbox("Original sale", labels, index=0)
    root_row = roots.loc[roots["label"] == selected_label].iloc[0]

    metric_label = st.radio("Performance measure", ["Total reallocation", "Stock selection only"], index=0)
    metric = "relative_total" if metric_label == "Total reallocation" else "relative_selection"
    st.session_state["layout_mode"] = st.radio("Horizontal layout", ["Generation", "Calendar time"], index=0)
    curve_strength = st.slider("Up/down curve strength", 0.15, 2.0, 0.75, 0.05)
    min_flow = st.slider("Hide selected-root flows below (% portfolio)", 0.0, 0.50, 0.02, 0.01)
    confidence_choices = st.multiselect(
        "Match confidence",
        ["Exact", "High", "Good", "Approximate", "Uncertain"],
        default=["Exact", "High", "Good", "Approximate", "Uncertain"],
    )

root_edges = selected_root_edges(lineage, edges_perf, root_row["root_id"])
if not root_edges.empty:
    root_edges = root_edges[(root_edges["root_flow"] >= min_flow) & (root_edges["confidence"].isin(confidence_choices))].copy()
    root_edges = prune_to_reachable(root_row["node_id"], root_edges)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Original sale", f"{root_row['amount']:.3f}%")
c2.metric("Decision edges", f"{len(root_edges):,}")
descendants = set(root_edges["target_company"].dropna()) if not root_edges.empty else set()
c3.metric("Descendant companies", f"{len(descendants):,}")
valid = root_edges.dropna(subset=[metric]) if not root_edges.empty else pd.DataFrame()
if len(valid):
    weighted = np.average(valid[metric], weights=np.maximum(valid["root_flow"], 1e-9))
    c4.metric("Flow-weighted decision benefit", f"{weighted:+.1%}")
else:
    c4.metric("Flow-weighted decision benefit", "n/a")

if root_edges.empty:
    st.warning("No descendant flows meet the current filters. Widen the funding gap, confidence set or minimum-flow threshold.")
else:
    fig = make_lineage_figure(root_row, root_edges, lineage["nodes"], metric, curve_strength)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    st.caption(
        "Line width is the amount of the selected original-sale ancestry flowing through that decision. "
        "Green/upward = relative benefit; red/downward = relative opportunity cost; grey = insufficient return data."
    )

    table = root_edges[[
        "source_company", "target_company", "sale_date", "buy_date", "root_flow", "allocation_amount",
        "gap_days", "confidence", "actual_return", "counterfactual_return", "relative_total",
        "relative_selection", "performance_coverage", "performance_truncated",
    ]].copy()
    table.columns = [
        "Sold", "Bought", "Sale date", "Buy date", "Selected-root flow (%)", "Total edge allocation (%)",
        "Gap days", "Confidence", "Bought return", "Sold-stock counterfactual", "Total reallocation benefit",
        "Selection-only benefit", "Performance coverage", "Counterfactual truncated",
    ]
    table = table.sort_values(["Sale date", "Buy date", "Selected-root flow (%)"], ascending=[True, True, False])

    st.subheader("Decision audit trail")
    display_table = table.copy()
    for col in ["Bought return", "Sold-stock counterfactual", "Total reallocation benefit", "Selection-only benefit", "Performance coverage"]:
        display_table[col] = display_table[col] * 100.0
    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Bought return": st.column_config.NumberColumn(format="%.1f%%"),
            "Sold-stock counterfactual": st.column_config.NumberColumn(format="%.1f%%"),
            "Total reallocation benefit": st.column_config.NumberColumn(format="%.1f%%"),
            "Selection-only benefit": st.column_config.NumberColumn(format="%.1f%%"),
            "Performance coverage": st.column_config.NumberColumn(format="%.0f%%"),
            "Selected-root flow (%)": st.column_config.NumberColumn(format="%.3f"),
            "Total edge allocation (%)": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    csv = table.to_csv(index=False).encode("utf-8")
    st.download_button("Download selected lineage CSV", csv, file_name="selected_capital_lineage.csv", mime="text/csv")

with st.expander("Model assumptions and diagnostics"):
    st.markdown(
        """
**Capital matching**

- Sales are made available before purchases on the same date.
- The closest prior sale date is matched first.
- If several sales occurred on the same date, their remaining proceeds are prorated across purchases.
- Purchases that cannot be matched inside the selected gap are attributed to `External / existing cash`.

**Position lineage**

- `% Portfolio Order` is the capital unit. It is not literal cash across decades.
- Partial sales release the known ancestry of a position pro rata.
- Complete sales release all tracked ancestry.
- A first-observed partial sale creates a synthetic residual legacy lot so later sales can retain that original ancestry.

**Performance**

- `Total reallocation` compares the bought stock from purchase date with the sold stock from sale date, so the cash-wait period is included.
- `Stock selection only` compares both securities from the purchase date.
- Return data use the workbook's daily total-return indices.
- If either return series ends before the natural decision horizon, the comparison is truncated rather than extrapolated.
        """
    )
    st.write(f"Latest return date in workbook: **{latest_return_date.date()}**")
    st.write(f"Unallocated matched-sale capital still sitting in the model: **{lineage['unallocated_sale_cash']:.3f}% portfolio-order units**")
    st.subheader("Return-series mapping")
    st.dataframe(diagnostics, use_container_width=True, hide_index=True)
