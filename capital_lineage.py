from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import difflib
import math
import re

import numpy as np
import pandas as pd


EPS = 1e-10


MANUAL_SERIES_ALIASES = {
    "AMERICA LATINA LOGISTICA": "RUMO ON",
    "AMERICA LATINA LOGISTICA UNITS": "RUMO ON",
    "BEIGENE LTD": "BEONE MEDICINES",
    "CATL STOCK CONNECT": "CONTEMPORARY AMPEREX TECHNOLOGY",
    "FACEBOOK CL": "META PLATFORMS",
    "META PLATFORMS INC": "META PLATFORMS",
    "INDITEX": "INDUSTRIA DE DISENO TEXTIL",
    "PINDUODUO INC ADR": "PDD HOLDINGS",
    "PDD HOLDINGS INC": "PDD HOLDINGS",
    "PROGRESSIVE": "PROG HOLDINGS",
    "SEATTLE GENETICS": "SEAGEN",
    "TSMC": "TAIWAN SEMICON",
    "FERGUSON OLD": "FERGUSON ENTERPRISES",
    "Q CELLS": "HANWHA Q CELLS",
    "SCP POOL CORPORATION": "POOL",
    "WRIGLEY": "WILLIAM WRIGLEY",
    "PETROBRAS COMMON ADR": "PETROLEO BRASILEIRO",
    "NEW ORIENTAL EDUCATION TECHNOLOGY ADR": "NEW ORNTL",
    "NEW ORIENTAL EDUCATION TECHNOLOGY SPONSORED ADR": "NEW ORNTL",
    "SHOPIFY": "SHOPIFY SUBORDINATE",
    "VCA ANTECH": "VCA DEAD",
    "LUKOIL ADR": "LUKOIL OAO",
    "PULTE HOMES": "PULTEGROUP",
    "SAMSUNG ELEC GDR": "SAMSUNG ELECTRONICS",
    "SAMSUNG ELEC COMMON GDR REG": "SAMSUNG ELECTRONICS",
    "SEA LTD ADR": "SEA 'A'",
    "DIS DEUTSCHER INDUSTRIE": "DIS DT.INDUSTRIE",
    "HERMES INTERNATIONAL": "HERMES INTL",
    "VALE PREF ADR": "VALE PREFERRED ADR",
    "VESTAS WIND SYSTEMS": "VESTAS WINDSYSTEMS",
    "THE TRADE DESK": "TRADE DESK",
    "MEITUAN DIANPING": "MEITUAN",
    "E L F BEAUTY": "ELF BEAUTY",
    "ASML": "ASML HOLDING",
    "ALIBABA HK LINE": "ALIBABA GROUP HOLDING - TOT RETURN IND",
    "NEW ORIENTAL EDUCATION AND TECHNOLOGY SPONSORED ADR": "NEW ORNTL",
}


def _clean_company_key(value: str) -> str:
    s = str(value).upper().replace("&", " AND ")
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    stop = {
        "INC", "CORP", "CORPORATION", "LTD", "LIMITED", "PLC", "GROUP",
        "HOLDINGS", "HOLDING", "CLASS", "SPONSORED", "COMMON", "REG", "S",
        "ADR", "ADS", "ORD", "NV", "SA", "AG", "CO", "COMPANY", "A", "B", "C",
    }
    tokens = [t for t in s.split() if t not in stop]
    return " ".join(tokens)


def _clean_series_key(value: str) -> str:
    s = str(value).upper().replace("&", " AND ")
    s = re.sub(r"\s+-\s+TOT\s+RETURN\s+IND.*$", "", s)
    s = re.sub(r"\s+DEAD\s+-.*$", "", s)
    s = re.sub(r"\s+DEAD.*$", "", s)
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    stop = {
        "INC", "CORP", "CORPORATION", "LTD", "LIMITED", "PLC", "GROUP",
        "HOLDINGS", "HOLDING", "CLASS", "SPONSORED", "COMMON", "REG", "S",
        "ADR", "ADS", "ORD", "NV", "SA", "AG", "CO", "COMPANY", "A", "B", "C",
        "TOT", "RETURN", "IND",
    }
    tokens = [t for t in s.split() if t not in stop and not t.isdigit()]
    return " ".join(tokens)


def _similarity(company: str, series: str) -> float:
    a = _clean_company_key(company)
    b = _clean_series_key(series)
    if not a or not b:
        return 0.0
    sa, sb = set(a.split()), set(b.split())
    jac = len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0
    seq = difflib.SequenceMatcher(None, a, b).ratio()
    contains = 1.0 if (a in b or b in a) else 0.0
    return 0.50 * seq + 0.45 * jac + 0.05 * contains


def _find_manual_alias(company: str) -> Optional[str]:
    key = _clean_company_key(company)
    if key in MANUAL_SERIES_ALIASES:
        return MANUAL_SERIES_ALIASES[key]
    # Some keys retain meaningful short words after cleaning.
    for raw_alias_key, target in MANUAL_SERIES_ALIASES.items():
        alias_key = _clean_company_key(raw_alias_key)
        if key == alias_key or (len(alias_key) >= 7 and alias_key in key):
            return target
    return None


def read_workbook(source: Union[str, Path, bytes, bytearray, BytesIO]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the trade sheet and total-return matrix from the supplied workbook."""
    if isinstance(source, (bytes, bytearray)):
        source = BytesIO(source)

    trades = pd.read_excel(source, sheet_name="Export")
    if hasattr(source, "seek"):
        source.seek(0)
    raw = pd.read_excel(source, sheet_name="Sheet1", header=None)

    required = ["Earliest Trade Date", "Order Direction", "Transaction Type", "Instrument Name", "% Portfolio Order"]
    missing = [c for c in required if c not in trades.columns]
    if missing:
        raise ValueError(f"Export sheet is missing required columns: {', '.join(missing)}")

    trades = trades[required].copy()
    trades["Earliest Trade Date"] = pd.to_datetime(trades["Earliest Trade Date"], errors="coerce")
    trades["% Portfolio Order"] = pd.to_numeric(trades["% Portfolio Order"], errors="coerce")
    trades = trades.dropna(subset=["Earliest Trade Date", "Order Direction", "Instrument Name", "% Portfolio Order"])
    trades["Order Direction"] = trades["Order Direction"].astype(str).str.upper().str.strip()
    trades["Transaction Type"] = trades["Transaction Type"].astype(str).str.strip()
    trades["Instrument Name"] = trades["Instrument Name"].astype(str).str.strip()
    trades["trade_id"] = np.arange(1, len(trades) + 1)

    series_headers = [str(x) for x in raw.iloc[0, 4:].tolist()]
    dates = pd.to_datetime(raw.iloc[1:, 3], errors="coerce")
    values = raw.iloc[1:, 4:].copy()
    values.columns = series_headers
    values.index = dates
    values = values[~values.index.isna()]
    values = values.apply(pd.to_numeric, errors="coerce")
    values = values.sort_index()
    values = values[~values.index.duplicated(keep="last")]

    source_labels = raw.iloc[:, 0].dropna().astype(str).to_frame("source_label")
    return trades, values, source_labels


@dataclass
class ReturnStore:
    data: pd.DataFrame
    mapping: Dict[str, Optional[str]]
    diagnostics: pd.DataFrame

    @property
    def latest_date(self) -> pd.Timestamp:
        if self.data.empty:
            return pd.Timestamp.today().normalize()
        return pd.Timestamp(self.data.index.max()).normalize()

    def _series(self, company: str) -> Optional[pd.Series]:
        col = self.mapping.get(company)
        if not col or col not in self.data.columns:
            return None
        s = self.data[col].dropna()
        return s if not s.empty else None

    def last_date(self, company: str) -> Optional[pd.Timestamp]:
        s = self._series(company)
        return pd.Timestamp(s.index[-1]).normalize() if s is not None and len(s) else None

    def wealth(self, company: str, start: pd.Timestamp, end: pd.Timestamp) -> Optional[float]:
        """Total-return wealth multiple from the first observation on/after start to last on/before end."""
        s = self._series(company)
        if s is None:
            return None
        start = pd.Timestamp(start).normalize()
        end = pd.Timestamp(end).normalize()
        if end < start:
            return None

        idx = s.index.values.astype("datetime64[ns]")
        start_np = np.datetime64(start.to_datetime64())
        end_np = np.datetime64(end.to_datetime64())
        i0 = int(np.searchsorted(idx, start_np, side="left"))
        i1 = int(np.searchsorted(idx, end_np, side="right") - 1)
        if i0 >= len(s) or i1 < 0 or i1 < i0:
            return None
        v0 = float(s.iloc[i0])
        v1 = float(s.iloc[i1])
        if not np.isfinite(v0) or not np.isfinite(v1) or v0 == 0:
            return None
        return v1 / v0


def build_return_store(trades: pd.DataFrame, return_data: pd.DataFrame, threshold: float = 0.43) -> ReturnStore:
    headers = list(return_data.columns)
    mapping: Dict[str, Optional[str]] = {}
    rows: List[Dict[str, Any]] = []

    for company in sorted(trades["Instrument Name"].dropna().unique()):
        alias = _find_manual_alias(company)
        chosen = None
        score = 0.0
        method = "unmatched"

        if alias:
            candidates = [(h, _similarity(alias, h)) for h in headers if alias.upper() in str(h).upper()]
            if not candidates:
                candidates = [(h, _similarity(alias, h)) for h in headers]
            if candidates:
                chosen, score = max(candidates, key=lambda x: x[1])
                method = "alias"

        if chosen is None:
            scored = [(h, _similarity(company, h)) for h in headers]
            chosen, score = max(scored, key=lambda x: x[1]) if scored else (None, 0.0)
            method = "fuzzy"

        # Manual aliases are explicit; fuzzy mappings have to clear the safety threshold.
        if method == "fuzzy" and score < threshold:
            chosen = None
            method = "unmatched"

        mapping[company] = chosen
        rows.append({
            "Company": company,
            "Return series": chosen,
            "Match score": score,
            "Method": method,
            "Available": chosen is not None,
        })

    diagnostics = pd.DataFrame(rows).sort_values(["Available", "Match score", "Company"], ascending=[True, True, True])
    return ReturnStore(return_data, mapping, diagnostics)


def confidence_from_gap(gap_days: Optional[int]) -> str:
    if gap_days is None:
        return "External"
    if gap_days == 0:
        return "Exact"
    if gap_days <= 3:
        return "High"
    if gap_days <= 7:
        return "Good"
    if gap_days <= 30:
        return "Approximate"
    return "Uncertain"


def _scale_map(values: Dict[str, float], factor: float) -> Dict[str, float]:
    return {k: v * factor for k, v in values.items() if abs(v * factor) > EPS}


def _add_maps(target: Dict[str, float], addition: Dict[str, float]) -> None:
    for k, v in addition.items():
        target[k] = target.get(k, 0.0) + v


def build_lineage(
    trades: pd.DataFrame,
    max_gap_days: int = 90,
    legacy_partial_residual_multiple: float = 1.0,
    partial_sale_max_fraction: float = 0.95,
) -> Dict[str, Any]:
    """
    Build a chronological, lot-aware capital lineage.

    Important approximations:
    - % Portfolio Order is treated as the capital unit.
    - Partial sales release all known position ancestries pro rata.
    - If a partial sale is larger than tracked nominal lot mass, the removal fraction is capped,
      because order weights at different dates are not directly additive.
    - A first-observed partial sale creates a synthetic legacy residual so later sales of the
      pre-history position can keep the same root ancestry.
    """
    t = trades.copy()
    t["amount"] = t["% Portfolio Order"].abs().astype(float)
    t["date"] = pd.to_datetime(t["Earliest Trade Date"]).dt.normalize()
    # Same-day sales are made available before same-day purchases.
    t["direction_order"] = t["Order Direction"].map({"SALE": 0, "PURCHASE": 1}).fillna(2)
    t = t.sort_values(["date", "direction_order", "trade_id"]).reset_index(drop=True)

    positions: Dict[str, List[Dict[str, Any]]] = {}
    cash_lots: List[Dict[str, Any]] = []
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []
    roots: Dict[str, Dict[str, Any]] = {}
    exit_events: Dict[str, List[Dict[str, Any]]] = {}
    edge_counter = 0

    for current_date, day in t.groupby("date", sort=True):
        sales = day[day["Order Direction"] == "SALE"]
        purchases = day[day["Order Direction"] == "PURCHASE"]

        # 1) Release sale proceeds into dated cash lots.
        for _, row in sales.iterrows():
            company = row["Instrument Name"]
            sale_amount = float(row["amount"])
            trade_id = int(row["trade_id"])
            tx_type = row["Transaction Type"]
            lots = [lot for lot in positions.get(company, []) if lot["remaining_mass"] > EPS]
            total_mass = sum(lot["remaining_mass"] for lot in lots)

            if not lots or total_mass <= EPS:
                root_id = f"root_sale_{trade_id}"
                node_id = root_id
                nodes[node_id] = {
                    "node_id": node_id,
                    "company": company,
                    "node_type": "root_sale",
                    "date": current_date,
                    "trade_id": trade_id,
                    "transaction_type": tx_type,
                    "amount": sale_amount,
                }
                roots[root_id] = {
                    "root_id": root_id,
                    "node_id": node_id,
                    "company": company,
                    "date": current_date,
                    "trade_id": trade_id,
                    "transaction_type": tx_type,
                    "amount": sale_amount,
                    "root_type": "sale",
                }
                cash_lots.append({
                    "source_node": node_id,
                    "sale_trade_id": trade_id,
                    "sale_company": company,
                    "sale_date": current_date,
                    "remaining_amount": sale_amount,
                    "root_amounts": {root_id: sale_amount},
                })

                # Keep a synthetic residual for a first-observed partial sale so later releases
                # continue to inherit the same root instead of starting a new unrelated root.
                if str(tx_type).lower() == "partial sale" and legacy_partial_residual_multiple > 0:
                    residual = sale_amount * legacy_partial_residual_multiple
                    positions.setdefault(company, []).append({
                        "node_id": node_id,
                        "company": company,
                        "initial_mass": sale_amount + residual,
                        "remaining_mass": residual,
                        "root_amounts": {root_id: residual},
                        "synthetic_legacy": True,
                    })
                continue

            is_complete = str(tx_type).lower() == "complete sale"
            if is_complete:
                removal_fraction = 1.0
            else:
                raw_fraction = sale_amount / total_mass if total_mass > EPS else partial_sale_max_fraction
                removal_fraction = max(0.0, min(raw_fraction, partial_sale_max_fraction))

            for lot in lots:
                lot_weight = lot["remaining_mass"] / total_mass
                proceeds_share = sale_amount * lot_weight
                root_total = sum(lot["root_amounts"].values())
                if root_total <= EPS:
                    continue
                out_roots = {k: proceeds_share * (v / root_total) for k, v in lot["root_amounts"].items()}
                cash_lots.append({
                    "source_node": lot["node_id"],
                    "sale_trade_id": trade_id,
                    "sale_company": company,
                    "sale_date": current_date,
                    "remaining_amount": proceeds_share,
                    "root_amounts": out_roots,
                })

                removed_nominal = lot["remaining_mass"] * removal_fraction
                initial_mass = max(float(lot["initial_mass"]), EPS)
                exit_events.setdefault(lot["node_id"], []).append({
                    "date": current_date,
                    "fraction_of_initial": removed_nominal / initial_mass,
                    "sale_trade_id": trade_id,
                    "sale_amount_share": proceeds_share,
                })
                lot["remaining_mass"] *= (1.0 - removal_fraction)
                lot["root_amounts"] = _scale_map(lot["root_amounts"], 1.0 - removal_fraction)

        # 2) Allocate the closest prior sale pools into purchases. Same-date sale pools are prorated.
        for _, row in purchases.iterrows():
            company = row["Instrument Name"]
            buy_amount = float(row["amount"])
            trade_id = int(row["trade_id"])
            tx_type = row["Transaction Type"]
            node_id = f"buy_{trade_id}"
            nodes[node_id] = {
                "node_id": node_id,
                "company": company,
                "node_type": "purchase",
                "date": current_date,
                "trade_id": trade_id,
                "transaction_type": tx_type,
                "amount": buy_amount,
            }

            need = buy_amount
            target_roots: Dict[str, float] = {}

            eligible_dates = sorted({
                lot["sale_date"] for lot in cash_lots
                if lot["remaining_amount"] > EPS
                and lot["sale_date"] <= current_date
                and (current_date - lot["sale_date"]).days <= max_gap_days
            }, reverse=True)

            for sale_date in eligible_dates:
                if need <= EPS:
                    break
                group = [lot for lot in cash_lots if lot["remaining_amount"] > EPS and lot["sale_date"] == sale_date]
                group_total = sum(lot["remaining_amount"] for lot in group)
                if group_total <= EPS:
                    continue
                take_total = min(need, group_total)
                # Prorate across all sale lots on the same date.
                snapshots = [(lot, lot["remaining_amount"], dict(lot["root_amounts"])) for lot in group]
                for lot, before_amount, before_roots in snapshots:
                    take = take_total * before_amount / group_total
                    if take <= EPS:
                        continue
                    frac = take / before_amount
                    piece_roots = _scale_map(before_roots, frac)
                    lot["remaining_amount"] = max(0.0, lot["remaining_amount"] - take)
                    lot["root_amounts"] = {k: max(0.0, v - piece_roots.get(k, 0.0)) for k, v in lot["root_amounts"].items()}
                    lot["root_amounts"] = {k: v for k, v in lot["root_amounts"].items() if v > EPS}
                    _add_maps(target_roots, piece_roots)

                    gap = int((current_date - lot["sale_date"]).days)
                    edge_counter += 1
                    edges.append({
                        "edge_id": f"edge_{edge_counter}",
                        "source_node": lot["source_node"],
                        "target_node": node_id,
                        "source_company": lot["sale_company"],
                        "target_company": company,
                        "source_sale_trade_id": lot["sale_trade_id"],
                        "target_trade_id": trade_id,
                        "sale_date": lot["sale_date"],
                        "buy_date": current_date,
                        "allocation_amount": take,
                        "root_amounts": piece_roots,
                        "gap_days": gap,
                        "confidence": confidence_from_gap(gap),
                        "target_transaction_type": tx_type,
                    })
                need -= take_total

            # Unfunded purchases become explicit external/existing-cash roots. They stay out of the
            # default root-sale selector but preserve ancestry if that capital is later sold.
            if need > EPS:
                root_id = f"external_{trade_id}"
                ext_node = root_id
                nodes[ext_node] = {
                    "node_id": ext_node,
                    "company": "External / existing cash",
                    "node_type": "external",
                    "date": current_date,
                    "trade_id": trade_id,
                    "transaction_type": "External funding",
                    "amount": need,
                }
                roots[root_id] = {
                    "root_id": root_id,
                    "node_id": ext_node,
                    "company": "External / existing cash",
                    "date": current_date,
                    "trade_id": trade_id,
                    "transaction_type": "External funding",
                    "amount": need,
                    "root_type": "external",
                }
                piece_roots = {root_id: need}
                _add_maps(target_roots, piece_roots)
                edge_counter += 1
                edges.append({
                    "edge_id": f"edge_{edge_counter}",
                    "source_node": ext_node,
                    "target_node": node_id,
                    "source_company": "External / existing cash",
                    "target_company": company,
                    "source_sale_trade_id": None,
                    "target_trade_id": trade_id,
                    "sale_date": current_date,
                    "buy_date": current_date,
                    "allocation_amount": need,
                    "root_amounts": piece_roots,
                    "gap_days": None,
                    "confidence": "External",
                    "target_transaction_type": tx_type,
                })

            positions.setdefault(company, []).append({
                "node_id": node_id,
                "company": company,
                "initial_mass": buy_amount,
                "remaining_mass": buy_amount,
                "root_amounts": target_roots,
                "synthetic_legacy": False,
            })

    # Terminal fraction for each purchase/legacy node.
    terminal_fractions: Dict[str, float] = {}
    for lots in positions.values():
        for lot in lots:
            initial = max(float(lot["initial_mass"]), EPS)
            terminal_fractions[lot["node_id"]] = terminal_fractions.get(lot["node_id"], 0.0) + lot["remaining_mass"] / initial

    # Root-flow expansion makes filtering a selected original sale cheap in Streamlit.
    root_edge_rows: List[Dict[str, Any]] = []
    for edge in edges:
        for root_id, amount in edge["root_amounts"].items():
            if amount > EPS:
                root_edge_rows.append({
                    "root_id": root_id,
                    "edge_id": edge["edge_id"],
                    "root_flow": amount,
                })

    edges_df = pd.DataFrame(edges)
    roots_df = pd.DataFrame(list(roots.values()))
    nodes_df = pd.DataFrame(list(nodes.values()))
    root_edges_df = pd.DataFrame(root_edge_rows)

    return {
        "trades": t,
        "nodes": nodes_df,
        "edges": edges_df,
        "roots": roots_df,
        "root_edges": root_edges_df,
        "exit_events": exit_events,
        "terminal_fractions": terminal_fractions,
        "unallocated_sale_cash": sum(lot["remaining_amount"] for lot in cash_lots if lot["remaining_amount"] > EPS),
    }


def add_edge_performance(lineage: Dict[str, Any], returns: ReturnStore, as_of: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Attach both attribution-style and intuitive absolute performance metrics to each edge.

    The absolute metrics compare the bought company with the company sold over the full
    holding horizon of the bought lot: from the relevant trade date to the last point
    that any of that purchase lot is held. If the lot is still held, the horizon is the
    latest return date in the workbook. Absolute outperformance is the percentage-point
    difference between those two cumulative total returns.
    """
    edges = lineage["edges"].copy()
    if edges.empty:
        return edges
    exit_events = lineage["exit_events"]
    terminal = lineage["terminal_fractions"]
    as_of = pd.Timestamp(as_of or returns.latest_date).normalize()

    perf_rows = []
    for _, edge in edges.iterrows():
        source = edge["source_company"]
        target = edge["target_company"]
        target_node = edge["target_node"]
        sale_date = pd.Timestamp(edge["sale_date"]).normalize()
        buy_date = pd.Timestamp(edge["buy_date"]).normalize()

        # Last point held for the bought lot. A remaining terminal fraction means the
        # lot is still held through the latest date in the return workbook.
        exit_dates = [
            pd.Timestamp(ev["date"]).normalize()
            for ev in exit_events.get(target_node, [])
            if float(ev.get("fraction_of_initial", 0.0)) > EPS
        ]
        rem = max(0.0, float(terminal.get(target_node, 0.0)))
        if rem > EPS:
            intended_last_held = as_of
        elif exit_dates:
            intended_last_held = max(exit_dates)
        else:
            intended_last_held = as_of
        intended_last_held = min(intended_last_held, as_of)

        absolute_fields = {
            "absolute_target_return": np.nan,
            "absolute_sold_return": np.nan,
            "absolute_outperformance": np.nan,
            "absolute_performance_end": intended_last_held,
            "absolute_performance_available": False,
        }

        if source != "External / existing cash":
            source_last = returns.last_date(source)
            target_last = returns.last_date(target)

            # The bought-company return can be displayed whenever that security has
            # full data through the last-held date. The sold-company comparison and
            # outperformance require both securities to cover that same horizon.
            if target_last is not None and target_last >= intended_last_held and intended_last_held >= buy_date:
                target_wealth = returns.wealth(target, buy_date, intended_last_held)
                if target_wealth is not None:
                    absolute_fields["absolute_target_return"] = target_wealth - 1.0

            if (
                source_last is not None
                and target_last is not None
                and source_last >= intended_last_held
                and target_last >= intended_last_held
                and intended_last_held >= buy_date
            ):
                target_wealth = returns.wealth(target, buy_date, intended_last_held)
                sold_wealth = returns.wealth(source, sale_date, intended_last_held)
                if target_wealth is not None and sold_wealth is not None:
                    target_return = target_wealth - 1.0
                    sold_return = sold_wealth - 1.0
                    absolute_fields.update({
                        "absolute_target_return": target_return,
                        "absolute_sold_return": sold_return,
                        "absolute_outperformance": target_return - sold_return,
                        "absolute_performance_available": True,
                    })

        if source == "External / existing cash":
            perf_rows.append({
                "edge_id": edge["edge_id"],
                "relative_total": np.nan,
                "relative_selection": np.nan,
                "actual_return": np.nan,
                "counterfactual_return": np.nan,
                "performance_coverage": 0.0,
                "performance_end": pd.NaT,
                "performance_truncated": False,
                **absolute_fields,
            })
            continue

        # Existing cohort-weighted attribution calculation. This remains available in
        # the audit data even though the main visual now uses the absolute comparison.
        cohorts = []
        for ev in exit_events.get(target_node, []):
            w = max(0.0, float(ev.get("fraction_of_initial", 0.0)))
            if w > EPS:
                cohorts.append((w, pd.Timestamp(ev["date"]).normalize()))
        if rem > EPS:
            cohorts.append((rem, as_of))
        if not cohorts:
            cohorts = [(1.0, as_of)]

        source_last = returns.last_date(source)
        target_last = returns.last_date(target)
        if source_last is None or target_last is None:
            perf_rows.append({
                "edge_id": edge["edge_id"],
                "relative_total": np.nan,
                "relative_selection": np.nan,
                "actual_return": np.nan,
                "counterfactual_return": np.nan,
                "performance_coverage": 0.0,
                "performance_end": pd.NaT,
                "performance_truncated": False,
                **absolute_fields,
            })
            continue

        valid_weight = 0.0
        actual_total_sum = 0.0
        cf_total_sum = 0.0
        actual_sel_sum = 0.0
        cf_sel_sum = 0.0
        end_dates = []
        truncated = False

        for weight, intended_end in cohorts:
            effective_end = min(intended_end, source_last, target_last, as_of)
            if effective_end < buy_date:
                continue
            actual = returns.wealth(target, buy_date, effective_end)
            cf_total = returns.wealth(source, sale_date, effective_end)
            cf_sel = returns.wealth(source, buy_date, effective_end)
            if actual is None or cf_total is None or cf_sel is None or cf_total <= 0 or cf_sel <= 0:
                continue
            valid_weight += weight
            actual_total_sum += weight * actual
            cf_total_sum += weight * cf_total
            actual_sel_sum += weight * actual
            cf_sel_sum += weight * cf_sel
            end_dates.append(effective_end)
            if effective_end < intended_end:
                truncated = True

        if valid_weight <= EPS:
            result = {
                "edge_id": edge["edge_id"],
                "relative_total": np.nan,
                "relative_selection": np.nan,
                "actual_return": np.nan,
                "counterfactual_return": np.nan,
                "performance_coverage": 0.0,
                "performance_end": pd.NaT,
                "performance_truncated": truncated,
                **absolute_fields,
            }
        else:
            actual_total = actual_total_sum / valid_weight
            cf_total = cf_total_sum / valid_weight
            actual_sel = actual_sel_sum / valid_weight
            cf_sel = cf_sel_sum / valid_weight
            result = {
                "edge_id": edge["edge_id"],
                "relative_total": actual_total / cf_total - 1.0,
                "relative_selection": actual_sel / cf_sel - 1.0,
                "actual_return": actual_total - 1.0,
                "counterfactual_return": cf_total - 1.0,
                "performance_coverage": min(1.0, valid_weight),
                "performance_end": max(end_dates) if end_dates else pd.NaT,
                "performance_truncated": truncated,
                **absolute_fields,
            }
        perf_rows.append(result)

    perf = pd.DataFrame(perf_rows)
    return edges.merge(perf, on="edge_id", how="left")

def selected_root_edges(lineage: Dict[str, Any], edges_with_perf: pd.DataFrame, root_id: str) -> pd.DataFrame:
    rel = lineage["root_edges"]
    if rel.empty:
        return pd.DataFrame()
    rel = rel[rel["root_id"] == root_id][["edge_id", "root_flow"]]
    return rel.merge(edges_with_perf, on="edge_id", how="left")


def root_selector_table(lineage: Dict[str, Any], include_external: bool = False) -> pd.DataFrame:
    roots = lineage["roots"].copy()
    if roots.empty:
        return roots
    if not include_external:
        roots = roots[roots["root_type"] == "sale"]
    roots = roots.sort_values(["date", "company", "trade_id"])
    roots["label"] = roots.apply(
        lambda r: f"{pd.Timestamp(r['date']).date()} | {r['company']} | {r['transaction_type']} | {r['amount']:.3f}%",
        axis=1,
    )
    return roots


def sale_selector_table(lineage: Dict[str, Any]) -> pd.DataFrame:
    """Return every sale trade as a selectable starting point for a forward lineage view."""
    trades = lineage["trades"].copy()
    if trades.empty:
        return trades

    sales = trades[trades["Order Direction"] == "SALE"].copy()
    if sales.empty:
        return sales

    sales = sales.sort_values(["date", "company" if "company" in sales.columns else "Instrument Name", "trade_id"], ascending=[False, True, False])
    sales["node_id"] = sales["trade_id"].map(lambda x: f"selected_sale_{int(x)}")
    sales["label"] = sales.apply(
        lambda r: (
            f"{pd.Timestamp(r['date']).date()} | {r['Instrument Name']} | "
            f"{r['Transaction Type']} | {float(r['amount']):.3f}%"
        ),
        axis=1,
    )
    return sales


def selected_sale_edges(
    lineage: Dict[str, Any],
    edges_with_perf: pd.DataFrame,
    sale_trade_id: int,
) -> pd.DataFrame:
    """
    Trace capital forward from one specific sale trade.

    The selected sale becomes a synthetic display root. Direct purchases funded by that
    sale are seeded at their full allocation amount. At later nodes, the selected-sale
    ancestry is propagated pro rata through subsequent sales and reallocations.
    """
    if edges_with_perf.empty:
        return pd.DataFrame()

    sale_trade_id = int(sale_trade_id)
    root_node = f"selected_sale_{sale_trade_id}"
    node_amounts = lineage["nodes"].set_index("node_id")["amount"].astype(float).to_dict()

    selected_fraction: Dict[str, float] = {}
    rows: List[Dict[str, Any]] = []

    # build_lineage creates edges in chronological order. add_edge_performance preserves
    # that left-hand edge order, which gives us a topological traversal of the DAG.
    for edge in edges_with_perf.itertuples(index=False):
        source_sale_trade_id = getattr(edge, "source_sale_trade_id", None)
        is_direct = False
        if source_sale_trade_id is not None and not pd.isna(source_sale_trade_id):
            is_direct = int(source_sale_trade_id) == sale_trade_id

        allocation = float(getattr(edge, "allocation_amount"))
        if is_direct:
            selected_flow = allocation
        else:
            selected_flow = allocation * selected_fraction.get(getattr(edge, "source_node"), 0.0)

        if selected_flow <= EPS:
            continue

        row = edge._asdict()
        row["original_source_node"] = row["source_node"]
        row["root_flow"] = selected_flow
        row["is_direct_from_selected_sale"] = is_direct
        if is_direct:
            row["source_node"] = root_node
        rows.append(row)

        target_node = getattr(edge, "target_node")
        target_amount = float(node_amounts.get(target_node, 0.0) or 0.0)
        if target_amount > EPS:
            selected_fraction[target_node] = min(
                1.0,
                selected_fraction.get(target_node, 0.0) + selected_flow / target_amount,
            )

    return pd.DataFrame(rows)
