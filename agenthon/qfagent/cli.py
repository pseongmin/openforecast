"""The ``forecast`` verb the Agenthon harness invokes.

    forecast --panels /input/panels --text /input/text --asof YYYY-MM-DD --out /output/forecast.parquet

Writes ``forecast.parquet``, ``forecast_meta.json`` and ``forecast_rationale.md``
next to ``--out``. The card (``card.toml``) is located by walking up from the
panels directory, which is where a staged unit keeps it.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tomllib

import pandas as pd

from .engine import ForecastRequest, forecast_draws
from .text_signal import read_corpus, scenarios_from_corpus


def _find_card(panels_dir: pathlib.Path) -> pathlib.Path | None:
    for candidate in [panels_dir, *panels_dir.parents]:
        card = candidate / "card.toml"
        if card.exists():
            return card
    return None


def _load_panels(panels_dir: pathlib.Path, asof: pd.Timestamp) -> pd.DataFrame:
    files = sorted(panels_dir.glob("*.parquet"))
    if not files and panels_dir.parent.exists():
        files = sorted(panels_dir.parent.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet panels under {panels_dir}")
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        cols = {c.lower(): c for c in df.columns}
        if not {"date", "asset", "value"} <= set(cols):
            continue
        df = df.rename(columns={cols["date"]: "date", cols["asset"]: "asset", cols["value"]: "value"})
        frames.append(df[["date", "asset", "value"]])
    long = pd.concat(frames, ignore_index=True)
    long["date"] = pd.to_datetime(long["date"])
    long = long[long["date"] <= asof]  # the cutoff is a hard rule, not a suggestion
    wide = long.pivot_table(index="date", columns="asset", values="value", aggfunc="last")
    return wide.sort_index()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forecast")
    parser.add_argument("verb", nargs="?", default="forecast")
    parser.add_argument("--panels", required=True, type=pathlib.Path)
    parser.add_argument("--text", required=False, type=pathlib.Path)
    parser.add_argument("--asof", required=True)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--draws", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.verb != "forecast":
        parser.error(f"unknown verb {args.verb!r}")

    asof = pd.Timestamp(args.asof)
    card_path = _find_card(args.panels)
    if card_path is None:
        raise SystemExit("card.toml not found above --panels")
    card = tomllib.loads(card_path.read_text())
    targets = card["targets"]
    unit_id = card["task"]["id"]

    panel = _load_panels(args.panels, asof)
    corpus = read_corpus(args.text) if args.text and args.text.exists() else []
    scenarios, notes = scenarios_from_corpus(corpus, targets["asset_ids"])

    request = ForecastRequest(
        asset_ids=list(targets["asset_ids"]),
        horizons=[int(h) for h in targets["horizons"]],
        target_type=targets["target_type"],
        n_draws=max(200, args.draws),
        scenarios=scenarios,
    )
    frame = forecast_draws(panel, request)

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    meta = {
        "unit_id": unit_id,
        "asof": asof.strftime("%Y-%m-%d"),
        "asset_ids": request.asset_ids,
        "horizons": request.horizons,
        "representation": "samples",
        "n_draws": request.n_draws,
    }
    (out.parent / "forecast_meta.json").write_text(json.dumps(meta, indent=2))
    (out.parent / "forecast_rationale.md").write_text(_rationale(unit_id, request, panel, notes))
    print(
        f"wrote {out.name}, forecast_meta.json, forecast_rationale.md: "
        f"{len(request.asset_ids)} asset(s) x {len(request.horizons)} horizon(s), {request.n_draws} draws"
    )
    return 0


def _rationale(unit_id: str, request: ForecastRequest, panel: pd.DataFrame, notes: list[str]) -> str:
    last = panel.ffill().iloc[-1]
    lines = [
        f"# Forecast rationale — {unit_id}",
        "",
        f"As-of {panel.index[-1].date()}; targets {request.asset_ids} at horizons {request.horizons} "
        f"({request.target_type}).",
        "",
        "## Numerical backbone",
        "Stationary block bootstrap (mean block 10 business days) over the panel's own history, "
        "sampled jointly across assets so cross-asset dependence survives. Centre = last observed "
        "level (or zero cumulative log return); spread = the empirical h-step distribution.",
        "",
        "Anchor levels: " + ", ".join(f"{a}={last[a]:.4g}" for a in request.asset_ids if a in last.index),
        "",
        "## What the text corpus contributed",
    ]
    if request.scenarios:
        for s in request.scenarios:
            tag = "established" if s.established else "inferred"
            lines.append(
                f"- {s.name}: weight {s.weight:.2f}, drift {s.drift_in_sd:+.2f} sd, "
                f"spread x{s.spread_multiple:.2f} ({tag})"
            )
    else:
        lines.append("- No scenario adjustment: the corpus carried no signal the rules let us use.")
    note_lines = [f"- {n}" for n in notes] or ["- none"]
    lines += ["", "## Notes", *note_lines, ""]
    lines += [
        "## What would change this forecast",
        "A dated document in the corpus stating a decision (not a tone) inside the horizon, or a "
        "regime change in realised volatility large enough to move the empirical spread.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
