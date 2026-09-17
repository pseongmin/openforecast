"""Cache public time-series history used by the ForecastBench dataset questions.

Sources: Yahoo Finance (yfinance), the FRED public CSV endpoint, and the DBnomics
public API. All three are free public endpoints and no credential is used.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

CACHE = Path(__file__).resolve().parents[1] / "cache"
UA = {"User-Agent": "openforecast/0.1 (public benchmark research)"}


def cache_yfinance(tickers: list[str], out: Path) -> dict:
    import yfinance as yf

    out.mkdir(parents=True, exist_ok=True)
    report = {"ok": 0, "empty": 0, "error": 0}
    todo = [t for t in tickers if not (out / f"{t}.csv").exists()]
    for i in range(0, len(todo), 40):
        batch = todo[i : i + 40]
        try:
            df = yf.download(
                batch, period="10y", interval="1d", auto_adjust=True,
                group_by="ticker", progress=False, threads=True,
            )
        except Exception as exc:
            print(f"batch {i} error {type(exc).__name__}: {exc}", flush=True)
            report["error"] += len(batch)
            continue
        for t in batch:
            try:
                sub = df[t] if len(batch) > 1 else df
                sub = sub.dropna(how="all")
                if sub.empty:
                    report["empty"] += 1
                    continue
                sub[["Close"]].to_csv(out / f"{t}.csv")
                report["ok"] += 1
            except Exception:
                report["empty"] += 1
        print(f"yfinance {i + len(batch)}/{len(todo)} ok={report['ok']}", flush=True)
        time.sleep(2.0)
    return report


def cache_fred(series_ids: list[str], out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    report = {"ok": 0, "empty": 0, "error": 0}
    for sid in series_ids:
        path = out / f"{sid}.csv"
        if path.exists():
            continue
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
        try:
            # FRED's edge drops requests carrying a non-default User-Agent
            # (measured 2026-09-17: custom UA = 20 s timeout, default UA = 0.2 s).
            r = requests.get(url, timeout=30)
        except Exception as exc:
            print(f"fred {sid} error {type(exc).__name__}", flush=True)
            report["error"] += 1
            continue
        if r.status_code != 200 or not r.text.startswith("observation_date"):
            report["empty"] += 1
            continue
        path.write_text(r.text)
        report["ok"] += 1
        time.sleep(0.3)
    print(f"fred done {report}", flush=True)
    return report


def split_dbnomics(code: str):
    """ForecastBench dbnomics ids look like ``provider_DATASET_series.code``."""
    if "_" not in code:
        return None, None, None
    provider, rest = code.split("_", 1)
    if "_" not in rest:
        return None, None, None
    dataset, series = rest.split("_", 1)
    return provider, dataset, series


def cache_dbnomics(series_codes: list[str], out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    report = {"ok": 0, "empty": 0, "error": 0}
    for code in series_codes:
        path = out / f"{code.replace('/', '__')}.csv"
        if path.exists():
            continue
        provider, dataset, series = split_dbnomics(code)
        if provider is None:
            report["empty"] += 1
            continue
        url = (
            f"https://api.db.nomics.world/v22/series/{provider}/{dataset}/{series}"
            "?observations=1"
        )
        try:
            docs = requests.get(url, headers=UA, timeout=30).json()["series"]["docs"]
        except Exception as exc:
            print(f"dbnomics {code} error {type(exc).__name__}", flush=True)
            report["error"] += 1
            continue
        if not docs:
            report["empty"] += 1
            continue
        d = docs[0]
        pd.DataFrame({"date": d["period"], "value": d["value"]}).to_csv(path, index=False)
        report["ok"] += 1
        time.sleep(0.25)
    print(f"dbnomics done {report}", flush=True)
    return report


def main() -> int:
    ids = json.loads(Path(sys.argv[1]).read_text())
    which = sys.argv[2] if len(sys.argv) > 2 else "all"
    if which in ("all", "yfinance"):
        print("yfinance:", cache_yfinance(ids.get("yfinance", []), CACHE / "yfinance"), flush=True)
    if which in ("all", "dbnomics"):
        cache_dbnomics(ids.get("dbnomics", []), CACHE / "dbnomics")
    if which in ("all", "fred"):
        cache_fred(ids.get("fred", []), CACHE / "fred")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
