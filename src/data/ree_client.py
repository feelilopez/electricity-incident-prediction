from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import requests

BASE_URL = "https://apidatos.ree.es/en/datos/demanda/evolucion"


@dataclass
class REEClient:
    timeout: int = 30

    def fetch_demand(
        self, start_date: str, end_date: str, time_trunc: str = "hour"
    ) -> pd.DataFrame:
        # REE API expects this query structure for the demand evolution endpoint.
        # https://apidatos.ree.es/es/datos/demanda/evolucion?start_date=2026-01-01T00:00&end_date=2026-03-3T23:59&time_trunc=month&geo_trunc=electric_system&geo_limit=peninsular&geo_ids=8741
        params = {
            "start_date": start_date,
            "end_date": end_date,
            "time_trunc": time_trunc,
            "geo_trunc": "electric_system",
            "geo_limit": "peninsular",
            "geo_ids": "8741",
        }

        response = requests.get(BASE_URL, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()

        values = self._extract_values(payload)
        if not values:
            raise ValueError("REE API returned no values for the requested period")

        frame = pd.DataFrame(values)
        if "datetime" not in frame.columns or "value" not in frame.columns:
            raise ValueError("Unexpected REE response format: missing datetime/value")

        frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
        frame = frame.rename(columns={"value": "demand_mw"})
        # Keep only standardized columns used by the downstream pipeline.
        return (
            frame[["datetime", "demand_mw"]]
            .sort_values("datetime")
            .reset_index(drop=True)
        )

    @staticmethod
    def _extract_values(payload: dict) -> list[dict]:
        included = payload.get("included", [])
        for element in included:
            attrs = element.get("attributes", {})
            values = attrs.get("values", [])
            if values:
                return values
        return []


def month_ranges(start: str, end: str) -> Iterable[tuple[pd.Timestamp, pd.Timestamp]]:
    # Output: Yields tuples (left, right) where both are pd.Timestamp boundaries for each chunk.
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    print(f"Generating month ranges from {start_ts} to {end_ts}...")

    # Split long downloads into monthly API calls to reduce request failures
    # and simplify retries if one period fails.
    cursor = start_ts
    while cursor <= end_ts:
        month_end = (cursor + pd.offsets.MonthEnd(0)).replace(hour=23, minute=59)
        right = min(month_end, end_ts)
        yield cursor, right
        cursor = right + pd.Timedelta(minutes=1)
