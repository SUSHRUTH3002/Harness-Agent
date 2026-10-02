"""
Centralized Victoria Metrics (PromQL) client for all MIT services.

Key difference from the logs client: VictoriaMetrics cluster select URLs are
**multi-tenant**. Every read is scoped by `/select/<accountID>:<projectID>`.
The tenant is not global config - it is a property of the data being queried and
is normally carried on the alert/resource itself (`vm_account_id` /
`vm_project_id` labels). Every method therefore accepts an optional tenant, and
falls back to the configured default only when none is supplied.

Usage:

    from client.victoria import query_metrics_range, VictoriaMetricsClient

    # Module-level helper (safe: returns [] when not configured)
    series = query_metrics_range(
        promql='hf_fs_use_pct{host_id="693de8..."}',
        start_time=start,
        end_time=end,
        step="1h",
        account_id="1",
        project_id="396",
    )

    # Or the singleton directly
    client = VictoriaMetricsClient.new()
    value = client.query_instant('hf_fs_avail_mb{...}', account_id="1", project_id="396")

Configuration is read from client.config.VICTORIA_METRICS_CONFIG:
    - metrics_select: VictoriaMetrics select endpoint base URL
    - tenant_path_template / account_id / project_id
    - query_path, query_range_path, series_path, label_values_path
    - timeout, default_step, max_points, max_series, max_retry_attempts, headers
"""

import logging
import math
import time
from datetime import datetime, timezone
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import pandas as pd
import requests

from client.config import VICTORIA_METRICS_CONFIG

logger = logging.getLogger(__name__)

# Suffix multipliers for Prometheus-style duration strings.
_STEP_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def parse_duration_seconds(duration: str, default: int = 60) -> int:
    """Convert a Prometheus duration string ('30s', '5m', '1h', '7d') to seconds."""
    if duration is None:
        return default
    text = str(duration).strip().lower()
    if not text:
        return default
    try:
        if text[-1] in _STEP_UNITS:
            return max(1, int(float(text[:-1]) * _STEP_UNITS[text[-1]]))
        return max(1, int(float(text)))
    except (ValueError, TypeError):
        logger.warning(f"Could not parse duration '{duration}', defaulting to {default}s")
        return default


class VictoriaMetricsClient:
    """
    Centralized Victoria Metrics client with singleton pattern.

    All query methods return plain Python structures (the Prometheus
    `data.result` list) so they can be serialised directly for LLM consumption.
    Failures are logged and surfaced as empty results rather than exceptions, so
    a metrics outage degrades an analysis instead of failing it.
    """

    _instance: ClassVar[Optional["VictoriaMetricsClient"]] = None

    def __init__(self) -> None:
        """Initialize Victoria Metrics client with configuration."""
        self.config = VICTORIA_METRICS_CONFIG
        self.metrics_select_url = (self.config.get("metrics_select") or "").rstrip("/")
        self.tenant_path_template = self.config["tenant_path_template"]
        self.account_id = self.config["account_id"]
        self.project_id = self.config["project_id"]
        self.query_path = self.config["query_path"]
        self.query_range_path = self.config["query_range_path"]
        self.series_path = self.config["series_path"]
        self.label_values_path = self.config["label_values_path"]
        self.timeout = self.config["timeout"]
        self.default_step = self.config["default_step"]
        self.max_points = self.config["max_points"]
        self.max_series = self.config["max_series"]
        self.max_retry_attempts = self.config["max_retry_attempts"]
        self.headers = self.config["headers"]

        if not self.metrics_select_url:
            logger.warning(
                "Victoria Metrics endpoint is not configured "
                "(VICTORIAMETRICS_ENDPOINT). Metric queries will return empty results."
            )
        else:
            logger.info("Victoria Metrics client initialized.")

    # ------------------------------------------------------------------
    # new() - process-level singleton constructor
    # ------------------------------------------------------------------

    @classmethod
    def new(cls) -> "VictoriaMetricsClient":
        """Return the process-level singleton, constructing it on first call."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def is_configured(self) -> bool:
        """True when a select endpoint is available."""
        return bool(self.metrics_select_url)

    # ------------------------------------------------------------------
    # URL construction
    # ------------------------------------------------------------------

    def _tenant_prefix(
        self,
        account_id: Optional[Union[str, int]] = None,
        project_id: Optional[Union[str, int]] = None,
    ) -> str:
        """Build the `/select/<account>:<project>` prefix for a tenant."""
        resolved_account = str(account_id) if account_id not in (None, "") else str(self.account_id)
        resolved_project = str(project_id) if project_id not in (None, "") else str(self.project_id)
        return self.tenant_path_template.format(
            account_id=resolved_account,
            project_id=resolved_project,
        )

    def _build_url(
        self,
        path: str,
        account_id: Optional[Union[str, int]] = None,
        project_id: Optional[Union[str, int]] = None,
    ) -> str:
        """Compose a full tenant-scoped URL for the given API path."""
        return f"{self.metrics_select_url}{self._tenant_prefix(account_id, project_id)}{path}"

    # ------------------------------------------------------------------
    # Low-level request helper
    # ------------------------------------------------------------------

    def _request(self, url: str, params: Dict[str, Any]) -> List[Any]:
        """
        Execute a Prometheus API GET with retries and uniform error handling.

        Returns the `data.result` payload, or [] on any failure.
        """
        if not self.is_configured:
            logger.warning("Victoria Metrics not configured; returning empty result.")
            return []

        last_error: Optional[Exception] = None

        for attempt in range(self.max_retry_attempts + 1):
            try:
                response = requests.get(
                    url,
                    params=params,
                    headers=self.headers,
                    timeout=self.timeout,
                )

                if response.status_code != 200:
                    logger.error(
                        f"Victoria Metrics query failed with status "
                        f"{response.status_code}: {response.text[:300]}"
                    )
                    return []

                body = response.json()
                if body.get("status") != "success":
                    logger.error(
                        f"Victoria Metrics returned non-success status: "
                        f"{body.get('status')} - {body.get('error', '')[:300]}"
                    )
                    return []

                data = body.get("data", {})

                # /series returns data as a list; query endpoints nest under 'result'.
                if isinstance(data, list):
                    return data
                return data.get("result", [])

            except Exception as e:
                last_error = e
                if attempt < self.max_retry_attempts:
                    logger.warning(
                        f"Victoria Metrics request failed (attempt "
                        f"{attempt + 1}/{self.max_retry_attempts + 1}): {e}. Retrying..."
                    )
                    time.sleep(1)
                    continue

        logger.error(f"Victoria Metrics request failed for {url}: {last_error}")
        return []

    # ------------------------------------------------------------------
    # Query Methods
    # ------------------------------------------------------------------

    @staticmethod
    def _epoch(value: datetime) -> int:
        """
        Convert to a Unix timestamp, treating naive datetimes as UTC.

        Timestamps in this pipeline are naive UTC (parsed from Victoria Logs then
        stripped of tzinfo). datetime.timestamp() would read them as local time
        and silently shift every query by the host's UTC offset.
        """
        if value.tzinfo is None:
            return int(value.replace(tzinfo=timezone.utc).timestamp())
        return int(value.timestamp())

    def query_instant(
        self,
        promql: str,
        at_time: Optional[datetime] = None,
        account_id: Optional[Union[str, int]] = None,
        project_id: Optional[Union[str, int]] = None,
    ) -> List[Dict]:
        """
        Run an instant PromQL query.

        Args:
            promql: PromQL expression
            at_time: Evaluation timestamp (defaults to server 'now')
            account_id / project_id: VictoriaMetrics tenant

        Returns:
            List of {"metric": {...}, "value": [ts, "val"]} dicts
        """
        params: Dict[str, Any] = {"query": promql}
        if at_time is not None:
            params["time"] = self._epoch(at_time)

        url = self._build_url(self.query_path, account_id, project_id)
        logger.info(f"VM instant query: {promql}")
        return self._request(url, params)

    def query_range(
        self,
        promql: str,
        start_time: datetime,
        end_time: datetime,
        step: Optional[str] = None,
        account_id: Optional[Union[str, int]] = None,
        project_id: Optional[Union[str, int]] = None,
    ) -> List[Dict]:
        """
        Run a PromQL range query.

        The step is automatically coarsened when the requested window would
        exceed `max_points`, so long lookbacks (weeks) stay within limits.

        Returns:
            List of {"metric": {...}, "values": [[ts, "val"], ...]} dicts
        """
        resolved_step = self._resolve_step(start_time, end_time, step)

        params = {
            "query": promql,
            "start": self._epoch(start_time),
            "end": self._epoch(end_time),
            "step": resolved_step,
        }

        url = self._build_url(self.query_range_path, account_id, project_id)
        logger.info(
            f"VM range query: {promql} "
            f"[{start_time.isoformat()} -> {end_time.isoformat()} step={resolved_step}]"
        )
        return self._request(url, params)

    def _resolve_step(
        self,
        start_time: datetime,
        end_time: datetime,
        step: Optional[str],
    ) -> str:
        """Clamp the step so the range query stays under `max_points` samples."""
        requested = step or self.default_step
        step_seconds = parse_duration_seconds(requested)
        window_seconds = max(1, int((end_time - start_time).total_seconds()))

        if window_seconds / step_seconds <= self.max_points:
            return requested

        needed = math.ceil(window_seconds / self.max_points)
        logger.info(
            f"Requested step {requested} would exceed {self.max_points} points "
            f"over {window_seconds}s; coarsening to {needed}s"
        )
        return f"{needed}s"

    def series(
        self,
        match: Union[str, List[str]],
        start_time: datetime,
        end_time: datetime,
        account_id: Optional[Union[str, int]] = None,
        project_id: Optional[Union[str, int]] = None,
    ) -> List[Dict]:
        """
        Discover series (label sets) matching one or more selectors.

        Useful for finding which devices/mounts/processes exist on a host before
        building a targeted query.
        """
        matchers = [match] if isinstance(match, str) else list(match)
        params = {
            "match[]": matchers,
            "start": self._epoch(start_time),
            "end": self._epoch(end_time),
            "limit": self.max_series,
        }

        url = self._build_url(self.series_path, account_id, project_id)
        logger.info(f"VM series lookup: {matchers}")
        return self._request(url, params)

    def label_values(
        self,
        label: str,
        start_time: datetime,
        end_time: datetime,
        match: Optional[Union[str, List[str]]] = None,
        account_id: Optional[Union[str, int]] = None,
        project_id: Optional[Union[str, int]] = None,
    ) -> List[str]:
        """
        List values for a label (e.g. `__name__` to enumerate metric names).
        """
        params: Dict[str, Any] = {
            "start": self._epoch(start_time),
            "end": self._epoch(end_time),
        }
        if match:
            params["match[]"] = [match] if isinstance(match, str) else list(match)

        url = self._build_url(
            self.label_values_path.format(label=label), account_id, project_id
        )
        logger.info(f"VM label values: {label}")
        result = self._request(url, params)
        return [str(v) for v in result] if isinstance(result, list) else []

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def query_scalar(
        self,
        promql: str,
        at_time: Optional[datetime] = None,
        account_id: Optional[Union[str, int]] = None,
        project_id: Optional[Union[str, int]] = None,
    ) -> Optional[float]:
        """Run an instant query and return the first sample as a float."""
        result = self.query_instant(promql, at_time, account_id, project_id)
        if not result:
            return None
        try:
            return float(result[0]["value"][1])
        except (KeyError, IndexError, TypeError, ValueError) as e:
            logger.warning(f"Could not coerce instant query result to float: {e}")
            return None

    def range_to_points(
        self,
        promql: str,
        start_time: datetime,
        end_time: datetime,
        step: Optional[str] = None,
        account_id: Optional[Union[str, int]] = None,
        project_id: Optional[Union[str, int]] = None,
    ) -> List[Tuple[datetime, float, Dict[str, str]]]:
        """
        Run a range query and flatten it into (timestamp, value, labels) tuples,
        sorted by time. Non-numeric samples (NaN) are dropped.
        """
        result = self.query_range(
            promql, start_time, end_time, step, account_id, project_id
        )

        points: List[Tuple[datetime, float, Dict[str, str]]] = []
        for series in result:
            labels = series.get("metric", {}) or {}
            for ts, raw_value in series.get("values", []) or []:
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                if math.isnan(value):
                    continue
                points.append(
                    (datetime.utcfromtimestamp(float(ts)), value, labels)
                )

        points.sort(key=lambda item: item[0])
        return points


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def get_victoria_metrics_client() -> Optional[VictoriaMetricsClient]:
    """
    Return the singleton client, or None when Victoria Metrics is not configured.
    """
    client = VictoriaMetricsClient.new()
    if not client.is_configured:
        logger.warning("Victoria Metrics client requested but not configured.")
        return None
    return client


def query_metrics_range(
    promql: str,
    start_time: datetime,
    end_time: datetime,
    step: Optional[str] = None,
    account_id: Optional[Union[str, int]] = None,
    project_id: Optional[Union[str, int]] = None,
) -> List[Dict]:
    """Convenience wrapper for a range query. Returns [] when unconfigured."""
    client = get_victoria_metrics_client()
    if client is None:
        return []
    return client.query_range(promql, start_time, end_time, step, account_id, project_id)


def query_metrics_instant(
    promql: str,
    at_time: Optional[datetime] = None,
    account_id: Optional[Union[str, int]] = None,
    project_id: Optional[Union[str, int]] = None,
) -> List[Dict]:
    """Convenience wrapper for an instant query. Returns [] when unconfigured."""
    client = get_victoria_metrics_client()
    if client is None:
        return []
    return client.query_instant(promql, at_time, account_id, project_id)


def metric_range_to_dataframe(result: List[Dict]) -> pd.DataFrame:
    """
    Convert a Prometheus range-query result into a tidy DataFrame with columns:
    timestamp, value, and one column per label.
    """
    rows: List[Dict[str, Any]] = []
    for series in result or []:
        labels = series.get("metric", {}) or {}
        for ts, raw_value in series.get("values", []) or []:
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            rows.append(
                {
                    "timestamp": datetime.utcfromtimestamp(float(ts)),
                    "value": value,
                    **labels,
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return df


def series_by_group(
    result: List[Dict],
    group_by: Optional[str],
) -> Dict[str, List[Tuple[datetime, float]]]:
    """Convert a Prometheus range result into {group_value: [(ts, value), ...]}."""
    grouped: Dict[str, List[Tuple[datetime, float]]] = {}

    for series in result or []:
        labels = series.get("metric", {}) or {}
        key = labels.get(group_by) if group_by else None
        if key is None:
            key = labels.get("device") or labels.get("name") or "__all__"

        bucket = grouped.setdefault(str(key), [])
        for raw_ts, raw_value in series.get("values", []) or []:
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if math.isnan(value):
                continue
            bucket.append((datetime.utcfromtimestamp(float(raw_ts)), value))

    for key in grouped:
        grouped[key].sort(key=lambda item: item[0])
    return grouped


if __name__ == '__main__':
    start_time = datetime.strptime("2026-09-07T11:47:06Z", "%Y-%m-%dT%H:%M:%SZ")
    end_time = datetime.strptime("2026-09-07T13:47:06Z", "%Y-%m-%dT%H:%M:%SZ")
    vm_account_id = '1'
    vm_project_id = '1227'
    promql = 'hf_process_cpu_pct{parent_resource_id="3645922432649511163/service:telemetry-sync-prod"}'
    result = query_metrics_range(promql, start_time, end_time, step='5m', account_id=vm_account_id, project_id=vm_project_id)
    result_df = metric_range_to_dataframe(result)
    print(result_df)