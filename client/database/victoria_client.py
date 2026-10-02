"""
Centralized Victoria Logs/Metrics client for all MIT services.

This module provides a unified, singleton-based interface for querying and inserting
data into Victoria Logs, designed for easy reuse across all MIT services.

KEY FEATURES:
   Singleton pattern with process-level instance (VictoriaClient.new())
   Execute custom LogsQL queries with automatic retry logic
   Query changelogs by Victoria Metrics IDs (bulk queries)
   Bulk insert operations for log records
   Streaming JSON (NDJSON) response parsing
   Configurable timeouts, batch sizes, and retry attempts
   Simple module-level helper functions for common operations

RECOMMENDED USAGE (Module-level helpers):

    from client.database.victoria_client import (
        query_changelogs_by_ids,
        query_victoria_logs,
        insert_log_records,
        get_victoria_client
    )

    # 1. Query changelogs by Victoria Metrics IDs
    changelogs = query_changelogs_by_ids(['vm_id_1', 'vm_id_2', 'vm_id_3'])
    # Returns: List of changelog dictionaries

    # 2. Execute custom LogsQL query
    logs = query_victoria_logs(
        '_stream:{object="Changelog"} AND msg.severity:"Critical"'
    )
    # Returns: List of log entries as dictionaries

    # 3. Insert log records (bulk operation)
    records = [
        {"msg": {"id": "abc", "fields": {...}}, "_ts": 1704067200, "object": "Changelog"},
        {"msg": {"id": "def", "fields": {...}}, "_ts": 1704067201, "object": "Changelog"}
    ]
    success = insert_log_records(records)
    # Returns: True if successful, False otherwise

ADVANCED USAGE (Direct Client Access):

    from client.database.victoria_client import VictoriaClient

    # Get singleton instance
    client = VictoriaClient.new()
    
    # Execute custom LogsQL query with retry logic
    logs = client.fetch_changelogs_with_retry(
        query='_stream:{object="Changelog"} AND msg.isActorHuman:true'
    )
    
    # Query changelogs by VM IDs with retry logic
    changelogs = client.fetch_changelogs_by_ids_with_retry(['id1', 'id2', 'id3'])
    
    # Bulk insert log records
    success = client.insert_log_records([record1, record2, record3])

CONFIGURATION:
    The client reads configuration from client.config.VICTORIA_LOGS_CONFIG:
    
    - logs_select: Victoria Logs query endpoint URL
    - logs_insert: Victoria Logs insert endpoint URL
    - query_path: Query API path (default: /select/logsql/query)
    - insert_path: Insert API path (default: /insert/jsonline?_stream_fields=object)
    - timeout: Request timeout in seconds (default: 30)
    - max_results: Maximum results per query (default: 10000)
    - max_retry_attempts: Number of retry attempts (default: 2)
    - bulk_batch_size: Batch size for bulk operations (default: 100)
    - time_column_name: Time column name for responses (default: create_date)
    - headers: Query request headers (Content-Type, Authorization)
    - insert_headers: Insert request headers (Content-Type, Authorization)

RESPONSE FORMAT:
    Victoria Logs returns NDJSON (newline-delimited JSON):
    - One JSON object per line
    - Flattened field names using dot notation (e.g., "msg.id", "msg.fields.key")
    - Special fields: _time, _ts, _stream, object

USAGE PATTERN (Similar to postgres_db.py):
    Just like postgres_db.py provides centralized database access, victoria_client.py
    provides centralized Victoria Logs access. Import and use anywhere in your services:

    # In any service
    from client.database.victoria_client import query_victoria_logs, insert_log_records
    
    # Query data
    results = query_victoria_logs('your_logsql_query_here')
    
    # Insert data
    success = insert_log_records([your_log_records])
"""

import json
import logging
import time
from typing import List, Dict, Optional, ClassVar

import requests
import pandas as pd

from client.config import VICTORIA_LOGS_CONFIG

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# VictoriaClient - Singleton Class
# ---------------------------------------------------------------------------

class VictoriaClient:
    """
    Centralized Victoria Logs client with singleton pattern.
    
    Combines query and update functionality for Victoria Logs operations.
    Supports bulk operations for efficiency and provides automatic data transformation.
    
    Usage:
        client = VictoriaClient.new()
        changelogs = client.fetch_changelogs_from_victoria(num_minutes=30)
        results = client.batch_update_changelogs_with_service_requests(...)
    """
    
    _instance: ClassVar[Optional["VictoriaClient"]] = None
    
    def __init__(self) -> None:
        """Initialize Victoria Logs client with configuration."""
        self.config = VICTORIA_LOGS_CONFIG
        self.logs_select_url = self.config["logs_select"]
        self.logs_insert_url = self.config["logs_insert"]
        self.query_path = self.config["query_path"]
        self.insert_path = self.config["insert_path"]
        self.timeout = self.config["timeout"]
        self.headers = self.config["headers"]
        self.insert_headers = self.config["insert_headers"]
        self.bulk_batch_size = self.config["bulk_batch_size"]
        self.max_results = self.config["max_results"]
        self.max_retry_attempts = self.config["max_retry_attempts"]
        self.time_column_name = self.config["time_column_name"]
        
        logger.info("Victoria Logs client initialized.")
    
    # ------------------------------------------------------------------
    # new() - process-level singleton constructor
    # ------------------------------------------------------------------
    
    @classmethod
    def new(cls) -> "VictoriaClient":
        """Return the process-level singleton, constructing it on first call."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    # ------------------------------------------------------------------
    # Query Methods
    # ------------------------------------------------------------------
    
    def query_logs(
        self,
        query: str,
    ) -> List[Dict]:
        """
        Query Victoria Logs using LogsQL.
        
        Args:
            query: LogsQL query string
            
        Returns:
            List of log entries as dictionaries
        """
        try:    
            # Build the complete LogsQL query
            full_query = f"{query} | limit {self.max_results}"
            url = f"{self.logs_select_url}{self.query_path}"
            
            logger.info(f"Query: {full_query}")
            
            # Use POST with form data as per production approach
            response = requests.post(
                url,
                data={'query': full_query},
                headers=self.headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                response_text = response.text
                logger.debug(f"Victoria Logs response: {response_text[:200]}...")
                
                # Parse streaming JSON response
                return self._parse_streaming_json(response_text)
            else:
                logger.error(f"Victoria Logs query failed with status {response.status_code}")
                logger.error(f"Error: {response.text}...")
                return []
            
        except Exception as e:
            logger.error(f"Victoria Logs query failed for endpoint {url}: {e}")
            raise
    
    def query_changelog_by_vm_ids(self, changelog_vm_ids: List[str]) -> Optional[Dict]:
        """
        Query the latest version of a changelog record by VM IDs.
        
        Args:
            changelog_vm_ids: List of changelog VM IDs to query
            
        Returns:
            The latest changelog record as a dictionary, or None if not found
        """
        try:
            # Build OR conditions for all changelog IDs
            id_conditions = ' OR '.join([f'msg.id:"{id_str}"' for id_str in changelog_vm_ids])

            # Build LogsQL query for specific IDs
            # Uses stats to get all fields per changelog ID
            query = (
                    f'_stream:{{object="Changelog"}} '
                    f'!msg.display:false AND msg.ignore:false AND msg.isActive:true '
                    f'AND ({id_conditions})'
                    f' | sort by (_ts) desc partition by (msg.id) limit 1'
                )
            
            # Call query_logs to execute the query
            results = self.query_logs(query)

            return results
                
        except Exception as e:
            logger.error(f"Failed to query changelog {changelog_vm_ids}: {e}")
            return None
    
    # ------------------------------------------------------------------
    # Insert/Update Methods
    # ------------------------------------------------------------------
    
    def insert_log_records(self, log_records: List[Dict]) -> bool:
        """
        Insert/update multiple log records in a single bulk operation.
        
        Args:
            log_records: List of complete log records to insert
            
        Returns:
            True if bulk insert was successful, False otherwise
        """
        if not log_records:
            logger.warning("insert_log_records: empty log_records list — skipped.")
            return True
        
        try:
            url = f"{self.logs_insert_url}{self.insert_path}"
            
            # Victoria Logs expects newline-delimited JSON (NDJSON)
            # Each record on a new line
            payload = "\n".join([json.dumps(record) for record in log_records]) + "\n"
            
            logger.debug(f"Bulk inserting {len(log_records)} log records to Victoria Logs")
            logger.debug(f"Payload size: {len(payload)} bytes")
            
            response = requests.post(
                url,
                data=payload,
                headers=self.insert_headers,
                timeout=self.timeout * 2  # Increase timeout for bulk insert
            )
            
            if response.status_code in [200, 204]:
                logger.info(f"Successfully bulk inserted {len(log_records)} changelogs to Victoria Logs")
                return True
            else:
                logger.error(f"Victoria Logs bulk insert failed with status {response.status_code}")
                logger.error(f"Error: {response.text[:300]}...")
                return False
                
        except Exception as e:
            logger.error(f"Failed to bulk insert log records: {e}")
            return False
    
    # ------------------------------------------------------------------
    # Utility/Processing Methods
    # ------------------------------------------------------------------
    
    def _parse_streaming_json(self, response_text: str) -> List[Dict]:
        """Parse VictoriaLogs streaming JSON response (one JSON object per line)."""
        results = []
        if not response_text.strip():
            return results
        
        for line_num, line in enumerate(response_text.strip().split('\n'), 1):
            if line.strip():
                try:
                    result = json.loads(line)
                    results.append(result)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num}: {line[:100]}... Error: {e}")
        
        return results
    
    # ------------------------------------------------------------------
    # Retry Logic
    # ------------------------------------------------------------------
    
    def fetch_changelogs_with_retry(
        self,
        query: str
    ) -> List[Dict]:
        """
        Fetch changelogs from Victoria Logs with retry logic.
        
        Args:
            query: LogsQL query string to execute
            
        Returns:
            List of changelog entries
        """
        for attempt in range(self.max_retry_attempts + 1):
            try:
                return self.query_logs(query)
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retry_attempts:
                    logger.warning(f"Victoria Logs attempt {attempt + 1} failed, retrying: {e}")
                    time.sleep(1)  # Brief delay before retry
                else:
                    logger.error(f"All {self.max_retry_attempts} attempts to fetch from Victoria Logs failed.")
                    raise e
                
    def fetch_changelogs_by_ids_with_retry(
        self,
        changelog_vm_ids: List[str]
    ) -> List[Dict]:
        """
        Fetch changelogs by VM IDs with retry logic.
        
        Args:
            changelog_vm_ids: List of changelog VM IDs to query
        
        Returns:
            Dictionary mapping changelog IDs to their latest records
        """
        for attempt in range(self.max_retry_attempts + 1):
            try:
                return self.query_changelog_by_vm_ids(changelog_vm_ids)
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} to fetch changelogs by IDs failed: {e}")
                if attempt < self.max_retry_attempts:
                    logger.warning(f"Victoria Logs attempt {attempt + 1} failed, retrying: {e}")
                    time.sleep(1)  # Brief delay before retry
                else:
                    logger.error(f"All {self.max_retry_attempts} attempts to fetch changelogs by IDs failed.")
                    raise e


# ---------------------------------------------------------------------------
# Module-level helper functions
# ---------------------------------------------------------------------------

def get_victoria_client() -> Optional[VictoriaClient]:
    """
    Get the Victoria Logs client instance if configuration is available.
    
    Returns:
        VictoriaClient instance if configured, None otherwise
    """
    try:
        config = VICTORIA_LOGS_CONFIG
        if config["logs_insert"] and config["logs_select"]:
            return VictoriaClient.new()
        else:
            logger.warning("Victoria Logs not configured - operations will be skipped")
            return None
    except Exception as e:
        logger.warning(f"Failed to initialize Victoria Logs client: {e}")
        return None


def query_changelogs_by_ids(changelog_ids: List[str]) -> Dict[str, Dict]:
    """
    Query multiple changelog records by IDs (bulk operation).
    
    Args:
        changelog_ids: List of changelog IDs to query
        
    Returns:
        Dictionary mapping changelog IDs to their latest records
    
    Usage:
        changelogs = query_changelogs_by_ids(['id1', 'id2', 'id3'])
        for changelog_id, record in changelogs.items():
            print(f"Changelog {changelog_id}: {record}")
    """
    client = get_victoria_client()
    if client:
        return client.fetch_changelogs_by_ids_with_retry(changelog_ids)
    return {}


def query_victoria_logs(query: str) -> List[Dict]:
    """
    Query Victoria Logs with a custom LogsQL query.
    
    Args:
        query: LogsQL query string
        
    Returns:
        List of log entries as dictionaries
    
    Usage:
        logs = query_victoria_logs('_stream:{object="Changelog"} AND msg.severity:"Critical"')
        for log in logs:
            print(log)
    """
    client = get_victoria_client()
    if client:
        return client.fetch_changelogs_with_retry(query)
    return []


def insert_log_records(log_records: List[Dict]) -> bool:
    """
    Insert multiple log records in a single bulk operation.
    
    Args:
        log_records: List of complete log records to insert
        
    Returns:
        True if bulk insert was successful, False otherwise
    
    Usage:
        records = [
            {"msg": {"id": "abc", "fields": {"key": "value"}}, "_ts": 1704067200},
            {"msg": {"id": "def", "fields": {"key": "value2"}}, "_ts": 1704067201}
        ]
        success = insert_log_records(records)
    """
    client = get_victoria_client()
    if client:
        return client.insert_log_records(log_records)
    return False


if __name__ == '__main__':
    # Example usage
    client = get_victoria_client()
    if client:
        query = 'msg.id:4d46218c4de5e2c74c05e4774018f719'
        # Example: Query changelogs from the last 30 minutes
        changelogs = client.fetch_changelogs_with_retry(
            query=query
        )
        print(f"Fetched {len(changelogs)} changelogs from Victoria Logs.")
        print(changelogs[:2])  # Print first 2 changelogs as sample