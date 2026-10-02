"""
Configurations for all client connections, including database and API clients.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Attempt to load .env file from the server path first, then fall back to workspace root
server_dotenv_path = "/server/.env"
# Use absolute path: config.py is in client/, so parent.parent is workspace root
workspace_root = Path(__file__).parent.parent
local_dotenv_path = workspace_root / ".env"

if os.path.exists(server_dotenv_path):
    load_dotenv(dotenv_path=server_dotenv_path, override=True)
else:
    load_dotenv(dotenv_path=str(local_dotenv_path), override=True)

DB_CONFIG = {
    "host":     os.getenv("DB_HOST"),
    "database": os.getenv("DB_NAME"),
    "user":     os.getenv("DB_USER"),
    "password": os.getenv("DB_PWD"),
    "port":     os.getenv("DB_PORT"),
    # Read-replica (falls back to primary when DB_HOST_RR is not set)
    "host_rr":     os.getenv("DB_HOST_RR",  os.getenv("DB_HOST")),
    "password_rr": os.getenv("DB_PWD_RR",   os.getenv("DB_PWD")),
    "port_rr":     os.getenv("DB_PORT_RR",   os.getenv("DB_PORT")),
    # Connection tuning
    "connect_timeout":     5,
    "keepalives":          1,
    "keepalives_idle":     30,
    "keepalives_interval": 10,
    "keepalives_count":    5,
    "options": "-c statement_timeout=120000 -c idle_in_transaction_session_timeout=60000",
    # Connection pool sizing (mirrors Go SetMaxIdleConns / SetMaxOpenConns)
    "pool_min_conns": int(os.getenv("DB_POOL_MIN_CONNS", "2")),
    "pool_max_conns": int(os.getenv("DB_POOL_MAX_CONNS", "10")),
}


# Keys that belong to our pool config, not psycopg2 connect()
_TUNING      = ("connect_timeout", "keepalives", "keepalives_idle",
                "keepalives_interval", "keepalives_count", "options")


MANIFEST_API_CONFIG = {
    "enabled": os.getenv("USE_MANIFEST_API", "false").lower() == "true",
    "base_url": os.getenv("MANIFEST_API_URL", "https://dev.api.manifestit.tech/"),
    "api_key": os.getenv("MANIFEST_API_KEY"),
    "org_key": os.getenv("MANIFEST_ORG_KEY", "dev"),
    "org_id": os.getenv("MANIFEST_ORG_ID", "1"),
    "subscription_id": os.getenv("MANIFEST_SUBSCRIPTION_ID", "4"),
    "timeout": int(os.getenv("MANIFEST_API_TIMEOUT", "30")),
    "timeout_max_retries": int(os.getenv("MANIFEST_API_TIMEOUT_MAX_RETRIES", "3")),

    # API Mode Selection
    # If True: use client API with static API key (/client/ endpoints)
    # If False: use internal API with token auth (/org/ endpoints)
    "use_client_api": os.getenv("AU_USE_CLIENT_API", "true").lower() == "true",

    # Log fetching preferences
    "use_as_fallback": os.getenv("MANIFEST_API_AS_FALLBACK", "true").lower() == "true",
    "resource_by_tags": os.getenv("MANIFEST_API_RESOURCE_BY_TAGS", "false").lower() == "true",
    "attachments_data": os.getenv("MANIFEST_API_ATTACHMENTS_DATA", "true").lower() == "true",

    # Token generation config (for internal API mode)
    "auth_issuer": os.getenv("KEYCLOAK_ISSUER", "https://auth.manifestit.io/realms/dev"),
    "client_id": os.getenv("KEYCLOAK_CLIENT_ID"),
    "client_secret": os.getenv("KEYCLOAK_CLIENT_SECRET"),
    "grant_type": os.getenv("KEYCLOAK_GRANT_TYPE", "client_credentials"),
    "username": os.getenv("KEYCLOAK_USERNAME"),
    "password": os.getenv("KEYCLOAK_PASSWORD"),

    # Tenant configuration
    "tenant": os.getenv("TENANT", os.getenv("ORG_KEY", "dev")),
}


VICTORIA_LOGS_CONFIG = {
    # Victoria Logs endpoints - support port forwarding
    "logs_insert": os.getenv("VICTORIA_LOGS_INSERT"),
    "logs_select": os.getenv("VICTORIALOGS_ENDPOINT"),
    
     # Query configuration
    "query_path": os.getenv("VICTORIALOGS_QUERY_PATH", "/select/logsql/query"),
    "insert_path": os.getenv("VICTORIA_LOGS_INSERT_PATH", "/insert/jsonline?_stream_fields=object"),
    "timeout": int(os.getenv("VICTORIA_LOGS_TIMEOUT", "30")),
    "max_results": int(os.getenv("VICTORIA_LOGS_MAX_RESULTS", "10000")),
    "bulk_batch_size": int(os.getenv("VICTORIA_LOGS_BULK_BATCH_SIZE", "100")),
    "max_retry_attempts": int(os.getenv("VICTORIA_LOGS_MAX_RETRY_ATTEMPTS", "2")),
    "time_column_name": os.getenv("TIME_COLUMN_NAME", "create_date"),

    # Headers configuration
    "headers": {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    },
    "insert_headers": {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Bearer {os.getenv('VICTORIA_LOGS_TOKEN')}" if os.getenv('VICTORIA_LOGS_TOKEN') else None
    }
}


VICTORIA_METRICS_CONFIG = {
    # Victoria Metrics select endpoint - supports port forwarding
    "metrics_select": os.getenv("VICTORIAMETRICS_ENDPOINT"),

    # Multi-tenancy: VictoriaMetrics cluster select URLs are prefixed with
    # /select/<accountID>:<projectID>.
    "tenant_path_template": os.getenv("VICTORIAMETRICS_TENANT_PATH_TEMPLATE", "/select/{account_id}:{project_id}"),
    "account_id": os.getenv("VICTORIAMETRICS_ACCOUNT_ID", "1"),
    "project_id": os.getenv("VICTORIAMETRICS_PROJECT_ID", "0"),

    # Query configuration (paths are appended after the tenant prefix)
    "query_path": os.getenv("VICTORIAMETRICS_QUERY_PATH", "/prometheus/api/v1/query"),
    "query_range_path": os.getenv("VICTORIAMETRICS_QUERY_RANGE_PATH", "/prometheus/api/v1/query_range"),
    "series_path": os.getenv("VICTORIAMETRICS_SERIES_PATH", "/prometheus/api/v1/series"),
    "label_values_path": os.getenv("VICTORIAMETRICS_LABEL_VALUES_PATH", "/prometheus/api/v1/label/{label}/values"),

    "timeout": int(os.getenv("VICTORIAMETRICS_TIMEOUT", "30")),
    "default_step": os.getenv("VICTORIAMETRICS_DEFAULT_STEP", "60s"),
    "max_points": int(os.getenv("VICTORIAMETRICS_MAX_POINTS", "11000")),
    "max_series": int(os.getenv("VICTORIAMETRICS_MAX_SERIES", "1000")),
    "max_retry_attempts": int(os.getenv("VICTORIAMETRICS_MAX_RETRY_ATTEMPTS", "2")),

    # Headers configuration
    "headers": {
        "Accept": "application/json",
        **(
            {"Authorization": f"Bearer {os.getenv('VICTORIAMETRICS_TOKEN')}"}
            if os.getenv("VICTORIAMETRICS_TOKEN") else {}
        ),
    },
}