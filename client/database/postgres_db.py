"""
Centralized PostgreSQL client with connection pooling for all MIT services.

Uses psycopg2.pool.ThreadedConnectionPool.
SetMaxIdleConns / SetMaxOpenConns pattern. Each direction (write / read)
gets its own pool; connections are checked out for the duration of a query
and returned immediately after for REUSE.

KEY FEATURES:
   Connection pooling with automatic reuse
   Separate pools for primary (write) and read-replica
   Thread-safe concurrent access
   Auto-detection of read vs write queries
   Health checks with retry logic
   Configurable pool sizing (DB_POOL_MIN_CONNS / DB_POOL_MAX_CONNS)
   Dedicated single-use connections (for special cases)

RECOMMENDED USAGE (Normal Operations - Pooled Connections):

    from client.database.postgres_db import execute_query, execute_batch_insert, close_all_connections

    # 1. Read queries (automatic read-replica routing)
    rows = execute_query("SELECT * FROM table WHERE id = %s", (123,))
    row = execute_query("SELECT name FROM users LIMIT 1", fetch_one=True)

    # 2. Write queries (automatic primary routing)
    execute_query("INSERT INTO table (col) VALUES (%s)", ('value',))
    count = execute_query("UPDATE table SET col = %s WHERE id = %s", (val, id))

    # 3. Batch inserts (high performance)
    data = [(1, 'a'), (2, 'b'), (3, 'c')]
    execute_batch_insert("INSERT INTO table (id, name) VALUES %s", data)

    # 4. Shutdown cleanup (call when process ends)
    close_all_connections()

CONNECTION REUSE (Pooled Connections):
   Every call to execute_query() or execute_batch_insert():
   - Checks out a connection from the pool (2-30 connections available)
   - Executes your query
   - Returns the connection to the pool (NOT closed, ready for reuse)
   - Next query may reuse the same physical database connection
   
   Result: Fast query execution, no connection creation overhead!

SPECIAL CASES (Dedicated Single-Use Connection):

    from client.database.postgres_db import execute_with_dedicated_connection, get_dedicated_connection
    
    # 1. Simple queries with dedicated connection (connection closed after use)
    rows = execute_with_dedicated_connection("SELECT * FROM table WHERE id = %s", (123,), fetch=True)
    row = execute_with_dedicated_connection("SELECT * FROM users LIMIT 1", fetch_one=True)
    count = execute_with_dedicated_connection("UPDATE table SET col = %s WHERE id = %s", (val, id))

ADVANCED USAGE (Direct Pool Access):

    from client.database.postgres_db import Postgres
    
    db = Postgres.new()
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
    # Connection automatically returned to pool
"""

from client.config import DB_CONFIG, _TUNING
import psycopg2
import psycopg2.pool
from psycopg2.extras import RealDictCursor, execute_values
from contextlib import contextmanager
import logging
from typing import Generator, Optional, ClassVar, Any, List
import time
import pandas as pd

logger = logging.getLogger(__name__)

def _psycopg2_params(
    host_key: str = "host",
    password_key: str = "password",
    port_key: str = "port",
) -> dict:
    """Build a dict safe to pass directly to psycopg2.connect() / ThreadedConnectionPool."""
    return {
        "host":     DB_CONFIG[host_key],
        "database": DB_CONFIG["database"],
        "user":     DB_CONFIG["user"],
        "password": DB_CONFIG[password_key],
        "port":     DB_CONFIG[port_key],
        **{k: DB_CONFIG[k] for k in _TUNING},
    }


# ---------------------------------------------------------------------------
# Postgres — connection pool singleton
# ---------------------------------------------------------------------------

class Postgres:
    """
    Manages two ThreadedConnectionPools (primary write + read-replica).

    Pool sizing:
        pool_min_conns  — minimum live connections kept open (SetMaxIdleConns)
        pool_max_conns  — maximum connections in the pool    (SetMaxOpenConns)
    Both are read from DB_POOL_MIN_CONNS / DB_POOL_MAX_CONNS env vars.

    Thread safety: ThreadedConnectionPool is thread-safe; concurrent
    callers share the same pool without contention.
    """

    _instance: ClassVar[Optional["Postgres"]] = None

    def __init__(self) -> None:
        min_c = DB_CONFIG["pool_min_conns"]
        max_c = DB_CONFIG["pool_max_conns"]

        self._write_pool = psycopg2.pool.ThreadedConnectionPool(
            min_c, max_c, **_psycopg2_params()
        )
        self._read_pool = psycopg2.pool.ThreadedConnectionPool(
            min_c, max_c,
            **_psycopg2_params("host_rr", "password_rr", "port_rr")
        )
        logger.info(
            "DB pools initialised — write & read-replica (min=%d max=%d).",
            min_c, max_c,
        )

    # ------------------------------------------------------------------
    # new() — process-level singleton constructor
    # ------------------------------------------------------------------

    @classmethod
    def new(cls) -> "Postgres":
        """Return the process-level singleton, constructing it on first call."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # connection() / read_connection() — context managers
    # ------------------------------------------------------------------

    @contextmanager
    def connection(self) -> Generator[psycopg2.extensions.connection, None, None]:
        """
        Check out a write connection from the pool.
        Always returns it — even if an exception is raised.
        """
        conn = self._write_pool.getconn()
        try:
            yield conn
        except Exception:
            if not conn.closed:
                conn.rollback()
            raise
        finally:
            self._write_pool.putconn(conn)

    @contextmanager
    def read_connection(self) -> Generator[psycopg2.extensions.connection, None, None]:
        """
        Check out a read-replica connection from the pool.
        Always returns it — even if an exception is raised.
        """
        conn = self._read_pool.getconn()
        try:
            yield conn
        finally:
            self._read_pool.putconn(conn)

    # ------------------------------------------------------------------
    # health() — retries with back-off, returns bool
    # ------------------------------------------------------------------

    def health(self) -> bool:
        """
        Return True if the primary DB responds to SELECT true.
        Retries three times with increasing back-off (100 ms × attempt).
        """
        for attempt in range(1, 4):
            try:
                with self.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT true")
                        row = cur.fetchone()
                        return bool(row and row[0])
            except Exception as exc:
                logger.warning("DB health check failed (attempt %d): %s", attempt, exc)
                time.sleep(attempt * 0.1)
        return False

    # ------------------------------------------------------------------
    # close() — drain both pools and reset the singleton
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Drain both pools and reset the singleton."""
        try:
            self._write_pool.closeall()
        except Exception as exc:
            logger.warning("Error closing write pool: %s", exc)
        try:
            self._read_pool.closeall()
        except Exception as exc:
            logger.warning("Error closing read pool: %s", exc)
        Postgres._instance = None
        logger.info("DB pools closed.")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def get_connection(read_only: bool = False) -> psycopg2.extensions.connection:
    """
    Check out a raw connection from the pool.

    The caller MUST call release_connection(conn, read_only) when done.
    Prefer get_db_connection() (context manager) or execute_query() instead.
    """
    db = Postgres.new()
    pool = db._read_pool if read_only else db._write_pool
    return pool.getconn()


def release_connection(conn: psycopg2.extensions.connection, read_only: bool = False) -> None:
    """Return a connection obtained via get_connection() back to its pool."""
    db = Postgres.new()
    pool = db._read_pool if read_only else db._write_pool
    pool.putconn(conn)


@contextmanager
def get_db_connection(
    read_only: bool = False,
) -> Generator[psycopg2.extensions.connection, None, None]:
    """
    Safe context manager: checks out a pooled connection and always returns it.
    
    Connection Lifecycle:
    1. Checks out connection from pool (may reuse existing connection)
    2. Yields connection to your code
    3. Automatically returns connection to pool (NOT closed!)
    4. Connection ready for reuse by next query
    
    Rolls back on unhandled exceptions for write connections.
    """
    db = Postgres.new()
    ctx = db.read_connection() if read_only else db.connection()
    with ctx as conn:
        yield conn


def execute_query(
    query: str,
    params=None,
    *,
    fetch: bool = False,
    fetch_one: bool = False,
    fetch_all: bool = False,
    fetch_df: bool = False,
    commit: bool = True,
    many: bool = False,
    read_only: Optional[bool] = None,
) -> Any:
    """
    Execute SQL query using a pooled connection with automatic connection reuse.

    Connection Management:
    - Checks out a connection from the pool
    - Automatically returns it to the pool after query completes
    - Connection is REUSED by subsequent calls (not closed)
    - Thread-safe and handles concurrent requests

    Args:
        query: SQL query string
        params: Query parameters (tuple/list/dict)
        fetch: Return all results as list of dicts 
        fetch_one: Return single result as dict or None
        fetch_all: Return all results as list of dicts 
        fetch_df: Return results as pandas DataFrame 
        commit: Auto-commit for write queries (default: True)
        many: Use executemany for batch operations
        read_only: Force read-replica (auto-detected from query if None)

    Returns:
        - DataFrame if fetch_df=True
        - Dict if fetch_one=True
        - List[Dict] if fetch=True or fetch_all=True
        - Row count otherwise

    Usage:
        # Read queries (uses read-replica pool)
        rows = execute_query("SELECT * FROM table WHERE id = %s", (123,))
        row = execute_query("SELECT * FROM table LIMIT 1", fetch_one=True)
        df = execute_query("SELECT * FROM table", fetch_df=True)
        
        # Write queries (uses primary pool)
        count = execute_query("UPDATE table SET col = %s WHERE id = %s", (val, id))
        execute_query("INSERT INTO table VALUES (%s, %s)", (a, b))
        
        # Batch operations
        execute_query("INSERT INTO table VALUES (%s)", data_list, many=True)
        
        # Connection returned to pool after each call - ready for reuse! ✓
    """
    if read_only is None:
        prefix = query.strip()[:10].upper()
        read_only = prefix.startswith(("SELECT", "WITH", "SHOW", "EXPLAIN"))

    with get_db_connection(read_only=read_only) as conn:

        if fetch_df:
            df = pd.read_sql_query(query, conn, params=params)
            return df
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if many and params:
                cur.executemany(query, params)
            elif params:
                cur.execute(query, params)
            else:
                cur.execute(query)

            if fetch_one:
                row = cur.fetchone()
                return dict(row) if row else None
            elif fetch or fetch_all:
                return [dict(r) for r in cur.fetchall()]
            else:
                if commit and not read_only:
                    conn.commit()
                return cur.rowcount


def execute_batch_insert(query: str, data_list: List, page_size: int = 1000) -> int:
    """
    Bulk-insert rows using psycopg2 execute_values for high performance.

    Connection Management:
    - Uses pooled connection (automatically returned after insert)
    - Connection is REUSED by subsequent calls

    Args:
        query: SQL INSERT template (e.g., "INSERT INTO table (col1, col2) VALUES %s")
        data_list: List of tuples containing row data
        page_size: Batch size for execute_values (default: 1000)

    Returns:
        Number of rows inserted

    Usage:
        data = [(1, 'a'), (2, 'b'), (3, 'c')]
        count = execute_batch_insert(
            "INSERT INTO table (id, name) VALUES %s",
            data,
            page_size=1000
        )
        # Connection returned to pool - ready for reuse! ✓
    """
    if not data_list:
        logger.warning("execute_batch_insert: empty data_list — skipped.")
        return 0

    with get_db_connection(read_only=False) as conn:
        with conn.cursor() as cur:
            execute_values(cur, query, data_list, page_size=page_size)
            conn.commit()
            count = cur.rowcount
            logger.info("Batch insert: %d rows.", count)
            return count


def close_all_connections() -> None:
    """
    Close all database connections in both pools (write + read-replica).
    
    Call this ONLY when your process is shutting down.
    This drains both connection pools and resets the singleton.
    After calling this, new queries will create a fresh pool.

    Usage:
        # Throughout your app
        execute_query("SELECT ...")
        execute_query("INSERT ...")
        execute_query("UPDATE ...")
        # ... many queries, all reusing pooled connections ...
        
        # When process ends (shutdown/cleanup)
        close_all_connections()  # Close all pools
    """
    if Postgres._instance:
        Postgres._instance.close()


# ---------------------------------------------------------------------------
# Single-use connection (for one-off operations)
# ---------------------------------------------------------------------------

@contextmanager
def get_dedicated_connection(
    read_only: bool = False,
) -> Generator[psycopg2.extensions.connection, None, None]:
    """
    Create a single dedicated connection and close it after use.
    
    Unlike get_db_connection() which uses pooled connections, this creates
    a fresh connection and completely closes it (not returned to pool).
    
    Connection Lifecycle:
    1. Creates a new dedicated database connection
    2. Yields connection to your code
    3. Automatically closes connection completely (NOT returned to pool!)
    
    Rolls back on unhandled exceptions for write connections.
    
    Args:
        read_only: If True, connects to read-replica; otherwise primary
    
    Yields:
        psycopg2.extensions.connection: Dedicated connection (closed on exit)
    
    Usage:
        with get_dedicated_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM table")
                results = cur.fetchall()
            conn.commit()
        # Connection is CLOSED (not returned to pool)
    """
    conn_params = _psycopg2_params(
        "host_rr" if read_only else "host",
        "password_rr" if read_only else "password",
        "port_rr" if read_only else "port",
    )
    
    conn = None
    try:
        conn = psycopg2.connect(**conn_params)
        logger.debug("Dedicated connection opened (will close after use)")
        yield conn
    except Exception:
        if conn and not conn.closed:
            conn.rollback()
        raise
    finally:
        if conn and not conn.closed:
            conn.close()
            logger.info("Dedicated connection closed")


def execute_with_dedicated_connection(
    query: str,
    params=None,
    *,
    fetch: bool = False,
    fetch_one: bool = False,
    fetch_all: bool = False,
    commit: bool = True,
    many: bool = False,
    read_only: Optional[bool] = None,
) -> Any:
    """
    Execute SQL query using a single dedicated connection that is closed after use.
    
    Unlike pooled connections, this creates a NEW connection and CLOSES it
    after use (not returned to pool). Use this ONLY for special cases like:
    - Long-running transactions that shouldn't hold a pool connection
    - Operations that need complete isolation
    - Debugging/testing scenarios
    
    For normal operations, use execute_query() which reuses pooled connections.
    
    Args:
        query: SQL query string
        params: Query parameters (tuple/list/dict)
        fetch: Return all results as list of dicts (sr-stamping style)
        fetch_one: Return single result as dict or None
        fetch_all: Return all results as list of dicts (synth/risk-scoring style)
        commit: Auto-commit for write queries (default: True)
        many: Use executemany for batch operations
        read_only: Force read-replica (auto-detected from query if None)

    Returns:
        - Dict if fetch_one=True
        - List[Dict] if fetch=True or fetch_all=True
        - Row count otherwise
    
    Usage:
        # Same as execute_query but connection is closed after use
        rows = execute_with_dedicated_connection("SELECT * FROM table WHERE id = %s", (123,), fetch=True)
        row = execute_with_dedicated_connection("SELECT * FROM table LIMIT 1", fetch_one=True)
        count = execute_with_dedicated_connection("UPDATE table SET col = %s", (val,))
    """
    if read_only is None:
        prefix = query.strip()[:10].upper()
        read_only = prefix.startswith(("SELECT", "WITH", "SHOW", "EXPLAIN"))

    with get_dedicated_connection(read_only=read_only) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if many and params:
                cur.executemany(query, params)
            elif params:
                cur.execute(query, params)
            else:
                cur.execute(query)

            if fetch_one:
                row = cur.fetchone()
                return dict(row) if row else None
            elif fetch or fetch_all:
                return [dict(r) for r in cur.fetchall()]
            else:
                if commit and not read_only:
                    conn.commit()
                return cur.rowcount