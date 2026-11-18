"""Create or upgrade the harness tables. Uses the app's own PostgresStore.migrate(),
so this can never drift from the schema the app expects (see MIGRATIONS in postgres.py)."""

import asyncio
import os

from agent_harness.stores.postgres import PostgresStore


async def main() -> None:
    url = os.environ["HARNESS_DATABASE_URL"]
    schema = os.environ.get("HARNESS_DB_SCHEMA", "public")
    # migrate=False: the pool would otherwise apply the schema on connect and hide the result here.
    store = PostgresStore(url, schema=schema, migrate=False)
    try:
        applied = await store.migrate()
    finally:
        await store.close()
    if applied:
        print(f"harness tables created/upgraded in schema '{schema}': versions {applied}")
    else:
        print(f"harness tables already up to date in schema '{schema}'")


if __name__ == "__main__":
    asyncio.run(main())
