"""PostgreSQL client for the sandbox agent."""

from __future__ import annotations

import psycopg2

from sandbox_agent.aio_settings import aiosettings


class PostgresClient:
    def __init__(self):
        self.conn = psycopg2.connect(
            dbname=aiosettings.postgres_database,
            user=aiosettings.postgres_user,
            password=aiosettings.postgres_password,
            host=aiosettings.postgres_host,
            port=aiosettings.postgres_port,
        )
        self.cur = self.conn.cursor()

    def execute(self, query, params=None):
        self.cur.execute(query, params)
        return self.cur.fetchall()
