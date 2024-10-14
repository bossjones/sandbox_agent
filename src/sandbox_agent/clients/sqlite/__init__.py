"""SQLite client for the sandbox agent."""

from __future__ import annotations

import sqlite3


class SQLiteClient:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cur = self.conn.cursor()

    def execute(self, query, params=None):
        self.cur.execute(query, params)
        return self.cur.fetchall()
