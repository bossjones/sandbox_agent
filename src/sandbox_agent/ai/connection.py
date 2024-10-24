"""DB connection utility functions"""

from __future__ import annotations

import subprocess
import sys

from functools import lru_cache
from urllib import parse

from langchain_community.vectorstores.pgembedding import CollectionStore
from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.engine import Connection
from sqlalchemy.exc import DatabaseError
from sqlalchemy.orm import Session

from sandbox_agent.aio_settings import aiosettings


# borrowed from brevia


def connection_string() -> str:
    """Postgres+pgvector db connection string"""
    conf = aiosettings
    if conf.postgres_url:
        logger.info(f"Using postgres_url: conf.postgres_url={conf.postgres_url}")
        return conf.postgres_url

    driver = conf.postgres_driver
    host = conf.postgres_host
    port = conf.postgres_port
    database = conf.postgres_database
    user = conf.postgres_user
    password = parse.quote_plus(conf.postgres_password)

    # return f"postgresql+{driver}://{user}:{password}@{host}:{port}/{database}"
    logger.info(f"Using postgres_url: aiosettings.postgres_url={aiosettings.postgres_url}")
    return aiosettings.postgres_url


@lru_cache
def get_engine():
    """Return db engine (using lru_cache)"""
    return create_engine(
        connection_string(),
        pool_size=aiosettings.pgvector_pool_size,
    )


def db_connection() -> Connection:
    """SQLAlchemy db connection"""
    return get_engine().connect()


def test_connection() -> bool:
    """Test db connection with a simple query"""
    try:
        with Session(db_connection()) as session:
            (session.query(CollectionStore.uuid).limit(1).all())
        return True
    except DatabaseError:
        print("Error performing a simple SQL query", file=sys.stderr)
        return False


def psql_command(
    cmd: str,
) -> bool:
    """Perform command on db using `psql`"""
    conf = aiosettings
    host = conf.postgres_host
    port = conf.postgres_port
    database = conf.postgres_database
    user = conf.postgres_user
    password = conf.postgres_password
    logger.info(f"psql command: host={host}, port={port}, database={database}, user={user}")

    cmd = ["psql", "-U", user, "-h", host, "-p", f"{port}", "-c", cmd, database]

    with subprocess.Popen(cmd, env={"PGPASSWORD": password}, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as proc:
        res = proc.communicate()
        output = res[0].decode("UTF-8")
        print(f"psql output: {output}")
        if proc.returncode != 0:
            print("Error launching psql command", file=sys.stderr)
            return False

    print("Psql command executed successfully")
    return True
