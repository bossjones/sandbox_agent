"""DB migrations commands """
from __future__ import annotations

from os import getcwd

from alembic import command
from alembic.config import Config


ALEMBIC_CFG = Config(f'{getcwd()}/alembic.ini')


def current(verbose=False):
    command.current(ALEMBIC_CFG, verbose=verbose)


def upgrade(revision="head"):
    command.upgrade(ALEMBIC_CFG, revision)


def downgrade(revision):
    command.downgrade(ALEMBIC_CFG, revision)
