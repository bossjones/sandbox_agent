"""Migration to JSONB format

Revision ID: 95d93297832c
Revises: b34017752e45
Create Date: 2024-05-10 18:25:42.596761

"""
from __future__ import annotations

import sqlalchemy as sa

from alembic import op
from sqlalchemy.dialects.postgresql import JSON, JSONB


# revision identifiers, used by Alembic.
revision = '95d93297832c'
down_revision = 'b34017752e45'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        'langchain_pg_embedding',
        'cmetadata',
        existing_type=sa.TEXT,
        type_=JSONB,
        postgresql_using='cmetadata::jsonb'
    )


def downgrade() -> None:
    op.alter_column(
        'langchain_pg_embedding',
        'cmetadata',
        existing_type=sa.TEXT,
        type_=JSON,
        postgresql_using='cmetadata::json'
    )
