"""add story relation evidence table

Revision ID: 0007_add_story_relation_evidence
Revises: 0006_add_story_relation_provenance
"""

from alembic import op
import sqlalchemy as sa


revision = "0007_add_story_relation_evidence"
down_revision = "0006_add_story_relation_provenance"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "story_relation_evidence",
        sa.Column(
            "story_id",
            sa.String(length=60),
            sa.ForeignKey(
                "stories.id",
                ondelete="CASCADE",
            ),
            primary_key=True,
            nullable=False,
        ),
        sa.Column(
            "relation_id",
            sa.String(length=32),
            sa.ForeignKey(
                "relations.id",
                ondelete="CASCADE",
            ),
            primary_key=True,
            nullable=False,
        ),
        sa.Column(
            "paragraph_id",
            sa.String(length=32),
            sa.ForeignKey(
                "paragraphs.id",
                ondelete="CASCADE",
            ),
            primary_key=True,
            nullable=False,
        ),
        sa.Column(
            "lang",
            sa.String(length=2),
            primary_key=True,
            nullable=False,
            server_default="ar",
        ),
        sa.Column(
            "evidence_source",
            sa.String(length=60),
            nullable=True,
        ),
        sa.Column(
            "source_relation",
            sa.String(length=100),
            nullable=True,
        ),
        sa.Column(
            "resolution_rule",
            sa.String(length=120),
            nullable=True,
        ),
    )

    op.create_index(
        "ix_story_relation_evidence_paragraph",
        "story_relation_evidence",
        ["paragraph_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_story_relation_evidence_paragraph",
        table_name="story_relation_evidence",
    )
    op.drop_table("story_relation_evidence")
