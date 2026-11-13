import sqlalchemy as sa
from alembic import op


revision = "0006_add_story_relation_provenance"
down_revision = "0005_add_group_entity_type"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "story_relations",
        sa.Column(
            "evidence_source",
            sa.String(length=60),
            nullable=True,
        ),
    )

    op.add_column(
        "story_relations",
        sa.Column(
            "source_relation",
            sa.String(length=100),
            nullable=True,
        ),
    )

    op.add_column(
        "story_relations",
        sa.Column(
            "resolution_rule",
            sa.String(length=120),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column(
        "story_relations",
        "resolution_rule",
    )

    op.drop_column(
        "story_relations",
        "source_relation",
    )

    op.drop_column(
        "story_relations",
        "evidence_source",
    )
