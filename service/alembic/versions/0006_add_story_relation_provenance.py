import sqlalchemy as sa
from alembic import op

revision = "0006_story_rel_provenance"
down_revision = "0005_add_group_entity_type"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    columns = {
        c["name"]
        for c in inspector.get_columns("story_relations")
    }

    additions = (
        ("evidence_source", 60),
        ("source_relation", 100),
        ("resolution_rule", 120),
    )

    for name, length in additions:
        if name not in columns:
            op.add_column(
                "story_relations",
                sa.Column(
                    name,
                    sa.String(length=length),
                    nullable=True,
                ),
            )


def downgrade() -> None:
    op.drop_column("story_relations", "resolution_rule")
    op.drop_column("story_relations", "source_relation")
    op.drop_column("story_relations", "evidence_source")
