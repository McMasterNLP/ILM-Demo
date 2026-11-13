from alembic import op


revision = "0005_add_group_entity_type"
down_revision = "0004_question_embeddings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        "ck_entity_type",
        "entities",
        type_="check",
    )

    op.create_check_constraint(
        "ck_entity_type",
        "entities",
        "type IN ('person', 'place', 'group')",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_entity_type",
        "entities",
        type_="check",
    )

    op.create_check_constraint(
        "ck_entity_type",
        "entities",
        "type IN ('person', 'place')",
    )
