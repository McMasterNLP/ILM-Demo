import sqlalchemy as sa
from alembic import op

revision = "0003_quiz_shape"
down_revision = "0002_open_answers"
branch_labels = None
depends_on = None


def _columns(inspector, table):
    return {c["name"] for c in inspector.get_columns(table)}


def _checks(inspector, table):
    return {c["name"] for c in inspector.get_check_constraints(table) if c.get("name")}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "ck_question_type" in _checks(inspector, "questions"):
        op.drop_constraint("ck_question_type", "questions", type_="check")

    op.execute(
        "UPDATE questions "
        "SET type = 'identification' "
        "WHERE type = 'character'"
    )

    op.create_check_constraint(
        "ck_question_type",
        "questions",
        "type IN ('relationship', 'identification', 'comprehension')",
    )

    inspector = sa.inspect(bind)
    story_columns = _columns(inspector, "stories")

    for column, default in (
        ("quiz_size", "6"),
        ("quiz_relationship", "0"),
        ("quiz_identification", "0"),
        ("quiz_comprehension", "0"),
    ):
        if column not in story_columns:
            op.add_column(
                "stories",
                sa.Column(
                    column,
                    sa.Integer(),
                    nullable=False,
                    server_default=default,
                ),
            )


def downgrade() -> None:
    for column in (
        "quiz_comprehension",
        "quiz_identification",
        "quiz_relationship",
        "quiz_size",
    ):
        op.drop_column("stories", column)

    op.drop_constraint("ck_question_type", "questions", type_="check")
    op.execute(
        "UPDATE questions "
        "SET type = 'character' "
        "WHERE type = 'identification'"
    )
    op.create_check_constraint(
        "ck_question_type",
        "questions",
        "type IN ('relationship', 'character', 'comprehension')",
    )
