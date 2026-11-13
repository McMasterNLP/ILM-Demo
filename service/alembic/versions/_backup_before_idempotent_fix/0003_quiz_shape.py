import sqlalchemy as sa
from alembic import op

revision = "0003_quiz_shape"
down_revision = "0002_open_answers"
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.drop_constraint("ck_question_type", "questions", type_="check")
    op.execute("UPDATE questions SET type = 'identification' WHERE type = 'character'")
    op.create_check_constraint(
        "ck_question_type",
        "questions",
        "type IN ('relationship', 'identification', 'comprehension')",
    )

    for column, default in (
        ("quiz_size", "6"),
        ("quiz_relationship", "0"),
        ("quiz_identification", "0"),
        ("quiz_comprehension", "0"),
    ):
        op.add_column(
            "stories",
            sa.Column(column, sa.Integer(), nullable=False, server_default=default),
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
    op.execute("UPDATE questions SET type = 'character' WHERE type = 'identification'")
    op.create_check_constraint(
        "ck_question_type",
        "questions",
        "type IN ('relationship', 'character', 'comprehension')",
    )
