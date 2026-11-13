import sqlalchemy as sa
from alembic import op

revision = "0002_open_answers"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def _columns(inspector, table):
    return {c["name"] for c in inspector.get_columns(table)}


def _checks(inspector, table):
    return {c["name"] for c in inspector.get_check_constraints(table) if c.get("name")}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    question_columns = _columns(inspector, "questions")
    if "answer_mode" not in question_columns:
        op.add_column(
            "questions",
            sa.Column(
                "answer_mode",
                sa.String(length=20),
                nullable=False,
                server_default="choice",
            ),
        )

    inspector = sa.inspect(bind)
    if "ck_question_answer_mode" not in _checks(inspector, "questions"):
        op.create_check_constraint(
            "ck_question_answer_mode",
            "questions",
            "answer_mode IN ('choice', 'open')",
        )

    if "reference_answer" not in _columns(inspector, "question_translations"):
        op.add_column(
            "question_translations",
            sa.Column(
                "reference_answer",
                sa.Text(),
                nullable=False,
                server_default="",
            ),
        )

    attempt_columns = _columns(inspector, "attempt_answers")
    if "text_answer" not in attempt_columns:
        op.add_column(
            "attempt_answers",
            sa.Column("text_answer", sa.Text(), nullable=True),
        )

    inspector = sa.inspect(bind)
    if "feedback" not in _columns(inspector, "attempt_answers"):
        op.add_column(
            "attempt_answers",
            sa.Column("feedback", sa.Text(), nullable=True),
        )


def downgrade() -> None:
    op.drop_column("attempt_answers", "feedback")
    op.drop_column("attempt_answers", "text_answer")
    op.drop_column("question_translations", "reference_answer")
    op.drop_constraint("ck_question_answer_mode", "questions", type_="check")
    op.drop_column("questions", "answer_mode")
