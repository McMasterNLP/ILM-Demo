import sqlalchemy as sa
from alembic import op

revision = "0002_open_answers"
down_revision = "0001_initial"
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.add_column(
        "questions",
        sa.Column(
            "answer_mode",
            sa.String(length=20),
            nullable=False,
            server_default="choice",
        ),
    )
    op.create_check_constraint(
        "ck_question_answer_mode", "questions", "answer_mode IN ('choice', 'open')"
    )

    op.add_column(
        "question_translations",
        sa.Column("reference_answer", sa.Text(), nullable=False, server_default=""),
    )

    op.add_column("attempt_answers", sa.Column("text_answer", sa.Text(), nullable=True))
    op.add_column("attempt_answers", sa.Column("feedback", sa.Text(), nullable=True))

def downgrade() -> None:
    op.drop_column("attempt_answers", "feedback")
    op.drop_column("attempt_answers", "text_answer")
    op.drop_column("question_translations", "reference_answer")
    op.drop_constraint("ck_question_answer_mode", "questions", type_="check")
    op.drop_column("questions", "answer_mode")
