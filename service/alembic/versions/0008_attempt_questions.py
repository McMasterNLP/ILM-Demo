"""persist selected questions for quiz attempts

Revision ID: 0008_attempt_questions
Revises: 0007_add_story_relation_evidence
"""

from alembic import op
import sqlalchemy as sa


revision = "0008_attempt_questions"
down_revision = "0007_add_story_relation_evidence"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "attempt_questions",
        sa.Column(
            "attempt_id",
            sa.String(length=32),
            nullable=False,
        ),
        sa.Column(
            "question_id",
            sa.String(length=32),
            nullable=False,
        ),
        sa.Column(
            "position",
            sa.Integer(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["attempt_id"],
            ["quiz_attempts.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["question_id"],
            ["questions.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "attempt_id",
            "question_id",
        ),
        sa.UniqueConstraint(
            "attempt_id",
            "position",
            name="uq_attempt_question_position",
        ),
    )

    op.create_index(
        "ix_attempt_questions_attempt_position",
        "attempt_questions",
        ["attempt_id", "position"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_attempt_questions_attempt_position",
        table_name="attempt_questions",
    )
    op.drop_table("attempt_questions")
