import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision = "0004_question_embeddings"
down_revision = "0003_quiz_shape"
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.create_table(
        "question_embeddings",
        sa.Column("question_id", sa.String(length=32), nullable=False),
        sa.Column("model", sa.String(length=120), nullable=False),
        sa.Column("lang", sa.String(length=2), nullable=False, server_default="en"),
        sa.Column("embedding", Vector(1024), nullable=False),
        sa.ForeignKeyConstraint(["question_id"], ["questions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("question_id", "model"),
    )
    op.execute(
        "CREATE INDEX ix_question_embedding_hnsw ON question_embeddings "
        "USING hnsw (embedding vector_cosine_ops)"
    )

def downgrade() -> None:
    op.drop_table("question_embeddings")
