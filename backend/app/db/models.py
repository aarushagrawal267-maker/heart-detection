"""SQLite schema (SQLAlchemy 2.0 ORM)."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text,
                        create_engine, event)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from .. import config as app_config


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Experiment(Base):
    __tablename__ = "experiments"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    kind: Mapped[str] = mapped_column(String(20))           # cnn | baseline
    model_name: Mapped[str] = mapped_column(String(60))     # ECGNet | random_forest | ...
    status: Mapped[str] = mapped_column(String(20), default="queued")
    config: Mapped[dict] = mapped_column(JSON, default=dict)        # hyperparameters
    pipeline: Mapped[dict] = mapped_column(JSON, default=dict)      # preprocessing config
    pipeline_hash: Mapped[str] = mapped_column(String(32), default="")
    dataset_version: Mapped[str] = mapped_column(String(20), default=app_config.DATASET_VERSION)
    split: Mapped[dict] = mapped_column(JSON, default=dict)         # records per partition
    architecture: Mapped[list] = mapped_column(JSON, default=list)
    n_params: Mapped[int | None] = mapped_column(Integer, nullable=True)
    environment: Mapped[dict] = mapped_column(JSON, default=dict)
    checkpoint_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_epoch: Mapped[int] = mapped_column(Integer, default=0)
    total_epochs: Mapped[int] = mapped_column(Integer, default=0)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str] = mapped_column(Text, default="")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[dict] = mapped_column(JSON, default=dict)       # headline validation metrics
    cv_results: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    notes: Mapped[str] = mapped_column(Text, default="")

    epochs: Mapped[list["EpochMetric"]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan", order_by="EpochMetric.epoch")
    evaluations: Mapped[list["Evaluation"]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan")


class EpochMetric(Base):
    __tablename__ = "epoch_metrics"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id", ondelete="CASCADE"))
    fold: Mapped[int] = mapped_column(Integer, default=-1)
    epoch: Mapped[int] = mapped_column(Integer)
    train_loss: Mapped[float] = mapped_column(Float)
    val_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    train_macro_f1: Mapped[float | None] = mapped_column(Float, nullable=True)
    val_macro_f1: Mapped[float | None] = mapped_column(Float, nullable=True)
    val_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    val_per_class_f1: Mapped[dict] = mapped_column(JSON, default=dict)
    lr: Mapped[float | None] = mapped_column(Float, nullable=True)
    duration_s: Mapped[float] = mapped_column(Float, default=0.0)
    experiment: Mapped[Experiment] = relationship(back_populates="epochs")


class Evaluation(Base):
    __tablename__ = "evaluations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id", ondelete="CASCADE"))
    partition: Mapped[str] = mapped_column(String(20))   # train | validation | test
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    per_subject: Mapped[list] = mapped_column(JSON, default=list)
    bootstrap: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    inference: Mapped[dict] = mapped_column(JSON, default=dict)
    records: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    experiment: Mapped[Experiment] = relationship(back_populates="evaluations")


class TestAccess(Base):
    """Every evaluation on held-out test subjects is logged (no silent test peeking)."""
    __tablename__ = "test_access_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(String(32))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    purpose: Mapped[str] = mapped_column(Text, default="")


class ClassificationEvent(Base):
    __tablename__ = "classification_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    kind: Mapped[str] = mapped_column(String(20))         # beat | record
    record: Mapped[str] = mapped_column(String(10))
    beat_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    experiment_id: Mapped[str] = mapped_column(String(32))
    predicted: Mapped[str | None] = mapped_column(String(4), nullable=True)
    reference: Mapped[str | None] = mapped_column(String(4), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    summary: Mapped[dict] = mapped_column(JSON, default=dict)


class Setting(Base):
    __tablename__ = "settings"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text)


_engine = None
_Session = None


def engine():
    global _engine, _Session
    if _engine is None:
        app_config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(f"sqlite:///{app_config.DB_PATH}",
                                connect_args={"check_same_thread": False, "timeout": 30})

        @event.listens_for(_engine, "connect")
        def _pragma(dbapi_conn, _):  # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

        Base.metadata.create_all(_engine)
        _Session = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def session():
    engine()
    return _Session()


def reset_engine() -> None:
    """Used by tests that point QRS_DB_PATH somewhere else."""
    global _engine, _Session
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _Session = None


def get_setting(key: str, default: str | None = None) -> str | None:
    with session() as s:
        row = s.get(Setting, key)
        return row.value if row else default


def set_setting(key: str, value: str) -> None:
    with session() as s:
        row = s.get(Setting, key)
        if row:
            row.value = value
        else:
            s.add(Setting(key=key, value=value))
        s.commit()
