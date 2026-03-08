import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, User, UserProfileDB


@pytest.fixture
def db_engine():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine):
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.close()


def test_tables_created(db_engine):
    tables = inspect(db_engine).get_table_names()
    assert "users" in tables
    assert "user_profiles" in tables


def test_user_insert_and_fetch(db_session):
    db_session.add(User(user_id="user-1"))
    db_session.commit()
    fetched = db_session.query(User).filter_by(user_id="user-1").first()
    assert fetched is not None
    assert fetched.user_id == "user-1"


def test_user_timestamps_set_on_insert(db_session):
    db_session.add(User(user_id="user-2"))
    db_session.commit()
    fetched = db_session.query(User).filter_by(user_id="user-2").first()
    assert fetched.created_at is not None
    assert fetched.updated_at is not None


def test_user_profile_defaults(db_session):
    db_session.add(User(user_id="user-3"))
    db_session.flush()
    db_session.add(UserProfileDB(user_id="user-3"))
    db_session.commit()
    fetched = db_session.query(UserProfileDB).filter_by(user_id="user-3").first()
    assert fetched.feature_preferences == "{}"
    assert fetched.liked_products == "[]"
    assert fetched.disliked_products == "[]"
    assert fetched.session_history == "[]"
    assert fetched.updated_at is not None


def test_user_profile_stores_custom_values(db_session):
    db_session.add(User(user_id="user-4"))
    db_session.flush()
    db_session.add(UserProfileDB(
        user_id="user-4",
        feature_preferences='{"battery_life": 0.8}',
        liked_products='["prod-1"]',
    ))
    db_session.commit()
    fetched = db_session.query(UserProfileDB).filter_by(user_id="user-4").first()
    assert fetched.feature_preferences == '{"battery_life": 0.8}'
    assert fetched.liked_products == '["prod-1"]'
