from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config import settings

# Create database engine
# This establishes the connection pool to PostgreSQL
engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG  # Log SQL queries when DEBUG=True
)

# Create SessionLocal class
# Each instance will be a database session
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Create Base class for models
# All database models will inherit from this
Base = declarative_base()


def get_db():
    """
    Dependency function that provides database sessions to routes.
    
    Usage in routes:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            # Use db here
            pass
    
    The session is automatically closed after the request.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
