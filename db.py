from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "mysql+pymysql://root:root@localhost:3306/photo_enhancer"
# create_engine is responsible for creating a database engine instance, which is the core interface for interacting with the database.
engine = create_engine(SQLALCHEMY_DATABASE_URL, echo=True)
# create database sessions, They provide an interface for working with your model objects and performing database operations in an object-oriented way.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
#Any class that inherits from Base will be considered a SQLAlchemy model, representing a table in the database.
Base = declarative_base()
