# models.py
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Enum, Table
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from db import Base
import enum

# Optional: Define an enum for model pipeline type
class PipelineType(str, enum.Enum):
    deblur = "deblur"
    dehaze = "dehaze"
    hybrid = "hybrid"

# Association table for many-to-many relationship between users and images
user_images = Table(
    "user_images",
    Base.metadata,
    Column("user_id", Integer, ForeignKey("users.id"), primary_key=True),
    Column("image_id", Integer, ForeignKey("images.id"), primary_key=True),
)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Establish many-to-many relationship to images using the association table
    images = relationship("Image", secondary=user_images, back_populates="users")

class Image(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True, index=True)
    file_path = Column(String(255), nullable=False)
    upload_time = Column(DateTime(timezone=True), server_default=func.now())
    image_metadata = Column(JSON, nullable=True)

    # Many-to-many relationship with users
    users = relationship("User", secondary=user_images, back_populates="images")
    model_outputs = relationship("ModelOutput", back_populates="image")

class ModelOutput(Base):
    __tablename__ = "model_outputs"
    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)  # New direct mapping
    pipeline = Column(Enum(PipelineType), nullable=False)
    output_file_path = Column(String(255), nullable=False)
    processed_at = Column(DateTime(timezone=True), server_default=func.now())
    parent_model_output_id = Column(Integer, ForeignKey("model_outputs.id"), nullable=True)
    pipeline_stage = Column(String(50), nullable=True)

    image = relationship("Image", back_populates="model_outputs")
    user = relationship("User")   # Direct relationship to User


