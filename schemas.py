# schemas.py
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime
import enum

# Authentication schemas
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: str | None = None

class PipelineType(str, enum.Enum):
    deblur = "deblur"
    dehaze = "dehaze"
    hybrid = "hybrid"  # Modified hybrid enum value

class UserBase(BaseModel):
    username: str
    email: EmailStr

class UserCreate(UserBase):
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class User(UserBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True

class ImageBase(BaseModel):
    file_path: str
    image_metadata: Optional[dict] = None

class ImageCreate(ImageBase):
    pass

class UserOut(UserBase):
    id: int

    class Config:
        from_attributes = True

class Image(ImageBase):
    id: int
    upload_time: datetime
    users: List[UserOut] = []

    class Config:
        from_attributes = True

class ModelOutputBase(BaseModel):
    image_id: int
    user_id: int 
    pipeline: PipelineType
    output_file_path: str

class ModelOutputCreate(ModelOutputBase):
    parent_model_output_id: Optional[int] = None
    pipeline_stage: Optional[str] = None

class ModelOutput(ModelOutputBase):
    id: int
    processed_at: datetime
    parent_model_output_id: Optional[int] = None
    pipeline_stage: Optional[str] = None

    class Config:
        from_attributes = True



class CheckLoginRequest(BaseModel):
    token: str
