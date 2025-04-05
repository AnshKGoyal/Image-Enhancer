import hashlib
from fastapi import UploadFile
from sqlalchemy import func
from models import Image
from passlib.context import CryptContext
import torch
from sqlalchemy.orm import Session
from helpers import save_upload_file, preprocess_image, postprocess_tensor
import models
import os

# Initialize the CryptContext for bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def calculate_image_hash(image_path: str):
    """Calculates the MD5 hash of an image file from its path."""
    try:
        with open(image_path, 'rb') as image_file: # Open file in binary read mode
            contents = image_file.read() # Read entire file content
            md5_hash = hashlib.md5(contents).hexdigest() # Calculate MD5 hash
            return md5_hash
            
    except Exception as e:
        print(f"Error calculating image hash: {e}")
        return None

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_or_create_image_record(db: Session, user_id: int, file: UploadFile) -> models.Image:
    """
    Save uploaded file, compute its hash, and either return an existing
    image record associated with the user or create a new record if needed.
    """
    # Save the uploaded file
    file_path = save_upload_file(file)
    # Calculate hash after saving
    image_hash = calculate_image_hash(file_path)
    
    if image_hash is not None:
        # Search for an existing image that has the same hash
        existing_image = db.query(Image).filter(
            func.json_extract(Image.image_metadata, '$.hash') == image_hash
        ).first()
        if existing_image:
            # Associate the user to this image if not already associated.
            if existing_image not in db.query(Image).filter(Image.users.any(id=user_id)).all():
                user = db.query(models.User).filter(models.User.id == user_id).first()
                if user:
                    user.images.append(existing_image)
                    db.commit()
            # Remove the duplicate file on disk.
            os.remove(file_path)
            return existing_image

    # If no duplicate is found, create a new image record.
    new_image = Image(
        file_path=file_path,
        image_metadata={'hash': image_hash}
    )
    db.add(new_image)
    db.commit()
    db.refresh(new_image)
    # Associate with the user.
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if user:
        user.images.append(new_image)
        db.commit()
    return new_image

def create_image_record(db: Session, user_id: int, file_path: str) -> models.Image:
    """
    Create a new Image record after saving a file.
    """
    new_image = models.Image(
        user_id=user_id,
        file_path=file_path,
        image_metadata=None  # You can update this if you need metadata
    )
    db.add(new_image)
    db.commit()
    db.refresh(new_image)
    return new_image

def run_inference(model, input_image_path: str):
    """
    Preprocess the image, run the model, and postprocess the output.
    Returns the output image file path.
    """
    # Preprocess
    input_tensor = preprocess_image(input_image_path)
    # Inference (ensure no gradient computation)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    # Postprocess to get an output image saved to disk
    output_image_path = postprocess_tensor(output_tensor)
    return output_image_path

def create_model_output_record(db: Session, image_id: int, user_id: int, pipeline: models.PipelineType, output_file_path: str, parent_model_output_id: int = None, pipeline_stage: str = None) -> models.ModelOutput:
    """
    Create a ModelOutput record in the database.
    """
    new_output = models.ModelOutput(
        image_id=image_id,
        user_id=user_id,
        pipeline=pipeline,
        output_file_path=output_file_path,
        parent_model_output_id=parent_model_output_id,
        pipeline_stage=pipeline_stage
    )
    db.add(new_output)
    db.commit()
    db.refresh(new_output)
    return new_output

def process_single_pipeline(
    db: Session, user_id: int, upload_file, model, pipeline_type: models.PipelineType, parent_model_output_id: int = None, pipeline_stage: str = None
):
    """
    Modular function to process a single pipeline:
      1. Save the file.  
      2. Create an image record.
      3. Run inference using the provided model.
      4. Create a model output record.
    Returns the original Image record and the ModelOutput record.
    """
    # Save file from UploadFile and create an image record.
    input_file_path = save_upload_file(upload_file)
    image_record = create_image_record(db, user_id, input_file_path)

    # Run model inference on input image.
    output_file_path = run_inference(model, input_file_path)

    # Create a record for the processed image.
    output_record = create_model_output_record(db, image_record.id, pipeline_type, output_file_path, parent_model_output_id, pipeline_stage)

    return image_record, output_record
