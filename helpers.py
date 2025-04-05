# helpers.py
import os
import uuid
import torch
from torchvision import transforms
from fastapi import UploadFile
from PIL import Image

# Define these directories as needed
UPLOAD_DIRECTORY = r".\uploaded_images"
OUTPUT_DIRECTORY = r".\output_images"

os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

def save_upload_file(upload_file: UploadFile) -> str:
    """
    Save the uploaded file to disk, reading content within this function.
    """
    file_extension = os.path.splitext(upload_file.filename)[1]
    file_name = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_DIRECTORY, file_name)
    contents = upload_file.file.read() # Read file contents here, within save_upload_file
    # Use buffer to write content to disk
    with open(file_path, "wb") as buffer:
        buffer.write(contents)
    return file_path

def preprocess_image(image_path: str, target_size=(384, 384)) -> torch.Tensor:
    """
    Reads an image from disk and performs preprocessing.
    """
    img = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),  # converts to tensor (and scales pixels to [0,1])
    ])
    img_tensor = preprocess(img)
    # Add batch dimension: [C, H, W] -> [1, C, H, W]
    return img_tensor.unsqueeze(0)

def postprocess_tensor(output_tensor: torch.Tensor) -> str:
    """
    Convert the output tensor from the model to an image file.
    Returns the file path where the image is saved.
    """
    # Assuming output_tensor is of shape [1, C, H, W]
    output_tensor = output_tensor.squeeze(0).clamp(0, 1)
    # Convert to PIL Image; note: to_pil_image expects values in [0,1]
    to_pil = transforms.ToPILImage()
    output_image = to_pil(output_tensor.cpu())
    
    # Save the image file with a unique name
    output_file = f"{uuid.uuid4()}.png"
    output_path = os.path.join(OUTPUT_DIRECTORY, output_file)
    output_image.save(output_path)
    return output_path
