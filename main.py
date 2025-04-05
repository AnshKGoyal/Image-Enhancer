# main.py
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, status, Form
from contextlib import asynccontextmanager
from sqlalchemy import func
from sqlalchemy.orm import Session
import models
import schemas
from db import engine, SessionLocal, Base
from utils import get_password_hash, verify_password, create_model_output_record, get_or_create_image_record, run_inference
from fastapi.middleware.cors import CORSMiddleware
from deeplearning_models import LitUModel_deblurring, LitUModel_dehazing
from fastapi import Depends, HTTPException, status, Header
from fastapi.security import OAuth2PasswordRequestForm
from typing import List, Dict, Any
import secrets




MODEL_WEIGHTS_PATH = r"models\deblurring\model_state_19_new.pth"
MODEL_WEIGHTS_PATH_DEHAZING = r"models\dehazing\model_state_26_jan_final_epoch_plus25.pth"
deblurring_model = None
dehazing_model = None


TOKEN_STORE = {}  # e.g., { "user_email": "the_generated_token" }

# Create tables in the database
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Dual-Pipeline Restoration API")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global deblurring_model
    global dehazing_model
    if deblurring_model is None:
        deblurring_model = LitUModel_deblurring.load_for_inference(MODEL_WEIGHTS_PATH, device='cpu')
        deblurring_model.eval()
    if dehazing_model is None:
        dehazing_model = LitUModel_dehazing.load_for_inference(MODEL_WEIGHTS_PATH_DEHAZING, device='cpu')
        dehazing_model.eval()
    yield  # This allows the application to start and run
    # Optional: Add cleanup code here (e.g., model cleanup)
    # Shutdown: Release resources
    print("Releasing model resources...")
    del deblurring_model
    del dehazing_model
    deblurring_model = None
    dehazing_model = None
    print("Both model resources have been released.")
app.router.lifespan_context = lifespan


# (Optional) Allow CORS so that your Streamlit or frontend can access the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency: creates a new database session and closes it after the request
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- User Endpoints ---

@app.post("/login", response_model=dict)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # Using existing user lookup and verify logic
    user = db.query(models.User).filter(models.User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # Generate a token simply using secrets
    token = secrets.token_hex(16)
    TOKEN_STORE[user.email] = token
    return {"access_token": token, "token_type": "bearer", "user_id": user.id}

def get_current_user(token: str = Header(...), db: Session = Depends(get_db)):
    # Here we check that token exists in one of the TOKEN_STORE entries.
    for email, stored_token in TOKEN_STORE.items():
        if token == stored_token:
            user = db.query(models.User).filter(models.User.email == email).first()
            return {"email":email, "user_id": user.id}
    raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/user/detail/",response_model=dict)
def user_detail(current_user: str = Depends(get_current_user)):
    return current_user

@app.post("/users/", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(user.password)
    new_user = models.User(
        username=user.username,
        email=user.email,
        password_hash=hashed_password
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


@app.get("/users/", response_model=list[schemas.User])
def read_users(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    users = db.query(models.User).offset(skip).limit(limit).all()
    return users



# --- Image Endpoints ---



@app.post("/upload_image/", response_model=schemas.Image)
def upload_image(user_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    image_record = get_or_create_image_record(db, user_id, file)
    return image_record













# --- Single Pipeline Endpoints ---
#used just for testing purposes
@app.post("/predict_deblur/", response_model=dict)
async def predict_deblur(user_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Process a deblurring pipeline:
    - Save the input.
    - Run deblurring inference.
    - Store records in the database.
    """
    global deblurring_model
    if deblurring_model is None:
        raise HTTPException(status_code=500, detail="Deblurring model not loaded.")
    
    # Get (or create) the image record using the shared helper.
    image_record = get_or_create_image_record(db, user_id, file)
    
    # Run the deblurring model on the image's file path.
    output_file_path = run_inference(deblurring_model, image_record.file_path)
    
    # Create the ModelOutput record.
    output_record = create_model_output_record(
        db,
        image_id=image_record.id,
        user_id=user_id, # Pass user_id here
        pipeline=models.PipelineType.deblur,
        output_file_path=output_file_path,
        parent_model_output_id=None,
        pipeline_stage="deblurring"
    )
    
    return {
        "input_image": image_record.file_path,
        "deblur_output": output_record.output_file_path,
        "model_output_id": output_record.id
    }
#used just for testing purposes
@app.post("/predict_dehaze/", response_model=dict)
async def predict_dehaze(user_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Process a dehazing pipeline:
    - Save the input.
    - Run dehazing inference.
    - Store records in the database.
    """
    global dehazing_model
    if dehazing_model is None:
        raise HTTPException(status_code=500, detail="Dehazing model not loaded.")

    # Get (or create) the image record using the shared helper.
    image_record = get_or_create_image_record(db, user_id, file)

    # Run the dehazing model on the image's file path.
    output_file_path = run_inference(dehazing_model, image_record.file_path)

    # Create the ModelOutput record.
    output_record = create_model_output_record(
        db,
        image_id=image_record.id,
        user_id=user_id, # Pass user_id here
        pipeline=models.PipelineType.dehaze,
        output_file_path=output_file_path,
        parent_model_output_id=None,
        pipeline_stage="dehazing"
    )

    return {
        "input_image": image_record.file_path,
        "dehaze_output": output_record.output_file_path,
        "model_output_id": output_record.id
    }
#used just for testing purposes
@app.post("/predict_hybrid/", response_model=dict)
async def predict_hybrid(user_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Process a dual-pipeline restoration: deblurring followed by dehazing.
    - Save the input image.
    - Run deblurring and save output.
    - Run dehazing on deblurred output and save output.
    - Store records for each step in the database.
    """
    global deblurring_model, dehazing_model
    if deblurring_model is None or dehazing_model is None:
        raise HTTPException(status_code=500, detail="One or both models not loaded.")

    # Get (or create) the image record for the initially uploaded image.
    image_record = get_or_create_image_record(db, user_id, file)

    # --- Deblurring Stage ---
    deblur_output_file_path = run_inference(deblurring_model, image_record.file_path)
    deblur_output_record = create_model_output_record(
        db,
        image_id=image_record.id,
        user_id=user_id, # Pass user_id here
        pipeline=models.PipelineType.hybrid,
        output_file_path=deblur_output_file_path,
        parent_model_output_id=None,
        pipeline_stage="deblurring"
    )

    # --- Dehazing Stage ---
    dehaze_output_file_path = run_inference(dehazing_model, deblur_output_file_path)
    dehaze_output_record = create_model_output_record(
        db,
        image_id=image_record.id,
        user_id=user_id, # Pass user_id here
        pipeline=models.PipelineType.hybrid,
        output_file_path=dehaze_output_file_path,
        parent_model_output_id=deblur_output_record.id, # Link to deblur output
        pipeline_stage="dehazing"
    )

    return {
        "input_image": image_record.file_path,
        "deblur_output": deblur_output_record.output_file_path,
        "dehaze_output": dehaze_output_record.output_file_path,
        "deblur_model_output_id": deblur_output_record.id,
        "dehaze_model_output_id": dehaze_output_record.id
    }


@app.post("/predict_pipeline/", response_model=dict)
async def predict_pipeline(
    file: UploadFile = File(...),
    user_id: int = Form(...),
    pipeline: str = Form(...),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """
    Accept an image and a pipeline configuration (a JSON string that is a list of stages).
    Process each stage sequentially.
    """
    import json
    if not pipeline:
        raise HTTPException(status_code=400, detail="Missing required 'pipeline' parameter")
        
    try:
        pipeline_config: List[Dict[str, Any]] = json.loads(pipeline)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid JSON format in pipeline parameter: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid pipeline configuration: {str(e)}"
        )

    if not isinstance(pipeline_config, list):
        raise HTTPException(
            status_code=400,
            detail="Pipeline config must be a list of stage objects"
        )

    for idx, stage in enumerate(pipeline_config):
        if not isinstance(stage, dict) or 'model' not in stage:
            raise HTTPException(
                status_code=400,
                detail=f"Stage {idx+1} is missing required 'model' field"
            )
    
    # Get (or create) the image record (handles duplicate checking, etc.)
    image_record = get_or_create_image_record(db, user_id, file)
    
    # Determine pipeline type
    if len(pipeline_config) > 1:
        pipeline_type = models.PipelineType.hybrid
    elif pipeline_config:
        first_stage_model = pipeline_config[0].get("model", "").lower()
        if first_stage_model == "deblur":
            pipeline_type = models.PipelineType.deblur
        elif first_stage_model == "dehaze":
            pipeline_type = models.PipelineType.dehaze
        else:
            pipeline_type = models.PipelineType.hybrid  # Default to hybrid if unknown
    else:
        pipeline_type = models.PipelineType.hybrid # Default if no stages

    # Process stages recursively/sequentially:
    previous_output_path = image_record.file_path
    stage_outputs = []
    parent_output_id = None
    
    for idx, stage in enumerate(pipeline_config):
        model_type = stage.get("model")   # e.g., "deblur", "dehaze"
        # Determine the appropriate model
        if model_type.lower() == "deblur":
            model = deblurring_model
        elif model_type.lower() == "dehaze":
            model = dehazing_model
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model {model_type}")
        
        # Run inference on the previous stage's output
        output_file_path = run_inference(model, previous_output_path)
        
        # Create ModelOutput record and link stages via parent_model_output_id
        output_record = create_model_output_record(
            db,
            image_id=image_record.id,
            user_id=user_id,
            pipeline=pipeline_type,  # Use dynamically determined pipeline_type
            output_file_path=output_file_path,
            parent_model_output_id=parent_output_id,
            pipeline_stage=f"stage_{idx+1}_{model_type}"
        )
        
        stage_outputs.append({
            "stage": idx+1,
            "model": model_type,
            "output_file_path": output_file_path,
            "model_output_id": output_record.id
        })
        
        # For next iteration, the output of current stage becomes input
        previous_output_path = output_file_path
        parent_output_id = output_record.id
    
    return {
        "input_image": image_record.file_path,
        "pipeline_results": stage_outputs,
        "final_output": previous_output_path
    }

@app.get("/model_outputs/global/", response_model=List[Dict[str, Any]])
def read_global_model_outputs(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    outputs = db.query(models.ModelOutput).offset(skip).limit(limit).all()

    serialized_outputs = []
    for output in outputs:
        serialized_output = schemas.ModelOutput.model_validate(output).model_dump()
        serialized_output["input_image"] = output.image.file_path
        serialized_outputs.append(serialized_output)
    return serialized_outputs

@app.get("/model_outputs/user/", response_model=List[Dict[str, Any]])
def read_user_model_outputs(skip: int = 0, limit: int = 100, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    user_id = current_user.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found")

    outputs = db.query(models.ModelOutput).filter(models.ModelOutput.user_id == user_id).offset(skip).limit(limit).all()

    serialized_outputs = []
    for output in outputs:
        serialized_output = schemas.ModelOutput.model_validate(output).model_dump()
        serialized_output["input_image"] = output.image.file_path
        serialized_outputs.append(serialized_output)
    return serialized_outputs


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
