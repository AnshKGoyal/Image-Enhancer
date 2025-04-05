# Hybrid Image Enhancement API and Frontend

This project provides a FastAPI backend and Streamlit frontend for **hybrid image enhancement** using deep learning models. It enables users to design **custom multi-stage pipelines** combining **deblurring** and **dehazing** in any sequence, for advanced and flexible image restoration workflows.

---

## Features

- **Configurable Hybrid Pipelines:**  
  Design custom image enhancement workflows by **adding multiple deblurring and dehazing stages sequentially**. This hybrid approach allows tailored, multi-step processing beyond single-model pipelines.

- **Deep Learning Models:**  
  - **Deblurring:** Custom-trained **NAFNet** model  
  - **Dehazing:** Custom-trained **U-Net** with mit_b3 encoder  
  Both models are integrated into the hybrid pipeline.

- **User Authentication:**  
  Secure login and registration system, with personalized output history.

- **Output History:**  
  View processed images, including intermediate and final outputs, linked to user accounts.

- **Interactive Streamlit Frontend:**  
  User-friendly interface for uploading images, configuring hybrid pipelines, running processing, and visualizing results.

- **Scalable Architecture:**  
  Modular design allows easy integration of new models and processing stages.

---

## Architecture

### Backend (FastAPI)

- **API Endpoints:**  
  Provides RESTful endpoints for:
  - User authentication (`/login`, `/users/`, `/user/detail/`)
  - Hybrid pipeline execution (`/predict_pipeline/`)
  - Model output management (`/model_outputs/global/`, `/model_outputs/user/`)

- **Business Logic:**  
  - **User Management:** Registration, login, token generation, and validation  
  - **Image Uploading:** Saves images, computes hashes to avoid duplicates, associates images with users  
  - **Hybrid Pipeline Execution:**  
    - Accepts a user-defined sequence of stages (deblur/dehaze)  
    - Runs each model sequentially, saving intermediate outputs  
    - Stores all outputs in the database, linked to the original image and user  
  - **Model Management:** Loads NAFNet and U-Net models at startup for inference  
  - **Database:** SQLAlchemy ORM with SQLite/MySQL, storing users, images, pipeline outputs, and relationships  
  - **Security:** Token-based authentication with per-user access control

### Frontend (Streamlit)

- **Pipeline Configuration:**  
  Users can **add multiple processing stages sequentially** via an "Add Stage" button. For each stage, they select either **deblur** or **dehaze**. The pipeline executes stages **in the order added**.

- **User Interface:**  
  - Register and login  
  - Upload images  
  - Configure hybrid pipelines  
  - Run processing and view outputs for each stage and final result  
  - Browse global and user-specific output histories with filtering options

- **API Client:**  
  Communicates with the FastAPI backend for authentication, pipeline execution, and data retrieval.

---

## Demo Video

Watch a demonstration of the application in action:  


https://github.com/user-attachments/assets/a55d0d8d-de94-45aa-8475-0440dacdf8bf



---

## Model Training

### Deblurring Model (NAFNet)

- **Dataset Preparation:**  
  - Loads a curated subset of the [A Curated List of Image Deblurring Datasets](https://www.kaggle.com/datasets/jishnuparayilshibu/a-curated-list-of-image-deblurring-datasets)  
  - Excludes datasets like 'CelebA'  
  - Maps paired blurry and sharp images  
  - Creates train/test DataFrames  

- **DataLoader Setup:**  
  - Custom PyTorch `Dataset` class loads blurry and sharp pairs  
  - Resizes images to 384x384, normalizes to [0,1]  
  - DataLoader with batch size 16, shuffling, prefetching, pin_memory, persistent_workers  
  - Batch visualization to verify data

- **Model & Loss:**  
  - NAFNet architecture with projected_in_channels=24, shallow-to-deep layers  
  - Composite loss: weighted sum of  
    - **PSNR Loss** (weight 0.7)  
    - **Perceptual Loss** (VGG16 feature-based, weight 0.3)  
    - No contribution from MSE or L1 losses (weights 0.0 and 0)

- **Training:**  
  - PyTorch Lightning framework  
  - Mixed precision (`precision='16-mixed'`)  
  - AdamW optimizer  
  - Progress bar callback  

- **Evaluation:**  
  - Runs inference on validation samples  
  - Visualizes blurry input, model output, and ground truth side by side  
  - Saves final model weights

---

### Dehazing Model (U-Net with mit_b3 Encoder)

- **Dataset Preparation:**  
  - Uses [RESIDE-OUT](https://www.kaggle.com/datasets/anshkgoyal/reside-out) dataset  
  - Pairs hazy and ground truth images, verifies filenames  
  - Saves paired paths as CSVs  
  - Loads CSVs into DataFrames  
  - Visualizes samples for verification

- **DataLoader Setup:**  
  - Custom PyTorch `Dataset` class loads hazy and GT pairs  
  - Resizes images to 384x384, normalizes to [0,1]  
  - DataLoader with batch size 16, shuffling, prefetching, pin_memory, persistent_workers  
  - Batch visualization to verify data

- **Model & Loss:**  
  - U-Net with mit_b3 encoder, pretrained on ImageNet, activation sigmoid  
  - Composite loss: weighted sum of  
    - **Perceptual Loss** (VGG16 feature-based, weight 0.4)  
    - **MSE Loss** (weight 0.2)  
    - **L1 Loss** (weight 0.4)

- **Training:**  
  - PyTorch Lightning framework  
  - Mixed precision  
  - AdamW optimizer  
  - Progress bar callback  

- **Evaluation:**  
  - Runs inference on validation samples  
  - Visualizes hazy input, model output, and ground truth side by side  
  - Saves final model weights

[kaggle notebook link](https://www.kaggle.com/code/anshkgoyal/dehazing)
---

---

## Project Structure

- `main.py`: FastAPI backend application with API endpoints and hybrid pipeline logic  
- `frontend/app.py`: Streamlit frontend application with pipeline configuration and visualization  
- `models.py`: SQLAlchemy models for users, images, and model outputs  
- `schemas.py`: Pydantic schemas for API validation  
- `utils.py`: Utilities for password hashing, image hashing, inference, and database helpers  
- `helpers.py`: File saving, image preprocessing, and postprocessing  
- `deeplearning_models.py`: PyTorch Lightning model classes for deblurring and dehazing  
- `notebooks/`:  
  - `deblurring.ipynb`: Training pipeline for NAFNet  
  - `dehazing.ipynb`: Training pipeline for U-Net  
- `models/`:  
  - `deblurring/model_state_19_new.pth`  
  - `dehazing/model_state_26_jan_final_epoch_plus25.pth`  
- `output_images/`: Saved outputs from pipeline stages  
- `uploaded_images/`: User-uploaded images  
- `requirements.txt`: Python dependencies

---

## Setup and Installation

### 0. Download Pre-trained Model Weights

- Download the pre-trained model weights from Google Drive:  
  [**Model Weights Download Link**](https://drive.google.com/drive/folders/1poqEPc01gX4ZgnO5plK3JLnyujLeJFnx?usp=sharing)

- After downloading, extract and place the files into the `models/` directory as follows:

```
models/
├── deblurring/
│   └── model_state_19_new.pth
├── dehazing/
│   └── model_state_26_jan_final_epoch_plus25.pth
```

- **Ensure the folder structure matches above** so the application can load the weights correctly.

- **Optional:** You can customize the model weights paths in `main.py` by editing:

```python
MODEL_WEIGHTS_PATH = r"models\\deblurring\\model_state_19_new.pth"
MODEL_WEIGHTS_PATH_DEHAZING = r"models\\dehazing\\model_state_26_jan_final_epoch_plus25.pth"
```

---

### 1. Clone the Repository
Download or clone this repository, then open the project directory in your preferred editor (e.g., VS Code).

---

### 2. Install MySQL Server
- **Download and install MySQL Community Server:**  
  https://dev.mysql.com/downloads/mysql/
- **Start the MySQL server** and ensure it is running.

---

### 3. Create the Database
Open a terminal or MySQL client and run:

```sql
CREATE DATABASE photo_enhancer;
```

- By default, the app connects with username `root` and password `root` on `localhost:3306`.
- **To customize credentials or host**, edit the connection string in `db.py`:

```python
SQLALCHEMY_DATABASE_URL = "mysql+pymysql://<username>:<password>@<host>:<port>/<database_name>"
```

---

### 4. (Optional) Set up a Virtual Environment
It is recommended to use a virtual environment to manage dependencies:

```bash
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

---

### 5. Install Python Dependencies
Install all required packages:

```bash
pip install -r requirements.txt
```

---

### 6. Initialize the Database Schema
No manual migration is needed.  
**The database tables will be created automatically** when you first run the backend server.

---

### 7. Run the Backend API Server
Start the FastAPI backend with:

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

---

### 8. Run the Streamlit Frontend
In a new terminal (with the virtual environment activated), run:

```bash
streamlit run frontend/app.py
```

The frontend will open in your browser, typically at `http://localhost:8501`.


---

## Usage

1. **Register/Login** via the frontend.
2. **Upload an image**.
3. **Configure your hybrid pipeline** by **adding multiple stages sequentially** and selecting the model for each.
4. **Run the pipeline** and view outputs for each stage and the final result.
5. **Explore output history** (global and user-specific) with filtering options.

---

## Datasets

- **Deblurring:**  
  [A Curated List of Image Deblurring Datasets](https://www.kaggle.com/datasets/jishnuparayilshibu/a-curated-list-of-image-deblurring-datasets) (filtered subset)

- **Dehazing:**  
  [RESIDE-OUT](https://www.kaggle.com/datasets/anshkgoyal/reside-out)

---

## Licensing

- **Code:** [Apache 2.0 License](https://github.com/AnshKGoyal/Image-Enhancer/blob/main/LICENSE)  
- **Models:** [Apache 2.0 License](https://github.com/AnshKGoyal/Image-Enhancer/blob/main/LICENSE)

---

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests.
