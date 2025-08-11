from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import tempfile
import shutil
from typing import Optional
import json
from image_agent import ModelContext, caption_generation, object_detection, detailed_description, feature_extraction, validate_outputs, refine_results, store_in_postgres

# Ensure offline mode
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppress parallelism warnings
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU mode for better compatibility
os.environ["TORCH_HOME"] = os.path.expanduser("~/.cache/torch")  # Local torch cache

app = FastAPI(
    title="Image Agent API",
    description="API for image analysis using multiple AI models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model context
model_context = None

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global model_context
    try:
        model_context = ModelContext()
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
        model_context = None

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup models on shutdown"""
    global model_context
    if model_context:
        del model_context
        model_context = None

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Image Agent API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    global model_context
    status = {
        "api_status": "running",
        "models_loaded": model_context is not None,
        "available_endpoints": [
            "GET /",
            "GET /health", 
            "POST /analyze",
            "POST /caption",
            "POST /detect-objects",
            "POST /extract-features",
            "GET /analyze/{image_id}"
        ]
    }
    return status

@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    store_in_db: bool = Form(True),
    include_validation: bool = Form(True)
):
    """
    Complete image analysis pipeline
    
    - **file**: Image file to analyze
    - **store_in_db**: Whether to store results in database
    - **include_validation**: Whether to include validation step
    """
    global model_context
    
    if not model_context:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        # Run analysis pipeline
        results = {}
        
        # 1. Caption Generation
        print("=== Caption Generation ===")
        caption = caption_generation(temp_path, model_context.blip_processor, model_context.blip_model)
        results["caption"] = caption
        
        # 2. Object Detection
        print("=== Object Detection ===")
        objects = object_detection(temp_path, model_context.yolo_model)
        results["objects"] = objects
        
        # 3. Detailed Description
        print("=== Detailed Description ===")
        description = detailed_description(caption, objects, model_context.llama_model, model_context.llama_tokenizer)
        results["description"] = description
        
        # 4. Feature Extraction
        print("=== Feature Extraction ===")
        embeddings = feature_extraction(temp_path, model_context.clip_model, model_context.clip_processor)
        results["embeddings"] = embeddings[:10]  # Return first 10 values for API response
        
        # 5. Validation (optional)
        if include_validation:
            print("=== Validation ===")
            validation_report = validate_outputs(caption, objects, description, embeddings)
            results["validation"] = validation_report
            
            # 6. Refinement
            print("=== Refinement ===")
            refined_results = refine_results(validation_report, caption, objects, description, embeddings)
            results["refined"] = refined_results
        
        # 7. Database Storage (optional)
        if store_in_db:
            print("=== Database Storage ===")
            storage_result = store_in_postgres(
                temp_path,
                results.get("refined", {}).get("caption", caption),
                results.get("refined", {}).get("objects", objects),
                results.get("refined", {}).get("description", description),
                results.get("refined", {}).get("embeddings", embeddings)
            )
            results["storage"] = storage_result
        
        # Cleanup temporary file
        os.unlink(temp_path)
        
        return JSONResponse(content=results, status_code=200)
        
    except Exception as e:
        # Cleanup on error
        if 'temp_path' in locals():
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/caption")
async def generate_caption(file: UploadFile = File(...)):
    """Generate caption for an image"""
    global model_context
    
    if not model_context:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        caption = caption_generation(temp_path, model_context.blip_processor, model_context.blip_model)
        
        os.unlink(temp_path)
        
        return {"caption": caption}
        
    except Exception as e:
        if 'temp_path' in locals():
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=f"Caption generation failed: {str(e)}")

@app.post("/detect-objects")
async def detect_objects(file: UploadFile = File(...)):
    """Detect objects in an image"""
    global model_context
    
    if not model_context:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        objects = object_detection(temp_path, model_context.yolo_model)
        
        os.unlink(temp_path)
        
        return objects
        
    except Exception as e:
        if 'temp_path' in locals():
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=f"Object detection failed: {str(e)}")

@app.post("/extract-features")
async def extract_features(file: UploadFile = File(...)):
    """Extract CLIP features from an image"""
    global model_context
    
    if not model_context:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        embeddings = feature_extraction(temp_path, model_context.clip_model, model_context.clip_processor)
        
        os.unlink(temp_path)
        
        return {"embeddings": embeddings[:10], "dimension": len(embeddings)}
        
    except Exception as e:
        if 'temp_path' in locals():
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")

@app.get("/analyze/{image_id}")
async def get_analysis_result(image_id: int):
    """Get analysis results by image ID from database"""
    try:
        import psycopg2
        from dotenv import load_dotenv
        load_dotenv()
        
        with psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", 5432)),
            database=os.getenv("DB_NAME", "image_data"),
            user=os.getenv("DB_USER", "agent"),
            password=os.getenv("DB_PASSWORD", "agent123")
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT filename, objects, clip_vector FROM images WHERE id = %s", (image_id,))
                result = cursor.fetchone()
                
                if not result:
                    raise HTTPException(status_code=404, detail="Image analysis not found")
                
                filename, objects_json, clip_vector = result
                
                return {
                    "id": image_id,
                    "filename": filename,
                    "objects": json.loads(objects_json) if objects_json else {},
                    "clip_vector": clip_vector[:10] if clip_vector else []  # Return first 10 values
                }
                
    except psycopg2.OperationalError:
        raise HTTPException(status_code=503, detail="Database connection failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving analysis: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
