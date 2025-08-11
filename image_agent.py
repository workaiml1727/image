import os
import json
import sys
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel, AutoModelForCausalLM, AutoTokenizer
from ultralytics import YOLO
from PIL import Image
import torch
import psycopg2
from dotenv import load_dotenv

# Ensure offline mode
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppress parallelism warnings

# Define constants
IMAGE = os.path.expanduser("~/Agentic/ImageCrew/Agents/People.jpg")
CAPTION = os.path.expanduser("~/Agentic/ImageCrew/Agents/caption.txt")
OBJECTS = os.path.expanduser("~/Agentic/ImageCrew/Agents/objects.txt")
DESCRIPTION = os.path.expanduser("~/Agentic/ImageCrew/Agents/description.txt")
FEATURES = os.path.expanduser("~/Agentic/ImageCrew/Agents/features.txt")
VALIDATION = os.path.expanduser("~/Agentic/ImageCrew/Agents/validation.txt")
REFINEMENT = os.path.expanduser("~/Agentic/ImageCrew/Agents/refinement.txt")
METADATA = os.path.expanduser("~/Agentic/ImageCrew/Agents/metadata.txt")

# Model Context Manager
class ModelContext:
    def __init__(self):
        self.llama_model_path = os.path.expanduser("~/llama3-8B")
        self.blip_model_path = os.path.expanduser("~/.cache/huggingface/hub/models--Salesforce--blip-image-captioning-base")
        self.yolo_model_path = os.path.expanduser("~/yolo/yolov8s.pt")
        self.clip_model_path = os.path.expanduser("~/.cache/huggingface/hub/models--openai--clip-vit-base-patch32")
        
        if not os.path.exists(self.llama_model_path):
            raise FileNotFoundError(f"LLaMA model directory not found at {self.llama_model_path}")
        if not os.path.exists(self.blip_model_path):
            raise FileNotFoundError(f"BLIP model not found at {self.blip_model_path}")
        if not os.path.exists(self.yolo_model_path):
            raise FileNotFoundError(f"YOLO model not found at {self.yolo_model_path}")
        if not os.path.exists(self.clip_model_path):
            raise FileNotFoundError(f"CLIP model not found at {self.clip_model_path}")
        
        self.llama_model = AutoModelForCausalLM.from_pretrained(self.llama_model_path, local_files_only=True)
        self.llama_tokenizer = AutoTokenizer.from_pretrained(self.llama_model_path, local_files_only=True)
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", local_files_only=True, use_fast=True)
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", local_files_only=True)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True, use_fast=True)
        self.yolo_model = YOLO(self.yolo_model_path)  # Load YOLO last to avoid tokenizer conflicts

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.llama_model, self.llama_tokenizer, self.blip_processor, self.blip_model
        del self.clip_model, self.clip_processor, self.yolo_model
        torch.cuda.empty_cache()

# Define Functions
def caption_generation(image, processor, model):
    with Image.open(image) as raw_image:
        inputs = processor(images=raw_image.convert('RGB'), return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    with open(CAPTION, 'w') as f:
        f.write(caption)
    return caption

def object_detection(image, model):
    results = model(image)
    objects = []
    for result in results:
        for box in result.boxes:
            objects.append({
                "class": result.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy.tolist()
            })
    objects_text = "Detected objects:\n" + "\n".join(
        [f"- {obj['class']} (confidence: {obj['confidence']:.2f}, bbox: {obj['bbox']})" for obj in objects]
    ) if objects else "No objects detected."
    with open(OBJECTS, 'w') as f:
        f.write(objects_text)
    return {"objects": objects}

def detailed_description(caption, objects, model, tokenizer):
    sorted_objects = sorted(objects["objects"], key=lambda x: x["confidence"], reverse=True)[:5]
    objects_text = "\n".join(
        [f"- {obj['class']} (confidence: {obj['confidence']:.2f})" for obj in sorted_objects]
    ) if sorted_objects else "No objects detected."
    prompt = (
        f"Based on the following image caption and detected objects, provide a detailed textual description of everything happening in the image, including objects, actions, and fine details:\n"
        f"Caption: {caption}\n"
        f"Objects:\n{objects_text}\n"
        f"Describe what people are doing, the setting, colors, positions, and any other observable details."
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1000)
    outputs = model.generate(**inputs, max_new_tokens=500)
    description = tokenizer.decode(outputs[0], skip_special_tokens=True)
    with open(DESCRIPTION, 'w') as f:
        f.write(description)
    return description

def feature_extraction(image, model, processor):
    with Image.open(image) as raw_image:
        inputs = processor(images=raw_image.convert('RGB'), return_tensors="pt")
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs).squeeze(0).tolist()  # Flatten to [512]
    embeddings_text = f"Extracted 512-dimensional feature vector: {embeddings[:5]}... (first 5 values)"
    with open(FEATURES, 'w') as f:
        f.write(embeddings_text)
    return embeddings

def validate_outputs(caption, objects, description, embeddings):
    report = []
    if not caption or len(caption.strip()) < 5:
        report.append("Caption is too short or empty.")
    else:
        report.append("Caption is valid.")
    if not objects.get("objects") or len(objects["objects"]) == 0:
        report.append("No objects detected.")
    else:
        report.append(f"Detected {len(objects['objects'])} objects with valid confidence scores.")
    if not description or len(description.strip()) < 20:
        report.append("Detailed description is too short or empty.")
    else:
        report.append("Detailed description is valid.")
    if not embeddings or len(embeddings) != 512:
        report.append("CLIP embeddings are missing or incorrect length.")
    else:
        report.append("CLIP embeddings are valid.")
    validation_text = "\n".join(report)
    with open(VALIDATION, 'w') as f:
        f.write(validation_text)
    return validation_text

def refine_results(validation_report, caption, objects, description, embeddings):
    refined = {"caption": caption, "objects": objects, "description": description, "embeddings": embeddings}
    if "Caption is too short or empty" in validation_report:
        refined["caption"] = "Default caption: Image content not clearly described."
    if "No objects detected" in validation_report:
        refined["objects"] = {"objects": [{"class": "unknown", "confidence": 0.0, "bbox": [0, 0, 0, 0]}]}
    if "Detailed description is too short or empty" in validation_report:
        refined["description"] = "Default description: No detailed observations available."
    if "CLIP embeddings are missing or incorrect length" in validation_report:
        refined["embeddings"] = [0.0] * 512
    refinement_text = (
        f"Refined caption: {refined['caption']}\n"
        f"Refined objects: {refined['objects']['objects']}\n"
        f"Refined description: {refined['description']}\n"
        f"Refined embeddings: {refined['embeddings'][:5]}... (first 5 values)"
    )
    with open(REFINEMENT, 'w') as f:
        f.write(refinement_text)
    return refined

def store_in_postgres(image, caption, objects, description, clip_embedding):
    with psycopg2.connect(host="localhost", port=5432, database="image_data", user="agent", password="agent123") as conn:
        with conn.cursor() as cursor:
            objects_with_data = objects.copy()
            objects_with_data["caption"] = caption
            objects_with_data["description"] = description
            objects_str = json.dumps(objects_with_data)
            filename = os.path.basename(image)
            cursor.execute("SELECT id FROM images WHERE filename = %s", (filename,))
            existing_record = cursor.fetchone()
            if existing_record:
                cursor.execute(
                    "UPDATE images SET objects = %s, clip_vector = %s WHERE filename = %s RETURNING id",
                    (objects_str, clip_embedding, filename)
                )
                image_id = cursor.fetchone()[0]
                action = "Updated"
            else:
                cursor.execute(
                    "INSERT INTO images (filename, objects, clip_vector) VALUES (%s, %s, %s) RETURNING id",
                    (filename, objects_str, clip_embedding)
                )
                image_id = cursor.fetchone()[0]
                action = "Stored"
            conn.commit()
            metadata_text = f"{action} image metadata with ID {image_id}"
            with open(METADATA, 'w') as f:
                f.write(metadata_text)
            return metadata_text

# Main Workflow
def main():
    image_path = IMAGE if len(sys.argv) == 1 else os.path.expanduser(sys.argv[1])
    if not image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = image_path + '.jpg'
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    with ModelContext() as ctx:
        print("=== Caption Agent ===")
        caption = caption_generation(image_path, ctx.blip_processor, ctx.blip_model)
        print(f"Caption Agent Output: {caption}")

        print("\n=== Object Detection Agent ===")
        objects = object_detection(image_path, ctx.yolo_model)
        print(f"Object Detection Agent Output: {open(OBJECTS).read()}")

        print("\n=== Detailed Description Agent ===")
        description = detailed_description(caption, objects, ctx.llama_model, ctx.llama_tokenizer)
        print(f"Detailed Description Agent Output: {description}")

        print("\n=== Feature Extraction Agent ===")
        embeddings = feature_extraction(image_path, ctx.clip_model, ctx.clip_processor)
        print(f"Feature Extraction Agent Output: {open(FEATURES).read()}")

        print("\n=== Validation Agent ===")
        validation_report = validate_outputs(caption, objects, description, embeddings)
        print(f"Validation Agent Output:\n{validation_report}")

        print("\n=== Refinement Agent ===")
        refined_results = refine_results(validation_report, caption, objects, description, embeddings)
        print(f"Refinement Agent Output:\n{open(REFINEMENT).read()}")

        print("\n=== Storage Agent ===")
        storage_confirmation = store_in_postgres(
            image_path,
            refined_results["caption"],
            refined_results["objects"],
            refined_results["description"],
            refined_results["embeddings"]
        )
        print(f"Storage Agent Output: {storage_confirmation}")

if __name__ == "__main__":
    main()
