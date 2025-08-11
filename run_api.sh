#!/bin/bash

# Image Agent API Runner Script

echo "=== Image Agent API ==="
echo "Starting API server..."

# Check if virtual environment exists
if [ ! -d "image_agent_env" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    echo "Run: python3 -m venv image_agent_env"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source image_agent_env/bin/activate

# Check if dependencies are installed
if ! python -c "import fastapi, torch, transformers" 2>/dev/null; then
    echo "❌ Dependencies not installed. Installing now..."
    pip install -r requirements.txt
fi

# Check if models are available (optional check)
echo "🔍 Checking model availability..."
python -c "
import os
models = {
    'LLaMA': '~/llama3-8B',
    'BLIP': '~/.cache/huggingface/hub/models--Salesforce--blip-image-captioning-base',
    'YOLO': '~/yolo/yolov8s.pt',
    'CLIP': '~/.cache/huggingface/hub/models--openai--clip-vit-base-patch32'
}

print('Model Status:')
for name, path in models.items():
    expanded_path = os.path.expanduser(path)
    if os.path.exists(expanded_path):
        print(f'✅ {name}: Found')
    else:
        print(f'❌ {name}: Not found at {expanded_path}')
"

echo ""
echo "🚀 Starting API server..."
echo "📖 API Documentation will be available at:"
echo "   - Swagger UI: http://localhost:8000/docs"
echo "   - ReDoc: http://localhost:8000/redoc"
echo ""
echo "🛑 Press Ctrl+C to stop the server"
echo ""

# Run the API
python router.py
