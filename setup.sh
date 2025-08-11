#!/bin/bash

# Image Agent API Setup Script

echo "=== Image Agent API Setup ==="
echo "This script will help you set up the Image Agent API"
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "image_agent_env" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv image_agent_env
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source image_agent_env/bin/activate

# Upgrade pip
echo "🔧 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "🔧 Installing dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "🔧 Creating .env file..."
    cat > .env << EOF
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=image_data
DB_USER=agent
DB_PASSWORD=agent123

# Model Paths (optional - will use defaults if not set)
LLAMA_MODEL_PATH=~/llama3-8B
BLIP_MODEL_PATH=~/.cache/huggingface/hub/models--Salesforce--blip-image-captioning-base
YOLO_MODEL_PATH=~/yolo/yolov8s.pt
CLIP_MODEL_PATH=~/.cache/huggingface/hub/models--openai--clip-vit-base-patch32
EOF
    echo "✅ .env file created"
else
    echo "✅ .env file already exists"
fi

# Check model availability
echo ""
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
missing_models = []
for name, path in models.items():
    expanded_path = os.path.expanduser(path)
    if os.path.exists(expanded_path):
        print(f'✅ {name}: Found')
    else:
        print(f'❌ {name}: Not found at {expanded_path}')
        missing_models.append(name)

if missing_models:
    print('\\n⚠️  Missing models detected. You may need to download them:')
    print('   - LLaMA: Download from Hugging Face or use a local model')
    print('   - BLIP: Will be downloaded automatically on first use')
    print('   - YOLO: Download from Ultralytics or use a local model')
    print('   - CLIP: Will be downloaded automatically on first use')
else:
    print('\\n✅ All models are available!')
"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Start the API: ./run_api.sh"
echo "2. Test the API: python test_api.py"
echo "3. View documentation: http://localhost:8000/docs"
echo ""
echo "📖 For more information, see README_API.md"
