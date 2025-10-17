# BFF Model Server

A FastAPI-based server that provides chat functionality with Ollama integration and speech-to-text capabilities using Whisper. This project serves as a backend for AI model interactions with support for both text and image inputs.

## Table of Contents
1. [Setup and Installation](#setup-and-installation)
2. [Usage](#usage)
3. [Testing](#testing)
4. [Models](#models)

## Setup and Installation

### Prerequisites
- macOS system
- Homebrew package manager
- Conda/Miniconda
- Python 3.11+

### 1. Install Homebrew

If Homebrew is not already installed, open Terminal and run:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Install Miniconda

Install Miniconda using Homebrew:

```bash
brew install --cask miniconda
```

After installation, initialize conda:

```bash
conda init
```

Restart your Terminal or run:

```bash
source ~/.zshrc
```

### 3. Create and Activate Conda Environment

Create a new conda environment named `bff`:

```bash
conda create --name bff python=3.11
```

Activate the environment:

```bash
conda activate bff
```

### 4. Install Ollama

Install Ollama to run language models locally:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 5. Install Project Dependencies

Navigate to the project directory and install requirements:

```bash
cd /path/to/bff-model
pip install -r requirements.txt
```

### 6. Pull Required Models

Download the default model:

```bash
ollama pull gemma3:4b
```

For vision capabilities, also pull:

```bash
ollama pull qwen2.5vl:3b
```

## Usage

### Start the Server

To run the server:

```bash
python server.py
```

The server will start on:
- **HTTP**: `http://localhost:8000` (if no SSL certificates)
- **HTTPS**: `https://localhost:8443` (if SSL certificates are present)

### API Endpoints

- `GET /` - Serves the web interface
- `GET /health` - Health check endpoint
- `GET /models` - List available Ollama models
- `POST /chat` - Chat with language models
- `POST /stt` - Speech-to-text conversion

### Environment Variables

Configure the server using these environment variables:

```bash
export OLLAMA_URL="http://localhost:11434/api/chat"
export OLLAMA_MODEL="gemma3:4b"
export WHISPER_SIZE="tiny.en"
export WHISPER_DEVICE="cpu"
```

## Testing

### 1. Test with Ollama CLI

You can test models directly using the Ollama command line interface:

#### Basic Chat Testing

```bash
ollama run gemma3:4b "Hello, how are you?"
```

#### Vision Model Testing with Images

Test image analysis capabilities:

```bash
# Interactive mode with image support
ollama run qwen2.5vl:3b

# Then in the interactive session:
>>> what do you see in this image: @/path/to/your/image.jpg
```

#### Testing with DeepSeek-R1

```bash
ollama run deepseek-r1 "Explain quantum computing"
```

### 2. Test Image Analysis

Using the provided test images:

```bash
# Test with dog image
time ollama run qwen2.5vl:3b "what does the robot dog see through this camera view: @/Volumes/Work/Projects/bff/code/bff-model/dog.jpg"

# Test with garage image  
time ollama run qwen2.5vl:3b "what does the robot dog see through this camera view: @/Volumes/Work/Projects/bff/code/bff-model/garage.jpg"
```

### 3. Performance Testing

Monitor response times:

```bash
# Test response time
time ollama run qwen2.5vl:3b "Describe this scene briefly: @/path/to/image.jpg"
```

## Models

### Currently Installed Models

Based on your system (`ollama list` output):

| Model | Size | Vision Support | Jetson Nano Compatible |
|-------|------|----------------|----------------------|
| `gemma3:latest` | 3.3 GB | ❌ | ✅ |
| `gemma3n:latest` | 7.5 GB | ❌ | ❌ |
| `gemma3n:e4b` | 7.5 GB | ❌ | ❌ |
| `llama3.2:3b` | 2.0 GB | ❌ | ✅ |
| `qwen2.5vl:7b` | 6.0 GB | ✅ | ❌ |
| `llama3:latest` | 4.7 GB | ❌ | ❌ |

### Vision Models (Accept Images)

- **`qwen2.5vl:3b`** - 3B parameter vision-language model
- **`qwen2.5vl:7b`** - 7B parameter vision-language model (currently installed)

To install additional vision models:

```bash
# Install smaller vision model for Jetson Nano
ollama pull qwen2.5vl:3b

# Install other vision models
ollama pull llava:7b
ollama pull bakllava:7b
```

### Jetson Nano Compatible Models

Models suitable for Jetson Nano (ARM64, limited RAM):

**Recommended for Jetson Nano:**
- `llama3.2:3b` (2.0 GB) - Good balance of performance and resource usage
- `gemma3:latest` (3.3 GB) - Efficient text model
- `qwen2.5vl:3b` (when available) - Vision capabilities in smaller size

**Install Jetson Nano compatible models:**

```bash
# Pull smaller models for Jetson Nano
ollama pull llama3.2:3b
ollama pull gemma3:4b
ollama pull qwen2.5vl:3b
```

### Model Performance Notes

- **Response Times**: Vision models typically take longer (10-50 seconds) compared to text-only models (1-5 seconds)
- **Memory Usage**: Larger models require more RAM - monitor system resources when running multiple models
- **Jetson Nano**: Stick to models under 4GB for optimal performance

### Additional Model Information

To get detailed information about any model:

```bash
ollama show <model-name>
```

Example:
```bash
ollama show qwen2.5vl:7b
```

## SSL Configuration (Optional)

To enable HTTPS, generate SSL certificates:

```bash
openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365
```

The server will automatically detect and use these certificates if present.

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**: Ensure Ollama is running (`ollama serve`)
2. **Model Not Found**: Pull the required model first (`ollama pull <model-name>`)
3. **Memory Issues**: Use smaller models or increase system memory
4. **SSL Certificate Issues**: Check file permissions and certificate validity

### Performance Optimization

- Use `tiny.en` Whisper model for faster speech-to-text
- Set `WHISPER_DEVICE=cpu` for systems without CUDA
- Choose appropriate model sizes based on available hardware resources