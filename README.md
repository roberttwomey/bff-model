# BFF Model Server

A FastAPI-based server that provides chat functionality with Ollama integration and speech-to-text capabilities using Whisper. This project serves as a backend for AI model interactions with support for both text and image inputs.

This is primarily focused on Jetson Orin Nano deployment (8GB), but can also be tested on mac silicon laptops.

## Table of Contents
1. [Setup and Installation](#setup-and-installation)
2. [Usage](#usage)
3. [Testing](#testing)
4. [Models](#models)

## Setup and Installation on Mac OS

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

or one of the other vision models listed in the [Models](#models) section. I prefer gemma3:4b for the vision model (which you can currently only test through the ollama CLI, not through the web interface)

## Usage of Web Interface

What's cool is, once you have started this interface on your laptop, you can connect to it with your phone or other devices on the current network, and use your headset. You can also use your headset through the browser on your laptop.

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

I don't ever use this so please ignore. 

The server code itself, including the web frontend allows you to select the model directly. But you will need to download models beforehand to use them—-with the `ollama pull` commands described above.

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

| Model | Size | Vision Support | Jetson Nano Compatible | Notes |
|-------|------|----------------|----------------------|-------|
| `deepseek-r1:1.5b` | 1.1 GB | ❌ | ✅ | Advanced reasoning with `<think>` process |
| `granite3.2-vision:latest` | 2.4 GB | ✅ | ✅ | IBM's vision model |
| `gemma3:1b` | 815 MB | ❌ | ✅ | Ultra-lightweight text model |
| `gemma3:270m` | 291 MB | ❌ | ✅ | Micro text model |
| `moondream:latest` | 1.7 GB | ✅ | ✅ | Lightweight vision model |
| `qwen2.5vl:7b` | 6.0 GB | ✅ | ❌ | Large vision-language model |
| `qwen2.5vl:3b` | 3.2 GB | ✅ | ✅ | Medium vision-language model |
| `gemma3:latest` | 3.3 GB | ✅ | ✅ | Standard vision model |
| `gemma3n:e4b` | 7.5 GB | ❌ | ❌ | Large text model (vision capable, not via Ollama) |
| `gemma3n:e2b` | 5.6 GB | ❌ | ❌ | Medium-large text model (vision capable, not via Ollama) |
| `gemma3:4b` | 3.3 GB | ✅ | ✅ | 4B parameter vision model |
| `snapper-01:latest` | 2.0 GB | ❌ | ✅ | Custom model |
| `llama3.2:3b` | 2.0 GB | ❌ | ✅ | Meta's 3B model |

### Vision Models (Accept Images)

**Currently Installed Vision Models:**

- **`granite3.2-vision:latest`** - IBM's vision model (2.4 GB) 
- **`gemma3:4b`** - 4B parameter vision model (3.3 GB)
- **`gemma3:latest`** - Standard vision model (3.3 GB)
- **`moondream:latest`** - Lightweight vision model (1.7 GB)
- **`qwen2.5vl:3b`** - 3B parameter vision-language model (3.2 GB)
- **`qwen2.5vl:7b`** - 7B parameter vision-language model (6.0 GB) - WON'T RUN ON JETSON

**Vision Model Characteristics:**

- **`granite3.2-vision:latest`**: IBM's enterprise-grade vision model
- **`gemma3:4b/latest`**: Google's efficient vision models with excellent image understanding
- **`moondream:latest`**: Optimized for mobile/edge devices
- **`qwen2.5vl:3b/7b`**: Alibaba's vision-language models with strong image understanding

To install additional vision models:

```bash
# Install other vision models
ollama pull llava:7b
ollama pull bakllava:7b
ollama pull llava:13b
```

### Jetson Nano Compatible Models

Models suitable for Jetson Nano (ARM64, limited RAM):

**Text Models (No Vision):**
- `gemma3:270m` (291 MB) - Ultra-lightweight text, fastest
- `gemma3:1b` (815 MB) - Very lightweight text
- `deepseek-r1:1.5b` (1.1 GB) - Advanced reasoning with `<think>` process
- `snapper-01:latest` (2.0 GB) - Custom model
- `llama3.2:3b` (2.0 GB) - Meta's efficient model
- `gemma3n:e4b` (7.5 GB) - Large text model (vision capable, not via Ollama)
- `gemma3n:e2b` (5.6 GB) - Medium-large text model (vision capable, not via Ollama)

**Vision Models for Jetson Nano:**
- `moondream:latest` (1.7 GB) - Lightweight vision optimized for edge
- `granite3.2-vision:latest` (2.4 GB) - IBM's enterprise vision
- `qwen2.5vl:3b` (3.2 GB) - Balanced vision-language model
- `gemma3:4b` (3.3 GB) - 4B parameter vision model
- `gemma3:latest` (3.3 GB) - Standard vision model

**Install Jetson Nano compatible models:**

```bash
# Ultra-lightweight text models
ollama pull gemma3:270m
ollama pull gemma3:1b

# Advanced reasoning text model
ollama pull deepseek-r1:1.5b

# Vision models for Jetson Nano
ollama pull granite3.2-vision:latest
ollama pull moondream:latest
ollama pull qwen2.5vl:3b
ollama pull gemma3:4b

# Standard text models
ollama pull llama3.2:3b
```

### Special Model Features

#### DeepSeek-R1 Reasoning Process

The `deepseek-r1:1.5b` model performs advanced self-reasoning before responding:

```bash
ollama run deepseek-r1:1.5b "Solve this math problem: 2x + 5 = 13"
```

**Example Output:**
```
<think>
I need to solve the equation 2x + 5 = 13.

First, I'll isolate the variable x by subtracting 5 from both sides:
2x + 5 - 5 = 13 - 5
2x = 8

Then I'll divide both sides by 2:
2x / 2 = 8 / 2
x = 4

Let me verify: 2(4) + 5 = 8 + 5 = 13 ✓
</think>

To solve 2x + 5 = 13:

1. Subtract 5 from both sides: 2x = 8
2. Divide by 2: x = 4

The solution is x = 4.
```

This reasoning process makes DeepSeek-R1 particularly good for:
- Complex problem solving
- Step-by-step analysis
- Mathematical reasoning
- Logical deduction

### Model Performance Notes

- **Response Times**: Vision models typically take longer (10-50 seconds) compared to text-only models (1-5 seconds)
- **Memory Usage**: Larger models require more RAM - monitor system resources when running multiple models
- **Jetson Nano**: Stick to models under 4GB for optimal performance
- **DeepSeek-R1**: May take longer due to reasoning process, but provides more thorough responses

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