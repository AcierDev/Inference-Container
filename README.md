# Inference Container

This container provides a microservice to classify images using a Roboflow classification model.

## Overview

The Inference Container is a Flask-based microservice that:

1. Accepts image uploads through a REST API endpoint
2. Processes images using a Roboflow classification model
3. Returns a binary classification result: "Good" or "Bad"
4. Organizes processed images into appropriate directories

## Getting Started

### Prerequisites

- Python 3.8+
- pip
- Roboflow account and API key (for model access)

### Installation

1. Clone this repository
2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

The container is configured through environment variables:

1. Create a `.env` file in the root directory
2. Set the following variables:
   ```
   CLASSIFICATION_MODEL_ID=your-model-id/version
   ```

You can also set these variables directly in your environment.

### Running the Container

Start the Flask server:

```bash
python main.py
```

The server will be available at `http://localhost:5000`.

## API Endpoints

### Health Check

```
GET /health
```

Returns the service health status and configuration.

### Detect Imperfection

```
POST /detect-imperfection
```

Upload an image for classification. The image should be sent as form data with the key `image`.

**Example response:**

```json
{
  "success": true,
  "data": {
    "predictions": [
      {
        "class_name": "Bad",
        "confidence": 0.95,
        "detection_id": 0,
        "is_classification": true
      }
    ]
  },
  "timestamp": "2023-11-07T12:34:56.789",
  "message": "Processing complete. File organization running in background.",
  "background_task_id": "task_20231107_123456_1234"
}
```

### Background Tasks

```
GET /background-tasks
```

Get the status of background organization tasks.

## Directory Structure

The container organizes processed images into the following directory structure:

```
storage/
├── uploads/            # Temporary storage for uploaded images
├── processed/
│   ├── annotated/      # Images with annotation overlays
│   └── classification/ # Organized by classification result
│       ├── good/       # Images classified as "Good"
│       ├── bad/        # Images classified as "Bad"
│       └── unknown/    # Images with uncertain classification
└── archive/            # Older images moved here by cleanup process
```

## Integrating with Router-Slaved

This container is designed to work with the Router-Slaved system. The classification results are used directly to make ejection decisions:

1. "Good" classification → No ejection
2. "Bad" classification → Eject the part

The confidence threshold for making decisions can be configured in the Router-Slaved settings.
