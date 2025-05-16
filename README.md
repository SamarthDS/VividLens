# VividLens

VividLens is a comprehensive video analysis and enhancement platform that combines advanced video captioning and frame augmentation capabilities. The project consists of two main components:

## Components

### 1. Video Analysis System
Located in the `Captioning` directory, this component provides:
- Real-time video analysis and captioning
- Suspicious activity detection
- Frame-by-frame analysis
- Summary generation of detected events
- Export capabilities for analysis results

Key features:
- Uses ActionCLIP model for video understanding
- Supports multiple video formats (mp4, avi, mov)
- Generates detailed captions for each frame
- Identifies potential security threats
- Provides timestamp-based event tracking

### 2. Video Frame Augmentation (Remushter)
Located in the `Remushter` directory, this component offers:
- Video frame enhancement and augmentation
- Multiple processing options:
  - Grayscale conversion
  - Brightness adjustment
  - Color enhancement
  - AI-powered colorization
- Real-time preview of processed frames

## Setup Instructions

### Prerequisites
- Python 3.x
- CUDA-compatible GPU (recommended for better performance)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/SamarthDS/VividLens.git
cd VividLens
```

2. Install dependencies for Captioning:
```bash
cd Captioning
pip install -r requirements.txt
```

3. Install dependencies for Remushter:
```bash
cd ../Remushter
pip install -r requirements.txt
```

## Usage

### Video Captioning
1. Navigate to the Captioning directory
2. Run the Streamlit app:
```bash
streamlit run main.py
```
3. Upload a video through the web interface
4. View the generated captions and analysis

### Frame Augmentation
1. Navigate to the Remushter directory
2. Run the Streamlit app:
```bash
streamlit run app.py
```
3. Upload a video clip
4. View and compare different augmented versions of the frames

## Project Structure
```
VividLens/
├── Captioning/
│   ├── src/
│   ├── outputs/
│   ├── input_videos/
│   ├── frames/
│   ├── config/
│   └── main.py
├── Remushter/
│   ├── data/
│   ├── app.py
│   ├── train.py
│   ├── model.py
│   └── utils.py
└── README.md
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
