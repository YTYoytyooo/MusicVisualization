# Music Visualization

A system that transforms audio into dynamic visual output based on emotion recognition and audio features.

---

## Installation

### 1.Rrepository

```bash
git clone https://github.com/YTYoytyooo/MusicVisualization.git
cd MusicVisualization
```

---

### 2. Requirements

```bash
pip install -r requirements.txt
```

---

## Dependencies

### FFmpeg (strongly recommended)

FFmpeg is used for audio decoding and final video encoding.

#### With FFmpeg (recommended)

The program outputs a final `.mp4` video.

Install:

* Windows: https://ffmpeg.org (add to PATH)
* macOS:

```bash
brew install ffmpeg
```

* Linux:

```bash
sudo apt install ffmpeg
```

---

#### Without FFmpeg (fallback mode)

If FFmpeg is not installed, the program will still run but outputs:

* `.avi` (video)
* `.mp3` (audio)

---

### Hugging Face (optional)

This project uses a CLAP-based model from Hugging Face.

No login is required, but authentication improves speed and avoids rate limits.

Optional setup:

```bash
pip install huggingface_hub
hf auth login
```

Notes:

* Model downloads automatically on first run
* Authentication is optional but recommended

---

## Usage

### Basic usage

```bash
python main.py input1.mp3 input2.wav input3.flac
```

### Specify model file (optional)

```bash
python main.py input1.wav input2.mp3 --model emotion_model.pth
```

---

### Notes

* Multiple input files are supported
* Each input file generates a corresponding output video
* Output files are saved in the current directory
* Supported formats: `.mp3`, `.wav`, `.flac`

---

## Project Structure

```text
main.py                  # Entry point
feature_extraction.py    # Audio analysis
emotion_model.py         # Emotion prediction
mcts.py                  # Decision algorithm
renderer.py              # Visualization rendering
requirements.txt         # Dependencies
README.md                # Documentation
```

---

## How It Works

1. Audio is loaded and processed into features
2. CLAP-based model estimates emotional characteristics
3. MCTS determines visual transitions and states
4. Renderer generates frames based on emotion and structure
5. Frames are encoded into a video file

---

## License

MIT License
