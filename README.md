## Dependencies

### 1. Python packages

```bash
pip install -r requirements.txt
```

---

### 2. FFmpeg (strongly recommended)

This project uses FFmpeg for audio decoding and final video encoding.

## With FFmpeg (recommended)

If FFmpeg is installed, the program will generate a final `.mp4` video file directly.

Install FFmpeg:

* Windows: download from https://ffmpeg.org and add to PATH
* macOS:

```bash
brew install ffmpeg
```

* Linux:

```bash
sudo apt install ffmpeg
```

Verify installation:

```bash
ffmpeg -version
```


## Without FFmpeg (fallback mode)

If FFmpeg is not available, the program will still run, but outputs will be separated:

* `.avi` (video without proper encoding)
* `.mp3` (audio track)

These files can be manually merged later using FFmpeg.


---

### 3. Hugging Face (optional)

This project uses a CLAP-based model from Hugging Face for audio understanding.

No login is required to run the project. However, unauthenticated requests may be slower and subject to rate limits.

## Optional: login for better performance

```bash
pip install huggingface_hub
hf auth login
```

Or set your token manually:

```bash
export HF_TOKEN=your_token_here   # Linux / macOS
setx HF_TOKEN your_token_here     # Windows
```

With authentication, model downloads are faster and more reliable.

## Notes

* The model will be automatically downloaded on first run
* No manual setup is required unless you encounter rate limits


