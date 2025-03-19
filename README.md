# LLM-Audio-Summarization


## Summary
- This Adept English podcast episode focuses on health and nutrition in the new year (2025).
- The episode introduces nutrition as the science of what we eat.
- The podcast aims to initiate a discussion and provide information related to health and nutrition.

## Transcript
Hi there and welcome to this podcast from Adept English. Today let's get fully into the New Year feel. It'll be 2025 by the time you're listening to this. So as ever, income's the New Year. Happy New Year. Hope yours was a good one. And suddenly we're all encouraged to be on a drive for our health. So today I thought I would start the ball rolling, begin that process by doing a health- and nutrition-related topic. That word, nutrition, N-U-T-R-I-T-I-O-N, is the science of what you eat.

---

## Code
This repository contains the necessary scripts for processing and summarizing audio transcriptions using Whisper, LangChain, and Gemini AI.

### Technologies Used
- **Python** for script execution
- **Whisper** for speech-to-text transcription
- **Librosa** for audio processing
- **LangChain** for AI-driven text summarization
- **Gemini AI** for generating concise summaries
- **Requests** for API interactions
- **S3 Storage** for file handling

### Key Features
- Fetch audio files from URLs.
- Process audio files and transcribe them.
- Summarize transcripts using AI models.
- Generate Markdown-based structured summaries.

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/repository-name.git
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Set up environment variables in a `.env` file:
   ```sh
   GOOGLE_API_KEY=your_google_api_key
   LANGCHAIN_API_KEY=your_langchain_api_key
   ```

### Usage
Run the script to process an audio file and generate a summary:
```sh
python script.py --audio_url "https://example.com/audio.mp3"
```

### Contributing
Feel free to submit pull requests or report issues.

### License
This project is licensed under the MIT License.
