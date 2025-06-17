
---

#  English-to-Hindi Neural Machine Translation & Audio Conversion Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project, developed for the NLP course (AIM 829) at the International Institute of Information Technology, Bangalore, implements and compares two deep learning architectures for English-to-Hindi machine translation and integrates the superior model into a full-fledged audio-to-audio translation pipeline.

> ### Problem Statement
> A considerable number of non-native English speakers encounter difficulties comprehending spoken English educational content, particularly when delivered with unfamiliar accents. This project aims to bridge that gap by converting English-spoken video content into Hindi, making it more accessible.

## 🎥 Demo: The Final Pipeline in Action

The final system takes an English video, translates the speech to Hindi, and generates a new video with synchronized Hindi audio.

*   **[▶️ Watch the Original Input Video (English)](https://iiitbac-my.sharepoint.com/:v:/r/personal/abhinav_deshpande_iiitb_ac_in/Documents/Group_4_Model_files_NLP_Proj/my_name_is_gora.mp4?csf=1&web=1&e=gAqdhg)**
*   **[▶️ Watch the Final Output Video (Translated to Hindi)](https://iiitbac-my.sharepoint.com/:v:/r/personal/abhinav_deshpande_iiitb_ac_in/Documents/Group_4_Model_files_NLP_Proj/end_result_slow_video.mp4?csf=1&web=1&e=zrQ224)**

---

## 🚀 Features & Pipeline

This repository contains two main components:
1.  **Two NMT Models:** An LSTM-based Seq2Seq model and a more advanced Transformer model.
2.  **An End-to-End Pipeline:** A comprehensive system that translates spoken English from a video file into spoken Hindi.

The final pipeline operates in the following stages:

1.  **Audio Extraction & Speech Recognition**: Extracts the audio from the input video and uses Google's Speech Recognition API to convert English speech to text.
2.  **Punctuation Restoration**: Applies the Silero TTS Engine to restore proper punctuation to the transcribed text, which is crucial for accurate translation.
3.  **Neural Machine Translation (NMT)**: Uses our trained **Transformer model** to translate the English sentences into Hindi.
4.  **Text-to-Speech (TTS) Synthesis**: Employs a high-quality `facebook/mms-tts-hin` model to generate natural-sounding Hindi audio from the translated text.
5.  **Audio Synchronization & Video Generation**: Aligns the newly generated Hindi audio segments with the original video's timing (preserving silences) and creates the final output video.

---

## 🏆 Performance & Architectural Comparison

We evaluated both models using the **BLEU score**, a standard metric for translation quality. The Transformer architecture demonstrates a massive improvement over the baseline Seq2Seq model.
---

Here is a direct statistical comparison of the two models. The Transformer architecture provides a significant improvement in translation quality, even with fewer training epochs.

| Metric | Seq2Seq (LSTM) Model | **Transformer Model (Final)** |
| :--- | :--- | :--- |
| **BLEU-1 Score** | 0.1512 (15.1%) | **0.6174 (61.7%)**  *Excellent professional quality*|
| **BLEU-4 Score** | 0.0181 (1.8%) | **0.3402 (34.0%)**  |
| **Architecture** | RNN-based Encoder-Decoder | See Architecture Details Below |
| **Training Time** | ~4+ hours | ~7.5 hours (on P100 GPU) |
| **Epochs Trained** | 15 (with Early Stopping) | 10 |
| **Translation Quality** | Basic, often incorrect | **Coherent & Contextually Aware** |

---

### 🏗️ Transformer Architecture Details

Our final model is based on the architecture from "Attention Is All You Need" by Vaswani et al., implemented from scratch with the following key hyperparameters:

*   **Encoder/Decoder Stacks**: The model consists of **3 identical layers** in both the encoder and decoder stacks.
*   **Multi-Head Attention**: Each self-attention and cross-attention mechanism utilizes **8 parallel attention heads**, allowing the model to jointly attend to information from different representation subspaces.
*   **Embedding Dimension (`d_model`)**: All embedding layers and sub-layer outputs produce vectors of dimension **512**.
*   **Feed-Forward Networks**: The position-wise feed-forward network within each layer consists of two linear transformations with a dimensionality of **2048**.
*   **Regularization**: A **Dropout rate of 0.1** is applied to the output of each sub-layer before it is added to the sub-layer input (residual connection).
*   **Vocabulary Size**: The source (English) and target (Hindi) vocabularies are capped at approximately **10,000 tokens** each.

### Approach 1: Seq2Seq with LSTMs

This model served as our baseline. While it learned some word-level associations, it struggled with grammar and sentence structure.

*   **BLEU-1 Score:** 0.1512 (15.1%)
*   **BLEU-4 Score:** 0.0181 (1.81%)

**Sample Translation (Seq2Seq):**
| English | Reference Hindi | Model Output (Incorrect) |
|---|---|---|
| I can see | मैं देख सकता हूँ | मैं देख सकते |

### Approach 2: Transformer (Final Model)

The Transformer model, based on the "Attention Is All You Need" paper, significantly outperformed the baseline. It effectively captures contextual relationships, leading to vastly superior translations.

*   **BLEU-1 Score:** 0.6174 (61.7%) - *Excellent professional quality*
*   **BLEU-4 Score:** 0.3402 (34.02%) - *Good to high-quality translations*

**Sample Translations (Transformer):**
| English | Hindi Translation (Correct) |
|---|---|
| I love to eat delicious food. | मैं स्वादिष्ट खाना खाता हूँ। |
| What is your name? | आपका नाम क्या है ? |
| India is a beautiful country... | भारत समृद्ध सांस्कृतिक विरासत वाला एक सुंदर देश है । |

---

## 🛠️ Getting Started

To get a local copy up and running, follow these steps.

### Prerequisites

You will need Python 3.8+ and `pip` installed. The pipeline also requires `ffmpeg` for audio processing.

*   **Install ffmpeg:**
    ```sh
    # On Ubuntu/Debian
    sudo apt update && sudo apt install ffmpeg

    # On MacOS (using Homebrew)
    brew install ffmpeg
    ```

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/bajoriya-vaibhav/Neural-Machine-Translation-Transformer.git
    cd Neural-Machine-Translation-Transformer
    ```
2.  **Install Python dependencies:**
    ```sh
    pip install -r requirements.txt
    ```
3.  **Download SpaCy model:**
    ```sh
    python -m spacy download en_core_web_sm
    ```

### Model Weights

The pre-trained model weights are required to run the translation pipeline.
*   **[🔗 Download Transformer Model Weights](https://drive.google.com/file/d/1zrpxw-1YFI2aOWqAaQuR6GXlcBiESzMR/view?usp=sharing)**
*   **[🔗 Download Seq2Seq Model Weights](https://iiitbac-my.sharepoint.com/:f:/g/personal/abhinav_deshpande_iiitb_ac_in/EpyvfpRM-DpNmPBXdvy_D90BDxhpWW2LkzVjiCJoDfdLQQ?e=GsZBlb)**

Place the downloaded model files in a `models/` directory in the project root.

---

## ⚙️ Usage

The primary entry point for the audio-to-audio translation is the main pipeline script.

```sh
python main_pipeline.py --input_video_path /path/to/your/video.mp4 --output_video_path /path/to/output/video.mp4
```
Check the respective notebooks for details on how to train the models or run inference on text-only data.

---

## 🗂️ Datasets

The models were trained on publicly available parallel corpora:
*   **[IIT Bombay English-Hindi Corpus](https://huggingface.co/datasets/iitb)**: A large-scale corpus consisting of aligned sentence pairs.
*   **[TED Talks Hindi-English Truncated Corpus](https://www.clarin.eu/resource-families/parallel-corpora)**: A high-quality dataset derived from translated TED talk transcripts.

---

## 🧑‍💻 The Team

*    ([**Abhinav Kumar**](https://github.com/Abhinav-Kumar012)) - Seq2Seq model training, report preparation.
*   ([**Abhinav Deshpande**](https://github.com/Abhinav-gh)) - Seq2Seq & Transformer model definition, audio pipeline architecture.
*    ([**Vaibhav Bajoriya**](https://github.com/bajoriya-vaibhav)) - Transformer model training & definition, TTS model integration.
*    ([**Shashank Devarmani**](https://github.com/standing-on-giants)) - Test video preparation, speech-to-text integration, pipeline debugging.

### Acknowledgments
*   This project was completed as part of our coursework at the **International Institute of Information Technology, Bangalore**.
*   We thank the creators of the open-source libraries and datasets that made this work possible.

