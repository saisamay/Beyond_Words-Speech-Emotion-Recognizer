# Speech Emotion Recognition (SER) --- Emotion2Vec+ Base

A local, terminal-based Speech Emotion Recognition pipeline that
predicts one of **seven emotion classes** from an audio or video file.

The pipeline uses a pretrained **Emotion2Vec+ Base** encoder to extract
a 768-dimensional utterance-level audio embedding, followed by the
project's trained MLP classifier. It reuses the normalization parameters
saved in the training checkpoint.

## 1. Emotion classes

    Index Emotion
  ------- ----------
        0 Angry
        1 Disgust
        2 Fear
        3 Happy
        4 Neutral
        5 Sad
        6 Surprise

## 2. Pipeline overview

``` text
Audio / Video file
       |
       v
FFmpeg: extract audio and convert
       |
       v
Mono, 16 kHz WAV
       |
       v
Emotion2Vec+ Base encoder
       |
       v
768-dimensional utterance embedding
       |
       v
Saved training-time normalization
       |
       v
Trained MLP classifier
       |
       v
Predicted emotion + seven class scores
```

A video is used only as an audio source. The video frames are not
analysed.

## 3. Project structure

Keep these files in the same project directory:

``` text
Emotion2Vec_Final_Model_Backup/
├── .venv/                       # Local virtual environment (do not commit)
├── emotion2vec_final_model.pth  # Trained classifier and saved scaler values
├── predict_emotion.py           # Complete terminal inference application
├── verify_model.py              # Step-by-step verification script
├── extract_embedding.py         # Standalone embedding extraction test
├── convert_audio.py              # Standalone audio conversion utility
├── experiment_config.json       # Experiment configuration
├── training_history.csv         # Training history
├── confusion_matrix.npy         # Saved test confusion matrix
├── test_predictions.npy         # Saved test predictions
└── test_targets.npy             # Saved test targets
```

For normal inference, the essential project files are
`predict_emotion.py` and `emotion2vec_final_model.pth`. FFmpeg and the
Python dependencies must also be installed.

**Important:** The current `predict_emotion.py` uses the relative
checkpoint path `emotion2vec_final_model.pth`. Run it from the project
directory, or change `CHECKPOINT_PATH` in the script.

## 4. Requirements

-   Ubuntu or another Linux distribution
-   Python 3 and `venv`
-   FFmpeg
-   PyTorch, NumPy and FunASR
-   Internet access on the first run to download Emotion2Vec+ Base from
    Hugging Face

The current inference script loads the encoder on **CPU**. A GPU is not
required for this script.

## 5. Installation on Ubuntu

### Step 1 --- Install system packages

``` bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip ffmpeg
```

Verify FFmpeg:

``` bash
ffmpeg -version
```

### Step 2 --- Open the project folder

If the folder is in Downloads:

``` bash
cd ~/Downloads/Emotion2Vec_Final_Model_Backup
```

If you cloned the project from GitHub, use the directory created by
`git clone` instead.

### Step 3 --- Create and activate a virtual environment

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

Your terminal prompt should now show `(.venv)`.

### Step 4 --- Install PyTorch

Use the official PyTorch installation selector to choose the correct
command for your operating system and hardware:

https://pytorch.org/get-started/locally/

### Step 5 --- Install the remaining Python dependencies

``` bash
python3 -m pip install --upgrade pip
python3 -m pip install numpy funasr modelscope
```

Verify the imports:

``` bash
python3 -c "import torch, numpy, funasr; print('PyTorch:', torch.__version__); print('NumPy:', numpy.__version__); print('FunASR:', funasr.__version__)"
```

## 6. Run the complete application

The recommended entry point is `predict_emotion.py`. It performs
conversion, embedding extraction, normalization and classification in
one run.

From the project folder, with `.venv` activated:

``` bash
python3 predict_emotion.py
```

When prompted:

``` text
Enter audio/video file path:
```

Enter the full path to your recording. Example:

``` text
/home/<your-username>/Downloads/check1.mp4
```

A WAV file can also be supplied:

``` text
/home/<your-username>/Downloads/sample.wav
```

The script will:

1.  Check that the input file and model checkpoint exist.
2.  Use FFmpeg to extract the audio and convert it to mono, 16 kHz PCM
    WAV.
3.  Load Emotion2Vec+ Base through FunASR.
4.  Extract a 768-dimensional utterance embedding.
5.  Load the trained MLP checkpoint.
6.  Apply the scaler mean and scale saved during training.
7.  Predict one emotion and print scores for all seven classes.
8.  Remove the temporary WAV file after processing.

### Example output format

``` text
Speech Emotion Recognition
-----------------------------------
Enter audio/video file path: /path/to/recording.mp4

[1/5] Converting audio to 16 kHz mono WAV...
[2/5] Loading Emotion2Vec+ Base encoder...
[3/5] Extracting 768-D audio embedding...
[4/5] Loading trained classifier and scaler...
[5/5] Predicting emotion...

=============================================
       SPEECH EMOTION RECOGNITION
=============================================

Predicted Emotion: HAPPY

Emotion Probability Distribution:
angry        2.10%
disgust      0.50%
fear         1.20%
happy       82.00%
neutral      9.00%
sad          3.20%
surprise     2.00%
=============================================
```

The values above are illustrative only. Actual results depend on the
input recording.

## 7. Run individual verification scripts

These scripts are useful for debugging. They are not required for normal
use once `predict_emotion.py` works.

### A. Convert a file to WAV

``` bash
python3 convert_audio.py "/full/path/to/input.mp4"
```

The current utility writes `converted_audio.wav` in the project
directory and converts the audio to mono, 16 kHz PCM WAV.

### B. Extract an Emotion2Vec embedding

After creating `converted_audio.wav`:

``` bash
python3 extract_embedding.py
```

The extraction test should report an embedding shape of `(1, 768)` and
no NaN or infinite values.

### C. Verify the trained classifier and run inference

``` bash
python3 verify_model.py
```

This is the staged verification script. It expects `converted_audio.wav`
in the project directory and should contain the checkpoint loading,
embedding extraction, saved-scaler normalization and prediction sections
developed during setup.

## 8. Checkpoint and normalization

The `emotion2vec_final_model.pth` checkpoint contains the trained
classifier state and training-time normalization values, including:

-   `model_state_dict`
-   `scaler_mean`
-   `scaler_scale`
-   `emotion_to_idx`
-   `idx_to_emotion`
-   Model architecture metadata and saved experiment metrics

The inference script reads the scaler arrays directly from the
checkpoint. It does **not** fit a new scaler on the input recording.

Do not fit a new scaler on a single test recording. That would change
the feature distribution supplied to the classifier and would not
reproduce the training preprocessing.

The separate `emotion2vec_scaler.pkl` file is not required by the
current inference script.

## 9. Troubleshooting

### `python3: command not found`

Install Python 3:

``` bash
sudo apt install -y python3
```

### `ffmpeg: command not found`

Install FFmpeg:

``` bash
sudo apt install -y ffmpeg
```

### `ModuleNotFoundError`

Activate the virtual environment and install the missing package:

``` bash
source .venv/bin/activate
python3 -m pip install numpy funasr modelscope
```

For PyTorch, use the official installation selector:
https://pytorch.org/get-started/locally/

### `Model checkpoint not found`

Make sure `emotion2vec_final_model.pth` is in the project directory and
run:

``` bash
cd ~/Downloads/Emotion2Vec_Final_Model_Backup
python3 predict_emotion.py
```

### First run takes longer

On the first run, FunASR downloads Emotion2Vec+ Base from Hugging Face.
Later runs can use the cached model, provided the cache remains
available.

### FFmpeg conversion failed

Check that the input path is correct and that the file contains an audio
stream. Put quotation marks around paths containing spaces:

``` text
"/home/<your-username>/Downloads/my recording.mp4"
```

## 10. Scope and limitations

-   The current application produces **one utterance-level prediction**
    for the supplied recording.
-   It does not currently perform speaker diarization, timestamped
    emotion tracking, speech separation or noise cancellation.
-   Video frames are not analysed; only the audio stream is used.
-   Softmax outputs are classifier scores and should not automatically
    be interpreted as calibrated probabilities.
-   Performance on arbitrary recordings may differ from the saved test
    results due to speakers, microphones, languages, recording
    conditions and background noise.
-   Saved test metrics describe the project's recorded evaluation setup;
    they are not a guarantee of universal real-world accuracy.

## 11. Model details

-   **Audio encoder:** Emotion2Vec+ Base (`iic/emotion2vec_plus_base`),
    loaded through FunASR using the Hugging Face hub.
-   **Embedding:** 768-dimensional utterance-level representation.
-   **Classifier:** MLP with 768 input features, hidden layers of 256
    and 128 units, Batch Normalization, ReLU and Dropout, followed by
    seven output classes.
-   **Inference device:** CPU in the current script.

## 12. License and attribution

Before redistributing this repository or using it commercially, check
the licenses and usage conditions for Emotion2Vec, its dependencies and
each dataset used in the research. Add the project's own license and
dataset citations here before publishing.
