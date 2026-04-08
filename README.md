A frequency spectrum analysis program which process the digital signal using the Fast Fourier Transform (FFT) or bank of the bandpass filters (FIR).

## Installation: 
  Requirements:
  
  - Python 3.8>=
  - RAM 4GB>=
  - Internet Browser

## Dependencies:
    To install the needed dependecies use the command: 
    ```bash
    pip install -r requirements.txt
    ```
    If any error will occure try on the Linux system: 
    ```bash
    sudo apt-get install libsndfile1 ffmpeg
    ``` 
    or for a macOS: 
    ```bash
    brew install libsndfile ffmpeg
    ```

In order to run a program type: ```streamlit run app.py.```. Upload your file (WAV, FLAC, OGG, MP3), and then choose the method FFT/FIR. Adjust the speed
of your analysis by using a sidebar and click th "Start" button.

You can stop the analysis at any time and save your histogram.

## Project status

The app works but *needs fixes* in the performance of UI.
