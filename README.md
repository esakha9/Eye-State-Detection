# Eye State Detection (Open / Closed)

This project detects whether a person's eyes are **open or closed** using a deep learning model.

The model is trained using images and then tested on video to predict eye state frame-by-frame.

---

## How it works

1. Train a model on eye images (open / closed)
2. Load the trained model
3. Run it on video frames
4. Predict eye state for each frame

---

## Files in this project

- train.py → used to train the model  
- inference.py → used to test on video  
- eye_classifier.pth → trained model  
- input.mp4 → test video  
- output_video.mp4 → result video  
