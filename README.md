# PPG-to-BP-Prediction-convnets

This repository contains the code for the AIME submission #65

Here we have two approaches to predict BP from Photoplethysmograms (PPG signal)

A sample data that this code works on can be found from: https://www.kaggle.com/mkachuee/noninvasivebp.

1. **Feature extraction based approach**: the feature extraction code can be found in code/data_preprocessing.py
2. **Image representation based approach**: the image representation code can be found in code/signal_to_spectogram.py and the file that runs that ML pipeline to make predictions from the image representations can be found in code/predictions_from_representations.py. 

Additionally, two other files spec.py and scalograms.py allows you to play around with the spectogram and scalogram image representations of the PPG signal. 

For spectogram, we use the in build matplotlib function specgram function. 
For scalogram, we use the continous wavelet transform (CWT) function from PYWT (https://pywavelets.readthedocs.io/en/latest/)

For getting the image embedding we use the following library: https://github.com/christiansafka/img2vec, which currently supports resnet18 and alexnet. 
