# Baseline model for Task 2 of DCASE 2018 aka Kaggle Challenge: FreeSound General-Purpose Audio Tagging

WARNING: the code and documentation are a work in progress. In case of questions, contact
Manoj Plakal, plakal@google.com

Rough code layout

* `make_class_map.py`: generates a map from class index to class name from the
  training dataset.
* `input.py`, `model.py`: libraries for input and model functions.
* `train.py`, `eval.py`: training and evaluation using labeled data.
* `inference.py`: run inference on a trained model and produce predictions for
   all audio files in the test set in a form submittable to Kaggle.
