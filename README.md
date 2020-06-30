<<<<<<< HEAD
README for streamline detection using an attention U-net model
==============================================
Data description: Training, validation, and testing dataset (LiDAR derived feature maps), scripts, pretrained models, and results for the hydrological streamline detection of Rowan Creek, NC, watershed area. The dataset is generated and provided by USGS (author: Larry Stanislawski (lstan@usgs.gov) and Ethan Shavers (eshavers@usgs.gov)).
Contents:  

- data.zip: 
  - mask.npy: The mask of research area. 1 indicates research area. 0 indicates outside area. 
=======
# streamline_detection
The hydrological streamline detection using deep learning

README for streamline detection using an attention U-net model
==============================================
Data description: Training, validation, and testing dataset (LiDAR derived feature maps), scripts, pretrained models, and results for the hydrological streamline detection of Rowan Creek, NC, watershed area. The dataset is generated and provided by USGS (author: Larry Stanislawski (lstan@usgs.gov) and Ethan Shavers (eshavers@usgs.gov)).
Contents:

- data.zip:
  - mask.npy: The mask of research area. 1 indicates research area. 0 indicates outside area.
>>>>>>> a8d1f539545f7dfcc8520fc5d05e8e201421ddf8
  - reference.npy: numpy array format of reference data. 1 indicates streamlines, 0 indicates non-streamlines
  - prediction_data.npy: The prediction data of research area
  - train*.npy: training data with data augmentation.
  - vali*.npy: validation data for model tuning.
  - train*label*.npy: the labels for training data.
  - vali*label*.npy: the labels for validation data.

- script:
  - train_U_net.py: training script of the U_net model for streamline detection.
  - train_attention_U_net.py: training script of the attention U_net model for streamline detection.
  - accuracy_evaluation.py: accuracy evaluation for the prediction results.
  - generate_resultant_tifs.py: convert result to geo-referenced .tiffs.

- result:
  - *.pickle: stores the intermediate training statistics and printouts.
  - preds_test_*.npy: predicted results orginzed as patches.
  - preds_test_*.tif: Organized.tiffs of predicted results.

- models:
  - *.h5: stores the trained model.

Environment: Python3.7
Library dependencies: cuda10.0, pdal 1.2, sklearn 0.18.1, keras 2.0, tensorflow 1.0, osgeo(gdal) 3.0.4.

Steps:
1. Download the data from https://doi.org/10.6084/m9.figshare.12584975.v1, unzip, and put it in the same directory as models, scripts, and result folders.
1. Run train_U_net.py or train_attention_U_net.py for model training by specifying the input directory and training scenario ('up','down','left', or 'right').
2. Run generate_resultant_tifs.py to transfer predicted results to geo-referenced tiffs.
3. Run accuracy_evaluation.py to generate the evaluation statistics of the four scenarios' prediction maps using the U-net (0) or the attention U-net model(1).

Note: if you don't want to train from scratch, you can simply load the pretrained models stored in the models folder to generate your predictions.



Publication:
----------------
