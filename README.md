# Supervised Multi-Label Action Localisation of video frame using Faster-r-3D-CNN and Triplet Loss

This code is able to provide you a supervised multi-label action localisation using Faster R-3D-CNN ResNet of 50 neural layers and triplet loss function.
In order be able to execute properly the code, we suggest you to follow carefuly steps below.

<img src="https://i.imgur.com/zriD6GZ.png">


## Installation and Preparing the environment

<b>1- </b> Clone our git repository
<pre>
 git clone https://github.com/phnhub/Projet-Advanced-Machine-Learning-Sorbonne-University-.git
</pre>

<b>2- </b> Create and Activate a new virtual environment in the file that you have cloned the git (using your terminal)
* In the script below we chose the virtual environment name as "mla" 
<pre>
python -m venv mla
.\mla\Scripts\activate
</pre> 

<b>3- </b> Install dependencies (using your terminal)
<pre>
python -m pip install --upgrade pip
pip install ipykernel
python -m ipykernel install --user --name=mla
</pre> 

<b>4- </b> Run jupyter (using your terminal) and be sure selecting the virtual environment << mla >>

## Usage

The code is divided in two parts :
Part 1 --> Preparing the database <br />
Part 2 --> Application of the ML model <br />

* The code prepares you some file directory and some folder but there is also other folder that have to be created manually. Because of the large size of the project (4.12 GB (4,427,279,723 bytes)), we could just insert some of very sample of the project on the git. However, if you follow attentively the ReadMe, you are able to re-implement our model and our results.

### Part 1: Preparing the database

Download the database (Database.rar) from git which contains two serie of frames so they will all be labelized for two label "Frotter" and "Marcher".
It has therefore two folder "Frotter" and "Marcher" that each of them contains 31 frame of the corresponding label followed by the rest of the video frames.

The "serie 1" folder contains 31 frames of "Frotter" label plus the rest of the video frames.
The "serie 2" folder contains 31 frames of "Marcher" label plus the rest of the video frames. 
So, We will use the frames of these two series to train the model. In this case, for each label, choose some frame (we choosed 3 per label) per label and and paste them in train folder. You can also choose some frame (we choosed 1 per label) per label and put them in test folder. 

*At the end we apply the trained method on these two series folder (serie 1, serie 2).

Go to code << Part 1 >> and run the code step-by-step until "Labelimg" block. This block is the last step of the code << Part 1 >> wich execute a python application via git that could be used for adding and labelizing the bounding box of the images from train folder. 
In this case, use the application to place your bounding box by precising its label (like image below). Then extract and put its annotation file in the train folder. 

<img src="https://i.imgur.com/UU8xP6Z.png">

### Part 2: Training and labelizing

Go to code << Part 2 >> and run the code step-by-step.
This code do some installation for Tensorflow Object Detection, prepare the model, train the model and apply it on the series of the frames to show the result of detection ,image labeling and differents evalutions using some loss function (Faster R-CNN loss and triplet loss).

*For analysing the trained model, its precision or its loss values, you can use Tensorboard tool:
 
 Usage :
 
 After training the Faster R-CNN model, you can use its generated file by the code << Part 2 >> and run the Tensorboard tool by executing the following code from the folder (ex: models -> faster_r_3D_cnn -> train) with contain the saved model.
  <pre>tensorboard --logdir=. </pre> 

