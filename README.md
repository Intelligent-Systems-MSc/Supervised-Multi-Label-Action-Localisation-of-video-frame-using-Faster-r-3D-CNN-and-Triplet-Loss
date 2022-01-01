# Projet-Advanced-Machine-Learning-Sorbonne-University
# Supervised multi-label action localisation using Faster R-3D-CNN ResNet of 50 neural layers and triplet loss

This code is able to provide you a supervised multi-label action localisation using Faster R-3D-CNN ResNet of 50 neural layers and triplet loss function.
In order be able to execute properly the code, we suggest you to follow carefuly steps below.

## Installation and Preparing the environment

<b>1- </b> Clone our git repository
<pre>
 git clone https://github.com/phnhub/Projet-Advanced-Machine-Learning-Sorbonne-University-.git
</pre>

<b>2- </b> Create and Activate a new virtual environment in the file that you have cloned the git (using your terminal)
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
Part 1 --> Preparing the database
Part 2 --> Application of the ML model

### Part 1: Preparing the database

Download the database from ... which contains two serie of frames so they will all be labelized for two label "Frotter" and "Marcher".
It has therefore two folder "Frotter" and "Marcher" that each of them contains 31 frame of the corresponding label followed by the rest of the video frames.

For the "Fortter" folder, the first 31 frames are all of the "Fortter" label and for the "Marcher" folder the last 31 frames are all of the "Marcher" label. So, We will use the frames of these two series to train the model. In this case, for each label, choose some frame (we choosed 4 per label) per label and and paste them in their  

Go to code << Part 1 >> and run the code step-by-step until 




