# rw_dataset_qe_deep_speaker_verification
## Read write dataset workflow for qe_deep_speaker_verification

Follow instructions to run the workflow for read-write datasets

#### 1. Download the dataset in the dataset directory 
```sh
mv domino/datasets/local/abishek_rw_dataset_workflow
```

  For e.g. in the directory ```/domino/datasets/local/abishek_rw_dataset_workflow``` (where ```abishek_rw_dataset_workflow``` is the project name) download the VoxCeleb1 dataset with the command
  
```sh 
    wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_test_wav.zip
```
Now unzip the file with the command
```sh
    unzip vox1_test_wav.zip
```

#### 2. Install some dependencies
```sh
    pip install -r requirements.txt
```
 
#### 3. Download the pyAudioAnalysis tar file
```sh
   tar -xvzf pyAudioAnalysis.tar.gz
``` 

#### 4. Navigate to the pyAudioAnalysis directory and install some more dependencies
```sh
   cd pyAudioAnalysis
   pip install -e .
```

#### 5. Run the preprocessing script (preferably in a tmux session)
```sh
   python data_preprocess_mod_train.py
 ```
