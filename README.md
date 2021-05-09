# rw_dataset_qe_deep_speaker_verification
Read write dataset for qe_deep_speaker_verification

Follow instructions to run the workflow for read-write datasets

1. Download the dataset in the dataset directory 
  mv domino/datasets/local/abishek_rw_dataset_workflow
  For e.g. in the directory "/domino/datasets/local/abishek_rw_dataset_workflow" (where abishek_rw_dataset_workflow is the project name) 
  download the VoxCeleb1 dataset with the comand 
  wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_test_wav.zip
 
2. Install some dependencies
   pip install -r requirements.txt
 
3. Download the pyAudioAnalysis tar file
   tar -xvzf pyAudioAnalysis.tar.gz
 

4. Navigate to the pyAudioAnalysis directory and install some more dependencies
   cd pyAudioAnalysis
   pip install -e .
  
5. Run the preprocessing script (preferably in a tmux session)
   python datadata_preprocess_mod_train.py
 
 



