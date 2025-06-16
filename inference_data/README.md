## Creating the dataset

In the folder **Example_JOBS_Snellius** example jobs are found that were used for the training and evaluation.

**environment_local.yml** is the conda environment generated from a local machine. In the case of any differences between snellius and a local machine the folder **train_and_evaluate** has a additional conda environment file as a backup.


**split_data.py** can be used to split a folder with data into a train, val and test set.

**create_mantra_dataset.py** can be used to process the OpenMantra into the right format. When wanting Chinese sentences use zh and when English ones en. This has to be changed manually in a couple of places inside the file.

The following files need to be present but don't have to be changed/used: **utils.py**, **models.py**, **vison_transformer.py**


## Execute the following in order to convert the data into the right format:

The **export_metainfo.py** can be run to generate the meta information needed for the other functions/files that will follow below. This needs to be run for the train, val and test folders of the data. 

The **create_hdf5.py** can be run to generate hdf5 files for the folders with the data. This needs to be run for the train, val and test folders of the data. There is a cell that needs to be uncommented when working with all text combined.

The **inference.py** can be run to inference the embeddings from the hdf5 files. In order to switch between Clip and Bert/Dino embeddings the changes need to be made in lines: 20-26 and 148-156.

Finally the function generate_candidate_sampling_key in the file **data.py** can be used to generate sampling files. In the file there is an example on how to run this function.

Now one can proceed to training and evaluating of models. For the OpenMantra dataset this has to be obviously done for this dataset instead of the COMICS dataset.

In the **data.py** file when clicking (ctrl f) you can find str_data2 thee times. You can either write str_data1 (dialogue) or str_data2 (narration) here. I do this for each and then concatenate the files and then split them up in the engine.py file. This is not the most convenient solution, however this prevented me from reworking the entire ASTERX code.


There might exist a slight difference between the data.py files in the directories train_and_evaluate and inference_data. Best to stick to files already present in the directories.

The comics with the following number have been left out of the COMICS dataset due to empty folders/or incompatible characters:  2453, 3608, 3864 and 3958
