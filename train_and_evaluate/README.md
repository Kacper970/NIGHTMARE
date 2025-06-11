In the folder **Example_JOBS_Snellius** example jobs are found that were used for the training and evaluation.

**environment_snellius.yml** is the conda environment generated from snellius. In the case of any differences between snellius and a local machine the folder **inference_data** has a additional conda environment file as a backup.

The files **data.py**, **models.py**, **utils.py** and **vison_transformer** are the same for all models. The other folders here have model specific files that need to be used when using those models. Here below we explain any additional steps that need to be taken into account when using those files. Each of the model folders has three files: the **engine.py** used for the main loops, **training.py** file used for training and a **evaluate_retrieval.py** used for evaluation. (When there is no training involved the training.py file is not included)


The headers below correspond to the names of the models as can be found in the thesis' tables. Checkpoints also have such names.


### ASTERX (Image Only) 

The files can be used without additional steps according to the job files in the folder Example_JOBS_Snellius.


### ASTERX (Text Only)

The files can be used without additional steps according to the job files in the folder Example_JOBS_Snellius.


### ASTERX (Multimodal, Element-wise +) 

The files can be used without additional steps according to the job files in the folder Example_JOBS_Snellius.


### ASTERX (Multimodal, Concatenation)

The files can be used without additional steps according to the job files in the folder Example_JOBS_Snellius.


### ASTERX (Multimodal, Combiner) & ASTERX (Multimodal, FNN-Based)  

For using ASTERX (Multimodal, Combiner):/
use the line: combiner = SequenceCombiner(384, 768, 384) and comment out: combiner = Combinerpaper(384, 768, 384)
  
For using ASTERX (Multimodal, FNN-Based):/
use the line: combiner = Combinerpaper(384, 768, 384) and comment out: combiner = SequenceCombiner(384, 768, 384)

The changes need to be done in the files training.py and evaluation_retrieval.py, keep in mind to select the right checkpoint.


### NIGHTMARE

The files can be used without additional steps according to the job files in the folder Example_JOBS_Snellius.


### ComicBERT (Multimodal)

The files can be used without additional steps according to the job files in the folder Example_JOBS_Snellius.


### Pretrained Clip (Multimodal) & Pretrained BertDino (Multimodal)  

When using Dino and Bert embeddings use the data folder: data_better_text/
When using Clip embeddings use the data folder: data_clip_better

Apply this change in the file evaluation_retrieval.py


### Pretrained Clip (Image Only) & Pretrained BertDino (Image Only) 

When using Dino and Bert embeddings use the data folder: data_better_text/
When using Clip embeddings use the data folder: data_clip_better

Apply this change in the file evaluation_retrieval.py


### NIGHTMARE, (Single Digit Condition) & NIGHTMARE, (Element Wise Condition)

For using : NIGHTMARE, (Single Digit Condition)/
use **feature_dim = 1**

For using : NIGHTMARE, (Element Wise Condition)/
use **feature_dim = 384**

The changes need to be done in the files training.py and evaluation_retrieval.py, keep in mind to select the right checkpoint.


### NIGHTMARE, (Masking)

The files can be used without additional steps according to the job files in the folder Example_JOBS_Snellius.


### NIGHTMARE, (FILM) 

The files can be used without additional steps according to the job files in the folder Example_JOBS_Snellius.


### NIGHTMARE, (3 Channels) & NIGHTMARE, (3 Channels, Double Cross Wrap) & NIGHTMARE, (3 Channels, Double Straight Wrap) 

For using: NIGHTMARE, (3 Channels)/
use the line: channel_proc = ModalityProcessor2()

For using: NIGHTMARE, (3 Channels, Double Cross Wrap)/
use the line: channel_proc = ModalityProcessor3()

For using: NIGHTMARE, (3 Channels, Double Straight Wrap)/
use the line: channel_proc = ModalityProcessor7()

The changes need to be done in the files training.py and evaluation_retrieval.py, keep in mind to select the right checkpoint.


### NIGHTMARE, (3 Channels, Text Channels +) 

The files can be used without additional steps according to the job files in the folder Example_JOBS_Snellius.


### NIGHTMARE, (2 Channels, Plus of Text Features) & NIGHTMARE, (3 Channels, Text Channels Gate) & NIGHTMARE, (3 Channels, Text Channels Bilinear)

For using: NIGHTMARE, (2 Channels, Plus of Text Features)/
use the line: channel_proc = ModalityProcessor4()

For using: NIGHTMARE, (3 Channels, Text Channels Gate)/
use the line: channel_proc = ModalityProcessor5()

For using: NIGHTMARE, (3 Channels, Text Channels Bilinear)/
use the line: channel_proc = ModalityProcessor6()

The changes need to be done in the files training.py and evaluation_retrieval.py, keep in mind to select the right checkpoint.


### ASTERX (Multimodal → Image Only, Element Wise +) 

The files can be used without additional steps according to the job files in the folder Example_JOBS_Snellius.


### ASTERX (Image Only → Multimodal, Element Wise +)

The files can be used without additional steps according to the job files in the folder Example_JOBS_Snellius.


### NIGHTMARE (Multimodal → Image Only) 

The files can be used without additional steps according to the job files in the folder Example_JOBS_Snellius.


### All the experiments for the OpenMantra dataset can be done by using the above methods with the OpenMantra data.
