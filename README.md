# AI Receipt Scanner with OCR


## Overview

<div align="justify">
    
> This repository presents a receipt processing system that combines **Document classification and OCR detection** to streamline the extraction of relevant details from receipts. The initial phase involves training a model to check whether a document is a receipt or not. This model is trained on a diverse dataset sourced from the Roboflow workspace, ensuring performance across various document types. Secondly, the system seamlessly integrates with Optical Character Recognition (OCR) for precise extraction of vital information such as vendor name, date, items purchased, and amounts from the receipt. The extracted details are then presented in an organized tabular form, offering users a clear and structured overview of the receipt information.

> The project comprises two primary components: `model_training.ipynb` handles data loading, preprocessing, model architecture, training, and saving, while `app.py` is responsible for loading the trained model. Additionally, it performs document detection, and if the document is identified as a receipt, it utilizes Optical Character Recognition (OCR) to extract and detect relevant details.

</div>


<hr style="border: 2px solid grey;">

## Project Structure


### 1. Model Training (`model_training.ipynb`)

- **Data Loading:** The initial step involves bringing in the data required for training our model. I use the `datasets.ImageFolder` class from Pytorch to load the data from the specified directory (`data_dir`).

- **Data Preprocessing:** In this step, process and prepare the data to ensure it's in the best possible shape for our model to learn from. Data transforms using the `transforms` module from PyTorch.

- **Train and Test Split** The dataset has been partitioned into training and testing sets to facilitate model training and evaluation, adhering to standard practices in machine learning.

- **Model Architecture:** In this section, define the structure of our model, specifying how it will learn from the provided data. The `DocumentClassifier` class is designed for a classification problem. In this example, the ResNet-18 architecture is utilized and fine-tuned for the specific classification task.

- **Model Training:** The training process takes place in this section, where the model learns to distinguish between different types of documents, with a specific focus on identifying receipts. 


- **Model Saving:** After training, we save the trained model so it can be easily used later for predictions.

  ### Results
   - The system attains remarkable accuracy by  leveraging the ResNet model in discerning between receipts and non-receipt documents. This highlights the model's efficacy in accurately classifying document types.

  ### Usage
  - Ensure you have the necessary dependencies installed. (List dependencies in a `requirements.txt` file, if applicable).
  - Download the [Dataset](https://drive.google.com/file/d/179BmDniUP74p4C4FcYytLhE8lxJrkhC7/view?usp=sharing) and put it in the same directory.
   - Run `model_training.ipynb` to perform all of the above steps.



### 2. Application (`app.py`)

- **Model Loading:** This part of the application involves loading the pre-trained model that we saved during the training phase.

- **Detection:** The application then uses the loaded model to perform document detection, determining whether the given document is a receipt.

- **OCR for Receipts:** If the document is identified as a receipt, the application utilizes Optical Character Recognition (OCR) to extract details like vendor name, date, items purchased, and amounts.


  ### Usage
    - Ensure you have the necessary dependencies installed. (List dependencies in a `requirements.txt` file, if applicable).
    - Download the [Trained_Model](https://drive.google.com/file/d/16ZOHyB2Ebck3Zd6qLdHN0oiT1uQXwfJ-/view?usp=sharing) and put it in the same directory.
    - Run `app.py` to start the application for testing purposes.
    - Input the images in `Test_Images` folder and let the model perform detection and extract the info.

</div>


<hr style="border: 2px solid grey;">


## File & Folder Structure 
- `DataSet`: You can find the data set [here](https://drive.google.com/file/d/179BmDniUP74p4C4FcYytLhE8lxJrkhC7/view?usp=sharing).
- `Trained_Model`: you can find the trained model [here](https://drive.google.com/file/d/16ZOHyB2Ebck3Zd6qLdHN0oiT1uQXwfJ-/view?usp=sharing).
- `model_training.ipynb`: file handles data loading, preprocessing, model architecture, training, and saving.
- `Test_Images`: Images used in the app.py for testing purposes.
- `app.py`: File loads the trained model, performs document detection, and utilizes Optical Character Recognition (OCR).
- `eng.traineddata`: To run the OCR this file must present in the same directory.
- `requirements.txt`: Includes the necessary dependencies to install.
