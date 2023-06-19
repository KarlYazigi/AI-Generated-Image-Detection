# AI-Generated-Image-Detection
This GitHub project is a personal endeavor aimed at exploring the classification of AI-generated images using a Convolutional Neural Network (CNN). The CNN is trained to discern between three distinct classes:  DALL-E-2 - generated images, Stable Diffusion-generated images, and Human-made images. 

The DALL-E-2 - generated images were scraped from the source https://dalle2.gallery
The Stable Diffusion-generated images were accessed from ["Link." ](https://huggingface.co/datasets/poloclub/diffusiondb)
The Human-generated images were both scraped from google and accessed from the source https://huggingface.co/datasets/poloclub/diffusiondb

## Files Overview:

* **WebScraper.ipynb**: Python code utilizing the Selenium library for web scraping, specifically designed to extract images generated by DALL-E 2 from https://dalle2.galley and images generated by humans from google images with different search prompts.
* **DataProcessing.ipynb**: Jupyter Notebook containing the data preprocessing pipeline, which includes tasks such as data partitioning, normalization, and categorization into three distinct classes.
* **ModelTrain.py**: Implementation of the model architecture and training procedure, ensuring efficient and effective training of the chosen machine learning model.

## Results
The total training dataset comprises approximately 150,000 images, with an even distribution of 50,000 images per class.

The following plot illustrates the training and test accuracy. The data was divided into 68 batches and trained for 10 epochs, resulting in a total of 680 epochs.


<img src= "https://github.com/KarlYazigi/AI-Generated-Image-Detection/assets/66206934/5dbfe1bd-f710-4c22-9613-e5a3c92dd13d" width="550" height="412.5">


Additionally, the following plot presents the relative accuracy across the three classes in the testing dataset. The model demonstrates proficient detection of AI-generated images, particularly for Stable Diffusion and DALL-E 2 images. However, there is an approximately 23% occurrence of misclassifying human-generated images as AI-generated images. Considering the primary objective of accurately identifying AI-generated images, this is acceptable.

<img src="https://github.com/KarlYazigi/AI-Generated-Image-Detection/assets/66206934/02aced8b-947f-4752-8ce3-4fcc536c15eb" width="700" height="500">



## Image Classification Example

### Class 1: DALLE-generated Images
![pasted image 0 (2)](https://github.com/KarlYazigi/AI-Generated-Image-Detection/assets/66206934/fcc6f16f-cc48-4c6e-acdd-a9653aa09522)        ![pasted image 0 (1)](https://github.com/KarlYazigi/AI-Generated-Image-Detection/assets/66206934/f6d0d06a-1808-4404-8b51-1ba7824c4ded)


### Class 2: Stable Diffusion-generated Images

![sd_2](https://github.com/KarlYazigi/AI-Generated-Image-Detection/assets/66206934/9f9c3557-36a9-4ba0-a0e7-14690f051ac5)        ![sd_1](https://github.com/KarlYazigi/AI-Generated-Image-Detection/assets/66206934/01363718-cff3-45db-9d60-b3c59a71c3fe)

### Class 3: Human-made Images
![j_animals_3054](https://github.com/KarlYazigi/AI-Generated-Image-Detection/assets/66206934/57517bce-75ba-4dae-a8c9-7e598edaeec5)       ![pasted image 0](https://github.com/KarlYazigi/AI-Generated-Image-Detection/assets/66206934/ef18671a-70c6-41db-9474-139cc2a61f48)


This Project is a collaborative effort with Santiago Miro Trejo : https://github.com/SantMiro
