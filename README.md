#  Artwork Classification Project  
AI Skills — Deep Learning Project

---

 1) About the Project  
This project classifies different types of artworks (such as paintings, sketches, and photography) using Convolutional Neural Networks (CNNs) and Transfer Learning.

We trained and evaluated multiple models, applied Grad-CAM for explainability, and built a GUI to make predictions on uploaded images.

---

 2) Project Structure  
- **data/** → Contains README with dataset instructions  
- **gui/** → Contains the GUI app, utils, and requirements  
- **notebooks/** → Contains Jupyter notebooks for training and evaluation  
- **src/** → Python scripts for loading data, training models, evaluating, and GradCAM  
- **README.md** → Main documentation  
- **.gitignore** → Files to be ignored by GitHub  

---

 3) Models Used  
We implemented 3 CNN architectures as required:

- VGG16  
- ResNet50  
- EfficientNetB0  

Each model is trained and evaluated separately inside the `notebooks/` folder.

---

 4) Evaluation  
We evaluate all models using:

- Accuracy  
- Precision, Recall, and F1-score  
- Confusion Matrix  
- Grad-CAM heatmaps

Full evaluation is inside:  
`notebooks/evaluation.ipynb`

---
 5) How to Run the GUI  
1. Go to the `gui/` folder  
2. Install requirements:



pip install -r requirements.txt

3. Run the app:


python app.py

The GUI allows the user to upload an image and view predictions with Grad-CAM.

---

 6) Dataset  
We use the **WikiArt Dataset**.

🔗 Dataset Link:  
(https://www.kaggle.com/datasets/steubk/wikiart/data)

---

 7) Team Members & Roles  
- Member 1: Data & Preprocessing  
- Member 2: VGG16 Training  
- Member 3: ResNet50 Training  
- Member 4: EfficientNetB0 Training  
- Member 5: Evaluation & Grad-CAM  
- Member 6: GUI Development 


---

 Bonus (Optional)  
We added an additional model for Artist Classification.

Notebook:
`notebooks/multitask_model.ipynb`




Project structure ready.
