#!/usr/bin/env python
# coding: utf-8

# # ML Task-03

# Develop a hand gesture recognition model that can accurately identify and classify different hand gestures from image or video data, enabling intuitive human-computer interaction and gesture-based control systems.
# 
# Dataset :- https://www.canva.com/link?target=https%3A%2F%2Fwww.kaggle.com%2Fc%2Fdogs-vs-cats%2Fdata&design=DAFpRxy47kU&accessRole=viewer&linkSource=document

# In[1]:


# extract dataset
from zipfile import ZipFile

dataset_train = "train.zip"
    
with ZipFile(dataset_train, 'r') as zip:
    zip.extractall()


# In[2]:


import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
from sklearn.model_selection import GridSearchCV
import cv2
import seaborn as sns
import time
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


# In[3]:


folder_path = f"Dataset/"
os.makedirs(folder_path, exist_ok=True)

# define path
confusion_image_path = os.path.join(folder_path, 'confusion matrix.png')
classification_file_path = os.path.join(folder_path, 'classification_report.txt')
model_file_path = os.path.join(folder_path, "svm_model.pkl")

# Path dataset
dataset_dir = "Dataset/"
train_dir = os.path.join(dataset_dir, "train")
test_dir = os.path.join(dataset_dir, "test1")


# In[4]:


# load data, preprocessing data, and labeling
# dog = 1, cat = 0
train_images = os.listdir(train_dir)
features = []
labels = []
image_size = (50, 50)

# Proses train images
for image in tqdm(train_images, desc="Processing Train Images"):
    if image[0:3] == 'cat' :
        label = 0
    else :
        label = 1
    image_read = cv2.imread(train_dir+"/"+image)
    image_resized = cv2.resize(image_read, image_size)
    image_normalized = image_resized / 255.0
    image_flatten = image_normalized.flatten()
    features.append(image_flatten)
    labels.append(label)


# In[5]:


del train_images


# In[6]:


features = np.asarray(features)
labels = np.asarray(labels)

# train test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=True, random_state=42)


# In[7]:


del features
del labels


# In[8]:


# PCA, SVM, & Pipeline
n_components = 0.8
pca = PCA(n_components=n_components)
svm = SVC()
pca = PCA(n_components=n_components, random_state=42)
pipeline = Pipeline([
    ('pca', pca),
    ('svm', svm)
])


# In[9]:


param_grid = {
    'pca__n_components': [2, 1, 0.9, 0.8],
    'svm__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
}


# In[10]:


# Hitung waktu training
start_time = time.time()

grid_search = GridSearchCV(pipeline, param_grid, cv=3, verbose=4)
grid_search.fit(X_train, y_train)

# Hitung waktu training
end_time = time.time()


# In[11]:


del X_train
del y_train


# In[12]:


# Mendapatkan model terbaik dan parameter terbaik
best_pipeline = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters: ", best_params)
print("Best Score: ", best_score)


# The grid search identified the best SVM model configuration: using 90% of principal components and the Radial Basis Function (RBF) kernel. This setup yielded an accuracy score of approximately 67.57%, demonstrating the effectiveness of these parameters in accurately classifying cats and dogs images.

# In[13]:


# Evaluation on test dataset
accuracy = best_pipeline.score(X_test, y_test)
print("Accuracy:", accuracy)


# The model attained an accuracy score of approximately 67.62%, indicating its ability to correctly classify images of cats and dogs from the Kaggle dataset. This accuracy score reflects the proportion of correctly predicted classifications out of the total dataset, showcasing the model's overall performance.

# In[15]:


y_pred = best_pipeline.predict(X_test)

# classification report
target_names = ['Cat', 'Dog']
classification_rep = classification_report(y_test, y_pred, target_names=target_names)
print("Classification Report:\n", classification_rep)

with open(classification_file_path, 'w') as file:
    file.write(classification_rep)


# In the classification report, the model achieved an overall accuracy of 68% in distinguishing between cats and dogs. With precision and recall scores around 68%, it demonstrates consistent performance in identifying both classes. The F1-score, a balanced measure of precision and recall, also hovers around 68%, indicating a well-rounded classification capability for this SVM model.

# In[16]:


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.savefig(confusion_image_path)
plt.show()


# In summary, the SVM model accurately classified Kaggle dataset images of cats and dogs. The confusion matrix visualized its performance, revealing precise predictions for both classes. This underscores the SVM's effectiveness in image classification, emphasizing the need for robust evaluation tools. This success paves the way for future improvements and applications in real-world scenarios requiring precise image recognition.

# * Conclusion, the successful optimization of the Support Vector Machine (SVM) model highlights the importance of parameter tuning and feature selection in enhancing image classification accuracy. By leveraging 90% of principal components and the RBF kernel, the model achieved a commendable accuracy of approximately 67.57%. This accomplishment not only underscores the effectiveness of SVMs in image classification tasks but also emphasizes the significance of fine-tuning for maximizing predictive capabilities. This knowledge paves the way for improved algorithms and applications in the realm of computer vision.
