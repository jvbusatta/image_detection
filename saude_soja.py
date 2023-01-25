import os
from PIL import Image
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import numpy as np


# Function to process the image
def process_image(image_path):
    # Open image
    image = Image.open(image_path)
    image = np.array(image)
    # Image processing to detect and isolate individual plants
    # Convert to grayscale
    gray_image = rgb2gray(image)
    # Apply binary threshold
    thresh = threshold_otsu(gray_image)
    binary_image = gray_image > thresh
    # Remove connected borders
    cleared_image = clear_border(binary_image)
    # Label connected regions
    label_image = label(cleared_image)
    regions = regionprops(label_image)
    # Count number of plants found
    plant_count = len(regions)

    # Identify if the plant contains pests and is healthy using machine learning
    # Load pre-trained
    #classification model
    model = load_model("path/to/plant_classification_model.h5")
    plant_health = []
    for plant in regions:
    # Extract plant features
    # ...
    # Pre-processing of features
    # ...
    # Plant health classification with the model
        health = model.predict(plant_features)
        plant_health.append(health)

    # Paint individual photos according to health classification using RGB scale
    for idx, plant in enumerate(regions):
        # Retrieve the health of the plant
        health = plant_health[idx]
        # Paint the health on the image
        # ...

    # Save the image of the plantation with the health of the plants marked
    save_path = os.path.join("path/to/save/directory", "health_marked_"+os.path.basename(image_path))
    Image.fromarray(image).save(save_path)

    # Load image data and labels
    data = np.array(Image.open(image_path))
    labels = ['praga', 'saudavel']
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # train the model
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)

    # evaluate the model
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("Accuracy: {:.2f}%".format(accuracy*100))
    print("Precision: {:.2f}%".format(precision*100))
    print("Recall: {:.2f}%".format(recall*100))
    print("F1 score: {:.2f}".format(f1))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


    #Iterate through all image files in the directory
    for file in os.listdir(dir_path):
        if file.endswith(".jpeg"):
         image_
