# Apple Quality Prediction

## Overview
This project aims to analyze and predict the quality of apples based on various attributes such as ripeness, sweetness, size, and other factors. The project utilizes machine learning techniques, specifically K-Nearest Neighbors (KNN) and Random Forest Regressor, to classify and predict apple quality.

## Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn (sklearn)
- Random Forest Regressor
- K-Nearest Neighbors (KNN)

## Dataset
The dataset used in this project is stored in `Apple Quality.csv` and contains various attributes related to apple quality assessment. Some of the key attributes include:
- Ripeness
- Sweetness
- Size
- Weight
- Crunchiness
- Juiciness
- Acidity
- Quality (Target Variable)

## Questions
**Question 1:**  
*Which attributes should be used to accurately predict the quality of an apple?*  

**Question 2:**  
*Would it be possible to optimize the predictability of an apple’s quality by considering and comparing alternative distance algorithms - Manhattan, Euclidean, and Minkowski?*  

## Project Workflow
1. **Data Preprocessing**
   - Read and load the dataset.
   - Check for missing values and data types.
   - Generate descriptive statistics.

2. **Exploratory Data Analysis (EDA)**
   - Generate histograms for all variables.
   - Create pairplots and scatter matrices to analyze relationships between attributes.

3. **Feature Scaling and Splitting Data**
   - Assign features (X) and target variable (y).
   - Scale the feature values using StandardScaler.
   - Split the dataset into training and testing sets (75%-25%).

4. **Model Training & Evaluation**
   - **K-Nearest Neighbors (KNN)**
     - Train KNN classifier with Manhattan distance metric.
     - Perform hyperparameter tuning for the best K value.
     - Evaluate model accuracy using cross-validation.
   - **Random Forest Regressor**
     - Train a Random Forest model for feature importance analysis.
     - Visualize feature importance scores.

5. **Hyperparameter Optimization**
   - Test different values of K in KNN.
   - Measure accuracy and error rates for different hyperparameters.
   - Generate line plots to visualize training vs. test performance.

## Results
- Model accuracy scores for KNN classification.
- Feature importance scores from the Random Forest model.
- Visualization of training and test scores for various hyperparameters.
- Error rate analysis for the KNN model.

##Manhattan Distance Metric:
  - Achieved consistent high accuracy across training, testing, and overall metrics, indicating robust generalization.
  - Outperformed other distance metrics with an accuracy of **75.05%**.
  - Demonstrates that using attributes such as ripeness, sweetness, and size, the model can predict apple quality with great accuracy.

- **Euclidean Distance Metric:**  
  - Achieved an accuracy of **74.4%**.
  - Considered a worse generalization since, despite relatively high training and testing accuracies, it recorded the lowest overall accuracy and dipped at higher values of k, indicating underfitting.

- **Minkowski Distance Metric:**  
  - Achieved an accuracy of **74.5%**.
  - Demonstrated good generalization as the accuracy did not dip at higher values of k; however, the values fluctuated between 74% and 76%, suggesting less stability compared to Manhattan.

## Feature Importance & Analysis
- **Key Attributes:**  
  The bar graph shows that **size, sweetness, and ripeness** are the attributes most correlated with apple quality. These features are the primary indicators in determining whether an apple is of good quality.
- **Ideal Apple Characteristics:**  
  - **Size:** Average to high values  
  - **Sweetness:** High values  
  - **Ripeness:** Low to average values
- **Observations:**  
  - Sorting apples by quality (good vs. bad) reveals the combination of these key attributes that results in an exceptional apple.
  - Although most apples fall within a small radius of variation with few outliers, these variables, when compared with the target (quality), provide insight into the factors contributing to overall apple quality.
  - Even without strong correlation between every attribute, the relative influence of size, sweetness, and ripeness is evident.

## Contributing
Feel free to fork this repository and submit pull requests for improvements.
