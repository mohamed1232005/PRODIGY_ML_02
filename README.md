# PRODIGY_ML_02
## Customer Segmentation Using K-Means Clustering and Gradient Boosting Classifier
### Project Overview :
In today's competitive retail industry, customer segmentation plays a pivotal role in designing personalized marketing strategies and improving customer engagement. This project aims to group retail store customers based on their purchase history using K-Means clustering. Additionally, a predictive model is implemented using the Gradient Boosting Classifier to classify the clusters formed by the K-Means algorithm based on specific customer attributes.
### ** Objectives**:
The primary objective of this project is to identify distinct customer groups based on annual income and spending scores and to build a predictive model to classify these groups based on demographic and financial features. The technical objectives are:

- **Apply K-Means clustering to group customers based on purchase patterns.**
- **Visualize the resulting clusters and understand the characteristics of each group.**
- **Develop a Gradient Boosting Classifier to predict the group to which a customer belongs based on features like gender, age, income, and spending score.**
- **Evaluate the model's performance using accuracy, precision, recall, and F1-score metrics.**



### 1- Dataset Overview
The dataset used in this project is the "Mall Customers" dataset, which contains information about 200 customers of a retail store. The features in the dataset include:

- **CustomerID: Unique ID for each customer**
- **Gender: Gender of the customer (Male/Female)**
- **Age: Age of the customer**
- **Annual Income (k$): Annual income of the customer in thousands of dollars**
- **Spending Score (1-100): Score assigned to the customer based on their spending behavior (1 = lowest spender, 100 = highest spender)**
The project primarily focuses on using Annual Income and Spending Score for clustering and other features like Gender and Age for predictive modeling.


### 2-Data Exploration and Preprocessing
Before applying any clustering algorithms or building models, the dataset was explored to check for missing values, duplicates, and other irregularities:

- **Summary Statistics: dataset.describe() was used to get the mean, median, and range of all numerical features.**
- **Data Information: dataset.info() showed the data types of the columns and confirmed there were no missing values.**
- **Duplicates and Missing Values: No missing values or duplicate records were found in the dataset.**
- **Feature Extraction: For clustering, only the 'Annual Income' and 'Spending Score' were considered.**
  These features were extracted into a variable X_cluster for clustering purposes.

### 3-Optimal Cluster Identification (Elbow Method)
To identify the optimal number of clusters, the Elbow Method was applied using K-Means clustering. The method works by plotting the inertia (also called WCSS - Within-Cluster Sum of Squares) for different values of 'k' (number of clusters) and identifying the point where the WCSS decreases abruptly, which is considered the "elbow."


### 4-Data Scaling and Clustering
To ensure that all features contributed equally to the clustering process, the data was scaled using the MinMaxScaler. This scaler transforms the features to a range between 0 and 1. The scaled features were then clustered using K-Means with 5 clusters.


### 5-Cluster Visualization
To visualize the clusters, the scaled values of the features 'Annual Income' and 'Spending Score' were plotted, with each point representing a customer. Different colors were used for each cluster, and the cluster centroids were marked with a star (*).

### 6-Predictive Modeling Using Gradient Boosting Classifier
After clustering, the original dataset was enriched by adding the cluster labels from the K-Means output. This transformed the problem into a supervised classification task, where the goal was to predict the cluster (target variable) based on the demographic and financial features (independent variables).

The target variable Cluster was added to the dataset, and the following features were used for classification:

- **Gender: Coded as 0 (Female) or 1 (Male)**
- **Age**
- **Annual Income (k$)**
- **Spending Score (1-100)**




### 7- Model Evaluation
The performance of the Gradient Boosting model was evaluated using multiple metrics, including accuracy, precision, recall, and F1-score. These metrics provide a comprehensive view of the model's performance.

**Results:**

Accuracy: 0.97
Precision: 0.99
Recall: 0.93
F1 Score: 0.95
The high accuracy and F1-score indicate that the model is effective in classifying customers into their respective clusters based on their demographic and financial data.



### 8- Libraries Used
- **Pandas: For data loading, manipulation, and exploration.**
- **Numpy: For numerical computations.**
- **Matplotlib: For visualizing the clusters and model performance.**
- **Scikit-learn: For implementing K-Means clustering, Gradient Boosting Classifier, and evaluation metrics.**
- **MinMaxScaler and LabelEncoder: For feature scaling and encoding categorical variables.**



#### Finally :


This project successfully applied K-Means clustering to segment retail store customers based on their annual income and spending score, identifying five distinct customer groups. The clusters were visualized, and a predictive model was built using the Gradient Boosting Classifier to classify customers into these clusters based on demographic and financial features.

The project demonstrates the power of unsupervised learning in customer segmentation and the effectiveness of gradient boosting for classification tasks. Future work could include exploring other clustering algorithms or feature engineering techniques to further improve customer segmentation.
