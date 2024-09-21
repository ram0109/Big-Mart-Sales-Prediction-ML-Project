# Big-Mart-Sales-Prediction-ML-Project
The project on **Big Mart Sales Prediction** focuses on predicting sales using a machine learning model. Below is a detailed explanation of each step involved:

### 1. **Importing Dependencies**
   You began by importing essential libraries for the project:
   - **NumPy** and **Pandas**: For data manipulation.
   - **Matplotlib** and **Seaborn**: For data visualization.
   - **Scikit-learn**: For data preprocessing and model evaluation.
   - **XGBoost**: The model you used for prediction, which is efficient for regression problems.
   
### 2. **Data Collection and Processing**
   - The dataset, `Train.csv`, was loaded into a Pandas DataFrame.
   - Basic exploration was done using `.head()` and `.info()` to get an overview of the data.
   - Categorical columns like `Item_Identifier`, `Item_Fat_Content`, and `Outlet_Type` were identified.

### 3. **Handling Missing Values**
   - The **Item_Weight** column's missing values were filled with the **mean** value.
   - For the **Outlet_Size** column, missing values were filled with the **mode**, grouped by the **Outlet_Type**.
   - After this, you rechecked the data for missing values, ensuring they were all addressed.

### 4. **Data Analysis and Visualization**
   You visualized various features to understand their distributions:
   - **Item_Weight**, **Item_Visibility**, and **Item_MRP** were visualized using distribution plots.
   - The **Item_Outlet_Sales** was also visualized to study its distribution.
   - Count plots were used to explore categorical variables such as **Item_Fat_Content**, **Item_Type**, and **Outlet_Size**.

### 5. **Data Preprocessing**
   - Some inconsistencies in **Item_Fat_Content** were corrected by replacing values like `low fat`, `LF` with `Low Fat`, and `reg` with `Regular`.
   - **Label Encoding** was used to convert categorical variables into numerical form for machine learning. This was done using `LabelEncoder` on features such as `Item_Identifier`, `Outlet_Type`, and others.

### 6. **Splitting Features and Target**
   - Features (`X`) were separated from the target variable (`Item_Outlet_Sales`).
   - The data was split into training (80%) and testing (20%) sets using `train_test_split` from Scikit-learn.

### 7. **Model Training**
   - You chose the **XGBoost Regressor**, a powerful model for regression tasks.
   - The model was trained using the training data (`X_train`, `Y_train`).

### 8. **Model Evaluation**
   - **R-squared (R²) value** was calculated to measure the goodness of fit for both the training and testing sets.
     - Training data R²: Indicates how well the model fits the training data.
     - Testing data R²: Indicates how well the model generalizes to new, unseen data.

   The **R² score** you obtained represents the proportion of variance in sales that can be explained by the model.

This project covers essential machine learning steps from data preprocessing to model training and evaluation, making it a robust approach to predicting sales for Big Mart.
