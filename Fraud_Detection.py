#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load the dataset
file_path = "PS_20174392719_1491204439457_log.csv"
df = pd.read_csv(file_path)

# Display basic info about the dataset
df.info()


# In[2]:


df.head()


# # Data Preprocessing

# In[3]:


# Check for missing values
df.isnull().sum()


# In[4]:


# Fill or drop missing values as needed
df = df.dropna()


# ## Creating a Dataframe

# In[11]:


# Calculate the index that represents 1/8th of the data
quarter_index = len(df) // 8

# Create a new dataframe with the first 1/8th of the data
df_subset = df.iloc[:quarter_index]

# Display the first few rows of the new subset dataframe
df_subset.head()


# In[12]:


df_subset.info()


# ## Encoding Categorical Variables

# In[13]:


import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder

# Encoding categorical columns (for example 'type' and 'nameOrig')
label_encoder = LabelEncoder()

df_subset['type'] = label_encoder.fit_transform(df_subset['type'])
df_subset['nameOrig'] = label_encoder.fit_transform(df_subset['nameOrig'])
df_subset['nameDest'] = label_encoder.fit_transform(df_subset['nameDest'])


# ## Normalization/Standardization

# In[16]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Using MinMaxScaler for normalization
scaler = MinMaxScaler()
df_subset[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']] = scaler.fit_transform(
    df_subset[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']])


# ## Addressing Class Imbalance (SMOTE)

# In[17]:


from imblearn.over_sampling import SMOTE

# Separate features and target variable
X = df_subset.drop(['isFraud', 'isFlaggedFraud'], axis=1)
y = df_subset['isFraud']

# Apply SMOTE for balancing the class distribution
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_res, y_res = smote.fit_resample(X, y)


# # EDA Analysis

# In[18]:


# Show descriptive statistics for numerical features
df_subset.describe()


# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns

# Distribution of the target variable (isFraud)
sns.countplot(x='isFraud', data=df_subset)
plt.title('Distribution of Fraudulent vs Non-Fraudulent Transactions')
plt.show()


# In[21]:


# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_subset.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# In[69]:


# Boxplot to check for outliers in numerical features
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x=df_subset[feature], color='lightgreen')
    plt.title(f'Boxplot of {feature}')
    plt.tight_layout()

plt.show()


# ## PCA Clustering

# In[22]:


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Apply PCA to reduce dimensions to 2
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_res)

# Plot the PCA clustering results
plt.figure(figsize=(10, 6))

# Scatter plot with different colors for fraudulent (1) and non-fraudulent (0) transactions
plt.scatter(X_pca[y_res == 0, 0], X_pca[y_res == 0, 1], label='Non-Fraudulent', color='blue', alpha=0.5)
plt.scatter(X_pca[y_res == 1, 0], X_pca[y_res == 1, 1], label='Fraudulent', color='red', alpha=0.5)

# Adding labels and title
plt.title('PCA Clustering of Fraudulent and Non-Fraudulent Transactions')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Add legend
plt.legend()

# Show plot
plt.show()


# # Model Selection and Evaluation

# ## Random Forest

# In[59]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Initialize the model
rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=10, random_state=42)

# Train the model
rf.fit(X_res, y_res)


# In[60]:


# Predict and evaluate
y_pred_rf = rf.predict(X_res)
print('Random Forest Classification Report:')
print(classification_report(y_res, y_pred_rf))
print('Random Forest AUC:', roc_auc_score(y_res, y_pred_rf))


# ## LightGBM

# In[25]:


import lightgbm as lgb

# Initialize the model
lgbm = lgb.LGBMClassifier(n_estimators=100, random_state=42)

# Train the model
lgbm.fit(X_res, y_res)


# In[26]:


# Predict and evaluate
y_pred_lgbm = lgbm.predict(X_res)
print('LightGBM Classification Report:')
print(classification_report(y_res, y_pred_lgbm))
print('LightGBM AUC:', roc_auc_score(y_res, y_pred_lgbm))


# ## Neural Networks (Keras)

# In[32]:


import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Build a simple neural network model
model = Sequential()
model.add(Dense(64, input_dim=X_res.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # For binary classification


# In[34]:


# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])


# In[35]:


model.fit(X_res, y_res, epochs=15, batch_size=32, validation_split=0.2)


# In[36]:


# Evaluate the model
loss, accuracy = model.evaluate(X_res, y_res)
print(f'Neural Network Accuracy: {accuracy * 100:.2f}%')


# ## DNN Model

# In[37]:


# Build the Deep Neural Network (DNN)
dnn_model = Sequential()

# Input layer and first hidden layer
dnn_model.add(Dense(128, input_dim=X_res.shape[1], activation='relu'))

# Additional hidden layers
dnn_model.add(Dense(64, activation='relu'))
dnn_model.add(Dense(32, activation='relu'))
dnn_model.add(Dense(16, activation='relu'))


# In[38]:


# Output layer (binary classification)
dnn_model.add(Dense(1, activation='sigmoid'))  # For binary classification (fraud or not)


# In[39]:


# Compile the model
dnn_model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])


# In[40]:


# Train the model
dnn_model.fit(X_res, y_res, epochs=10, batch_size=32, validation_split=0.2)


# In[41]:


# Evaluate the model
loss, accuracy = dnn_model.evaluate(X_res, y_res)
print(f'DNN Accuracy: {accuracy * 100:.2f}%')


# # Visualizations

# In[47]:


# Fit the model and save the history
ann_history = model.fit(X_res, y_res, epochs=15, batch_size=32, validation_split=0.2)
# Fit the DNN model and save the history
dnn_history = dnn_model.fit(X_res, y_res, epochs=10, batch_size=32, validation_split=0.2)

# Function to plot the training history
def plot_training_history(history, model_name):
    history_keys = history.history.keys()
    
    # Check for the correct keys for accuracy and loss
    acc_key = 'accuracy' if 'accuracy' in history_keys else None
    val_acc_key = 'val_accuracy' if 'val_accuracy' in history_keys else None
    loss_key = 'loss' if 'loss' in history_keys else None
    val_loss_key = 'val_loss' if 'val_loss' in history_keys else None

    # Plot the training and validation accuracy and loss if available
    plt.figure(figsize=(12, 6))

    # Plot training and validation accuracy if available
    if acc_key:
        plt.subplot(1, 2, 1)
        plt.plot(history.history[acc_key], label='Train Accuracy')
        if val_acc_key:
            plt.plot(history.history[val_acc_key], label='Validation Accuracy')
        plt.title(f'{model_name} Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

    # Plot training and validation loss if available
    if loss_key:
        plt.subplot(1, 2, 2)
        plt.plot(history.history[loss_key], label='Train Loss')
        if val_loss_key:
            plt.plot(history.history[val_loss_key], label='Validation Loss')
        plt.title(f'{model_name} Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

    plt.tight_layout()
    plt.show()

# Plot for ANN
plot_training_history(ann_history, 'ANN')

# Plot for DNN
plot_training_history(dnn_history, 'DNN')


# ## Confusion Matrix

# In[61]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Confusion Matrix for RandomForest
cm_rf = confusion_matrix(y_res, y_pred_rf)

# Plotting the confusion matrix as a heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])

# Adding labels and title
plt.title('RandomForest Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Show plot
plt.show()


# In[70]:


# Confusion Matrix for LightGBM
cm_lgbm = confusion_matrix(y_res, y_pred_lgbm)

# Plotting the confusion matrix as a heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm_lgbm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])

# Adding labels and title
plt.title('LightGBM Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Show plot
plt.show()


# ## Comparison Between Models

# In[51]:


from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


# In[63]:


# Predict probabilities for RandomForest, LightGBM, ANN, and DNN
fpr_rf, tpr_rf, _ = roc_curve(y_res, rf.predict_proba(X_res)[:, 1])
fpr_lgbm, tpr_lgbm, _ = roc_curve(y_res, lgbm.predict_proba(X_res)[:, 1])


# In[64]:


# For ANN, predict probabilities using the `predict` method
fpr_ann, tpr_ann, _ = roc_curve(y_res, model.predict(X_res)[:, 0])

# For DNN, predict probabilities using the `predict` method
fpr_dnn, tpr_dnn, _ = roc_curve(y_res, dnn_model.predict(X_res)[:, 0])


# In[65]:


# Plot the ROC curves for all models
plt.figure(figsize=(8, 6))

plt.plot(fpr_rf, tpr_rf, color='blue', label='RandomForest ROC')
plt.plot(fpr_lgbm, tpr_lgbm, color='green', label='LightGBM ROC')
plt.plot(fpr_ann, tpr_ann, color='orange', label='ANN ROC')

# Diagonal line representing a random classifier
plt.plot([0, 1], [0, 1], color='red', linestyle='--')

# Set title and labels
plt.title('ROC Curve Comparison Between ANN and Other Models')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# Add legend
plt.legend()

# Show plot
plt.show()


# In[66]:


# Plot the ROC curves for all models
plt.figure(figsize=(8, 6))

plt.plot(fpr_rf, tpr_rf, color='blue', label='RandomForest ROC')
plt.plot(fpr_lgbm, tpr_lgbm, color='green', label='LightGBM ROC')
plt.plot(fpr_dnn, tpr_dnn, color='purple', label='DNN ROC')

# Diagonal line representing a random classifier
plt.plot([0, 1], [0, 1], color='red', linestyle='--')

# Set title and labels
plt.title('ROC Curve Comparison Between DNN and Other Models')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# Add legend
plt.legend()

# Show plot
plt.show()

