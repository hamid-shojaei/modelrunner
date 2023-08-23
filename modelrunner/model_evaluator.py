import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz

import graphviz

from xml.etree import ElementTree as ET
from IPython.display import display, SVG

from .utils import display_svg_with_zoom

from io import BytesIO
from PIL import Image

sns.set_theme()

class ModelEvaluator:
    def __init__(self, X, y, hyperparameters, test_size=0.2):
        """
        X: features dataframe
        y: target series/dataframe
        hyperparameters: dictionary of hyperparameters for models
        test_size: proportion of the dataset to include in the test split
        """
        self.X, self.encoders = self._label_encode_dataframe(X)
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
        self.hyperparameters = hyperparameters
        self.models = {}
        self.predictions = {}
        self.confusion_matrices = {}
        self.feature_importances = {}
        
    def _label_encode_dataframe(self, df):
        """
        Label encode categorical columns of a dataframe
        """
        df_encoded = df.copy()
        encoders = {}
        for column in df.columns:
            if df[column].dtype == 'object':
                encoder = LabelEncoder()
                df_encoded[column] = encoder.fit_transform(df[column])
                encoders[column] = encoder
        return df_encoded, encoders
    
    def _initialize_model(self, model_name, params):
        """
        Initialize the model based on the algorithm and hyperparameters
        """
        if params["algorithm"] == "DecisionTree":
            return DecisionTreeClassifier(max_depth=params.get("max_depth", None),
                                          min_samples_leaf=params.get("min_samples_leaf", 1))
        
        elif params["algorithm"] == "RandomForest":
            return RandomForestClassifier(n_estimators=params.get("num_trees", 100),
                                          max_depth=params.get("max_depth", None),
                                          min_samples_leaf=params.get("min_samples_leaf", 1))
        else:
            raise ValueError(f"Unsupported algorithm: {params['algorithm']}")
    
    def run_models(self):
        """
        Train models based on the provided hyperparameters and predict on the test set
        """
        for model_name, params in self.hyperparameters.items():
            model = self._initialize_model(model_name, params)
            model.fit(self.X_train, self.y_train)
            self.models[model_name] = model
            preds = model.predict(self.X_test)
            self.predictions[model_name] = preds
            self.confusion_matrices[model_name] = confusion_matrix(self.y_test, preds)
            # Check if the model has feature_importances_ attribute
            if hasattr(model, "feature_importances_"):
                self.feature_importances[model_name] = model.feature_importances_
    
    def get_predictions(self, model_name):
        return self.predictions.get(model_name, None)
    
    def get_confusion_matrix(self, model_name):
        return self.confusion_matrices.get(model_name, None)
    
    def get_feature_importances(self, model_name):
        """
        Return feature importances as a sorted dataframe
        """
        importances = self.feature_importances.get(model_name, None)
        if importances is None:
            return None
        
        df_importances = pd.DataFrame({
            "feature": self.X.columns,
            "importance": importances
        })
        return df_importances.sort_values(by="importance", ascending=False).reset_index(drop=True)
    
    def plot_confusion_matrix(self, model_name):
        """
        Plot confusion matrix as a heatmap
        """
        matrix = self.get_confusion_matrix(model_name)
        if matrix is None:
            print(f"No confusion matrix found for model: {model_name}")
            return
        
        # Check if target variable was label-encoded
        if self.y.name in self.encoders:
            labels = self.encoders[self.y.name].classes_
        else:
            labels = self.y.unique()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues', 
                    xticklabels=labels,
                    yticklabels=labels)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix for {model_name}')
        plt.show()

    def visualize_decision_tree(self, model_name):
        """
        Visualize the decision tree using graphviz
        """
        model = self.models.get(model_name, None)
        if model is None:
            print(f"No model found with name: {model_name}")
            return
        
        if not isinstance(model, DecisionTreeClassifier):
            print(f"Model {model_name} is not a DecisionTree. Visualization only supports DecisionTree.")
            return
        
        # Convert class names to string type
        str_class_names = [str(cls) for cls in model.classes_]
        
        dot_data = export_graphviz(model, out_file=None, 
                                   feature_names=self.X.columns,  
                                   class_names=str_class_names,  
                                   filled=True, rounded=True,  
                                   special_characters=True)  
        graph = graphviz.Source(dot_data)  
        return graph