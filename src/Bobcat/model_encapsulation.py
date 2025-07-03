{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "856b3cee-7865-4b2a-bc87-6a9ea569946f",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.experimental import enable_halving_search_cv\n",
    "from sklearn.model_selection import HalvingGridSearchCV, train_test_split\n",
    "from sklearn.multioutput import MultiOutputClassifier\n",
    "from sklearn.metrics import accuracy_score, classification_report"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "b5c3ccec-c79b-42ea-9538-087a7c65a474",
   "metadata": {},
   "outputs": [],
   "source": [
    "class FootballModel:\n",
    "    def __init__(self):\n",
    "        print(\"Hello world!\")\n",
    "\n",
    "    def fitModel(self, X_train, y_train):\n",
    "        param_grid = {\n",
    "            \"criterion\": [\"gini\"],  # Only use one criterion\n",
    "            \"max_depth\": [25, 30],  # Reduce depth options\n",
    "            \"n_estimators\": [100],  # Lower estimators\n",
    "            \"max_features\": [\"sqrt\"],  # Single selection\n",
    "            \"min_impurity_decrease\": [0.0],  # Fixed value\n",
    "            \"min_samples_split\": [10],  # One choice\n",
    "            \"min_samples_leaf\": [5],  # One choice\n",
    "            \"class_weight\": [\"balanced\"]  # Keep balanced\n",
    "        }\n",
    "\n",
    "        self.clf = HalvingGridSearchCV(\n",
    "            RandomForestClassifier(),\n",
    "            param_grid=param_grid,\n",
    "            scoring=\"f1_micro\",  # Keep optimizing for precision\n",
    "            n_jobs=-1,\n",
    "            verbose=1,\n",
    "            factor=3,\n",
    "            cv=5,\n",
    "        )\n",
    "\n",
    "        clf.fit(X_train, y_train)\n",
    "\n",
    "    def predict(self):\n",
    "        # Make predictions for Run/Pass on the test set\n",
    "        y_pred = self.clf.predict(X_test)\n",
    "\n",
    "        # Calculate accuracy\n",
    "        accuracy = accuracy_score(y_test, y_pred)\n",
    "        print(f\"\\n🔥 Run/Pass Model Accuracy: {accuracy:.2%}\")\n",
    "\n",
    "    def feature_importance(X_train, ):\n",
    "        #Feature Importance Analysis\n",
    "        best_model = clf.best_estimator_ # Get the best trained RandomForestClassifier\n",
    "        feature_importance_df = pd.DataFrame({ \"Feature\": X_train.columns, \"Importance\": best_model.feature_importances_ }).sort_values(by=\"Importance\", ascending=False)\n",
    "        #Display top features\n",
    "        print('All Feature Importances:', feature_importance_df)\n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "38e5369c-7e60-4244-9042-705eccfbac9d",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
