# MLOps introduction

## Part 1: Repository Setup and Version Control

### Task 1: Create a New Repository

- **Objective**: Introduce version control using GitHub.
- **Instructions:**
  1. Go to [GitHub](https://github.com/).
  2. Create a new repository titled `mlops-introduction`.
  3. Clone the repository to your local machine:

     ```bash
     git clone https://github.com/YOUR_USERNAME/mlops-introduction.git
     cd mlops-introduction
     ```

### Task 2: Initialize Git and Set Up `.gitignore`

- **Objective**: Initialize Git and set up a `.gitignore` file.
- **Instructions:**
  1. Initialize Git if necessary:

     ```bash
     git init
     ```

  2. Create a `.gitignore` file with:

     ```
     __pycache__/
     ```

  3. Add, commit, and push changes:

     ```bash
     git add .
     git commit -m "Initial commit"
     git push origin main
     ```

## Part 2: Setting Up Python Environment

### Task 1: Create and Activate Python Virtual Environment

- **Objective**: Set up a clean Python environment using `virtualenv`.
- **Instructions:**
  1. Install `virtualenv` if not installed:

     ```bash
     pip install virtualenv
     ```

  2. Create and activate environment:

     ```bash
     virtualenv mlops-lab1
     source mlops-lab1/bin/activate  # On Windows, use `mlops-lab1\Scripts\activate`
     ```

### Task 2: Create and Populate `requirements.txt`

- **Objective**: Document project dependencies using `requirements.txt`.
- **Instructions:**
  1. Create a `requirements.txt` file in the project directory:

     ```bash
     touch requirements.txt
     ```

  2. Add the required libraries to `requirements.txt`:

     ```
     scikit-learn
     pandas
     numpy
     matplotlib
     ```

  3. Install dependencies from `requirements.txt`:

     ```bash
     pip install -r requirements.txt
     ```

  4. Verify installation:

     ```bash
     python -c "import sklearn, pandas, numpy, matplotlib; print('Libraries installed successfully')"
     ```

## Part 3: Build a Simple ML Pipeline

If you find it easier, you can initially write and test your code in a Jupyter Notebook (`.ipynb`). This allows for interactive coding and testing. Once you have verified your code works correctly, copy it into Python functions and create a working script (`.py`).

### Task 1: Load and Explore the Iris Dataset

- **Objective**: Load and examine a public dataset. Then, train a simple model.
- **Instructions:**

  1. Create a new branch for this task:

     ```bash
     git checkout -b feature/load-data
     ```

  2. Create `iris_pipeline.py` with:

     ```python
     import pandas as pd
     from sklearn.datasets import load_iris

     def load_dataset():
         iris = load_iris()
         df = pd.DataFrame(iris.data, columns=iris.feature_names)
         df['species'] = iris.target
         df["species_name"] = df.apply(
             lambda x: str(iris.target_names[int(x["species"])]), axis=1
         )
         return df

     if __name__ == "__main__":
         iris_df = load_dataset()
         print(iris_df.head())
     ```

  3. Add, commit, and push your changes:

     ```bash
     git add iris_pipeline.py
     git commit -m "Load and explore Iris dataset"
     git push origin feature/load-data
     ```

  4. Open a pull request on GitHub to merge your changes into the `main` branch and review it before merging.

### Task 2: Train a Logistic Regression Model

- **Objective**: Build and evaluate a simple model.
- **Instructions:**

  1. Create a new branch for this task:

     ```bash
     git checkout -b feature/train-model
     ```

  2. Extend `iris_pipeline.py` to include:

     ```python
     from sklearn.model_selection import train_test_split
     from sklearn.linear_model import LogisticRegression
     from sklearn.metrics import accuracy_score

     def train(df):
         X_train, X_test, y_train, y_test = train_test_split(
             df.iloc[:, :-1], df["species"], test_size=0.2, random_state=42
         )

         model = LogisticRegression(max_iter=200)
         model.fit(X_train, y_train)

         return model, X_train, X_test, y_train, y_test

     def get_accuracy(model, X_test, y_test):
         predictions = model.predict(X_test)
         accuracy = accuracy_score(y_test, predictions)

         return accuracy

     if __name__ == "__main__":
         iris_df = load_dataset()
         model, X_train, X_test, y_train, y_test = train(iris_df)
         accuracy = get_accuracy(model, X_test, y_test)
         print(f"Accuracy: {accuracy:.2f}")
     ```

  3. Update `requirements.txt` if new dependencies are required:

     ```
     scikit-learn
     ```

  4. Add, commit, and push your changes:

     ```bash
     git add iris_pipeline.py requirements.txt
     git commit -m "Train logistic regression model"
     git push origin feature/train-model
     ```

  5. Open a pull request on GitHub to merge your changes into the `main` branch and review it before merging.

### Task 3: Write and Run a Simple Unit Test

- **Objective**: Introduce the concept of testing in ML pipelines.
- **Note**: The tests provided here are basic/dummy examples to demonstrate the concept of testing.
- **Instructions:**
  1. Create a new branch for this task:

     ```bash
     git checkout -b feature/add-tests
     ```

  2. Create a test script named `test_iris_pipeline.py`.

  3. Write test functions for each major component in your script. For example:
     ```python
     from iris_pipeline import load_dataset, train, get_accuracy

     def test_load_dataset():
         df = load_dataset()
         assert not df.empty, "The DataFrame should not be empty after loading the dataset."

     def test_model_accuracy():
         df = load_dataset()
         model, X_train, X_test, y_train, y_test = train(df)
         accuracy = get_accuracy(model, X_test, y_test)
         assert accuracy > 0.8, "Model accuracy is below 80%."
     ```

  4. Run the tests using a testing framework like `pytest`:

     ```bash
     pytest test_iris_pipeline.py
     ```

  5. Add, commit, and push your changes:

     ```bash
     git add test_iris_pipeline.py
     git commit -m "Add unit tests for data loading, training, and model accuracy"
     git push origin feature/add-tests
     ```

  6. Open a pull request on GitHub to merge your changes into the `main` branch and review it before merging.

### Task 4: Visualize Data and Model Performance

- **Objective**: Visualize data and model outputs.
- **Instructions:**

  1. Create a new branch for this task:

     ```bash
     git checkout -b feature/add-visualizations
     ```

  2. Update your `iris_pipeline.py` script to include functions for data visualization using `matplotlib`.

  3. Add the following visualization functions to `iris_pipeline.py`:

     ```python
     import matplotlib.pyplot as plt
     from sklearn.metrics import ConfusionMatrixDisplay

     def plot_feature(df, feature):
         # Plot a histogram of one of the features
         df[feature].hist()
         plt.title(f"Distribution of {feature}")
         plt.xlabel(feature)
         plt.ylabel("Frequency")
         plt.show()

     def plot_features(df):
         # Plot scatter plot of first two features.
         scatter = plt.scatter(
             df["sepal length (cm)"], df["sepal width (cm)"], c=df["species"]
         )
         plt.title("Scatter plot of the sepal features (width vs length)")
         plt.xlabel(xlabel="sepal length (cm)")
         plt.ylabel(ylabel="sepal width (cm)")
         plt.legend(
             scatter.legend_elements()[0],
             df["species_name"].unique(),
             loc="lower right",
             title="Classes",
         )
         plt.show()

     def plot_model(model, X_test, y_test):
         # Plot the confusion matrix for the model
         ConfusionMatrixDisplay.from_estimator(estimator=model, X=X_test, y=y_test)
         plt.title("Confusion Matrix")
         plt.show()
     ```

  4. Call these functions in the `__main__` block to visualize the data and model performance:

     ```python
     if __name__ == "__main__":
         iris_df = load_dataset()
         model, X_train, X_test, y_train, y_test = train(iris_df)
         accuracy = get_accuracy(model, X_test, y_test)
         print(f"Accuracy: {accuracy:.2f}")

         plot_feature(iris_df, "sepal length (cm)")
         plot_features(iris_df)
         plot_model(model, X_test, y_test)
     ```

  5. Update `requirements.txt` if new dependencies are required:

     ```
     matplotlib
     ```

  6. Add, commit, and push your changes:

     ```bash
     git add iris_pipeline.py requirements.txt
     git commit -m "Add data visualization functions and update script"
     git push origin feature/add-visualizations
     ```

  7. Open a pull request on GitHub to merge your changes into the `main` branch and review it before merging.




