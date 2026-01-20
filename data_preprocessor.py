import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """
    A reusable data preprocessing class for Machine Learning pipelines
    """

    def __init__(self, dataset_path):
        """
        Initialize with dataset path
        """
        self.dataset_path = dataset_path
        self.df = None
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def load_data(self):
        """
        Load dataset from CSV
        """
        self.df = pd.read_csv(self.dataset_path)
        print("Dataset loaded successfully")
        return self.df

    def handle_missing_values(self):
        """
        Handle missing values using median (numerical)
        and mode (categorical)
        """
        for column in self.df.columns:
            if self.df[column].dtype in ['int64', 'float64']:
                self.df[column].fillna(self.df[column].median(), inplace=True)
            else:
                self.df[column].fillna(self.df[column].mode()[0], inplace=True)

        print("Missing values handled")
        return self.df

    def encode_categorical(self):
        """
        Encode categorical columns using LabelEncoder
        """
        categorical_cols = self.df.select_dtypes(include=['object']).columns

        for col in categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le

        print("Categorical variables encoded")
        return self.df

    def scale_features(self, exclude_columns=None):
        """
        Scale numerical features using StandardScaler
        """
        if exclude_columns is None:
            exclude_columns = []

        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        numerical_cols = [col for col in numerical_cols if col not in exclude_columns]

        self.df[numerical_cols] = self.scaler.fit_transform(self.df[numerical_cols])

        print("Numerical features scaled")
        return self.df

    def split_data(self, target_column, test_size=0.2, random_state=42):
        """
        Split dataset into train and test sets
        """
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        print("Data split into train and test sets")
        return X_train, X_test, y_train, y_test

    def save_processed_data(self, output_path):
        """
        Save processed dataset to CSV
        """
        self.df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")


# ------------------ DEMONSTRATION ------------------
if __name__ == "__main__":
    preprocessor = DataPreprocessor("titanic_cleaned.csv")

    df = preprocessor.load_data()
    df = preprocessor.handle_missing_values()
    df = preprocessor.encode_categorical()
    df = preprocessor.scale_features(exclude_columns=['Survived'])

    X_train, X_test, y_train, y_test = preprocessor.split_data(
        target_column='Survived'
    )

    preprocessor.save_processed_data("titanic_processed.csv")

    print("\nPreprocessing completed successfully!")
