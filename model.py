import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib  # For saving the model

# Load the dataset
df = pd.read_csv('/home/kaderavan/Desktop/base_log/prepared_network_data.csv')

# Preprocessing
df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
df.fillna(df.mean(numeric_only=True), inplace=True)

# Ensure 'Label' is present and encode it
if 'Label' in df.columns:
    label_encoder = LabelEncoder()
    df['Label'] = label_encoder.fit_transform(df['Label'])
else:
    print("Label column not found!")

# Filter numeric features
numeric_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
if 'Label' in numeric_features:
    numeric_features.remove('Label')

X = df[numeric_features]
y = df['Label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Output evaluation
print("Accuracy of the model:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save the model
joblib.dump(model, '/home/kaderavan/Desktop/base_log/trained_model.pkl')
print("Model saved as 'trained_model.pkl'")
