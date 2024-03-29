import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load Exercise Details Dataset
def load_exercise_details():
    exercise_details = []
    with open('ExampleDB3.csv', 'r', encoding='utf-8', errors='ignore') as file:
        reader = csv.DictReader(file, delimiter=',')
        for row in reader:
            exercise_details.append(row)
    return exercise_details

# Prepare the dataset
def prepare_dataset(exercise_details):
    df = pd.DataFrame(exercise_details)

    # Encode categorical features (muscle group, level)
    label_encoder = LabelEncoder()
    df['BodyPart'] = label_encoder.fit_transform(df['BodyPart'])
    df['Level'] = label_encoder.fit_transform(df['Level'])

    # Features and target
    X = df[['BodyPart', 'Level']]
    y = df['Title']

    return X, y, label_encoder

# Train a machine learning model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Choose a machine learning model (Random Forest)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    return model

# Get recommendations
def get_recommendations(user_muscle_group, user_fitness_level, model, label_encoder):
    user_muscle_group_encoded = label_encoder.transform([user_muscle_group])
    user_fitness_level_encoded = label_encoder.transform([user_fitness_level])

    # Predict the workout
    recommendation = model.predict([[user_muscle_group_encoded[0], user_fitness_level_encoded[0]]])

    return recommendation[0]

# Main Program
if __name__ == '__main__':
    exercise_details = load_exercise_details()
    X, y, label_encoder = prepare_dataset(exercise_details)
    model = train_model(X, y)

    print("Available muscle groups:")
    muscle_groups = set(exercise['BodyPart'] for exercise in exercise_details)
    for group in muscle_groups:
        print(group)

    user_muscle_group = input("Enter your desired muscle group: ")
    user_fitness_level = input("Enter your fitness level (Beginner, Intermediate, etc.): ")

    recommendation = get_recommendations(user_muscle_group, user_fitness_level, model, label_encoder)

    if recommendation:
        print(f"Recommended workout for {user_muscle_group} ({user_fitness_level}): {recommendation}")
    else:
        print("No recommendation available for your input.")
