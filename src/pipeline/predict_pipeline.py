import numpy as np
import pandas as pd
import pickle

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def predict(sample):
    model = load_pickle("artifact/model.pkl")
    preprocessor = load_pickle("artifact/preprocessor.pkl")
    label_encoder = load_pickle("artifact/label_encoder.pkl")

    # Create DataFrame with 60 columns (like during training)
    input_df = pd.DataFrame([sample], columns=[str(i) for i in range(60)])

    # Transform and predict
    transformed_input = preprocessor.transform(input_df)
    prediction = model.predict(transformed_input)
    label = label_encoder.inverse_transform(prediction.astype(int))
    return label[0]

# Example input with 60 values
sample_input = [
    0.0200, 0.0371, 0.0428, 0.0207, 0.0954, 0.0986, 0.1539, 0.1601, 0.3109, 0.2111,
    0.1609, 0.1582, 0.2238, 0.0645, 0.0660, 0.2273, 0.3100, 0.2999, 0.5078, 0.4797,
    0.5783, 0.5071, 0.4328, 0.5550, 0.6711, 0.6415, 0.7104, 0.8080, 0.6791, 0.3857,
    0.1307, 0.2604, 0.5121, 0.7547, 0.8537, 0.8507, 0.6692, 0.6097, 0.4943, 0.2744,
    0.0510, 0.2834, 0.2825, 0.4256, 0.2641, 0.1386, 0.1051, 0.1343, 0.0383, 0.0324,
    0.0232, 0.0027, 0.0065, 0.0159, 0.0072, 0.0167, 0.0180, 0.0084, 0.0090, 0.0032
]

print("Predicted label:", predict(sample_input))
