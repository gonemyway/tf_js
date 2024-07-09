import tensorflow as tf

# Load the model from the HDF5 file
model_path = "models/best.hdf5"
model = tf.keras.models.load_model(model_path)

# Convert the model to JSON
model_json = model.to_json()

# Save the JSON to a file
json_path = "models/model.json"
with open(json_path, "w") as json_file:
    json_file.write(model_json)

print("Model has been converted to JSON and saved to", json_path)