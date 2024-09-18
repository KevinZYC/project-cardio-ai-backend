import numpy as np
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="calories_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare the input data (same as before)
data = [0, 19, 195, 88, 40, 140, 38]
data = np.array(data, dtype=np.float32).reshape(1, -1)

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], data)

# Run the model
interpreter.invoke()

# Get the output prediction
output_data = interpreter.get_tensor(output_details[0]['index']).tolist()

# Print the result
print(output_data)
