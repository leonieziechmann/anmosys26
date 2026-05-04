import os
# Wichtig: Das Backend muss gesetzt werden, BEVOR Keras importiert wird!
os.environ["KERAS_BACKEND"] = "jax" 

import keras

# Erstelle einen simplen Keras Random Tensor, um das Backend zu verifizieren
random_tensor = keras.random.normal((3, 3))

print("Keras Backend aktiv!")
print("Tensor Typ:", type(random_tensor))
print(random_tensor)
