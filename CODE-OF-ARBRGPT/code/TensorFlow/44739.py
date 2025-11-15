from tensorflow import keras

model_dir = 'C:\\IncorrectPathToSavedModel'

model = keras.models.load_model(model_dir)