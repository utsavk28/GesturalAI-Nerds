from tensorflow import keras

resnet_model = keras.models.load_model("./resnet_model.h5")
mobilenet_model = keras.models.load_model("./mobilenet_model.h5")
