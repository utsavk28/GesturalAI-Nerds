from tensorflow import keras

resnet_model = keras.models.load_model("./models/resnet_model.h5")
mobilenet_model = keras.models.load_model("./models/mobilenet_model.h5")
