import json
from tensorflow import keras
from keras.models import model_from_json

resnet_model = keras.models.load_model("./resnet_model_d3.h5")
model1 = keras.models.load_model("./model1_t.h5")
model2 = keras.models.load_model("./model2_t.h5")

f = open('./model-bw.json')
res = json.loads(f.read())
f.close()
js = json.dumps(res)

model_bw = model_from_json(js)
model_bw.save_weights('./model-bw.h5')
