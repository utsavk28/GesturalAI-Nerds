import pickle

model = pickle.load(open('./rf_model.sav', 'rb'))
le = pickle.load(open('./rf_le.sav','rb'))