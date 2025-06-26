import pandas as pd
import numpy as np
import warnings
from model import build_moel
from model_tools import model_predict,preproc
warnings.filterwarnings('ignore')
#title's string catgory
MAX_TITLE = 7975

#content's string category
MAX_CONTENT = 77490

#n target
N_PLACE = 55
N_PROPERTY = 79

# reading test data
data = pd.read_csv('./data/sample_data.csv')
data = data[['place_name','property_type','title_tf_idf','tf_idf']]
data = data.dropna()
data['property_type'] = data['property_type'].map(lambda x : x.replace(' ',''))
data['place_name'] = data['place_name'].map(lambda x : x.replace(' ',''))
print(data.columns)


# preproc_data
def preproc_data(data):
    preproc_data = pd.DataFrame()
    title,content,place,property_type = [],[],[],[]

    for i,v in data.iterrows():
        #print(v.title_tf_idf)
        title.append(v.title_tf_idf.split('@'))
        content.append(v.tf_idf.split('@'))
        place.append(v.place_name)
        property_type.append(v.property_type)
    preproc_data = pd.DataFrame([title, content, place, property_type]).T
    preproc_data.columns = ['title_tf_idf', 'content_tf_idf', 'place_name', 'property_type']
    return preproc_data

# encode_data
def encode_labels(data):
    teProcessed = pd.DataFrame()
    for _,v in data.iterrows():
        result = []
        result.append(np.sort(preproc['title'].transform(v['title_tf_idf'])))
        result.append(np.sort(preproc['content'].transform(v['content_tf_idf'])))
        result.append(preproc['place'].transform([[v['place_name']]]).toarray())
        result.append(preproc['property'].transform([[v['property_type']]]).toarray())
        teProcessed = teProcessed.append(pd.DataFrame([result],columns=['title_tf_idf','content_tf_idf','place','property']))
    return teProcessed



# 地區模型
place_model = build_moel(n_title = MAX_TITLE ,
                   n_tf_idf=MAX_CONTENT,
                   n_target=N_PLACE,
                   target_name="place",
                   modelDir="./models/place_model")
# 飯店類別模型
property_model = build_moel(n_title = MAX_TITLE ,
                   n_tf_idf=MAX_CONTENT,
                   n_target=N_PROPERTY,
                   target_name="property",
                   modelDir="./models/property_model")
# 資料
test_data = encode_labels(preproc_data(data))
print(test_data.columns)
# 結果
place_result = model_predict(place_model, test_data, target_name='place', model_path="place_model")
property_result = model_predict(property_model, test_data, target_name='property', model_path="property_model")

print('place_result:\n',place_result)
print('property_resut:\n',property_result)

