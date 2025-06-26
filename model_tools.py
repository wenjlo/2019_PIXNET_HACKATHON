import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
from collections import OrderedDict
preproc = pickle.load(open('./encoder/preproc.pickle', 'rb'))
def model_predict(model, data, target_name='place', model_path="./models/place_model"):
    """
    model: model 物件
    data : 欲預測的資料 (pd.Dataframe)
    data cols : 欲預測的資料欄位 (title_tf_idf,content_tf_idf)
    target_name : 要預測的目標 (place or property)
    model_path : 模型的路徑資料夾 (place_model or property_model)
    """
    preproc = pickle.load(open('./encoder/preproc.pickle', 'rb'))
    for i, (data) in enumerate(dataFn(data, None, n_batch=data.shape[0], shuffle=True)(), 2):

        break

    with tf.Session(graph=model.graph) as sess:
        model.ckpt(sess, model_path)
        pred = np.squeeze(model.predict(sess, data), 1)
        pred = preproc[target_name].inverse_transform(pred)
    return pred


def do_multi(df, multi_cols, target_col):
    pad = tf.keras.preprocessing.sequence.pad_sequences
    ret = OrderedDict()
    for colname, col in df.iteritems():
        if colname in multi_cols:
            lens = col.map(len)
            ret[colname] = list(pad(col, padding="post", maxlen=lens.max()))
            ret[colname + "_len"] = lens.values
        else:
            ret[colname] = col.values
    if target_col is None:
        return ret
    else:
        return ret, np.concatenate(ret[target_col])


def get_minibatches_idx(n, batch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32")
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    minibatch_start = 0
    for i in range(n // batch_size):
        minibatches.append(idx_list[minibatch_start: minibatch_start + batch_size])
        minibatch_start += batch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])
    return minibatches


def dataFn(data, target_col, n_batch=128, shuffle=False):
    def fn():
        dataInner = data.copy()
        indices = get_minibatches_idx(len(dataInner), n_batch, shuffle=shuffle)
        for ind in indices:
            yield do_multi(dataInner.iloc[ind], ["title_tf_idf", "content_tf_idf"], target_col)

    return fn