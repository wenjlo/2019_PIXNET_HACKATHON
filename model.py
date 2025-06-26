import tensorflow.compat.v1 as tf
import os,shutil,time
import numpy as np
import warnings
warnings.filterwarnings("ignore")
class base_model:
    def __init__(self, n_title, n_tf_idf):
        self.n_title = n_title
        self.n_tf_idf = n_tf_idf
        self.dim = 32
        self.learning_rate = 0.0001

        graph = tf.Graph()
        with graph.as_default():
            with tf.variable_scope("inputs"):
                # pixnet title
                self.title = tf.placeholder(tf.int32, [None, None], name="title")
                self.title_len = tf.placeholder(tf.int32, [None], name="title_len")
                # pixnet tf_idf
                self.tf_idf = tf.placeholder(tf.int32, [None, None], name="tf_idf")
                self.tf_idf_len = tf.placeholder(tf.int32, [None], name="tf_idf_len")

            with tf.variable_scope("embedding"):
                emb_init_fn = tf.glorot_uniform_initializer()
                self.w_title = tf.Variable(emb_init_fn(shape=[self.n_title, self.dim]), name="w_title")
                self.w_tf_idf = tf.Variable(emb_init_fn(shape=[self.n_tf_idf, self.dim]), name="w_tf_idf")
                # title embedding with l2_normalize :[X1*w1,X2*w2,....../sqrt(w1**2+w2**2+....)]
                self.title_emb = tf.nn.embedding_lookup(self.w_title, self.title)
                title_mask = tf.expand_dims(tf.nn.l2_normalize(tf.to_float(tf.sequence_mask(self.title_len)), 1), -1)
                self.title_emb = tf.reduce_sum(self.title_emb * title_mask, 1)

                # tf_idf embedding with l2_normalize :[X1*w1,X2*w2,....../sqrt(w1**2+w2**2+....)]
                self.tf_idf_emb = tf.nn.embedding_lookup(self.w_tf_idf, self.tf_idf)
                tf_idf_mask = tf.expand_dims(tf.nn.l2_normalize(tf.to_float(tf.sequence_mask(self.tf_idf_len)), 1), -1)
                self.tf_idf_emb = tf.reduce_sum(self.tf_idf_emb * tf_idf_mask, 1)

                # concat [title_emb,tf_idf_emb]
                self.layer_inputs = tf.concat([self.title_emb, self.tf_idf_emb], axis=1)
        self.graph = graph


class build_moel(object):
    def __init__(self, n_title, n_tf_idf, n_target, target_name, modelDir):
        # super(build_moel,self).__init__(n_title=max_n_title,n_tf_idf=max_n_content)

        self.n_title = n_title
        self.n_tf_idf = n_tf_idf
        self.dim = 32
        self.learning_rate = 0.0001

        self.target_name = target_name
        self.n_target = n_target
        self.target_name = target_name
        self.modelDir = modelDir
        init_fn = tf.glorot_uniform_initializer()
        graph = tf.Graph()
        with graph.as_default():
            with tf.variable_scope("inputs"):
                # pixnet title
                self.title = tf.placeholder(tf.int32, [None, None], name="title")
                self.title_len = tf.placeholder(tf.int32, [None], name="title_len")
                # pixnet tf_idf
                self.tf_idf = tf.placeholder(tf.int32, [None, None], name="tf_idf")
                self.tf_idf_len = tf.placeholder(tf.int32, [None], name="tf_idf_len")

            with tf.variable_scope("embedding"):
                emb_init_fn = tf.glorot_uniform_initializer()
                self.w_title = tf.Variable(emb_init_fn(shape=[self.n_title, self.dim]), name="w_title")
                self.w_tf_idf = tf.Variable(emb_init_fn(shape=[self.n_tf_idf, self.dim]), name="w_tf_idf")
                # title embedding with l2_normalize :[X1*w1,X2*w2,....../sqrt(w1**2+w2**2+....)]
                self.title_emb = tf.nn.embedding_lookup(self.w_title, self.title)
                title_mask = tf.expand_dims(tf.nn.l2_normalize(tf.to_float(tf.sequence_mask(self.title_len)), 1), -1)
                self.title_emb = tf.reduce_sum(self.title_emb * title_mask, 1)

                # tf_idf embedding with l2_normalize :[X1*w1,X2*w2,....../sqrt(w1**2+w2**2+....)]
                self.tf_idf_emb = tf.nn.embedding_lookup(self.w_tf_idf, self.tf_idf)
                tf_idf_mask = tf.expand_dims(tf.nn.l2_normalize(tf.to_float(tf.sequence_mask(self.tf_idf_len)), 1), -1)
                self.tf_idf_emb = tf.reduce_sum(self.tf_idf_emb * tf_idf_mask, 1)

                # concat [title_emb,tf_idf_emb]
                self.layer_inputs = tf.concat([self.title_emb, self.tf_idf_emb], axis=1)

            with tf.variable_scope("target"):
                self.target = tf.placeholder(tf.int32, [None, self.n_target], name=self.target_name)

            with tf.variable_scope("layers"):
                layer1 = tf.layers.dense(self.layer_inputs, 256, kernel_initializer=init_fn, activation=tf.nn.relu)
                layer2 = tf.layers.dense(layer1, 128, kernel_initializer=init_fn, activation=tf.nn.relu)
                self.output = tf.layers.dense(layer2, self.n_target)
                self.pred = tf.nn.softmax(self.output)
                self.pred = tf.one_hot(tf.nn.top_k(self.pred).indices, tf.shape(self.pred)[1])

            with tf.variable_scope("loss"):
                self.loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=self.target, logits=self.output))

            with tf.variable_scope("train"):
                self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
                correct_pred = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.target, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            self.saver = tf.train.Saver(tf.global_variables())
            self.graph = graph

    def fit(self, sess, trainGen, testGen, reset=False, nEpoch=50):

        sess.run(tf.global_variables_initializer())
        if reset:
            print("reset model: clean model dir: {} ...".format(self.modelDir))
            self.resetModel(self.modelDir)
        # try: 試著重上次儲存的model再次training
        self.ckpt(sess, self.modelDir)

        start = time.time()
        print("%s\t%s\t%s\t%s\t%s\t%s" % (
        "Epoch", "Train Error", "Train Accuracy", "Val Error", "Val Accuracy", "Elapsed Time"))
        minLoss = 1e7
        for ep in range(1, nEpoch + 1):
            tr_loss, tr_accuracy = [], []
            for i, (data, target) in enumerate(trainGen(), 1):
                loss, accuracy, _ = sess.run([self.loss, self.accuracy, self.train_op],
                                             feed_dict=self.feed_dict(data, target, mode="train"))
                # tt = self.predict(sess,data)
                # print(tt)
                tr_loss.append(loss)
                tr_accuracy.append(accuracy)
                print("\rtrain loss: %.3f,accuracy:%.3f" % (loss, accuracy), end="")

            if testGen is not None:
                epochLoss = self.epochLoss(sess, testGen)

            tpl = "\r%02d\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f secs"

            if minLoss > epochLoss[0]:
                tpl += ", saving ..."
                self.saver.save(sess, os.path.join(self.modelDir, 'model'), global_step=ep)
                minLoss = epochLoss[0]

            end = time.time()
            print(tpl % (ep, np.mean(tr_loss), np.mean(tr_accuracy), epochLoss[0], epochLoss[1], end - start))
            start = end
        return self

    def epochLoss(self, sess, dataGen):
        totLoss, totAcc = [], []
        for data, target in dataGen():
            lossTensor = self.loss
            accTensor = self.accuracy
            loss = sess.run(lossTensor, feed_dict=self.feed_dict(data, target, mode="eval"))
            acc = sess.run(accTensor, feed_dict=self.feed_dict(data, target, mode="eval"))

            totLoss.append(loss)
            totAcc.append(acc)
        return np.mean(totLoss), np.mean(totAcc)

    def feed_dict(self, data, target, mode="train"):
        ret = {
            self.title: data["title_tf_idf"],
            self.title_len: data["title_tf_idf_len"],
            self.tf_idf: data["content_tf_idf"],
            self.tf_idf_len: data["content_tf_idf_len"]
        }
        if mode == "train" or mode == "eval":
            ret[self.target] = target
        elif mode == "predict":
            pass
        else:
            print('error mode!!!')
        return ret

    def ckpt(self, sess, modelDir):
        """load latest saved model"""
        latestCkpt = tf.train.latest_checkpoint(modelDir)
        if latestCkpt:
            self.saver.restore(sess, latestCkpt)
        return latestCkpt

    def predict(self, sess, data):
        self.ckpt(sess, self.modelDir)
        ret = self.feed_dict(data, None, mode="predict")
        print(ret)
        return sess.run(self.output, feed_dict=ret)

    def predict(self, sess, data):
        self.ckpt(sess, self.modelDir)
        return sess.run(self.pred, feed_dict={
            self.title: data["title_tf_idf"],
            self.title_len: data["title_tf_idf_len"],
            self.tf_idf: data["content_tf_idf"],
            self.tf_idf_len: data["content_tf_idf_len"]
        })

    def resetModel(self, modelDir):

        shutil.rmtree(path=modelDir, ignore_errors=True)
        os.makedirs(modelDir)