
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf


class FM(object):

    def __init__(self):
        pass

    def model(self):
        """
        FM召回模型的特点：
            1.所有特征的embedding维度相同
            2.用户向量是 所有特征emb求和归一后得到的, 内容向量也类似
        """

        input_user_id = tf.keras.layers.Input(shape=1, name='input_user_id') 
        input_user_fea1 = tf.keras.layers.Input(shape=1, name='input_user_fea1') 

        input_item_id = tf.keras.layers.Input(shape=1, name='input_item_id') 
        input_item_fea1 = tf.keras.layers.Input(shape=1, name='input_item_fea1') 

        emb_user_id = tf.keras.layers.Embedding(input_dim=10000, output_dim=32, name='emb_user_id')
        emb_user_id_data = emb_user_id(input_user_id)
        emb_user_fea1 = tf.keras.layers.Embedding(input_dim=100, output_dim=32, name='emb_user_fea1')
        emb_user_fea1_data = emb_user_fea1(input_user_fea1)

        emb_item_id = tf.keras.layers.Embedding(input_dim=20000, output_dim=32, name='emb_item_id')
        emb_item_id_data = emb_item_id(input_item_id)
        emb_item_fea1 = tf.keras.layers.Embedding(input_dim=200, output_dim=32, name='emb_item_fea1')
        emb_item_fea1_data = emb_item_fea1(input_item_fea1)

        concat_user_data = tf.keras.layers.Concatenate(axis=1)([emb_user_id_data, emb_user_fea1_data])
        concat_item_data = tf.keras.layers.Concatenate(axis=1)([emb_item_id_data, emb_item_fea1_data])

        sum_user_data = tf.math.reduce_sum(concat_user_data, axis=1, keepdims=True)
        sum_item_data = tf.math.reduce_sum(concat_item_data, axis=1, keepdims=True)

        norm_user_data = tf.math.l2_normalize(sum_user_data, axis=-1)
        norm_item_data = tf.math.l2_normalize(sum_item_data, axis=-1)

        score = tf.math.reduce_sum(tf.math.multiply(norm_user_data, norm_item_data), axis=1, name='score', keepdims=True)
        temperature = 0.5
        score /= temperature

        output = tf.sigmoid(score, name='output')


        user_input_list = [input_user_id, input_user_fea1]
        item_input_list = [input_item_id, input_item_fea1]
        model = tf.keras.Model(inputs=user_input_list + item_input_list, outputs=output)

        model.__setattr__("user_input", user_input_list)
        model.__setattr__("user_embedding", norm_user_data)

        model.__setattr__("item_input", item_input_list)
        model.__setattr__("item_embedding", norm_item_data)

        model.__setattr__('score', score)

        return model

ins = FM()
model = ins.model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(),
)
model.summary()


