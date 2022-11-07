
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf

class DSSM(object):

    def __init__(self) -> None:
        pass

    def model(self):
        """
        DSSM双塔模型召回特点
            1.特征embedding向量维度可以不同
            2.embedding数据拼接后经过mlp层
            3.点积交叉前需要归一化
            4.点击交叉后需要加入温度系数衰减, 最后再经过sigmoid
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

        concat_user_data = tf.keras.layers.Concatenate(axis=-1)([emb_user_id_data, emb_user_fea1_data]) 
        dense_user_layer = tf.keras.layers.Dense(32, activation='relu', name='user_dense_layer_1')
        dense_user_data = dense_user_layer(concat_user_data)
        user_vec = tf.nn.l2_normalize(dense_user_data, axis=-1) 
        user_vec = tf.squeeze(user_vec, axis=1) 

        concat_item_data = tf.keras.layers.Concatenate(axis=-1)([emb_item_id_data, emb_item_fea1_data]) 

        concat_item_data = tf.keras.layers.Concatenate(axis=-1)([emb_item_id_data, emb_item_fea1_data]) 
        dense_item_layer = tf.keras.layers.Dense(32, activation='relu', name='item_dense_layer_1')
        dense_item_data = dense_item_layer(concat_item_data)
        item_vec = tf.nn.l2_normalize(dense_item_data, axis=-1) 
        item_vec = tf.squeeze(item_vec, axis=1) 

        temperature = 0.05
        user_vec /= temperature
        output = tf.linalg.matmul(user_vec, item_vec, transpose_b=True, name='output')
       
        user_input_list = [input_user_id, input_user_fea1]
        item_input_list = [input_item_id, input_item_fea1]
        model = tf.keras.Model(inputs=user_input_list + item_input_list, outputs=output)

        model.__setattr__("user_input", user_input_list)
        model.__setattr__("user_embedding", user_vec)

        model.__setattr__("item_input", item_input_list)
        model.__setattr__("item_embedding", item_vec)

        return model

def sampledsoftmaxloss(y_true, y_pred):
    labels = tf.linalg.diag(y_true)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y_pred)
    return loss

ins = DSSM()
model = ins.model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=sampledsoftmaxloss,
)
model.summary()