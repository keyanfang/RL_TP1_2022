import load
import tensorflow as tf

def models():
    inputL=tf.keras.layers.Input((2720,))
    x = tf.keras.layers.Dense(units=256,activation=tf.nn.relu)(inputL)
    x = tf.keras.layers.Dense(units=3, activation=tf.nn.softmax)(x)
    model = tf.keras.Model(inputL,x)

def scheduler(epoch):
  if epoch < 10:
    return 1e-4
  else:
    return 1e-5

def train_agent(save_path):
    x1,y1,x2,y2=load.load()
    newModel = models()
    newModel.compile(loss='catelorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    callback=tf.keras.callbacks.LearningRateScheduler(scheduler())
    newModel.fit(x1,y1,epochs=100,callbacks=[callback])
    newModel.save(save_path)
    return 0