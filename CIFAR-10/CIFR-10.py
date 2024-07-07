import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# CIFAR-10データセットをロードして、トレーニングデータとテストデータに分割
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# データの前処理：画像を正規化（0-255のピクセル値を0-1の範囲に変換）
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# CNNモデルの構築
cnn_model = models.Sequential()
cnn_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
cnn_model.add(layers.MaxPooling2D((2, 2)))
cnn_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
cnn_model.add(layers.MaxPooling2D((2, 2)))
cnn_model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# フラット化して全結合層に接続
cnn_model.add(layers.Flatten())
cnn_model.add(layers.Dense(64, activation='relu'))
cnn_model.add(layers.Dense(10, activation='softmax'))

# CNNモデルのコンパイル
cnn_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# CNNモデルのトレーニング
cnn_history = cnn_model.fit(train_images, train_labels, epochs=10, 
                            validation_data=(test_images, test_labels))

# DNNモデルの構築
dnn_model = models.Sequential()
dnn_model.add(layers.Flatten(input_shape=(32, 32, 3)))
dnn_model.add(layers.Dense(512, activation='relu'))
dnn_model.add(layers.Dense(256, activation='relu'))
dnn_model.add(layers.Dense(128, activation='relu'))
dnn_model.add(layers.Dense(10, activation='softmax'))

# DNNモデルのコンパイル
dnn_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# DNNモデルのトレーニング
dnn_history = dnn_model.fit(train_images, train_labels, epochs=10, 
                            validation_data=(test_images, test_labels))

# CNNモデルの評価
cnn_test_loss, cnn_test_acc = cnn_model.evaluate(test_images, test_labels, verbose=2)
print(f'\nCNN Test accuracy: {cnn_test_acc}')

# DNNモデルの評価
dnn_test_loss, dnn_test_acc = dnn_model.evaluate(test_images, test_labels, verbose=2)
print(f'\nDNN Test accuracy: {dnn_test_acc}')

# トレーニングと検証の精度をプロット
plt.plot(cnn_history.history['accuracy'], label='CNN accuracy')
plt.plot(cnn_history.history['val_accuracy'], label='CNN val_accuracy')
plt.plot(dnn_history.history['accuracy'], label='DNN accuracy')
plt.plot(dnn_history.history['val_accuracy'], label='DNN val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.0, 1])
plt.xticks(range(0, 11))  # 0から10までの範囲で1単位の目盛りを設定
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy on CIFAR-10')
plt.show()