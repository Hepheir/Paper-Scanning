from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar100

model = load_model('../models/CIFAR-100/empty.model')
model.summary()

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print(f'''
download complete:
trains: {x_train.shape},
tests: {x_test.shape}
''')

# About batch_size: https://nittaku.tistory.com/293
model.fit(x_train, y_train, epochs=250, batch_size=1)
model.evaluate(x_test, y_test)
model.save('./models/CIFAR-100/trained.model')