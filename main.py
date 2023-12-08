import numpy as np
import pandas as pd
import seaborn as sns
import scikitplot as skplt
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Embedding, LSTM, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

data = pd.read_csv('/content/malicious_phish.csv')

le = LabelEncoder()
data['type_encoded'] = le.fit_transform(data['type'])


X_train, X_test, y_train, y_test = train_test_split(data['url'], data['type_encoded'], test_size=0.25, shuffle=True, random_state=42)


tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
vocab_size = len(tokenizer.word_index) + 1


maxlen = 100 # or some other appropriate value
X_train_padded = pad_sequences(X_train_seq, maxlen=maxlen)
X_test_padded = pad_sequences(X_test_seq, maxlen=maxlen)

model = Sequential(name="Cnn-Lstm_model")
model.add(Embedding(input_dim=vocab_size, output_dim=8,
input_length=maxlen, name='layer_embedding'))
model.add(BatchNormalization())
model.add(Conv1D(filters = 32, kernel_size = 9, padding = 'same', activation = 'relu'))
model.add(MaxPooling1D(pool_size = 2))
model.add(LSTM(units=512, return_sequences=False, dropout=0.2))
model.add(Dense(units=4, activation='softmax'))
model.summary()

optimizer = Adam(learning_rate=.0001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

y_train_onehot = to_categorical(y_train)
y_test_onehot = to_categorical(y_test)

history = model.fit(X_train_padded, y_train_onehot, validation_split=0.2, epochs=150, batch_size=512)

model.save('categorical_model.h5')

y_pred = np.argmax(model.predict(X_test_padded), axis=-1)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
