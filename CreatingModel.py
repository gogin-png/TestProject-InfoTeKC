import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Загрузка данных
data = pd.read_csv('/home/denis/PycharmProjects/TestProject-InfoTeKC-/DGA_domains_dataset-master/dga_domains_full.csv', header=None, names=['class', 'label', 'domain'])

# 2. Предобработка данных
# Преобразование меток в бинарные (0 - легитимный, 1 - DGA)
data['class'] = LabelEncoder().fit_transform(data['class'])


# Используем Tokenizer для преобразования доменных имен в числовые последовательности
tokenizer = Tokenizer(char_level=True)  # char_level=True означает, что рассматриваем отдельные символы
tokenizer.fit_on_texts(data['domain'])
sequences = tokenizer.texts_to_sequences(data['domain'])

# Дополнение или усечение последовательностей до одинаковой длины
max_length = max([len(seq) for seq in sequences])
print(max_length)
X = pad_sequences(sequences, maxlen=max_length, padding='post')
y = data['class'].values

# 3. Разделение данных на тренировочные и тестовые
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Создание модели
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_length))
model.add(Bidirectional(LSTM(64, return_sequences=False)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 5. Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. Обучение модели
model.fit(X_train, y_train, epochs=8, batch_size=64, validation_data=(X_test, y_test))
model.save('my_model-2.keras')

# 7. Оценка модели
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")


