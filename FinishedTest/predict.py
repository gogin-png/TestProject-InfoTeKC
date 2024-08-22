import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# 1. Загрузка данных из файла
file_path = 'test.csv'
df = pd.read_csv(file_path, header=None, names=['domain'])

# 2. Предобработка данных
# Используем Tokenizer для преобразования доменных имен в числовые последовательности
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(df['domain'])
sequences = tokenizer.texts_to_sequences(df['domain'])

# Дополнение или усечение последовательностей до длины, использованной при обучении
max_length = 73
X = pad_sequences(sequences, maxlen=max_length, padding='post')

# 3. Загрузка модели
model = load_model('my_model.keras')

# 4. Предсказание
y_pred = model.predict(X)
y_pred_labels = (y_pred > 0.5).astype(int).flatten()

# 5. Запись предсказаний в prediction.csv
df_predictions = pd.DataFrame({'domain': df['domain'], 'is_dga': y_pred_labels})
df_predictions.to_csv('prediction.csv', index=False)

# 6. вывод результатов для каждого домена
for domain, pred_label in zip(df['domain'], y_pred_labels):
    class_label = 'Legit' if pred_label == 0 else 'DGA'
    print(f"Domain: {domain}, Predicted: {class_label}")
