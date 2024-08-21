import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# 1. Загрузка данных из файла
file_path = '/home/denis/PycharmProjects/TestProject-InfoTeKC-/DGA_domains_dataset-master/dga_domains_sample.csv'  # Укажите путь к вашему файлу
df = pd.read_csv(file_path, header=None, names=['class', 'type', 'domain'])

# 2. Предобработка данных
# Преобразование меток в бинарные (0 - легитимный, 1 - DGA)
df['class'] = LabelEncoder().fit_transform(df['class'])

# Используем Tokenizer для преобразования доменных имен в числовые последовательности
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(df['domain'])
sequences = tokenizer.texts_to_sequences(df['domain'])

# Дополнение или усечение последовательностей до длины, использованной при обучении
# Подставьте max_length из обучения, если точно известен
max_length = 73  # Замените на фактическое значение max_length из вашей модели
X = pad_sequences(sequences, maxlen=max_length, padding='post')
y_true = df['class'].values

# 3. Загрузка модели
model = load_model('/home/denis/PycharmProjects/TestProject-InfoTeKC-/my_model.keras')

# 4. Предсказание
y_pred = model.predict(X)

# 5. Оценка модели
y_pred_labels = (y_pred > 0.5).astype(int).flatten()

# 6. Расчет точности
accuracy = (y_pred_labels == y_true).mean()
print(f"Accuracy: {accuracy:.2f}")

# Опционально: вывод результатов для каждого домена
for domain, true_label, pred_label, clas in zip(df['domain'], y_true, y_pred_labels, df['class']):
    class_label = 'Legit' if pred_label else 'DGA'
    clas1 = 'DGA' if clas == 0 else 'Legit'
    print(f"Domain: {domain}, True: {true_label}, Predicted: {class_label}, {clas1}")
