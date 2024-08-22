import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# 1. Загрузка данных из файла
file_path = '/home/denis/PycharmProjects/TestProject-InfoTeKC-/DGA_domains_dataset-master/val.csv'
df = pd.read_csv(file_path, header=None, names=['domain', 'class'])

# 2. Предобработка данных (тут не нужна)
# Преобразование меток в бинарные (0 - легитимный, 1 - DGA)
# df['class'] = LabelEncoder().fit_transform(df['class'])

# 3.Используем Tokenizer для преобразования доменных имен в числовые последовательности
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(df['domain'])
sequences = tokenizer.texts_to_sequences(df['domain'])

# 4.Дополнение или усечение последовательностей до длины, использованной при обучении
max_length = 73
X = pad_sequences(sequences, maxlen=max_length, padding='post')
y_true = df['class'].values

# 5. Загрузка модели
model = load_model('/home/denis/PycharmProjects/TestProject-InfoTeKC-/my_model_3_finetuned.keras')

# 6. Предсказание
y_pred = model.predict(X)
y_pred_labels = (y_pred > 0.5).astype(int).flatten()

# 7. Оценка модели
accuracy = accuracy_score(y_true, y_pred_labels)
precision = precision_score(y_true, y_pred_labels)
recall = recall_score(y_true, y_pred_labels)
f1 = f1_score(y_true, y_pred_labels)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred_labels).ravel()

# 8. Запись метрик в validation.txt
with open('validation.txt', 'w') as f:
    f.write(f"True positive: {tp}\n")
    f.write(f"False positive: {fp}\n")
    f.write(f"False negative: {fn}\n")
    f.write(f"True negative: {tn}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1: {f1:.4f}\n")

# 9. вывод результатов для каждого домена
for domain, true_label, pred_label, clas in zip(df['domain'], y_true, y_pred_labels, df['class']):
    class_label = 'Legit' if pred_label == 0 else 'DGA'
    class1 = "DGA" if clas == 1 else "Legit"
    print(f"Domain: {domain}, True: {true_label}, Predicted: {class_label}, {class1}")
