import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Embedding, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from email.header import decode_header
from email.mime.text import MIMEText
import imaplib
import email
import smtplib
from imapclient import IMAPClient
from sklearn.metrics import classification_report

def preprocess_text(text):
    if isinstance(text, str):
        text = BeautifulSoup(text, "html.parser").get_text()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()
    return text
def load_and_preprocess_data(csv_file, encoding='utf-8'):
    data = pd.read_csv(csv_file, encoding=encoding)
    data = data.dropna(subset=['text'])
    data['text'] = data['text'].apply(preprocess_text)
    return data
def delete(EMAIL, PASSWORD, IMAP_SERVER, IMAP_PORT=993):
    mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
    mail.login(EMAIL, PASSWORD)
    mail.select('inbox')
    status, data = mail.search(None, 'SEEN')
    most_recent_email_id = data[0].split()[-1]
    status, message_data = mail.fetch(most_recent_email_id, '(RFC822)')
    raw_email = message_data[0][1]
    msg = email.message_from_bytes(raw_email)
    subject = decode_header(msg['Subject'])[0][0]
    sender = decode_header(msg['From'])[0][0]
    if isinstance(subject, bytes):
        subject = subject.decode()
    if isinstance(sender, bytes):
        sender = sender.decode()
    print('Subject:', subject)
    print('From:', sender)
    for part in msg.walk():
        if part.get_content_type() == "text/plain":
            content = part.get_payload(decode=True)
            charset = part.get_content_charset()
            if charset:
                content = content.decode(charset)
            else:
                content = content.decode()
    mail.store(most_recent_email_id, '+FLAGS', '\\Deleted')
    mail.expunge()
def send_confirmation_email(username, password, recipient, message):
    msg = MIMEText(message)
    msg['From'] = username
    msg['To'] = recipient
    msg['Subject'] = "Emailsink"
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(username, password)
        server.send_message(msg)
def process_unseen_email(email_text, msgid, username, password):
    preprocessed_text = preprocess_text(email_text)
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    padded_sequence = pad_sequences(sequence, maxlen=maxlen)
    class_probabilities = model.predict(padded_sequence)
    predicted_class_index = np.argmax(class_probabilities)
    predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]
    print("Predicted spam type:", predicted_class)
    if predicted_class == "Harassment" or predicted_class == "Suspicious" or predicted_class == "Fraudulent" or predicted_class == "spam":
        delete(EMAIL, PASSWORD, IMAP_SERVER, IMAP_PORT=993)
        send_confirmation_email(username, password, username, f"This email is {predicted_class} and has been deleted.")
    elif predicted_class == "ham":
        return
ham_data = load_and_preprocess_data("ham.csv", encoding='latin-1')
suspicious_data = load_and_preprocess_data("suspicious.csv", encoding='latin-1')
harassment_data = load_and_preprocess_data("harassment.csv", encoding='latin-1')
fraudulent_data = load_and_preprocess_data("fraudulent.csv", encoding='latin-1')
all_data = pd.concat([suspicious_data, harassment_data, fraudulent_data, ham_data], ignore_index=True)
max_words = 1000
tokenizer = Tokenizer(num_words=max_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True)
tokenizer.fit_on_texts(all_data['text'])
sequences = tokenizer.texts_to_sequences(all_data['text'])
maxlen = 50
X = pad_sequences(sequences, maxlen=maxlen)
y = all_data['label']
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
embedding_dim = 50
model = Sequential([
    Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=maxlen),
    SpatialDropout1D(0.2),
    LSTM(units=32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
    GRU(units=32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
    LSTM(units=32, dropout=0.2, recurrent_dropout=0.2),
    Dense(4, activation='softmax') 
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
epochs = 50
batch_size = 32
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
target_names = label_encoder.classes_
print(classification_report(y_test, y_pred_classes, target_names=target_names))
IMAP_SERVER = 'imap.gmail.com'  
EMAIL = 'teamspam1010@gmail.com' 
PASSWORD = 'zjlezvhaysqtpumi' 
mail = imaplib.IMAP4_SSL(IMAP_SERVER)
mail.login(EMAIL, PASSWORD)
mail.select('inbox')
status, data = mail.search(None, 'UNSEEN')
if status == 'OK':
    for num in data[0].split():
        status, message_data = mail.fetch(num, '(RFC822)')
        if status == 'OK':
            raw_email = message_data[0][1]
            msg = email.message_from_bytes(raw_email)
            subject = decode_header(msg['Subject'])[0][0]
            from_ = decode_header(msg['From'])[0][0]
            if isinstance(subject, bytes):
                subject = subject.decode()
            if isinstance(from_, bytes):
                from_ = from_.decode()            
            print(f"Subject: {subject}")
            print(f"From: {from_}")
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get('Content-Disposition'))
                if content_type == 'text/plain' and 'attachment' not in content_disposition:
                    body = part.get_payload(decode=True)
                    charset = part.get_content_charset()
                    
                    if charset:
                        body = body.decode(charset)
                    print("Body:")
                    process_unseen_email(body, num, EMAIL, PASSWORD)
                    print(body)
                    print("-" * 50)
                    
else:
    print("Failed to retrieve emails.")
mail.close()
mail.logout()
