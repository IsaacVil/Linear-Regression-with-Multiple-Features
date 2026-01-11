import tkinter as tk
from tkinter import ttk
import re
import string
import numpy as np
from sklearn.datasets import fetch_openml
import csv
import os
from classification import normX, sigmoid, gradientDescent, gradientDescentReg

FEATURE_NAMES = [
    "Total word count",
    "ALL CAPS words",
    "ALL CAPS ratio",
    "Numeric-only words",
    "Number of links",
    "Exclamation marks (!)",
    "Exclamation marks ratio (!)",
    "Dollar symbols ($)",
    "Spam Words Count",
    "Previous spam reports"
]
DEBUG = True #Esto es solo para significar como se configuraria el tema de reincidencias de reportes de spam, es para probar como se comporta el modelo con ellas.


def emailSender(w, b):
    root = tk.Tk()
    root.title("Email Spam Predictor")
    root.geometry("800x500")
    root.resizable(False, False)

    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill="both")

    tab1 = ttk.Frame(notebook, padding=20)
    tab2 = ttk.Frame(notebook, padding=15)

    notebook.add(tab1, text="Email")
    notebook.add(tab2, text="Prediction")

    ttk.Label(
        tab1,
        text="Compose Email",
        font=("Arial", 16, "bold")
    ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 15))

    ttk.Label(tab1, text="From:").grid(row=1, column=0, sticky="w", pady=5)
    email_entry = ttk.Entry(tab1, width=50)
    email_entry.grid(row=1, column=1, sticky="w", pady=5)

    ttk.Label(tab1, text="To:").grid(row=2, column=0, sticky="w", pady=5)
    receiver_entry = ttk.Entry(tab1, width=50)
    receiver_entry.grid(row=2, column=1, sticky="w", pady=5)

    ttk.Label(tab1, text="Subject:").grid(row=3, column=0, sticky="w", pady=5)
    subject_entry = ttk.Entry(tab1, width=50)
    subject_entry.grid(row=3, column=1, sticky="w", pady=5)

    ttk.Label(tab1, text="Message:").grid(row=4, column=0, sticky="nw", pady=5)
    body_text = tk.Text(tab1, width=60, height=12, wrap="word")
    body_text.grid(row=4, column=1, sticky="w", pady=5)

    spam_reports_var = tk.IntVar(value=0)

    if DEBUG:
        ttk.Label(
            tab1,
            text="Debug – Previous spam reports:"
        ).grid(row=5, column=0, sticky="w", pady=(15, 5))

        ttk.Spinbox(
            tab1,
            from_=0,
            to=100,
            width=10,
            textvariable=spam_reports_var
        ).grid(row=5, column=1, sticky="w", pady=(15, 5))

    summary_frame = ttk.Frame(tab2)
    summary_frame.pack(fill="x")

    prob_label = ttk.Label(
        summary_frame,
        text="Spam Probability: --",
        font=("Arial", 14, "bold")
    )
    prob_label.grid(row=0, column=0, sticky="w")

    decision_label = ttk.Label(
        summary_frame,
        text="Classification: --",
        font=("Arial", 14, "bold")
    )
    decision_label.grid(row=1, column=0, sticky="w", pady=(5, 5))

    ttk.Label(
        summary_frame,
        text="Decision boundary: 0.7",
        font=("Arial", 10)
    ).grid(row=2, column=0, sticky="w")

    ttk.Separator(tab2, orient="horizontal").pack(fill="x", pady=10)

    table_frame = ttk.Frame(tab2)
    table_frame.pack(fill="both", expand=True)

    columns = ("Feature", "Value", "Weight", "Contribution")

    feature_table = ttk.Treeview(
        table_frame,
        columns=columns,
        show="headings",
        height=8
    )

    feature_table.heading("Feature", text="Feature")
    feature_table.heading("Value", text="Value")
    feature_table.heading("Weight", text="Weight")
    feature_table.heading("Contribution", text="Contribution")

    feature_table.column("Feature", width=240, anchor="w")
    feature_table.column("Value", width=80, anchor="center")
    feature_table.column("Weight", width=80, anchor="center")
    feature_table.column("Contribution", width=100, anchor="center")

    feature_table.pack(fill="both", expand=True)

    def send_email_action():
        subject = subject_entry.get()
        body = body_text.get("1.0", "end-1c")
        full_text = subject + " " + body

        reincidence = spam_reports_var.get() if DEBUG else 0

        features = extractEmail(full_text, reincidence)
        x_vec = features[0]  # shape (7,)

        prob = sigmoid(np.dot(w, x_vec) + b)
        is_spam = prob >= 0.7

        prob_label.config(text=f"Spam Probability: {prob:.4f}")

        decision_label.config(
            text=f"Classification: {'SPAM 🚨' if is_spam else 'NOT SPAM ✅'}",
            foreground="red" if is_spam else "green"
        )

        for row in feature_table.get_children():
            feature_table.delete(row)

        for i in range(len(x_vec)):
            contribution = x_vec[i] * w[i]
            feature_table.insert(
                "",
                "end",
                values=(
                    FEATURE_NAMES[i],
                    f"{x_vec[i]:.2f}",
                    f"{w[i]:.3f}",
                    f"{contribution:.3f}"
                )
            )

    ttk.Button(
        tab1,
        text="Send Email",
        command=send_email_action
    ).grid(row=6, column=1, sticky="e", pady=20)

    root.mainloop()


def extractEmail(email, spamReport):
    spam_words = [
        "free", "win", "winner", "cash", "offer", "buy",
        "cheap", "click", "urgent", "money", "promo"
    ]
    sizeOfEmail = len(email.split())
    if sizeOfEmail == 0:
        sizeOfEmail = 1
    capsNum = 0
    digitsNum = 0
    spam_word_count = 0
    for item in email.split():
        if item.isupper():
            capsNum = capsNum + 1
        if item.isdigit():
            digitsNum = digitsNum + 1
        clean_word = re.sub(r"[^a-zA-Z]", "", item.lower())
        if clean_word in spam_words:
            spam_word_count += 1
    caps_ratio = capsNum / sizeOfEmail
    linksNum = len(re.findall(r"http[s]?://", email))
    linksNum = linksNum + len(re.findall(r"www\.", email))
    exclaNum = email.count("!")
    excla_ratio = (exclaNum*2) / sizeOfEmail
    dollarsNum = email.count("$")
    featurepatern = np.array([sizeOfEmail, capsNum, caps_ratio, digitsNum, linksNum, exclaNum, excla_ratio, dollarsNum, spam_word_count, spamReport]).reshape(1, -1)
    return featurepatern

def parse_csv_xy(relative_path):
    X, y = [], []

    base_dir = os.path.dirname(__file__)  # carpeta del .py
    full_path = os.path.join(base_dir, relative_path)

    with open(full_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            X.append(row[2])
            y.append(int(row[3]))

    return X, y

if __name__ == "__main__":
    x, y = parse_csv_xy(r"dataset\spam_ham_dataset.csv")
    X_extract = None

    for i in range(len(x)):
        emailtemp = extractEmail(x[i], 0)

        if X_extract is None:
            X_extract = emailtemp
        else:
            X_extract = np.vstack((X_extract, emailtemp))

    Normx, miu, sigma = normX(X_extract)
    w_initialize = np.zeros(Normx.shape[1])
    b_initialize = 0
    alpha = 0.1
    maxIter = 100000
    epsilon = 0.00001
    lambda_ = 1
    w, b, w_Hist, b_Hist = gradientDescentReg(Normx, y, w_initialize, b_initialize, alpha, maxIter, epsilon, lambda_)
    emailSender(w, b)