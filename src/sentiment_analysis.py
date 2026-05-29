import html
import os
import random
import re
from collections import Counter

import jieba
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from stopwordsiso import stopwords


RANDOM_STATE = 42

DATA_FILES = [
    "完美日记_好评.xlsx",
    "完美日记_差评.xlsx",
    "橘朵_好评.xlsx",
    "橘朵_差评.xlsx",
    "毛戈平_好评.xlsx",
    "毛戈平_差评.xlsx",
    "玛丽黛佳_好评.xlsx",
    "玛丽黛佳_差评.xlsx",
]

HTML_ENTS_PATTERN = re.compile(r"&[a-z]{2,7};")
TAG_PATTERN = re.compile(r"<[^>]+>")


def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_review_data(file_paths):
    frames = []

    for file_path in file_paths:
        try:
            df = pd.read_excel(file_path)
            filename = file_path.replace(".xlsx", "")
            brand, rating_type = filename.split("_")

            df["品牌名"] = brand
            df["评价类别"] = 1 if rating_type == "差评" else 0
            df["来源文件"] = filename

            frames.append(df)
        except Exception as exc:
            print(f"Skip {file_path}: {exc}")

    if not frames:
        raise FileNotFoundError("No valid Excel review files were loaded.")

    return pd.concat(frames, ignore_index=True)


def clean_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""

    text = html.unescape(text)
    text = TAG_PATTERN.sub(" ", text)
    text = HTML_ENTS_PATTERN.sub(" ", text)
    text = re.sub(r"[^\w\u4e00-\u9fa5]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def build_stopwords():
    cn_stop = stopwords("zh")
    custom_stopwords = {
        "hellip",
        "nbsp",
        "amp",
        "gt",
        "lt",
        "quot",
        "京东",
        "淘宝",
        "快递",
    }
    cn_stop.update(custom_stopwords)

    for word in ["完美日记", "橘朵", "毛戈平", "玛丽黛佳"]:
        jieba.add_word(word)

    return cn_stop


def tokenize(text, cn_stop):
    words = jieba.lcut(text)
    return [word for word in words if word not in cn_stop and len(word) > 1]


def prepare_dataset(raw_data):
    df = raw_data.copy()
    df.rename(columns={"评论内容": "text", "评论打分": "score"}, inplace=True)

    df.dropna(subset=["text", "score"], inplace=True)
    df.drop_duplicates(subset=["text", "评论者"], inplace=True)

    df["clean"] = df["text"].apply(clean_text)
    df = df[df["clean"].str.len() > 0].copy()

    return df


def print_top_keywords(df, cn_stop, top_n=20):
    df["tokens"] = df["clean"].apply(lambda text: tokenize(text, cn_stop))

    for label, name in [(0, "Positive"), (1, "Negative")]:
        words = [word for tokens in df[df["评价类别"] == label]["tokens"] for word in tokens]
        print(f"\nTop {name} keywords:")
        print(Counter(words).most_common(top_n))


def vectorize_text(X_train, X_test, cn_stop):
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words=list(cn_stop),
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return vectorizer, X_train_tfidf, X_test_tfidf


def build_models():
    logistic_regression = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    random_forest = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    naive_bayes = MultinomialNB()

    voting = VotingClassifier(
        estimators=[
            ("rf", random_forest),
            ("lr", logistic_regression),
        ],
        voting="soft",
    )

    stacking = StackingClassifier(
        estimators=[
            ("rf", random_forest),
            ("lr", logistic_regression),
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        stack_method="predict_proba",
        cv=5,
        n_jobs=-1,
    )

    return {
        "Logistic Regression": logistic_regression,
        "Random Forest": random_forest,
        "Naive Bayes": naive_bayes,
        "Voting Classifier": voting,
        "Stacking Classifier": stacking,
    }


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=["Positive", "Negative"]))

    return y_pred


def main():
    set_seed()

    raw_data = load_review_data(DATA_FILES)
    df = prepare_dataset(raw_data)
    cn_stop = build_stopwords()

    print(f"Cleaned dataset size: {len(df)}")
    print_top_keywords(df, cn_stop)

    X = df["clean"]
    y = df["评价类别"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    _, X_train_tfidf, X_test_tfidf = vectorize_text(X_train, X_test, cn_stop)

    models = build_models()
    for name, model in models.items():
        evaluate_model(name, model, X_train_tfidf, X_test_tfidf, y_train, y_test)


if __name__ == "__main__":
    main()
