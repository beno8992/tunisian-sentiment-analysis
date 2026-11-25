"""Train and use a binary sentiment classifier for Tunisian Arabizi comments."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def clean_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep nonempty, unambiguous, unique positive/negative texts."""
    required = {"InputText", "SentimentLabel"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {', '.join(sorted(missing))}")
    data = frame.loc[:, ["InputText", "SentimentLabel"]].copy()
    data["SentimentLabel"] = pd.to_numeric(data["SentimentLabel"], errors="coerce")
    data = data.dropna(subset=["InputText", "SentimentLabel"])
    data["InputText"] = data["InputText"].astype(str).str.strip()
    data = data.loc[
        data["SentimentLabel"].isin([-1, 1]) & data["InputText"].ne("")
    ].copy()
    data["SentimentLabel"] = data["SentimentLabel"].astype(int)
    # Remove contradictory annotations before removing duplicate texts.
    label_counts = data.groupby("InputText")["SentimentLabel"].transform("nunique")
    return (
        data.loc[label_counts.eq(1)]
        .drop_duplicates(subset=["InputText"])
        .reset_index(drop=True)
    )


def make_model() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def train(args: argparse.Namespace) -> None:
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size doit être compris strictement entre 0 et 1.")
    raw = pd.read_csv(args.data)
    data = clean_data(raw)
    counts = data["SentimentLabel"].value_counts()
    if set(counts.index) != {-1, 1} or int(counts.min()) < 2:
        raise ValueError("Il faut au moins deux textes distincts pour chacune des deux classes.")
    test_count = math.ceil(len(data) * args.test_size)
    if test_count < 2 or len(data) - test_count < 2:
        raise ValueError("Le train et le test doivent chacun contenir au moins deux exemples.")
    x_train, x_test, y_train, y_test = train_test_split(
        data["InputText"],
        data["SentimentLabel"],
        test_size=args.test_size,
        random_state=42,
        stratify=data["SentimentLabel"],
    )
    if set(y_train) != {-1, 1} or set(y_test) != {-1, 1}:
        raise ValueError("Une classe est absente du train ou du test ; ajustez le découpage.")
    model = make_model()
    # Vocabulary and IDF are learned only from the training split.
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    report = {
        "status": "evaluated",
        "random_state": 42,
        "rows_loaded": len(raw),
        "rows_after_cleaning": len(data),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "class_labels": {"-1": "negatif", "1": "positif"},
        "accuracy": float(accuracy_score(y_test, predictions)),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[-1, 1],
            target_names=["negatif", "positif"],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[-1, 1]).tolist(),
        "confusion_matrix_label_order": [-1, 1],
    }
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Modèle enregistré : {args.model}")
    print(f"Rapport enregistré : {args.report}")


def predict(args: argparse.Namespace) -> None:
    if any(not text.strip() for text in args.text):
        raise ValueError("Les commentaires ne doivent pas être vides.")
    # joblib uses pickle: load only your own model or one from a trusted source.
    model = joblib.load(args.model)
    predictions = model.predict(args.text)
    results = [
        {"text": text, "label": int(label), "sentiment": "positif" if label == 1 else "negatif"}
        for text, label in zip(args.text, predictions)
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Entraîner et évaluer sur un CSV TUNIZI")
    training.add_argument("--data", type=Path, required=True)
    training.add_argument("--test-size", type=float, default=0.2)
    training.add_argument("--model", type=Path, default=Path("models/sentiment.joblib"))
    training.add_argument("--report", type=Path, default=Path("reports/metrics.json"))
    training.set_defaults(run=train)
    prediction = commands.add_parser("predict", help="Classer de nouveaux commentaires")
    prediction.add_argument("--model", type=Path, default=Path("models/sentiment.joblib"))
    prediction.add_argument("--text", nargs="+", required=True)
    prediction.set_defaults(run=predict)
    args = parser.parse_args()
    try:
        args.run(args)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(2, f"Erreur : {error}\n")


if __name__ == "__main__":
    main()
