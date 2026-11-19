"""Command line: triage one message or several in a row."""

import argparse
import sys

from .assistant import handle
from .config import load_model, load_settings, model_name, prompt_version


def main() -> None:
    parser = argparse.ArgumentParser(description="Асистент техпідтримки")
    parser.add_argument("message", nargs="?", help="звернення клієнта; без нього — кілька звернень поспіль")
    args = parser.parse_args()

    load_settings()
    name = model_name()
    try:
        model = load_model(name)
    except Exception as e:
        sys.exit(f"Не вдалося створити модель {name}: {e}\nПеревірте .env (див. .env.example).")
    version = prompt_version()
    print(f"Модель: {name}, інструкції: {version}")

    def run(message: str) -> None:
        try:
            print(handle(model, message, version))
        except NotImplementedError as e:
            sys.exit(str(e))

    if args.message:
        run(args.message)
        return
    print("Порожній рядок або Ctrl+D — вихід.")
    while True:
        try:
            message = input("\nЗвернення клієнта: ").strip()
        except EOFError:
            break
        if not message:
            break
        run(message)
