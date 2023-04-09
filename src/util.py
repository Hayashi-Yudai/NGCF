from datetime import datetime


def simple_logger(message: str, file: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {file}\t{message}")
