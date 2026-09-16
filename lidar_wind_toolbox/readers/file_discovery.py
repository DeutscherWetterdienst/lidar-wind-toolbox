import datetime


def try_date(text: str) -> datetime.datetime:
    for fmt in ("%Y%m%dT%H", "%Y%m%dT%H%M%S"):
        try:
            return datetime.datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError("no valid date format found")
