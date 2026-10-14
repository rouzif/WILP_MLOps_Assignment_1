from datetime import datetime

def get_timestamp() -> str:
    """Generates a timestamp string in 'YYYYMMDDHHMMSS' format."""
    return datetime.now().strftime("%Y%m%d%H%M%S")


if __name__ == "__main__":
    print(get_timestamp())
