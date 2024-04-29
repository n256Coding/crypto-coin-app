def get_currency_name(raw_currency_name: str) -> str:
    return raw_currency_name.split('-')[0]