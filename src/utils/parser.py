import os, re, pathlib

COMPANY_ALIASES = {
    "GOOGLE": "GOOGL",
    "ALPHABET": "GOOGL",
    "GOOGL": "GOOGL",
    "MICROSOFT": "MSFT",
    "MSFT": "MSFT",
    "NVIDIA": "NVDA",
    "NVDA": "NVDA",
}

COMPANY_LIST = ["GOOGL", "MSFT", "NVDA"]

def parse_company_year_from_filename(path: str):
    """Expecting filenames like MSFT_2023.pdf (case-insensitive).
    Returns (company, year) or (None, None) if not parseable."""
    name = pathlib.Path(path).name.upper()
    m = re.search(r"(GOOGL|MSFT|NVDA)[^\d]*([12][0-9]{3})", name)
    if not m:
        # Try common name variants
        for alias, canon in COMPANY_ALIASES.items():
            if alias in name:
                year = re.search(r"([12][0-9]{3})", name)
                return canon, (year.group(1) if year else None)
        return None, None
    company = m.group(1)
    year = m.group(2)
    company = COMPANY_ALIASES.get(company, company)
    return company, year

def clean_whitespace(text: str) -> str:
    text = re.sub(r"\u00a0", " ", text) # non-breaking space
    text = re.sub(r"[\t\r]", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()

def dehyphenate(text: str) -> str:
    # join hyphenated words broken by line breaks: e.g., "oper-ating" -> "operating"
    return re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
