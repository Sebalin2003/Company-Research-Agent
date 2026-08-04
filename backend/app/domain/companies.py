from __future__ import annotations

import re


def normalize_company_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip()).lower()


def validate_company_name(name: str) -> str:
    normalized = normalize_company_name(name)
    if not normalized:
        raise ValueError("El nombre de la empresa es obligatorio.")
    if len(normalized) > 200:
        raise ValueError("El nombre de la empresa supera el limite permitido.")
    return normalized
