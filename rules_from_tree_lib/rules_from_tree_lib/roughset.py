"""
roughset.py
===========
Biblioteka do reprezentacji, ekstrakcji i analizy reguł decyzyjnych.

Przeznaczona do współpracy z ID3C45Classifier, ale niezależna —
reguły można też tworzyć ręcznie lub ładować z zewnątrz.

Główne klasy
------------
Condition
    Pojedynczy warunek reguły: (atrybut, operator, wartość).
    Operatory: '=', '<=', '>', '!='.

Rule
    Reguła decyzyjna: lista warunków + konkluzja (atrybut decyzyjny, wartość).
    Przechowuje też metadane: support, confidence, n_samples.

RuleSet
    Zbiór reguł z metodami eksportu, filtrowania, aplikowania i analizy.

Funkcje modułu
--------------
extract_rules_from_tree(clf)
    Ekstrahuje reguły z dopasowanego ID3C45Classifier.
    Zwraca RuleSet.

Formaty eksportu
----------------
RuleSet.to_series()
    pd.Series stringów: "IF f1='val' AND f2<=3.5 THEN class='tak'"

RuleSet.to_dataframe()
    pd.DataFrame: kolumny = atrybuty warunkowe + atrybut decyzyjny
    + kolumny support/confidence/coverage, wartości tylko dla atrybutów
    użytych w regule, reszta NaN.

Wczytywanie reguł z DataFrame / CSV
------------------------------------
RuleSet.from_dataframe(df, feature_cols, decision_col, weight_col)
    Tworzy RuleSet z pd.DataFrame (np. wczytanego z CSV).
    Komórki NaN oznaczają brak warunku dla danego atrybutu.
    Wartości numeryczne komórek obsługują formaty "<=2.5", ">1.0"
    oraz zwykłe wartości kategoryczne.

Klasyfikacja przez ważone głosowanie
--------------------------------------
RuleSet.predict(..., strategy='weighted_vote', weight='support')
    Dla każdego rekordu zbiera pasujące reguły, grupuje po klasie
    i sumuje wagi — wygrywa klasa z najwyższą sumą wag.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import io


# ─────────────────────────────────────────────────────────────────────────────
# Pomocnicza funkcja parsowania warunków z DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def _parse_condition(attribute: str, raw_value: str) -> "Condition":
    """
    Parsuje wartość komórki DataFrame do obiektu Condition.

    Obsługuje formaty:
    - "<=2.45"  → Condition(attr, "<=", 2.45)
    - ">1.0"    → Condition(attr, ">",  1.0)
    - "<=2.45 AND >1.0"  → złożone, zwraca pierwszą część
      (pełne zakresy są obsługiwane na poziomie Rule.get_condition_values)
    - "sunny"   → Condition(attr, "=",  "sunny")
    - "!=val"   → Condition(attr, "!=", "val")

    Parameters
    ----------
    attribute : str
        Nazwa atrybutu.
    raw_value : str
        Wartość z komórki DataFrame (po strip()).

    Returns
    -------
    Condition
    """
    import re
    s = raw_value.strip()

    # Wzorzec: operator + wartość numeryczna
    num_pattern = re.compile(r'^(<=|>=|!=|<|>)\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)$')
    m = num_pattern.match(s)
    if m:
        op_raw, val_str = m.group(1), m.group(2)
        # Normalizuj ">=" → ">" (uproszczenie; C4.5 używa tylko <= i >)
        op = op_raw if op_raw in {"<=", ">", "!=", "="} else (
            ">" if op_raw == ">=" else "<="
        )
        return Condition(attribute, op, float(val_str))

    # "!= wartość_kategoryczna"
    if s.startswith("!="):
        return Condition(attribute, "!=", s[2:].strip())

    # Czysta wartość kategoryczna
    return Condition(attribute, "=", s)


# ─────────────────────────────────────────────────────────────────────────────
# Condition
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Condition:
    """
    Pojedynczy warunek reguły decyzyjnej.

    Atrybuty
    --------
    attribute : str
        Nazwa atrybutu (np. "Outlook", "petal_length").
    operator : str
        Operator porównania: '=', '<=', '>', '!='.
    value : Any
        Wartość po prawej stronie operatora.

    Przykłady
    ---------
    >>> Condition("Outlook", "=", "sunny")
    Condition(Outlook = sunny)
    >>> Condition("petal_length", "<=", 2.45)
    Condition(petal_length <= 2.45)
    """

    attribute: str
    operator: str   # '=', '<=', '>', '!='
    value: Any

    _VALID_OPS = {"=", "<=", ">", "!="}

    def __post_init__(self):
        if self.operator not in self._VALID_OPS:
            raise ValueError(
                f"Nieznany operator '{self.operator}'. "
                f"Dozwolone: {self._VALID_OPS}"
            )

    # ── formatowanie ──────────────────────────────────────────────────────

    def __str__(self) -> str:
        val = self._fmt_value(self.value)
        return f"{self.attribute}{self.operator}{val}"

    def __repr__(self) -> str:
        return f"Condition({self.attribute} {self.operator} {self.value!r})"

    @staticmethod
    def _fmt_value(v: Any) -> str:
        """Formatuje wartość do czytelnej postaci stringowej."""
        if isinstance(v, float):
            return f"{v:.4g}"
        if isinstance(v, str):
            return f"'{v}'"
        return str(v)

    # ── ewaluacja ─────────────────────────────────────────────────────────

    def evaluate(self, record: Dict[str, Any]) -> bool:
        """
        Sprawdza, czy warunek jest spełniony dla podanego rekordu.

        Parameters
        ----------
        record : dict
            Słownik {nazwa_atrybutu: wartość}.

        Returns
        -------
        bool
            True jeśli warunek spełniony, False w p.p.
            Niezdefiniowany atrybut → False.
        """
        if self.attribute not in record:
            return False
        actual = record[self.attribute]
        try:
            if self.operator == "=":
                return actual == self.value
            elif self.operator == "!=":
                return actual != self.value
            elif self.operator == "<=":
                return float(actual) <= float(self.value)
            elif self.operator == ">":
                return float(actual) > float(self.value)
        except (TypeError, ValueError):
            return False
        return False

    # ── serializacja ──────────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        return {"attribute": self.attribute, "operator": self.operator,
                "value": self.value}

    @classmethod
    def from_dict(cls, d: Dict) -> "Condition":
        return cls(d["attribute"], d["operator"], d["value"])


# ─────────────────────────────────────────────────────────────────────────────
# Rule
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Rule:
    """
    Reguła decyzyjna postaci: IF warunki THEN konkluzja.

    Atrybuty
    --------
    conditions : list of Condition
        Lista warunków (część IF). Łączone koniunkcją (AND).
    decision_attribute : str
        Nazwa atrybutu decyzyjnego.
    decision_value : Any
        Wartość atrybutu decyzyjnego (konkluzja).
    support : int, default 0
        Liczba próbek treningowych pokrytych przez regułę.
    confidence : float, default 1.0
        Dokładność reguły na próbkach pokrytych (0.0–1.0).
    n_total : int, default 0
        Łączna liczba próbek treningowych (do obliczenia coverage).

    Właściwości
    -----------
    coverage : float
        support / n_total (frakcja pokrytych próbek).
    n_conditions : int
        Liczba warunków w regule.
    """

    conditions: List[Condition] = field(default_factory=list)
    decision_attribute: str = "decision"
    decision_value: Any = None
    support: int = 0
    confidence: float = 1.0
    n_total: int = 0

    # ── właściwości ───────────────────────────────────────────────────────

    @property
    def coverage(self) -> float:
        if self.n_total == 0:
            return 0.0
        return self.support / self.n_total

    @property
    def n_conditions(self) -> int:
        return len(self.conditions)

    # ── formatowanie ──────────────────────────────────────────────────────

    def __str__(self) -> str:
        """
        Format: IF f1='val1' AND f2<=2.5 THEN class='klasa'
        """
        if self.conditions:
            cond_str = " AND ".join(str(c) for c in self.conditions)
        else:
            cond_str = "TRUE"
        dec_val = Condition._fmt_value(self.decision_value)
        return f"IF {cond_str} THEN {self.decision_attribute}={dec_val}"

    def __repr__(self) -> str:
        return f"Rule(conditions={self.n_conditions}, decision={self.decision_value!r})"

    def to_string(
        self,
        include_stats: bool = False,
        number: Optional[int] = None,
    ) -> str:
        """
        Zwraca regułę jako string w formacie IF/THEN.

        Parameters
        ----------
        include_stats : bool
            Dołącz support/confidence w nawiasach kwadratowych.
        number : int or None
            Jeśli podany, prefiks "R{number}: " przed regułą.

        Returns
        -------
        str
            Np. "R1: IF Outlook='sunny' AND Humidity='high' THEN class='no'
                   [sup=3, conf=1.000]"
        """
        if self.conditions:
            cond_str = " AND ".join(str(c) for c in self.conditions)
        else:
            cond_str = "TRUE"
        dec_val = Condition._fmt_value(self.decision_value)
        base = f"IF {cond_str} THEN {self.decision_attribute}={dec_val}"
        if number is not None:
            base = f"R{number}: {base}"
        if include_stats:
            base += f"  [sup={self.support}, conf={self.confidence:.3f}]"
        return base

    # ── ewaluacja ─────────────────────────────────────────────────────────

    def covers(self, record: Dict[str, Any]) -> bool:
        """
        Sprawdza, czy reguła pokrywa dany rekord
        (wszystkie warunki spełnione).
        """
        return all(c.evaluate(record) for c in self.conditions)

    def fires(self, record: Dict[str, Any]) -> Optional[Any]:
        """
        Jeśli reguła pokrywa rekord — zwraca wartość decyzyjną.
        W przeciwnym razie zwraca None.
        """
        if self.covers(record):
            return self.decision_value
        return None

    # ── upraszczanie ──────────────────────────────────────────────────────

    def simplify(self) -> "Rule":
        """
        Zwraca uproszczoną kopię reguły przez eliminację nadmiarowych
        warunków numerycznych na tym samym atrybucie.

        Logika upraszczania
        -------------------
        Dla operatora ``<=`` (górne ograniczenie): zostaje **najmniejsza**
        wartość — silniejsza (węższa) granica.
        Przykład: ``x<=5 AND x<=3``  →  ``x<=3``

        Dla operatora ``>`` (dolne ograniczenie): zostaje **największa**
        wartość — silniejsza (węższa) granica.
        Przykład: ``x>2.45 AND x>4.75 AND x>5.1``  →  ``x>5.1``

        Warunki kategoryczne (``=``, ``!=``) pozostają bez zmian.
        Mieszanie ``>`` i ``<=`` na tym samym atrybucie (zakresy) jest
        przepuszczane poprawnie — oba ograniczenia mogą współistnieć.

        Returns
        -------
        Rule
            Nowa reguła z uproszczonymi warunkami (oryginał bez zmian).

        Przykład
        --------
        >>> r = Rule([
        ...     Condition("petal length (cm)", ">", 2.45),
        ...     Condition("petal length (cm)", ">", 4.75),
        ...     Condition("petal width (cm)",  "<=", 1.75),
        ...     Condition("petal length (cm)", ">", 5.1),
        ... ], decision_attribute="gatunek", decision_value="virginica")
        >>> print(r.simplify())
        # IF petal width (cm)<=1.75 AND petal length (cm)>5.1 THEN gatunek='virginica'
        """
        from collections import defaultdict as _defaultdict

        # Grupuj warunki numeryczne per (atrybut, operator)
        num_groups: Dict[Tuple[str, str], List[Condition]] = _defaultdict(list)
        other: List[Condition] = []

        for c in self.conditions:
            if c.operator in ("<=", ">"):
                try:
                    float(c.value)
                    num_groups[(c.attribute, c.operator)].append(c)
                except (TypeError, ValueError):
                    other.append(c)
            else:
                other.append(c)

        # Zachowaj kolejność pierwszego pojawienia się klucza (attr, op)
        seen_keys: List[Tuple[str, str]] = []
        for c in self.conditions:
            if c.operator in ("<=", ">"):
                try:
                    float(c.value)
                    key = (c.attribute, c.operator)
                    if key not in seen_keys:
                        seen_keys.append(key)
                except (TypeError, ValueError):
                    pass

        # Zbuduj listę uproszczonych warunków w oryginalnej kolejności
        simplified: List[Condition] = []
        for key in seen_keys:
            attr, op = key
            values = [float(c.value) for c in num_groups[key]]
            best = min(values) if op == "<=" else max(values)
            simplified.append(Condition(attr, op, best))

        # "other" (kategoryczne) — wstaw w pozycjach z oryginalnej listy
        # (przed pierwszym warunkiem numerycznym tego atrybutu)
        result: List[Condition] = []
        other_iter = iter(other)
        num_inserted = set()
        for c in self.conditions:
            if c.operator not in ("<=", ">"):
                # warunek kategoryczny — przenieś z other w tej samej kolejności
                result.append(next(other_iter))
            else:
                try:
                    float(c.value)
                except (TypeError, ValueError):
                    result.append(next(other_iter))
                    continue
                key = (c.attribute, c.operator)
                if key not in num_inserted:
                    # Wstaw uproszczony warunek dla tej grupy
                    idx = seen_keys.index(key)
                    result.append(simplified[idx])
                    num_inserted.add(key)
                # else: pomiń — to był duplikat

        return Rule(
            conditions=result,
            decision_attribute=self.decision_attribute,
            decision_value=self.decision_value,
            support=self.support,
            confidence=self.confidence,
            n_total=self.n_total,
        )

    # ── serializacja ──────────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        return {
            "conditions": [c.to_dict() for c in self.conditions],
            "decision_attribute": self.decision_attribute,
            "decision_value": self.decision_value,
            "support": self.support,
            "confidence": self.confidence,
            "n_total": self.n_total,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Rule":
        conditions = [Condition.from_dict(c) for c in d.get("conditions", [])]
        return cls(
            conditions=conditions,
            decision_attribute=d.get("decision_attribute", "decision"),
            decision_value=d.get("decision_value"),
            support=d.get("support", 0),
            confidence=d.get("confidence", 1.0),
            n_total=d.get("n_total", 0),
        )

    # ── pomocnicze ────────────────────────────────────────────────────────

    def get_condition_values(self, all_attributes: Sequence[str]) -> Dict[str, Any]:
        """
        Zwraca słownik {atrybut: wartość_lub_opis} dla podanej listy atrybutów.
        Atrybuty nieużywane w regule → None (będą NaN w DataFrame).

        Dla warunków numerycznych: "<=2.45" lub ">2.45".
        Dla kategorycznych: wartość bezpośrednia.
        """
        result: Dict[str, Any] = {a: None for a in all_attributes}
        # Grupuj warunki per atrybut (może być np. <=X i >Y dla tego samego)
        per_attr: Dict[str, List[Condition]] = {}
        for c in self.conditions:
            per_attr.setdefault(c.attribute, []).append(c)

        for attr, conds in per_attr.items():
            if attr not in result:
                result[attr] = None
                continue
            if len(conds) == 1:
                c = conds[0]
                if c.operator == "=":
                    result[attr] = c.value
                else:
                    val_str = Condition._fmt_value(c.value)
                    result[attr] = f"{c.operator}{val_str}"
            else:
                # Wiele warunków na tym samym atrybucie (zakres liczbowy)
                parts = []
                for c in sorted(conds, key=lambda x: x.operator):
                    val_str = Condition._fmt_value(c.value)
                    parts.append(f"{c.operator}{val_str}")
                result[attr] = " AND ".join(parts)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# RuleSet
# ─────────────────────────────────────────────────────────────────────────────

class RuleSet:
    """
    Zbiór reguł decyzyjnych z metodami eksportu, filtrowania i aplikowania.

    Parametry
    ---------
    rules : list of Rule
        Lista reguł.
    decision_attribute : str, default 'decision'
        Nazwa atrybutu decyzyjnego.
    feature_names : list of str or None
        Nazwy atrybutów warunkowych (potrzebne do to_dataframe).

    Przykład
    --------
    >>> rs = extract_rules_from_tree(clf)
    >>> series = rs.to_series()
    >>> df = rs.to_dataframe()
    """

    def __init__(
        self,
        rules: Optional[List[Rule]] = None,
        decision_attribute: str = "decision",
        feature_names: Optional[List[str]] = None,
    ):
        self.rules: List[Rule] = rules or []
        self.decision_attribute = decision_attribute
        self.feature_names: Optional[List[str]] = feature_names

    # ── podstawowe właściwości ────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.rules)

    def __iter__(self):
        return iter(self.rules)

    def __getitem__(self, idx):
        return self.rules[idx]

    def __repr__(self) -> str:
        return f"RuleSet({len(self.rules)} rules, decision='{self.decision_attribute}')"
    
    def __str__(self, include_stats = True):
        f = io.StringIO()
        with redirect_stdout(f):
            self.print_rules(include_stats=include_stats)
        return f.getvalue().strip()

    def add(self, rule: Rule) -> "RuleSet":
        """Dodaje regułę do zbioru. Zwraca self (fluent interface)."""
        self.rules.append(rule)
        return self

    # ── eksport Format 1: pd.Series stringów ─────────────────────────────

    def to_series(
        self,
        include_stats: bool = False,
        numbered: bool = False,
        name: str = "rule",
    ) -> pd.Series:
        """
        Eksportuje reguły jako pd.Series stringów w formacie IF/THEN.

        Każda reguła ma postać:
            IF f1='val1' AND f2<='niski' THEN class='żyje'

        Parameters
        ----------
        include_stats : bool
            Dołącz support/confidence w nawiasach kwadratowych.
        numbered : bool
            Poprzedź każdą regułę numerem (R1:, R2:, ...).
        name : str
            Nazwa kolumny/serii.

        Returns
        -------
        pd.Series
            Indeks: 0, 1, 2, ... (numer reguły).
        """
        strings = [
            r.to_string(
                include_stats=include_stats,
                number=(i + 1) if numbered else None,
            )
            for i, r in enumerate(self.rules)
        ]
        return pd.Series(strings, name=name, dtype=str)

    # ── eksport Format 2: pd.DataFrame ───────────────────────────────────

    def to_dataframe(
        self,
        feature_names: Optional[List[str]] = None,
        decision_column: Optional[str] = None,
        use_none_as_nan: bool = True,
        include_stats: bool = True,
    ) -> pd.DataFrame:
        """
        Eksportuje reguły jako pd.DataFrame.

        Każdy wiersz to jedna reguła.
        Kolumny = atrybuty warunkowe + atrybut decyzyjny
        + (opcjonalnie) support, confidence, coverage.
        Atrybuty nieużywane w regule → NaN (lub None).

        Dla warunków numerycznych wartość w komórce to string "<=2.45"
        lub ">2.45" (opis warunku). Dla kategorycznych — sama wartość.

        Parameters
        ----------
        feature_names : list of str or None
            Nazwy kolumn (atrybutów warunkowych).
            Jeśli None — użyte zostaną self.feature_names lub
            zebrane z warunków reguł.
        decision_column : str or None
            Nazwa kolumny decyzyjnej. Jeśli None — używa
            self.decision_attribute (np. "class", "klasa" itp.).
        use_none_as_nan : bool
            True → None zamień na np.nan (standardowe NaN w DataFrame).
        include_stats : bool, default True
            Dołącz kolumny 'support', 'confidence', 'coverage'.

        Returns
        -------
        pd.DataFrame
            Kolumny: [feat1, feat2, ..., decision_column,
                      (support, confidence, coverage)]
            Index: 0, 1, 2, ...
        """
        fn = feature_names or self.feature_names
        dec_col = decision_column if decision_column is not None else self.decision_attribute

        if fn is None:
            # Zbierz nazwy atrybutów ze wszystkich warunków
            seen = []
            for r in self.rules:
                for c in r.conditions:
                    if c.attribute not in seen:
                        seen.append(c.attribute)
            fn = seen

        stat_cols = ["support", "confidence", "coverage"] if include_stats else []
        columns = list(fn) + [dec_col] + stat_cols
        rows = []

        for rule in self.rules:
            cond_vals = rule.get_condition_values(fn)
            row = {col: cond_vals.get(col, None) for col in fn}
            row[dec_col] = rule.decision_value
            if include_stats:
                row["support"]    = rule.support
                row["confidence"] = round(rule.confidence, 4)
                row["coverage"]   = round(rule.coverage, 4)
            rows.append(row)

        if use_none_as_nan:
            # Zastąp None przez np.nan bezpośrednio w słownikach wierszy,
            # zanim trafią do DataFrame — unika FutureWarning z pandas replace()
            rows = [
                {k: (np.nan if v is None else v) for k, v in row.items()}
                for row in rows
            ]

        return pd.DataFrame(rows, columns=columns)

    # ── filtrowanie ───────────────────────────────────────────────────────

    def filter_by_decision(self, value: Any) -> "RuleSet":
        """Zwraca nowy RuleSet zawierający tylko reguły z daną konkluzją."""
        return RuleSet(
            [r for r in self.rules if r.decision_value == value],
            decision_attribute=self.decision_attribute,
            feature_names=self.feature_names,
        )

    def filter_by_confidence(self, min_confidence: float) -> "RuleSet":
        """Zwraca reguły z confidence >= min_confidence."""
        return RuleSet(
            [r for r in self.rules if r.confidence >= min_confidence],
            decision_attribute=self.decision_attribute,
            feature_names=self.feature_names,
        )

    def filter_by_support(self, min_support: int) -> "RuleSet":
        """Zwraca reguły z support >= min_support."""
        return RuleSet(
            [r for r in self.rules if r.support >= min_support],
            decision_attribute=self.decision_attribute,
            feature_names=self.feature_names,
        )

    def filter_by_length(self, max_conditions: int) -> "RuleSet":
        """Zwraca reguły z liczbą warunków <= max_conditions."""
        return RuleSet(
            [r for r in self.rules if r.n_conditions <= max_conditions],
            decision_attribute=self.decision_attribute,
            feature_names=self.feature_names,
        )

    def simplify(self) -> "RuleSet":
        """
        Zwraca nowy RuleSet, w którym każda reguła została uproszczona
        przez ``Rule.simplify()``.

        Eliminuje nadmiarowe warunki numeryczne na tym samym atrybucie:
        - ``<=``: zachowuje tylko najmniejszą (najsilniejszą) wartość górną
        - ``>`` : zachowuje tylko największą (najsilniejszą) wartość dolną

        Returns
        -------
        RuleSet
            Nowy zbiór z uproszczonymi regułami (oryginał bez zmian).

        Przykład
        --------
        >>> rs_simple = rs.simplify()
        >>> rs_simple.print_rules()
        """
        return RuleSet(
            [r.simplify() for r in self.rules],
            decision_attribute=self.decision_attribute,
            feature_names=self.feature_names,
        )

    # ── wczytywanie z DataFrame / CSV ─────────────────────────────────────

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        decision_col: str = "decision",
        support_col: Optional[str] = "support",
        confidence_col: Optional[str] = "confidence",
        coverage_col: Optional[str] = "coverage",
        decision_attribute: str = "decision",
    ) -> "RuleSet":
        """
        Tworzy RuleSet z pd.DataFrame (np. wczytanego z CSV).

        Każdy wiersz to jedna reguła. Komórki NaN/puste oznaczają brak
        warunku dla danego atrybutu. Wartości numerycznych warunków mogą
        mieć formaty "<=2.5", ">1.0" (są parsowane do Condition)
        lub zwykłe wartości kategoryczne (operator '=').

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame z regułami.
        feature_cols : list of str or None
            Kolumny atrybutów warunkowych. Jeśli None — wszystkie kolumny
            poza decision_col i kolumnami wag.
        decision_col : str, default 'decision'
            Nazwa kolumny z wartością decyzyjną.
        support_col : str or None, default 'support'
            Nazwa kolumny z wagą support. None → support=0 dla każdej reguły.
        confidence_col : str or None, default 'confidence'
            Nazwa kolumny z wagą confidence. None → confidence=1.0.
        coverage_col : str or None, default 'coverage'
            Nazwa kolumny coverage (pomijana przy budowie Rule, służy info).
        decision_attribute : str, default 'decision'
            Nazwa atrybutu decyzyjnego w tworzonych regułach.

        Returns
        -------
        RuleSet

        Przykład
        --------
        >>> df = pd.read_csv("rules.csv", na_values=[''], dtype=str)
        >>> rs = RuleSet.from_dataframe(df, decision_col="class",
        ...                             support_col="support",
        ...                             confidence_col="confidence")
        >>> rs.predict([{"Outlook": "sunny", "Humidity": "high"}])
        """
        # Ustal kolumny meta (nie-warunkowe)
        meta_cols = {decision_col}
        for c in [support_col, confidence_col, coverage_col]:
            if c and c in df.columns:
                meta_cols.add(c)

        if feature_cols is None:
            feature_cols = [c for c in df.columns if c not in meta_cols]

        rules: List[Rule] = []

        for _, row in df.iterrows():
            conditions: List[Condition] = []
            for col in feature_cols:
                raw = row.get(col, None)
                # Pomiń NaN / puste
                if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                    continue
                val_str = str(raw).strip()
                if val_str in ("", "nan", "NaN"):
                    continue
                # Parsuj operator numeryczny: "<=2.5", ">1.0", itp.
                # Wartości złożone (np. "<=5.1 AND >2.45") rozbijamy po " AND "
                # i parsujemy każdy człon osobno jako osobny Condition.
                for part in val_str.split(" AND "):
                    cond = _parse_condition(col, part.strip())
                    conditions.append(cond)

            # Wartość decyzyjna
            dec_val = row.get(decision_col, None)
            if dec_val is not None and not (isinstance(dec_val, float) and np.isnan(dec_val)):
                dec_val = str(dec_val).strip()

            # Wagi
            sup = 0
            if support_col and support_col in df.columns:
                try:
                    sup = int(float(row[support_col]))
                except (ValueError, TypeError):
                    sup = 0

            conf = 1.0
            if confidence_col and confidence_col in df.columns:
                try:
                    conf = float(row[confidence_col])
                except (ValueError, TypeError):
                    conf = 1.0

            rule = Rule(
                conditions=conditions,
                decision_attribute=decision_attribute,
                decision_value=dec_val,
                support=sup,
                confidence=conf,
            )
            rules.append(rule)

        return cls(
            rules=rules,
            decision_attribute=decision_attribute,
            feature_names=list(feature_cols),
        )

    @classmethod
    def from_csv(
        cls,
        path: str,
        feature_cols: Optional[List[str]] = None,
        decision_col: str = "decision",
        support_col: Optional[str] = "support",
        confidence_col: Optional[str] = "confidence",
        coverage_col: Optional[str] = "coverage",
        decision_attribute: str = "decision",
        **read_csv_kwargs,
    ) -> "RuleSet":
        """
        Wygodny skrót: wczytuje CSV i zwraca RuleSet.

        Parameters
        ----------
        path : str
            Ścieżka do pliku CSV.
        Pozostałe parametry jak w from_dataframe().
        **read_csv_kwargs
            Dodatkowe argumenty przekazywane do pd.read_csv().

        Returns
        -------
        RuleSet
        """
        read_csv_kwargs.setdefault("na_values", [""])
        read_csv_kwargs.setdefault("dtype", str)
        df = pd.read_csv(path, **read_csv_kwargs)
        return cls.from_dataframe(
            df,
            feature_cols=feature_cols,
            decision_col=decision_col,
            support_col=support_col,
            confidence_col=confidence_col,
            coverage_col=coverage_col,
            decision_attribute=decision_attribute,
        )

    # ── aplikowanie reguł ─────────────────────────────────────────────────

    def predict(
        self,
        records: Sequence[Dict[str, Any]],
        default: Any = None,
        strategy: str = "weighted_vote",
        weight: str = "support",
    ) -> List[Any]:
        """
        Klasyfikuje rekordy za pomocą reguł.

        Parameters
        ----------
        records : list of dict
            Rekordy do klasyfikacji.
        default : Any
            Wartość domyślna gdy żadna reguła nie pokrywa rekordu.
        strategy : {'weighted_vote', 'first', 'confidence', 'support'}
            'weighted_vote' – (domyślna) grupuje pasujące reguły po klasie
                              i sumuje wagi; wygrywa klasa z najwyższą sumą.
            'first'         – pierwsze dopasowanie (kolejność w zbiorze).
            'confidence'    – reguła z najwyższym confidence.
            'support'       – reguła z najwyższym support.
        weight : {'support', 'confidence', 'coverage'}
            Kolumna wagi używana przez strategię 'weighted_vote'.

        Returns
        -------
        list
        """
        _weight_fn = {
            "support":    lambda r: r.support,
            "confidence": lambda r: r.confidence,
            "coverage":   lambda r: r.coverage,
        }.get(weight, lambda r: r.support)

        results = []
        for rec in records:
            matching = [r for r in self.rules if r.covers(rec)]
            if not matching:
                results.append(default)
                continue

            if strategy == "weighted_vote":
                # Grupuj po klasie, sumuj wagi — wygrywa klasa z max sumą
                votes: Dict[Any, float] = defaultdict(float)
                for r in matching:
                    votes[r.decision_value] += _weight_fn(r)
                results.append(max(votes.items(), key=lambda kv: kv[1])[0])
            elif strategy == "first":
                results.append(matching[0].decision_value)
            elif strategy == "confidence":
                best = max(matching, key=lambda r: r.confidence)
                results.append(best.decision_value)
            elif strategy == "support":
                best = max(matching, key=lambda r: r.support)
                results.append(best.decision_value)
            else:
                results.append(matching[0].decision_value)
        return results

    def predict_dataframe(
        self,
        df: pd.DataFrame,
        default: Any = None,
        strategy: str = "weighted_vote",
        weight: str = "support",
        result_col: str = "predicted",
    ) -> list:
        """
        Klasyfikuje DataFrame i zwraca listę predykcji.

        Parameters
        ----------
        df : pd.DataFrame
            Dane do klasyfikacji.
        default : Any
            Wartość domyślna gdy żadna reguła nie pasuje.
        strategy : str
            Patrz predict().
        weight : str
            Patrz predict().
        result_col : str, default 'predicted'
            Nazwa nowej kolumny z predykcjami.

        Returns
        -------
        list:
            Lista predicts
        """
        records = df.to_dict(orient="records")
        preds = self.predict(records, default=default,
                             strategy=strategy, weight=weight)
        out = df.copy()
        out[result_col] = preds
        return preds

    def coverage_matrix(
        self,
        records: Sequence[Dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Macierz pokrycia (reguły × rekordy).

        Returns
        -------
        pd.DataFrame of bool
            Wiersze = reguły (indeks 0..n-1),
            Kolumny = rekordy (indeks 0..m-1).
        """
        data = {
            j: [r.covers(rec) for r in self.rules]
            for j, rec in enumerate(records)
        }
        return pd.DataFrame(data, index=range(len(self.rules)))

    def print_rules(self, include_stats: bool = False) -> None:
        """
        Wyświetla wszystkie reguły w czytelnej postaci IF/THEN.

        Parameters
        ----------
        include_stats : bool
            Dołącz [sup=..., conf=...] po każdej regule.
        """
        for i, r in enumerate(self.rules, 1):
            print(r.to_string(include_stats=include_stats, number=i))

    def evaluate(
        self,
        X: pd.DataFrame,
        y_true: "pd.Series",
        default: Any = None,
        strategy: str = "weighted_vote",
        weight: str = "support",
    ) -> float:
        """
        Oblicza dokładność (accuracy) klasyfikacji reguł względem prawdziwych etykiet.

        Parameters
        ----------
        X : pd.DataFrame
            Dane wejściowe.
        y_true : pd.Series
            Prawdziwe etykiety klas.
        default : Any
            Wartość domyślna gdy żadna reguła nie pasuje.
        strategy : str
            Patrz predict().
        weight : str
            Patrz predict().

        Returns
        -------
        float
            Accuracy (0.0–1.0).
        """
        records = X.to_dict(orient="records")
        y_pred = self.predict(records, default=default,
                              strategy=strategy, weight=weight)
        y_true_list = list(y_true)
        correct = sum(p == t for p, t in zip(y_pred, y_true_list))
        return correct / len(y_true_list) if y_true_list else 0.0

    def fidelity(
        self,
        X: pd.DataFrame,
        y_tree: List[Any],
        default: Any = None,
        strategy: str = "weighted_vote",
        weight: str = "support",
    ) -> float:
        """
        Oblicza zgodność klasyfikacji reguł z klasyfikacją drzewa (fidelity).

        Parameters
        ----------
        X : pd.DataFrame
            Dane wejściowe.
        y_tree : list
            Przewidywania drzewa decyzyjnego dla tych samych danych.
        default : Any
            Wartość domyślna gdy żadna reguła nie pasuje.
        strategy : str
            Patrz predict().
        weight : str
            Patrz predict().

        Returns
        -------
        float
            Odsetek przypadków, dla których reguły i drzewo przewidziały tę samą klasę.
        """
        records = X.to_dict(orient="records")
        y_pred = self.predict(records, default=default,
                              strategy=strategy, weight=weight)
        correct = sum(p == t for p, t in zip(y_pred, y_tree))
        return correct / len(y_tree) if y_tree else 0.0

    # ── statystyki ────────────────────────────────────────────────────────

    def summary(self) -> pd.DataFrame:
        """
        Zwraca DataFrame ze statystykami zbioru reguł.

        Kolumny: rule (IF/THEN), n_conditions, decision_value,
                 support, confidence, coverage.
        """
        rows = []
        for i, r in enumerate(self.rules):
            rows.append({
                "index":          i,
                "rule":           r.to_string(number=i + 1),
                "n_conditions":   r.n_conditions,
                "decision_value": r.decision_value,
                "support":        r.support,
                "confidence":     round(r.confidence, 4),
                "coverage":       round(r.coverage, 4),
            })
        return pd.DataFrame(rows).set_index("index")

    # ── serializacja ──────────────────────────────────────────────────────

    def to_list_of_dicts(self) -> List[Dict]:
        return [r.to_dict() for r in self.rules]

    @classmethod
    def from_list_of_dicts(cls, data: List[Dict], **kwargs) -> "RuleSet":
        rules = [Rule.from_dict(d) for d in data]
        return cls(rules, **kwargs)

    def to_json(self, **kwargs) -> str:
        """Serializuje do JSON."""
        import json

        def _default(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Nie można serializować {type(obj)}")

        payload = {
            "decision_attribute": self.decision_attribute,
            "feature_names": self.feature_names,
            "rules": self.to_list_of_dicts(),
        }
        return json.dumps(payload, default=_default, **kwargs)

    @classmethod
    def from_json(cls, json_str: str) -> "RuleSet":
        import json
        payload = json.loads(json_str)
        rs = cls.from_list_of_dicts(
            payload.get("rules", []),
            decision_attribute=payload.get("decision_attribute", "decision"),
            feature_names=payload.get("feature_names"),
        )
        return rs

    # ── format .rul (RSES) ────────────────────────────────────────────────

    def to_rul(
        self,
        filepath: str,
        ruleset_name: str = "RULE_SET",
        feature_types: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Zapisuje zbiór reguł do pliku w formacie RSES .rul.

        Format pliku
        ------------
        ::

            RULE_SET NazwaZbioru
            ATTRIBUTES n
             attr1 numeric 0
             attr2 symbolic
            DECISION_VALUES k
            val1
            val2
            RULES n
            (attr1=1)&(attr2=sunny)=>(dec=val[88]) 88
            ...

        Każda reguła zawiera:
        - warunki w postaci ``(attr=val)`` lub ``(attr<=val)`` / ``(attr>val)``
          połączone znakiem ``&``
        - konkluzję ``(decision_attribute=decision_value[support])``
        - support jako liczba po spacjii na końcu wiersza

        Parameters
        ----------
        filepath : str
            Ścieżka do pliku wyjściowego. Jeśli nie kończy się na ``.rul``,
            rozszerzenie jest dodawane automatycznie.
        ruleset_name : str, default ``'RULE_SET'``
            Nazwa zbioru reguł wpisywana w nagłówku.
        feature_types : dict {nazwa: typ} or None
            Nadpisanie typów atrybutów. Klucz = nazwa atrybutu,
            wartość = ``'symbolic'``, ``'numeric 0'`` lub ``'numeric 1'``.
            Jeśli None — typy są inferowane automatycznie:
            warunki z ``<=`` / ``>`` → ``numeric 0``,
            warunki z ``=`` i wartością tekstową → ``symbolic``,
            warunki z ``=`` i wartością liczbową → ``numeric 0``.

        Example
        -------
        >>> rs.to_rul("wyniki.rul", ruleset_name="Moje_Reguły")
        """
        import os

        base = filepath if filepath.lower().endswith('.rul') else filepath + '.rul'

        # Zbierz wszystkie użyte atrybuty warunkowe (zachowaj kolejność)
        seen_attrs: List[str] = []
        for rule in self.rules:
            for c in rule.conditions:
                if c.attribute not in seen_attrs:
                    seen_attrs.append(c.attribute)

        def _fmt_num(v: Any) -> str:
            """Formatuje wartość do zapisu w .rul z pełną precyzją IEEE 754."""
            if isinstance(v, (float, np.floating)):
                pv = float(v)
                if pv == int(pv) and (pv == pv):   # int-like, nie NaN
                    return str(int(pv))
                return repr(pv)   # pełna precyzja, np. 2.45
            if isinstance(v, (int, np.integer)):
                return str(int(v))
            return str(v)

        def _infer_type(attr: str) -> str:
            """
            Inferuje typ RSES atrybutu:
            - operatory <= / > z float niecelkowitym  -> 'numeric 1'
            - operatory <= / > z wartością całkowitą  -> 'numeric 0'
            - operator = z wartością całkowitą        -> 'numeric 0'
            - operator = z float niecelkowitym        -> 'numeric 1'
            - wartość tekstowa                        -> 'symbolic'
            Nadpisywane przez feature_types.
            """
            if feature_types and attr in feature_types:
                return feature_types[attr]
            for rule in self.rules:
                for c in rule.conditions:
                    if c.attribute == attr:
                        try:
                            fv = float(c.value)
                            return "numeric 1" if fv != int(fv) else "numeric 0"
                        except (TypeError, ValueError):
                            return "symbolic"
            return "symbolic"

        # Zbierz unikalne wartości decyzyjne (zachowaj kolejność)
        seen_decisions: List[Any] = []
        for rule in self.rules:
            if rule.decision_value not in seen_decisions:
                seen_decisions.append(rule.decision_value)

        # Inferuj typ atrybutu decyzyjnego
        dec_attr = self.decision_attribute
        dec_type = "symbolic"
        if seen_decisions:
            try:
                fv = float(seen_decisions[0])
                dec_type = "numeric 1" if fv != int(fv) else "numeric 0"
            except (TypeError, ValueError):
                dec_type = "symbolic"

        with open(base, "w", encoding="utf-8") as f:
            # nagłówek
            f.write(f"RULE_SET {ruleset_name}\n")

            # atrybuty: warunkowe + decyzyjny
            all_attrs = seen_attrs + [dec_attr]
            f.write(f"ATTRIBUTES {len(all_attrs)}\n")
            for attr in seen_attrs:
                f.write(f" {attr} {_infer_type(attr)}\n")
            f.write(f" {dec_attr} {dec_type}\n")

            # wartości decyzyjne
            f.write(f"DECISION_VALUES {len(seen_decisions)}\n")
            for dv in seen_decisions:
                f.write(f"{dv}\n")

            # reguły
            f.write(f"RULES {len(self.rules)}\n")
            for rule in self.rules:
                cond_parts = [
                    f"({c.attribute}{c.operator}{_fmt_num(c.value)})"
                    for c in rule.conditions
                ]
                cond_str = "&".join(cond_parts)
                dval_str = _fmt_num(rule.decision_value)
                sup = rule.support
                conclusion = f"({dec_attr}={dval_str}[{sup}])"
                f.write(f"{cond_str}=>{conclusion} {sup}\n")

        # print(f"Saved {len(self.rules)} rules to '{base}'.")

    @classmethod
    def from_rul(
        cls,
        filepath: str,
        decision_attribute: Optional[str] = None,
    ) -> "RuleSet":
        """
        Wczytuje zbiór reguł z pliku w formacie RSES .rul.

        Obsługiwany format
        ------------------
        ::

            RULE_SET NazwaZbioru
            ATTRIBUTES n
             attr1 numeric 0
             attr2 symbolic
            DECISION_VALUES k
            val1
            ...
            RULES n
            (attr1=1)&(attr2=sunny)=>(dec=val[88]) 88
            ...

        Parameters
        ----------
        filepath : str
            Ścieżka do pliku .rul.
        decision_attribute : str or None
            Nadpisanie nazwy atrybutu decyzyjnego (domyślnie wykrywana
            z nagłówka reguł lub ustawiana na ``'decision'``).

        Returns
        -------
        RuleSet

        Raises
        ------
        FileNotFoundError
            Gdy plik nie istnieje.
        ValueError
            Gdy format pliku jest nieprawidłowy.

        Example
        -------
        >>> rs = RuleSet.from_rul("wyniki.rul")
        >>> rs.print_rules(include_stats=True)
        """
        import re

        with open(filepath, "r", encoding="utf-8") as f:
            lines = [line.rstrip('\n') for line in f if line.strip()]

        idx = 0

        # ── RULE_SET ──────────────────────────────────────────────────────
        if not lines[idx].startswith("RULE_SET"):
            raise ValueError(f"Expected 'RULE_SET ...' at line 1, got: {lines[idx]!r}")
        ruleset_name = lines[idx].split(None, 1)[1] if len(lines[idx].split()) > 1 else ""
        idx += 1

        # ── ATTRIBUTES ────────────────────────────────────────────────────
        if not lines[idx].startswith("ATTRIBUTES"):
            raise ValueError(f"Expected 'ATTRIBUTES n', got: {lines[idx]!r}")
        n_attrs = int(lines[idx].split()[1])
        idx += 1

        # Parsuj definicje atrybutów
        # Format: " nazwa_atrybutu typ [precyzja]"
        # Nazwa może zawierać spacje, więc używamy podziału od końca
        import re as _re
        attr_types: Dict[str, str] = {}   # nazwa -> 'symbolic' | 'numeric'
        attr_precisions: Dict[str, int] = {}  # nazwa -> 0 (int) | 1 (float)
        last_attr = None
        _attr_line_pat = _re.compile(
            r"^\s*(.*?)\s+(symbolic|numeric)\s*(\d+)?\s*$", _re.IGNORECASE
        )
        for _ in range(n_attrs):
            m = _attr_line_pat.match(lines[idx])
            if m:
                attr_name = m.group(1)
                attr_type = m.group(2).lower()
                precision = int(m.group(3)) if m.group(3) is not None else 0
            else:
                # Fallback: ostatnie słowo to typ
                parts = lines[idx].strip().rsplit(None, 1)
                attr_name = parts[0].strip() if len(parts) > 1 else lines[idx].strip()
                attr_type = parts[1].lower() if len(parts) > 1 else "symbolic"
                precision = 0
            attr_types[attr_name] = attr_type
            attr_precisions[attr_name] = precision
            last_attr = attr_name
            idx += 1

        # Heurystycznie ustal atrybut decyzyjny: ostatni atrybut w sekcji
        inferred_dec_attr = decision_attribute or last_attr or "decision"

        # ── DECISION_VALUES ───────────────────────────────────────────────
        if not lines[idx].startswith("DECISION_VALUES"):
            raise ValueError(f"Expected 'DECISION_VALUES k', got: {lines[idx]!r}")
        n_dec_vals = int(lines[idx].split()[1])
        idx += 1
        decision_values = []
        for _ in range(n_dec_vals):
            decision_values.append(lines[idx].strip())
            idx += 1

        # ── RULES ─────────────────────────────────────────────────────────
        if not lines[idx].startswith("RULES"):
            raise ValueError(f"Expected 'RULES n', got: {lines[idx]!r}")
        n_rules = int(lines[idx].split()[1])
        idx += 1

        # Parsowanie reguły: (attr OP val) & ... => (dec=val[sup]) sup
        # Wzorzec warunku: (cokolwiek)(operator)(cokolwiek_do_nawiasu)
        # Operator może być: <=, >=, !=, <, >, =
        # Wartość może zawierać cyfry, kropki, litery, spacje (nazwy symboliczne)
        # Używamy greedy match od prawej strony nawiasu
        _cond_pat = _re.compile(
            r"\((.+?)(<=|>=|!=|<(?!=)|(?<!<)>(?!=)|(?<![<>!=])=(?!=))([^)]+)\)"
        )
        _conc_pat = _re.compile(
            r"\((.+?)=([^\[)]+?)(?:\[(\d+)\])?\)\s*(\d+)?\s*$"
        )

        def _cast_value(val_str: str, attr_name: str) -> Any:
            """
            Rzutuje wartość z pliku .rul na właściwy typ Python:
            - atrybuty numeric -> float lub int (wg precyzji w nagłówku)
            - atrybuty symbolic -> str
            Jeśli atrybut nieznany, próbuje float, fallback str.
            """
            atype = attr_types.get(attr_name, "")
            prec  = attr_precisions.get(attr_name, 0)
            v = val_str.strip()
            if atype == "numeric":
                try:
                    fv = float(v)
                    # Zachowaj float jeśli precision=1 lub wartość niecelkowita
                    if prec == 1 or fv != int(fv):
                        return fv
                    return int(fv)
                except ValueError:
                    return v
            elif atype == "symbolic":
                return v
            else:
                # Nieznany typ - próbuj float
                try:
                    fv = float(v)
                    return int(fv) if fv == int(fv) else fv
                except ValueError:
                    return v

        rules: List[Rule] = []
        for _ in range(n_rules):
            if idx >= len(lines):
                break
            line = lines[idx].strip()
            idx += 1
            if not line or "=>" not in line:
                continue

            # Podziel na część warunkową i konkluzję po ostatnim "=>"
            arrow_pos = line.rfind("=>")
            cond_part = line[:arrow_pos]
            rest = line[arrow_pos + 2:]

            # Parsuj warunki
            conditions: List[Condition] = []
            for m in _cond_pat.finditer(cond_part):
                attr = m.group(1).strip()
                op   = m.group(2)
                val_raw = m.group(3).strip()
                # Normalizuj operatory do zbioru dozwolonego przez Condition
                op_norm = {">=": ">", "<": "<="}.get(op, op)
                val = _cast_value(val_raw, attr)
                conditions.append(Condition(attr, op_norm, val))

            # Parsuj konkluzję: (dec_attr=dec_val[sup]) sup
            rest_stripped = rest.strip()
            # Usuń trailing support (liczba po spacji na końcu linii)
            trailing_sup_m = _re.search(r"\s+(\d+)\s*$", rest_stripped)
            trailing_sup = int(trailing_sup_m.group(1)) if trailing_sup_m else 0

            m_conc = _conc_pat.search(rest_stripped)
            if m_conc:
                dec_attr_parsed = m_conc.group(1).strip()
                dec_val_raw     = m_conc.group(2).strip()
                bracket_sup     = int(m_conc.group(3)) if m_conc.group(3) else trailing_sup
                sup = bracket_sup
                dec_val = _cast_value(dec_val_raw, dec_attr_parsed)
            else:
                dec_attr_parsed = inferred_dec_attr
                dec_val = None
                sup = trailing_sup

            rule = Rule(
                conditions=conditions,
                decision_attribute=decision_attribute or dec_attr_parsed,
                decision_value=dec_val,
                support=sup,
                confidence=1.0,
            )
            rules.append(rule)

        # Atrybuty warunkowe = wszystko poza atrybutem decyzyjnym
        _eff_dec = decision_attribute or inferred_dec_attr
        feature_names = [a for a in attr_types if a != _eff_dec]

        rs = cls(
            rules=rules,
            decision_attribute=_eff_dec,
            feature_names=feature_names,
        )
        # print(f"Loaded {len(rs.rules)} rules from '{filepath}'.")
        return rs
    
    def sort(self):
        # ── sortowanie reguł malejąco po support i confidence ───────────────────────
        self.rules.sort(key=lambda r: (r.support, r.confidence), reverse=True)
        return self



# ─────────────────────────────────────────────────────────────────────────────
# Ekstrakcja reguł z drzewa ID3C45Classifier
# ─────────────────────────────────────────────────────────────────────────────

def extract_rules_from_tree(
    clf,
    decision_attribute: str = "decision",
) -> RuleSet:
    """
    Ekstrahuje reguły decyzyjne z dopasowanego ID3C45Classifier.

    Każda ścieżka od korzenia do liścia generuje jedną regułę.
    Warunki na ścieżce stają się częścią IF, klasa liścia — konkluzją.

    Parameters
    ----------
    clf : ID3C45Classifier
        Dopasowany klasyfikator (po wywołaniu fit()).
    decision_attribute : str, default 'decision'
        Nazwa atrybutu decyzyjnego w wygenerowanych regułach.

    Returns
    -------
    RuleSet
        Zbiór wyekstrahowanych reguł.

    Raises
    ------
    RuntimeError
        Jeśli klasyfikator nie jest dopasowany (brak tree_).

    Przykład
    --------
    >>> clf = ID3C45Classifier(algorithm="ID3", feature_names=["f1","f2"])
    >>> clf.fit(X, y)
    >>> rs = extract_rules_from_tree(clf, decision_attribute="klasa")
    >>> print(rs.to_series())
    >>> print(rs.to_dataframe())
    """
    if clf.tree_ is None:
        raise RuntimeError("Klasyfikator nie jest dopasowany. Wywołaj fit() przed ekstrakcją reguł.")

    n_total = clf.tree_.n_samples
    feature_names = clf.feature_names or [f"X[{i}]" for i in range(clf.n_features_in_)]
    class_names = clf.class_names

    rules: List[Rule] = []

    def _get_class_label(encoded_label: int) -> Any:
        """Dekoduje zakodowaną etykietę klasy do oryginalnej wartości."""
        original = clf.classes_[encoded_label]
        if class_names is not None and encoded_label < len(class_names):
            return class_names[encoded_label]
        return original

    def _traverse(node, conditions: List[Condition]):
        """Rekurencyjne przejście drzewa — depth-first."""
        if node.is_leaf:
            # Oblicz confidence: frakcja próbek z klasą większościową
            total = node.n_samples
            majority_count = node.class_counts.get(node.class_label, 0)
            conf = majority_count / total if total > 0 else 1.0

            dec_val = _get_class_label(node.class_label)

            rule = Rule(
                conditions=list(conditions),          # kopia bieżącej ścieżki
                decision_attribute=decision_attribute,
                decision_value=dec_val,
                support=total,
                confidence=conf,
                n_total=n_total,
            )
            rules.append(rule)
            return

        feat_name = node.feature_name or f"X[{node.feature_index}]"

        if node.threshold is not None:
            # ── gałąź numeryczna ──────────────────────────────────────
            left_cond  = Condition(feat_name, "<=", node.threshold)
            right_cond = Condition(feat_name, ">",  node.threshold)

            _traverse(node.children["left"],
                      conditions + [left_cond])
            _traverse(node.children["right"],
                      conditions + [right_cond])
        else:
            # ── gałąź kategoryczna ───────────────────────────────────
            for val, child in node.children.items():
                cond = Condition(feat_name, "=", val)
                _traverse(child, conditions + [cond])

    _traverse(clf.tree_, [])
    
    # # ── sortowanie reguł malejąco po n_total i confidence ───────────────────────
    # rules.sort(key=lambda r: (r.n_total, r.confidence), reverse=True)
    
    return RuleSet(
        rules=rules,
        decision_attribute=decision_attribute,
        feature_names=feature_names,
    )
