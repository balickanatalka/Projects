"""
ExtDataframe.py
============
Podklasa pd.DataFrame z obsługą formatu RSES (.tab).

Import:
    from decision_tree_lib import DataFrame
    # lub bezpośrednio:
    from decision_tree_lib.Dataframe import DataFrame

Przykład:
    df = DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
    df.to_rses_tab('output.tab')       # zapis
    df2 = DataFrame.from_rses_tab('output.tab')  # odczyt

UWAGA:
    Floaty są zapisywane z pełną precyzją Pythona.
    Kolumny bool są traktowane jako symbolic (True/False jako stringi).
"""

import csv
import io
import os
import re

import numpy as np
import pandas as pd


# ─── helper: rozpoznaj typ kolumny dla RSES ──────────────────────────────────

def _rses_col_type(series: pd.Series) -> str:
    """
    Zwraca typ RSES kolumny:
    - 'symbolic'  dla string / object / category / bool
    - 'numeric 0' dla integer
    - 'numeric 1' dla float
    Kompatybilny z pandas 1.x (dtype object) i 3.x (StringDtype).
    """
    dt = series.dtype

    # pandas StringDtype (pandas >= 1.0, dominuje w 3.x)
    if isinstance(dt, pd.StringDtype):
        return "symbolic"

    # bool -> symbolic (True/False zapisane jako stringi)
    if dt == bool or (hasattr(np, 'bool_') and np.issubdtype(dt, np.bool_)):
        return "symbolic"

    # Stary object dtype (strings, mixed)
    if dt == object:
        return "symbolic"

    # Pandas CategoricalDtype
    if isinstance(dt, pd.CategoricalDtype):
        return "symbolic"

    # Pandas nullable integer (Int8, Int16, Int32, Int64, UInt*, ...)
    if isinstance(dt, pd.api.extensions.ExtensionDtype):
        dtype_str = str(dt).lower()
        if 'int' in dtype_str:
            return "numeric 0"
        if 'float' in dtype_str:
            return "numeric 1"
        return "symbolic"

    try:
        if np.issubdtype(dt, np.integer):
            return "numeric 0"
        if np.issubdtype(dt, np.floating):
            return "numeric 1"
    except TypeError:
        pass

    # Fallback
    return "symbolic"


# ─── ExtDataFrame ────────────────────────────────────────────────────────────────
class ExtDataFrame(pd.DataFrame):
    """
    Podklasa pd.DataFrame z dodatkowymi metodami dla formatu RSES (.tab).

    Metody
    ------
    to_rses_tab(filepath)
        Zapisuje DataFrame do pliku RSES .tab.
    from_rses_tab(filepath)  [classmethod]
        Wczytuje DataFrame z pliku RSES .tab.
    """

    def __init__(self, *args, **kwargs):
        pd.DataFrame.__init__(self, *args, **kwargs)

    @property
    def _constructor(self):
        """Zachowuje podklasę przy operacjach pandas."""
        return ExtDataFrame

    # ── zapis .tab ────────────────────────────────────────────────────────

    def to_rses_tab(self, filepath: str) -> None:
        """
        Zapisuje DataFrame do pliku RSES .tab.

        Format pliku
        ------------
        ::

            TABLE nazwa
            ATTRIBUTES n
             kolumna1 symbolic
             kolumna2 numeric 0
             kolumna3 numeric 1
            OBJECTS m
            val1 val2 val3
            ...

        Reguły typowania
        ----------------
        - string / object / bool / category -> ``symbolic``
        - integer (w tym pandas nullable Int64) -> ``numeric 0``
        - float -> ``numeric 1``

        Parameters
        ----------
        filepath : str
            Sciezka wyjsciowa. Rozszerzenie jest zastepowane przez ``.tab``.

        Notes
        -----
        Nazwy kolumn ze spacjami sa zamieniane na podkreslenia.
        """
        # Kopia zeby nie mutowac oryginalu
        df = self.copy()
        df.columns = df.columns.str.replace(' ', '_')

        base = filepath if not filepath.lower().endswith('.tab') else filepath[:-4]
        tab_filepath = base + '.tab'
        table_name = os.path.splitext(os.path.basename(base))[0]

        with open(tab_filepath, "w", encoding="utf-8") as f:
            f.write(f"TABLE {table_name}\n")
            f.write(f"ATTRIBUTES {len(df.columns)}\n")

            for col in df.columns:
                rtype = _rses_col_type(df[col])
                f.write(f" {col} {rtype}\n")

            f.write(f"OBJECTS {len(df)}\n")

            for row in df.itertuples(index=False, name=None):
                parts = []
                for col, val in zip(df.columns, row):
                    if pd.isna(val) if not isinstance(val, str) else False:
                        s = "?"
                    else:
                        if pd.api.types.is_integer_dtype(df[col]):
                            s = str(val)  # tutaj NIE będzie 1.0
                        elif pd.api.types.is_float_dtype(df[col]):
                            s = str(val)
                        else:
                            s = str(val)
                    if ' ' in s:
                        s = f'"{s}"'
                    parts.append(s)
                f.write(" ".join(parts) + "\n")

        print(f"File {tab_filepath} has been created.")

    # ── odczyt .tab ───────────────────────────────────────────────────────

    @classmethod
    def from_rses_tab(cls, filepath: str) -> "DataFrame":
        """
        Wczytuje DataFrame z pliku RSES .tab.

        Parameters
        ----------
        filepath : str
            Sciezka do pliku .tab.

        Returns
        -------
        DataFrame
            DataFrame z typami kolumn zgodnymi z naglowkiem RSES:
            ``symbolic``  -> ``object`` (str),
            ``numeric 0`` -> ``Int64``  (nullable integer),
            ``numeric 1`` -> ``float64``.

        Raises
        ------
        FileNotFoundError
            Gdy plik nie istnieje.
        ValueError
            Gdy naglowek pliku jest nieprawidlowy.

        Example
        -------
        >>> df = DataFrame.from_rses_tab('dane.tab')
        >>> print(df.dtypes)
        """
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read()

        lines = [line.rstrip('\n') for line in raw.splitlines()]
        non_empty = [l for l in lines if l.strip() != '']

        idx = 0

        # TABLE
        if not non_empty[idx].strip().startswith("TABLE"):
            raise ValueError(f"Expected 'TABLE ...', got: {non_empty[idx]!r}")
        idx += 1

        # ATTRIBUTES n
        if not non_empty[idx].strip().startswith("ATTRIBUTES"):
            raise ValueError(f"Expected 'ATTRIBUTES n', got: {non_empty[idx]!r}")
        n_attrs = int(non_empty[idx].split()[1])
        idx += 1

        # Definicje atrybutow: " nazwa typ [precyzja]"
        # Nazwa moze zawierac spacje - parsujemy od konca
        _attr_pat = re.compile(
            r"^\s*(.*?)\s+(symbolic|numeric)\s*(\d+)?\s*$", re.IGNORECASE
        )
        col_names = []
        col_types = []  # 'symbolic' | 'numeric_int' | 'numeric_float'

        for _ in range(n_attrs):
            line = non_empty[idx]
            m = _attr_pat.match(line)
            if m:
                name = m.group(1).strip()
                atype = m.group(2).lower()
                prec = int(m.group(3)) if m.group(3) is not None else 0
            else:
                parts = line.strip().rsplit(None, 1)
                name = parts[0].strip() if len(parts) > 1 else line.strip()
                atype = parts[1].lower() if len(parts) > 1 else "symbolic"
                prec = 0
            col_names.append(name)
            if atype == "symbolic":
                col_types.append("symbolic")
            elif atype == "numeric":
                col_types.append("numeric_int" if prec == 0 else "numeric_float")
            else:
                col_types.append("symbolic")
            idx += 1

        # OBJECTS n
        if not non_empty[idx].strip().startswith("OBJECTS"):
            raise ValueError(f"Expected 'OBJECTS n', got: {non_empty[idx]!r}")
        n_objects = int(non_empty[idx].split()[1])
        idx += 1

        # Wiersze danych: separator spacja, cudzyslow dla wartosci ze spacja
        data_lines = non_empty[idx: idx + n_objects]
        rows = []
        for dline in data_lines:
            reader = csv.reader(
                [dline],
                delimiter=' ',
                quotechar='"',
                skipinitialspace=True,
            )
            row = next(reader)
            # Usun puste tokeny (podwojne spacje)
            row = [tok for tok in row if tok != '']
            rows.append(row)

        df_raw = pd.DataFrame(rows, columns=col_names)

        # Rzutuj typy
        for col, ctype in zip(col_names, col_types):
            if ctype == "numeric_int":
                df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').astype('Int64')
            elif ctype == "numeric_float":
                df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').astype(float)
            # symbolic: zostaje jako object/str

        result = cls(df_raw)
        print(f"Loaded {len(result)} rows x {len(result.columns)} columns from '{filepath}'.")
        return result
