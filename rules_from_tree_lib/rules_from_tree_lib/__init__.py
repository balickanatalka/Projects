"""
rules_from_tree_lib
=================
Biblioteka do generowania drzew decyzyjnych ID3/C4.5,
indukowania reguł decyzyjnych oraz obsługi formatów RSES (.tab, .rul).

Szybki start
------------
::

    from rules_from_tree_lib import ID3C45Classifier, RuleSet, ExtDataFrame
    from rules_from_tree_lib import extract_rules_from_tree, remove_inconsistencies

    # 1. Wczytaj dane z CSV lub .tab
    import pandas as pd
    df_raw = pd.read_csv("dane.csv")
    # lub: df_raw = ExtDataFrame.from_rses_tab("dane.tab")

    # 2. Usuń niespójności (opcjonalnie)
    df_clean = remove_inconsistencies(df_raw, decision_col="klasa")

    # 3. Wytrenuj drzewo C4.5
    X = df_clean.drop(columns=["klasa"]).values
    y = df_clean["klasa"].values

    clf = ID3C45Classifier(
        algorithm="C4.5",
        feature_names=list(df_clean.columns[:-1]),
    )
    clf.fit(X, y)
    print(clf.export_text())

    # 4. Wyindukuj reguły
    rs = extract_rules_from_tree(clf, decision_attribute="klasa")
    rs = rs.simplify()
    rs.print_rules(include_stats=True)

    # 5. Zapisz do .rul
    rs.to_rul("wyniki.rul", ruleset_name="MojeReguły")

    # 6. Wczytaj reguły z .rul
    rs2 = RuleSet.from_rul("wyniki.rul")

    # 7. Zapisz ExtDataFrame do .tab
    rdf = ExtDataFrame(df_clean)
    rdf.to_rses_tab("dane_clean.tab")

Publiczne API
-------------
Klasy:
    ID3C45Classifier    - klasyfikator drzewa decyzyjnego ID3/C4.5
    RuleSet             - zbiór reguł decyzyjnych (z to_rul / from_rul)
    Rule                - pojedyncza reguła decyzyjna
    Condition           - pojedynczy warunek reguły
    ExtDataFrame           - pd.DataFrame z obsługą formatu RSES .tab

Funkcje:
    extract_rules_from_tree(clf, decision_attribute)  - ekstrakcja reguł z drzewa
    remove_inconsistencies(df, decision_col)          - usuwanie niespójności
    plot_tree(clf, **kwargs)                          - wizualizacja drzewa
    export_text(clf, **kwargs)                        - eksport drzewa jako tekst
    export_graphviz(clf, **kwargs)                    - eksport do formatu DOT
"""

# ── klasyfikator ──────────────────────────────────────────────────────────────
from .id3_c45_classifier import (
    ID3C45Classifier,
    plot_tree,
    export_text,
    export_graphviz,
)

# ── reguły ────────────────────────────────────────────────────────────────────
from .roughset import (
    Condition,
    Rule,
    RuleSet,
    extract_rules_from_tree,
)

# ── DataFrame z obsługą .tab ──────────────────────────────────────────────────
from .ExtDataFrame import ExtDataFrame

# ── preprocessing ─────────────────────────────────────────────────────────────
from .consistencies import remove_inconsistencies

__all__ = [
    # klasyfikator
    "ID3C45Classifier",
    "plot_tree",
    "export_text",
    "export_graphviz",
    # reguły
    "Condition",
    "Rule",
    "RuleSet",
    "extract_rules_from_tree",
    # I/O
    "ExtDataFrame",
    # preprocessing
    "remove_inconsistencies",
]

__version__ = "1.1.0"
__author__ = "decision_tree_lib"
