import pandas as pd

def remove_inconsistencies(df, decision_col: str = None):
    """
    Usuwa niespójne wiersze z tablicy decyzyjnej (DataFrame) oraz duplikaty, pozostawiając
    tylko unikalne wiersze z decyzją o największym wsparciu.

    Niespójne wiersze to takie, które mają identyczne wartości atrybutów, ale różne wartości decyzji.
    W przypadku wystąpienia niespójności, wiersze z najmniejszym wsparciem dla danej decyzji są usuwane.

    Parametry:
    ----------
    df : pd.DataFrame
        DataFrame zawierający tablicę decyzyjną, w której należy usunąć niespójności.
    decision_col : str, domyślnie None
        Nazwa kolumny zawierającej decyzję, której niespójność ma być usunięta.

    Zwraca:
    -------
    pd.DataFrame
        Nowy DataFrame bez niespójnych wierszy i duplikatów, z unikalnymi wartościami
        decyzji o największym wsparciu.

    Przykład:
    ---------
    >>> df = pd.DataFrame({
    ...     'attr1': [1, 1, 1, 2, 2, 2],
    ...     'attr2': [3, 3, 3, 4, 4, 4],
    ...     'decision': [0, 0, 1, 1, 1, 0]
    ... })
    >>> df_cleaned = remove_inconsistencies(df)
    >>> print(df_cleaned)

    """
    #decision_col = last column from df.columns
    if decision_col is None:
        decision_col = df.columns[-1]

    # Grupowanie wierszy na podstawie atrybutów (bez decyzji)
    attributes = df.columns.difference([decision_col])

    # # Opcja 1: Tylko kategorie, które faktycznie istnieją w danych (zalecane)
    # grouped = df.groupby(list(attributes), observed=True)
    # Opcja 2: Wszystkie kategorie, nawet puste (stare zachowanie)
    grouped = df.groupby(list(attributes), observed=False)
    
    # Znajdowanie niezgodnych grup
    inconsistent_indices = []
    for _, group in grouped:
        # Sprawdź, czy w grupie występują różne wartości decyzji
        if group[decision_col].nunique() > 1:
            #display(group)

            # Znajdź wsparcie dla każdej decyzji w grupie
            decision_counts = group[decision_col].value_counts()
            #display(decision_counts)

            # Znajdź decyzję o największym wsparciu
            max_support_decision = decision_counts.idxmax()
            # Dodaj do listy indeksy wierszy z mniejszym wsparciem (do usunięcia)
            inconsistent_indices.extend(group[group[decision_col] != max_support_decision].index)
    
    # Usuń wiersze z niezgodnymi decyzjami o mniejszym wsparciu
    df_cleaned = df.drop(inconsistent_indices)
    
    # Usuń duplikaty
    df_cleaned = df_cleaned.drop_duplicates()
    
    return df_cleaned


# # Przykład użycia
# # Zakładamy, że masz już DataFrame o nazwie `df` zawierający kolumnę "decision"
# df = pd.DataFrame({
#     'attr1':    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2],
#     'attr2':    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4],
#     'decision': [1, 2, 0, 1, 0, 2, 2, 2, 0, 1, 1, 1, 0]
# })

# df_cleaned = remove_inconsistencies(df)
# print("Tablica po usunięciu niespójności i duplikatów:")
# print(df_cleaned)
