def rimuovi_collinearita(df):
    """
    Rimuove dal DataFrame alcune feature fortemente collineari
    individuate durante l'analisi esplorativa dei dati (EDA).

    In particolare vengono eliminate:
    - blueCSPerMin
    - redCSPerMin
    - blueAssists
    - redAssists

    Le variabili CSPerMin risultano ridondanti rispetto ad altre
    informazioni economiche e di farming già presenti nel dataset.
    Gli assist mostrano inoltre una forte correlazione con altre
    variabili legate al combattimento, all'oro e all'esperienza.

    La rimozione riduce la multicollinearità e limita la presenza
    di informazioni fortemente sovrapposte durante l'addestramento
    dei modelli.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame originale contenente le feature delle partite.

    Returns
    -------
    pandas.DataFrame
        Una copia del DataFrame senza le feature collineari
        selezionate.
    """

    colonne_da_rimuovere = [
        "blueCSPerMin",
        "redCSPerMin",
        "blueAssists",
        "redAssists"
    ]

    return df.drop(
        columns=colonne_da_rimuovere,
        errors="ignore"
    )


def rimuovi_ridondanze_speculari(df):
    """
    Rimuove le feature della squadra Red che rappresentano
    informazioni perfettamente speculari rispetto alle
    corrispondenti feature della squadra Blue.

    Durante l'EDA sono state individuate correlazioni perfette
    pari a +1 o -1, dovute a dipendenze funzionali tra le feature.

    In particolare:
    - redFirstBlood è complementare a blueFirstBlood;
    - redKills corrisponde a blueDeaths;
    - redDeaths corrisponde a blueKills;
    - redGoldDiff è l'opposto di blueGoldDiff;
    - redExperienceDiff è l'opposto di blueExperienceDiff.

    Mantenere entrambe le rappresentazioni non aggiunge nuova
    informazione e introduce multicollinearità perfetta.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame originale contenente le feature delle partite.

    Returns
    -------
    pandas.DataFrame
        Una copia del DataFrame senza le feature speculari
        della squadra Red.
    """

    colonne_da_rimuovere = [
        "redFirstBlood",
        "redKills",
        "redDeaths",
        "redGoldDiff",
        "redExperienceDiff"
    ]

    return df.drop(
        columns=colonne_da_rimuovere,
        errors="ignore"
    )


def rimuovi_aggregati(df):
    """
    Rimuove le feature aggregate relative agli Elite Monsters.

    I controlli di consistenza effettuati durante l'EDA hanno
    mostrato che:

        blueEliteMonsters =
            blueDragons + blueHeralds

        redEliteMonsters =
            redDragons + redHeralds

    Le variabili EliteMonsters sono quindi combinazioni lineari
    esatte di Dragons e Heralds e non introducono nuova
    informazione nel dataset.

    Vengono mantenute separatamente le feature relative a Draghi
    e Araldi, poiché rappresentano obiettivi di gioco differenti.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame originale contenente le feature delle partite.

    Returns
    -------
    pandas.DataFrame
        Una copia del DataFrame senza le feature aggregate
        EliteMonsters.
    """

    colonne_da_rimuovere = [
        "blueEliteMonsters",
        "redEliteMonsters"
    ]

    return df.drop(
        columns=colonne_da_rimuovere,
        errors="ignore"
    )