from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# =========================================================
# CONFIGURAZIONE PAGINA
# =========================================================

# Deve essere una delle prime istruzioni Streamlit eseguite.
st.set_page_config(
    page_title="PredictLoL",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# PERCORSI E MODELLI
# =========================================================

# =========================================================
# PERCORSI E MODELLI
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"


@st.cache_resource
def load_models_and_scaler():
    """
    Carica i sei modelli addestrati e lo StandardScaler
    utilizzato durante il preprocessing.
    """

    model_paths = {
        "Random Forest": MODELS_DIR / "random_forest_best.pkl",
        "Logistic Regression": MODELS_DIR / "logistic_regression.pkl",
        "Support Vector Machine": MODELS_DIR / "svm.pkl",
        "Decision Tree": MODELS_DIR / "decision_tree.pkl",
        "K-Nearest Neighbors": MODELS_DIR / "knn_best.pkl",
        "Gaussian Naive Bayes": MODELS_DIR / "naive_bayes_best.pkl",
    }

    models = {
        model_name: joblib.load(model_path)
        for model_name, model_path in model_paths.items()
    }

    scaler = joblib.load(
        MODELS_DIR / "scaler.pkl"
    )

    return models, scaler


try:
    models, scaler = load_models_and_scaler()

except FileNotFoundError as error:
    st.error(
        "Impossibile caricare uno o più modelli. "
        "Controlla il contenuto della cartella models/."
    )

    st.code(str(error))
    st.stop()

@st.cache_data
def load_test_data():
    """
    Carica il Test Set originale e il relativo target.
    """

    x_test_path = PROJECT_ROOT / "data" / "X_test.csv"
    y_test_path = PROJECT_ROOT / "data" / "y_test.csv"

    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)

    return X_test, y_test


try:
    X_test_examples, y_test_examples = load_test_data()

except FileNotFoundError:
    X_test_examples = None
    y_test_examples = None


def load_example_match():

    if X_test_examples is None:
        return

    # Prende una partita casuale dal Test Set
    example_row = X_test_examples.sample(
        n=1
    ).iloc[0]
    example_index = example_row.name

    # BLUE TEAM
    st.session_state["blue_gold"] = int(
        example_row["blueTotalGold"]
    )

    st.session_state["blue_xp"] = int(
        example_row["blueTotalExperience"]
    )

    st.session_state["blue_level"] = float(
        example_row["blueAvgLevel"]
    )

    st.session_state["blue_kills"] = int(
        example_row["blueKills"]
    )

    st.session_state["blue_dragons"] = int(
        example_row["blueDragons"]
    )

    st.session_state["blue_heralds"] = int(
        example_row["blueHeralds"]
    )

    st.session_state["blue_towers"] = int(
        example_row["blueTowersDestroyed"]
    )

    st.session_state["blue_minions"] = int(
        example_row["blueTotalMinionsKilled"]
    )

    st.session_state["blue_jungle"] = int(
        example_row["blueTotalJungleMinionsKilled"]
    )

    st.session_state["blue_wards_placed"] = int(
        example_row["blueWardsPlaced"]
    )

    st.session_state["blue_wards_destroyed"] = int(
        example_row["blueWardsDestroyed"]
    )


    # RED TEAM
    st.session_state["red_gold"] = int(
        example_row["redTotalGold"]
    )

    st.session_state["red_xp"] = int(
        example_row["redTotalExperience"]
    )

    st.session_state["red_level"] = float(
        example_row["redAvgLevel"]
    )

    st.session_state["red_kills"] = int(
        example_row["blueDeaths"]
    )

    st.session_state["red_dragons"] = int(
        example_row["redDragons"]
    )

    st.session_state["red_heralds"] = int(
        example_row["redHeralds"]
    )

    st.session_state["red_towers"] = int(
        example_row["redTowersDestroyed"]
    )

    st.session_state["red_minions"] = int(
        example_row["redTotalMinionsKilled"]
    )

    st.session_state["red_jungle"] = int(
        example_row["redTotalJungleMinionsKilled"]
    )

    st.session_state["red_wards_placed"] = int(
        example_row["redWardsPlaced"]
    )

    st.session_state["red_wards_destroyed"] = int(
        example_row["redWardsDestroyed"]
    )
        # FIRST BLOOD
    if int(example_row["blueFirstBlood"]) == 1:
        st.session_state["first_blood_input"] = "Blue Team"
    else:
        st.session_state["first_blood_input"] = "Red Team"


    # VERO VINCITORE DELLA PARTITA
    actual_result = int(
        y_test_examples.loc[
            example_index,
            "blueWins"
        ]
    )

    if actual_result == 1:
        st.session_state["example_actual_winner"] = "BLUE TEAM"
    else:
        st.session_state["example_actual_winner"] = "RED TEAM"

    st.session_state["example_loaded"] = True


# =========================================================
# STILE
# =========================================================

st.markdown(
    """
    <style>

    /* =====================================================
       PAGINA PRINCIPALE
       ===================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at top,
                #162436 0%,
                #0b1420 45%,
                #050a0f 100%
            );
        color: #f0e6d2;
    }

    /* Nasconde alcuni elementi standard di Streamlit */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    /* =====================================================
       HEADER
       ===================================================== */

    .main-title {
        text-align: center;
        font-size: 3.4rem;
        font-weight: 700;
        letter-spacing: 4px;
        color: #c8aa6e;
        margin-bottom: 0;
        text-shadow:
            0 0 18px rgba(200, 170, 110, 0.20);
    }

    .subtitle {
        text-align: center;
        color: #a09b8c;
        font-size: 1.05rem;
        letter-spacing: 1px;
        margin-bottom: 1.5rem;
    }

    .minute-badge {
        width: fit-content;
        margin: 0 auto 2rem auto;
        padding: 7px 22px;

        border: 1px solid #785a28;
        background-color: rgba(10, 20, 30, 0.75);

        color: #c8aa6e;

        border-radius: 4px;

        font-weight: 600;
        letter-spacing: 2px;
    }


    /* =====================================================
       TITOLI DELLE SQUADRE
       ===================================================== */

    .blue-title {
        text-align: center;
        color: #5bb7ff;

        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: 2px;

        padding: 10px;

        border-bottom: 2px solid #1e90ff;
        margin-bottom: 20px;
    }

    .red-title {
        text-align: center;
        color: #ff6b6b;

        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: 2px;

        padding: 10px;

        border-bottom: 2px solid #c0392b;
        margin-bottom: 20px;
    }


    /* =====================================================
       RISULTATO
       ===================================================== */

    .result-box {
        text-align: center;

        padding: 28px;
        margin-top: 25px;
        margin-bottom: 25px;

        border: 1px solid #785a28;

        background:
            linear-gradient(
                180deg,
                rgba(18, 31, 46, 0.96),
                rgba(6, 13, 20, 0.96)
            );

        color: #c8aa6e;

        border-radius: 6px;

        box-shadow:
            0 0 25px rgba(200, 170, 110, 0.08);
    }

    .result-label {
        color: #a09b8c;
        font-size: 0.95rem;
        letter-spacing: 2px;
        margin-bottom: 8px;
    }

    .blue-winner {
        color: #5bb7ff;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: 3px;
    }

    .red-winner {
        color: #ff6b6b;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: 3px;
    }

    .confidence {
        margin-top: 15px;
        color: #f0e6d2;
        font-size: 1.2rem;
    }

    .model-name {
        margin-top: 7px;
        color: #a09b8c;
        font-size: 0.9rem;
    }


    /* =====================================================
       PULSANTI
       ===================================================== */

    div.stButton > button {
        width: 100%;

        background:
            linear-gradient(
                180deg,
                #c8aa6e 0%,
                #785a28 100%
            );

        color: #091428;

        border: 1px solid #c8aa6e;
        border-radius: 4px;

        padding: 0.8rem;

        font-weight: 700;
        letter-spacing: 1px;
    }

    div.stButton > button:hover {
        border-color: #f0e6d2;
        color: white;
    }


    /* =====================================================
       TABS
       ===================================================== */

    .stTabs [data-baseweb="tab-list"] {
        gap: 30px;
        justify-content: center;
    }

    .stTabs [data-baseweb="tab"] {
        color: #a09b8c;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">PREDICT LoL</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Early-game Machine Learning prediction
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="minute-badge">
        GAME STATE · 10:00
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# TABS
# =========================================================

tab_predict, tab_compare, tab_about = st.tabs(
    [
        "⚔ Predict",
        "◈ Compare Models",
        "ℹ About",
    ]
)


# =========================================================
# TAB 1 — PREDICT
# =========================================================

with tab_predict:

    st.write("")

    # =====================================================
    # LOAD EXAMPLE MATCH
    # =====================================================

    example_col1, example_col2 = st.columns(
        [1, 3]
    )

    with example_col1:
        st.button(
    "LOAD EXAMPLE MATCH",
    key="load_example_match",
    use_container_width=True,
    on_click=load_example_match,
)

    with example_col2:
        st.caption(
            "Carica una partita reale del Test Set "
            "e compila automaticamente le statistiche."
        )
        

            

    st.write("")

    blue_col, red_col = st.columns(
        2,
        gap="large",
    )


    # =====================================================
    # BLUE TEAM
    # =====================================================

    with blue_col:

        st.markdown(
            '<div class="blue-title">BLUE TEAM</div>',
            unsafe_allow_html=True,
        )

        blue_total_gold = st.number_input(
            "Total Gold",
            min_value=0,
            value=16500,
            step=100,
            key="blue_gold",
        )

        blue_total_experience = st.number_input(
            "Total Experience",
            min_value=0,
            value=18000,
            step=100,
            key="blue_xp",
        )

        blue_avg_level = st.number_input(
            "Average Level",
            min_value=1.0,
            max_value=18.0,
            value=7.0,
            step=0.1,
            key="blue_level",
        )

        blue_kills = st.number_input(
            "Kills",
            min_value=0,
            value=6,
            step=1,
            key="blue_kills",
        )

        blue_dragons = st.number_input(
            "Dragons",
            min_value=0,
            value=0,
            step=1,
            key="blue_dragons",
        )

        blue_heralds = st.number_input(
            "Rift Heralds",
            min_value=0,
            value=0,
            step=1,
            key="blue_heralds",
        )

        blue_towers = st.number_input(
            "Towers Destroyed",
            min_value=0,
            value=0,
            step=1,
            key="blue_towers",
        )

        blue_minions = st.number_input(
            "Lane Minions Killed",
            min_value=0,
            value=220,
            step=1,
            key="blue_minions",
        )

        blue_jungle_minions = st.number_input(
            "Jungle Minions Killed",
            min_value=0,
            value=50,
            step=1,
            key="blue_jungle",
        )

        blue_wards_placed = st.number_input(
            "Wards Placed",
            min_value=0,
            value=20,
            step=1,
            key="blue_wards_placed",
        )

        blue_wards_destroyed = st.number_input(
            "Wards Destroyed",
            min_value=0,
            value=3,
            step=1,
            key="blue_wards_destroyed",
        )


    # =====================================================
    # RED TEAM
    # =====================================================

    with red_col:

        st.markdown(
            '<div class="red-title">RED TEAM</div>',
            unsafe_allow_html=True,
        )

        red_total_gold = st.number_input(
            "Total Gold",
            min_value=0,
            value=16500,
            step=100,
            key="red_gold",
        )

        red_total_experience = st.number_input(
            "Total Experience",
            min_value=0,
            value=18000,
            step=100,
            key="red_xp",
        )

        red_avg_level = st.number_input(
            "Average Level",
            min_value=1.0,
            max_value=18.0,
            value=7.0,
            step=0.1,
            key="red_level",
        )

        red_kills = st.number_input(
            "Kills",
            min_value=0,
            value=6,
            step=1,
            key="red_kills",
        )

        red_dragons = st.number_input(
            "Dragons",
            min_value=0,
            value=0,
            step=1,
            key="red_dragons",
        )

        red_heralds = st.number_input(
            "Rift Heralds",
            min_value=0,
            value=0,
            step=1,
            key="red_heralds",
        )

        red_towers = st.number_input(
            "Towers Destroyed",
            min_value=0,
            value=0,
            step=1,
            key="red_towers",
        )

        red_minions = st.number_input(
            "Lane Minions Killed",
            min_value=0,
            value=220,
            step=1,
            key="red_minions",
        )

        red_jungle_minions = st.number_input(
            "Jungle Minions Killed",
            min_value=0,
            value=50,
            step=1,
            key="red_jungle",
        )

        red_wards_placed = st.number_input(
            "Wards Placed",
            min_value=0,
            value=20,
            step=1,
            key="red_wards_placed",
        )

        red_wards_destroyed = st.number_input(
            "Wards Destroyed",
            min_value=0,
            value=3,
            step=1,
            key="red_wards_destroyed",
        )


    # =====================================================
    # FIRST BLOOD
    # =====================================================

    st.write("")

    first_blood = st.radio(
        "First Blood",
        [
            "Blue Team",
            "Red Team",
        ],
        horizontal=True,
        key="first_blood_input",
    )
    


    # =====================================================
    # SELEZIONE MODELLO
    # =====================================================

    st.write("")

    selected_model = st.selectbox(
        "Prediction Model",
        [
            "Random Forest",
            "Logistic Regression",
            "Support Vector Machine",
            "Decision Tree",
            "K-Nearest Neighbors",
            "Gaussian Naive Bayes",
        ],
    )


    # =====================================================
    # PULSANTE PREDIZIONE
    # =====================================================

    st.write("")

    predict_button = st.button(
        "PREDICT WINNER"
    )


    # =====================================================
    # PREDIZIONE
    # =====================================================

    if predict_button:

        # =================================================
        # COSTRUZIONE DELLE FEATURE
        # =================================================

        blue_first_blood = (
            1
            if first_blood == "Blue Team"
            else 0
        )

        blue_gold_diff = (
            blue_total_gold
            - red_total_gold
        )

        blue_experience_diff = (
            blue_total_experience
            - red_total_experience
        )

        blue_gold_per_min = (
            blue_total_gold / 10
        )

        red_gold_per_min = (
            red_total_gold / 10
        )

        # Nel dataset:
        # blueDeaths = redKills
        blue_deaths = red_kills


        # =================================================
        # 27 FEATURE FINALI
        # =================================================

        features = {

            "blueWardsPlaced":
                blue_wards_placed,

            "blueWardsDestroyed":
                blue_wards_destroyed,

            "blueFirstBlood":
                blue_first_blood,

            "blueKills":
                blue_kills,

            "blueDeaths":
                blue_deaths,

            "blueDragons":
                blue_dragons,

            "blueHeralds":
                blue_heralds,

            "blueTowersDestroyed":
                blue_towers,

            "blueTotalGold":
                blue_total_gold,

            "blueAvgLevel":
                blue_avg_level,

            "blueTotalExperience":
                blue_total_experience,

            "blueTotalMinionsKilled":
                blue_minions,

            "blueTotalJungleMinionsKilled":
                blue_jungle_minions,

            "blueGoldDiff":
                blue_gold_diff,

            "blueExperienceDiff":
                blue_experience_diff,

            "blueGoldPerMin":
                blue_gold_per_min,

            "redWardsPlaced":
                red_wards_placed,

            "redWardsDestroyed":
                red_wards_destroyed,

            "redDragons":
                red_dragons,

            "redHeralds":
                red_heralds,

            "redTowersDestroyed":
                red_towers,

            "redTotalGold":
                red_total_gold,

            "redAvgLevel":
                red_avg_level,

            "redTotalExperience":
                red_total_experience,

            "redTotalMinionsKilled":
                red_minions,

            "redTotalJungleMinionsKilled":
                red_jungle_minions,

            "redGoldPerMin":
                red_gold_per_min,
        }


        # =================================================
        # DATAFRAME
        # =================================================

        input_df = pd.DataFrame(
            [features]
        )

        feature_order = list(
            scaler.feature_names_in_
        )

        input_df = input_df[
            feature_order
        ]


        # =================================================
        # VERSIONE STANDARDIZZATA
        # =================================================

        input_scaled = pd.DataFrame(
            scaler.transform(input_df),
            columns=feature_order,
        )


        # =================================================
        # SELEZIONE DEL MODELLO
        # =================================================

        model = models[
            selected_model
        ]


        # Il Decision Tree è stato addestrato
        # sulle feature NON standardizzate.
        if selected_model == "Decision Tree":

            model_input = input_df

        else:

            model_input = input_scaled


        # =================================================
        # PREDIZIONE
        # =================================================

        prediction = int(
            model.predict(model_input)[0]
        )


        # =================================================
        # CONFIDENCE / PROBABILITÀ
        # =================================================

        predicted_probability = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                model_input
            )[0]

            class_index = list(
                model.classes_
            ).index(prediction)

            predicted_probability = (
                probabilities[class_index]
            )


        # =================================================
        # RISULTATO
        # =================================================

        if prediction == 1:

            winner = "BLUE TEAM"
            winner_class = "blue-winner"

        else:

            winner = "RED TEAM"
            winner_class = "red-winner"


        # Alcuni modelli, come la SVM utilizzata
        # nel progetto, potrebbero non fornire
        # probabilità tramite predict_proba.
        if predicted_probability is not None:

            confidence_html = (
                f'<div class="confidence">'
                f'Confidence: '
                f'{predicted_probability:.1%}'
                f'</div>'
            )

        else:

            confidence_html = (
                '<div class="confidence">'
                'Confidence: not available'
                '</div>'
            )


        result_html = (
            '<div class="result-box">'

            '<div class="result-label">'
            'PREDICTED WINNER'
            '</div>'

            f'<div class="{winner_class}">'
            f'{winner}'
            '</div>'

            f'{confidence_html}'

            '<div class="model-name">'
            f'{selected_model}'
            '</div>'

            '</div>'
        )


        st.markdown(
            result_html,
            unsafe_allow_html=True,
        )

        actual_winner = st.session_state.get(
            "example_actual_winner"
        )

        if actual_winner is not None:

            st.write("")

            st.markdown(
                f"**Actual Winner:** {actual_winner}"
            )

            if winner == actual_winner:
                st.success("CORRECT PREDICTION")
            else:
                st.error("WRONG PREDICTION")


# =========================================================
# TAB 2 — COMPARE MODELS
# =========================================================

# =========================================================
# TAB 2 — COMPARE MODELS
# =========================================================

with tab_compare:

    st.markdown(
        "### Compare Models"
    )

    st.caption(
        "Confronta i sei classificatori utilizzando "
        "gli stessi dati inseriti nella sezione Predict."
    )

    # Recupera il vero vincitore, se è stata caricata
    # una partita dal Test Set
    actual_winner = st.session_state.get(
        "example_actual_winner"
    )

    if actual_winner is not None:
        st.info(
            f"Actual Winner of loaded match: {actual_winner}"
        )

    st.write("")

    compare_button = st.button(
        "COMPARE ALL MODELS",
        key="compare_all_models",
    )


    if compare_button:

        # =================================================
        # COSTRUZIONE DELLE FEATURE
        # =================================================

        blue_first_blood = (
            1
            if first_blood == "Blue Team"
            else 0
        )

        blue_gold_diff = (
            blue_total_gold
            - red_total_gold
        )

        blue_experience_diff = (
            blue_total_experience
            - red_total_experience
        )

        blue_gold_per_min = (
            blue_total_gold / 10
        )

        red_gold_per_min = (
            red_total_gold / 10
        )

        blue_deaths = red_kills


        # =================================================
        # 27 FEATURE FINALI
        # =================================================

        features = {

            "blueWardsPlaced":
                blue_wards_placed,

            "blueWardsDestroyed":
                blue_wards_destroyed,

            "blueFirstBlood":
                blue_first_blood,

            "blueKills":
                blue_kills,

            "blueDeaths":
                blue_deaths,

            "blueDragons":
                blue_dragons,

            "blueHeralds":
                blue_heralds,

            "blueTowersDestroyed":
                blue_towers,

            "blueTotalGold":
                blue_total_gold,

            "blueAvgLevel":
                blue_avg_level,

            "blueTotalExperience":
                blue_total_experience,

            "blueTotalMinionsKilled":
                blue_minions,

            "blueTotalJungleMinionsKilled":
                blue_jungle_minions,

            "blueGoldDiff":
                blue_gold_diff,

            "blueExperienceDiff":
                blue_experience_diff,

            "blueGoldPerMin":
                blue_gold_per_min,

            "redWardsPlaced":
                red_wards_placed,

            "redWardsDestroyed":
                red_wards_destroyed,

            "redDragons":
                red_dragons,

            "redHeralds":
                red_heralds,

            "redTowersDestroyed":
                red_towers,

            "redTotalGold":
                red_total_gold,

            "redAvgLevel":
                red_avg_level,

            "redTotalExperience":
                red_total_experience,

            "redTotalMinionsKilled":
                red_minions,

            "redTotalJungleMinionsKilled":
                red_jungle_minions,

            "redGoldPerMin":
                red_gold_per_min,
        }


        # =================================================
        # DATAFRAME
        # =================================================

        input_df = pd.DataFrame(
            [features]
        )

        feature_order = list(
            scaler.feature_names_in_
        )

        input_df = input_df[
            feature_order
        ]


        # =================================================
        # STANDARDIZZAZIONE
        # =================================================

        input_scaled = pd.DataFrame(
            scaler.transform(input_df),
            columns=feature_order,
        )


        # =================================================
        # PREDIZIONE DI TUTTI I MODELLI
        # =================================================

        comparison_results = []

        for model_name, model in models.items():

            # Decision Tree usa i valori originali.
            if model_name == "Decision Tree":
                model_input = input_df

            else:
                model_input = input_scaled


            # Predizione
            prediction = int(
                model.predict(model_input)[0]
            )


            # Vincitore
            if prediction == 1:
                winner = "BLUE TEAM"
                winner_color = "#5bb7ff"

            else:
                winner = "RED TEAM"
                winner_color = "#ff6b6b"


            # Probabilità, se disponibile
            confidence = None

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    model_input
                )[0]

                class_index = list(
                    model.classes_
                ).index(prediction)

                confidence = probabilities[
                    class_index
                ]


            comparison_results.append(
                {
                    "model": model_name,
                    "prediction": prediction,
                    "winner": winner,
                    "color": winner_color,
                    "confidence": confidence,
                }
            )


        # =================================================
        # RISULTATI GRAFICI
        # =================================================

        st.write("")
        st.markdown("#### Individual predictions")
        st.write("")

        card_columns = st.columns(3)

        for index, result in enumerate(
            comparison_results
        ):

            column = card_columns[
                index % 3
            ]

            # =============================================
            # CONFIDENCE
            # =============================================

            if result["confidence"] is not None:

                confidence_text = (
                    f'{result["confidence"]:.1%}'
                )

            else:

                confidence_text = "N/A"


            # =============================================
            # CORRECT / WRONG
            # =============================================

            if actual_winner is not None:

                if result["winner"] == actual_winner:

                    status_html = (
                        '<div style="'
                        'color:#6fcf97;'
                        'font-weight:700;'
                        'margin-top:12px;'
                        'letter-spacing:1px;'
                        '">'
                        'CORRECT'
                        '</div>'
                    )

                else:

                    status_html = (
                        '<div style="'
                        'color:#ff6b6b;'
                        'font-weight:700;'
                        'margin-top:12px;'
                        'letter-spacing:1px;'
                        '">'
                        'WRONG'
                        '</div>'
                    )

            else:

                status_html = ""


            # =============================================
            # CARD DEL MODELLO
            # =============================================

            card_html = (
                '<div style="'
                'border:1px solid #785a28;'
                'background:rgba(10,20,30,0.82);'
                'border-radius:6px;'
                'padding:22px;'
                'margin-bottom:18px;'
                'text-align:center;'
                'min-height:185px;'
                '">'

                '<div style="'
                'color:#a09b8c;'
                'font-size:0.85rem;'
                'letter-spacing:1px;'
                'margin-bottom:12px;'
                '">'
                f'{result["model"]}'
                '</div>'

                '<div style="'
                f'color:{result["color"]};'
                'font-size:1.45rem;'
                'font-weight:700;'
                'letter-spacing:2px;'
                'margin-bottom:14px;'
                '">'
                f'{result["winner"]}'
                '</div>'

                '<div style="'
                'color:#f0e6d2;'
                'font-size:1rem;'
                '">'
                f'Confidence: {confidence_text}'
                '</div>'

                f'{status_html}'

                '</div>'
            )


            # =============================================
            # MOSTRA LA CARD
            # =============================================

            with column:

                st.markdown(
                    card_html,
                    unsafe_allow_html=True,
                )


        # =================================================
        # MODEL CONSENSUS
        # =================================================

        blue_votes = sum(
            result["prediction"] == 1
            for result in comparison_results
        )

        red_votes = (
            len(comparison_results)
            - blue_votes
        )


        if blue_votes > red_votes:

            consensus_team = "BLUE TEAM"
            consensus_color = "#5bb7ff"
            consensus_votes = blue_votes

        elif red_votes > blue_votes:

            consensus_team = "RED TEAM"
            consensus_color = "#ff6b6b"
            consensus_votes = red_votes

        else:

            consensus_team = "TIE"
            consensus_color = "#c8aa6e"
            consensus_votes = 3


        st.write("")

        consensus_html = (
            '<div style="'
            'border-top:1px solid #785a28;'
            'border-bottom:1px solid #785a28;'
            'padding:30px;'
            'margin-top:20px;'
            'text-align:center;'
            'background:rgba(10,20,30,0.70);'
            '">'
            '<div style="'
            'color:#a09b8c;'
            'font-size:0.9rem;'
            'letter-spacing:3px;'
            '">'
            'MODEL CONSENSUS'
            '</div>'
            '<div style="'
            f'color:{consensus_color};'
            'font-size:2rem;'
            'font-weight:700;'
            'letter-spacing:3px;'
            'margin-top:10px;'
            '">'
            f'{consensus_team}'
            '</div>'
            '<div style="'
            'color:#f0e6d2;'
            'font-size:1.15rem;'
            'margin-top:12px;'
            '">'
            f'Blue {blue_votes} / 6'
            ' &nbsp;&nbsp;·&nbsp;&nbsp; '
            f'Red {red_votes} / 6'
            '</div>'
            '</div>'
        )

        st.markdown(
            consensus_html,
            unsafe_allow_html=True,
        )

        st.caption(
            "Il Model Consensus è soltanto un riepilogo "
            "dell'accordo tra i sei classificatori e non "
            "costituisce un ulteriore modello ensemble."
        )


# =========================================================
# TAB 3 — ABOUT
# =========================================================


with tab_about:

    # =====================================================
    # INTRO
    # =====================================================

 about_header_html = (
    '<div style="'
    'text-align:center;'
    'margin-top:15px;'
    'margin-bottom:30px;'
    '">'
    '<div style="'
    'color:#c8aa6e;'
    'font-size:2rem;'
    'font-weight:700;'
    'letter-spacing:3px;'
    '">'
    'ABOUT PREDICT LoL'
    '</div>'
    '<div style="'
    'color:#a09b8c;'
    'margin-top:8px;'
    'font-size:1rem;'
    '">'
    'Predicting League of Legends matches '
    'from the first 10 minutes'
    '</div>'
    '</div>'
)

st.markdown(
    about_header_html,
    unsafe_allow_html=True,
)


    # =====================================================
    # PROJECT METRICS
    # =====================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
        "Matches",
        "9,879",
    )

col2.metric(
        "Game snapshot",
        "10:00",
    )

col3.metric(
        "Final features",
        "27",
    )

col4.metric(
        "ML models",
        "6",
    )


st.write("")
st.markdown("---")


    # =====================================================
    # DESCRIPTION
    # =====================================================

left_col, right_col = st.columns(
        [1.3, 1],
        gap="large",
    )


with left_col:

        st.markdown(
            "### Project overview"
        )

        st.markdown(
            """
            **PredictLoL** è un progetto di Data Science
            finalizzato alla previsione dell'esito di partite
            ranked di **League of Legends** utilizzando
            esclusivamente le informazioni disponibili
            al **decimo minuto di gioco**.

            Il dataset contiene **9.879 partite di livello
            Diamond** e descrive lo stato dell'early game
            attraverso informazioni relative a:

            - oro;
            - esperienza;
            - eliminazioni;
            - farming;
            - controllo degli obiettivi;
            - torri;
            - visione della mappa.

            Dopo l'analisi esplorativa e la gestione delle
            ridondanze, il dataset è stato ridotto a
            **27 feature predittive**.
            """
        )


with right_col:

        st.markdown(
            "### Machine Learning pipeline"
        )

        st.markdown(
            """
            **1.** Exploratory Data Analysis

            ↓

            **2.** Feature selection & preprocessing

            ↓

            **3.** 80/20 Train-Test Split

            ↓

            **4.** Z-Score standardization

            ↓

            **5.** 5-Fold Cross-Validation

            ↓

            **6.** Model evaluation

            ↓

            **7.** Interpretability & error analysis
            """
        )


st.write("")
st.markdown("---")


    # =====================================================
    # MODELS
    # =====================================================

st.markdown(
        "### Models evaluated"
    )

model_col1, model_col2, model_col3 = st.columns(3)

with model_col1:

        st.markdown(
            """
            **Logistic Regression**

            Linear and highly interpretable
            """
        )

        st.markdown(
            """
            **Support Vector Machine**

            Margin-based classifier
            """
        )


with model_col2:

        st.markdown(
            """
            **Decision Tree**

            Rule-based and interpretable
            """
        )

        st.markdown(
            """
            **K-Nearest Neighbors**

            Distance-based classifier
            """
        )


with model_col3:

        st.markdown(
            """
            **Random Forest**

            Ensemble of decision trees
            """
        )

        st.markdown(
            """
            **Gaussian Naive Bayes**

            Probabilistic classifier
            """
        )


st.write("")
st.markdown("---")


    # =====================================================
    # RESULTS
    # =====================================================

st.markdown(
        "### Final Test Set performance"
    )

results_df = pd.DataFrame(
        {
            "Model": [
                "Random Forest",
                "Logistic Regression",
                "Decision Tree",
                "Support Vector Machine",
                "Gaussian Naive Bayes",
                "K-Nearest Neighbors",
            ],

            "Accuracy": [
                0.7247,
                0.7212,
                0.7206,
                0.7166,
                0.7115,
                0.7075,
            ],

            "Precision": [
                0.7251,
                0.7182,
                0.7343,
                0.7156,
                0.7158,
                0.7056,
            ],

            "Recall": [
                0.7221,
                0.7262,
                0.6897,
                0.7170,
                0.6998,
                0.7099,
            ],

            "F1-Score": [
                0.7236,
                0.7221,
                0.7113,
                0.7163,
                0.7077,
                0.7078,
            ],
        }
    )


st.dataframe(
        results_df.style.format(
            {
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1-Score": "{:.2%}",
            }
        ),
        width="stretch",
        hide_index=True,
    )


    # =====================================================
    # BEST MODEL
    # =====================================================

st.write("")

best_model_html = (
    '<div style="'
    'border:1px solid #785a28;'
    'background:rgba(10,20,30,0.82);'
    'padding:25px;'
    'text-align:center;'
    'border-radius:6px;'
    'margin-top:15px;'
    '">'
    '<div style="'
    'color:#a09b8c;'
    'font-size:0.85rem;'
    'letter-spacing:3px;'
    '">'
    'BEST OVERALL MODEL'
    '</div>'
    '<div style="'
    'color:#c8aa6e;'
    'font-size:2rem;'
    'font-weight:700;'
    'letter-spacing:2px;'
    'margin-top:8px;'
    '">'
    'RANDOM FOREST'
    '</div>'
    '<div style="'
    'color:#f0e6d2;'
    'margin-top:12px;'
    'font-size:1.1rem;'
    '">'
    'Accuracy: 72.47% &nbsp; · &nbsp; '
    'F1-Score: 72.36%'
    '</div>'
    '</div>'
)

st.markdown(
    best_model_html,
    unsafe_allow_html=True,
)


st.write("")
st.markdown("---")


    # =====================================================
    # MAIN FINDINGS
    # =====================================================

st.markdown(
        "### Main findings"
    )

finding_col1, finding_col2, finding_col3 = st.columns(3)


with finding_col1:

        st.markdown(
            """
            #### Gold

            `blueGoldDiff` è risultata la feature
            più importante sia per la Logistic
            Regression sia per la Random Forest.
            """
        )


with finding_col2:

        st.markdown(
            """
            #### Experience

            `blueExperienceDiff` rappresenta
            la seconda variabile più rilevante
            in entrambi i modelli analizzati.
            """
        )


with finding_col3:

        st.markdown(
            """
            #### Model complexity

            La Random Forest ottiene il risultato
            migliore, ma la Logistic Regression
            raggiunge prestazioni molto vicine con
            una maggiore interpretabilità.
            """
        )


st.write("")
st.markdown("---")


    # =====================================================
    # AUTHORS + GITHUB
    # =====================================================

author_col, github_col = st.columns(
        [2, 1]
    )


with author_col:

        st.markdown(
            """
            ### Project

            Data Science project developed using
            **Python**, **pandas**, **scikit-learn**
            and **Streamlit**.
            """
        )


with github_col:

        st.link_button(
            "View project on GitHub",
            "https://github.com/MelaniaLeonelli/predictLoL",
            use_container_width=True,
        )