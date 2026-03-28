import streamlit as st
import pandas as pd
import pickle

st.title("asgmd05-initial")


model = pickle.load(open("model/pipeline.pkl", "rb"))


HomePlanet = st.selectbox("HomePlanet", ["Earth", "Europa", "Mars", "Unknown"])
CryoSleep = st.selectbox("CryoSleep", [True, False])
Destination = st.selectbox("Destination", ["TRAPPIST-1e", "PSO J318.5-22", "55 Cancri e", "Unknown"])
VIP = st.selectbox("VIP", [True, False])
Age = st.number_input("Age", min_value=0, value=30)

RoomService = st.number_input("RoomService", min_value=0.0, value=0.0)
FoodCourt = st.number_input("FoodCourt", min_value=0.0, value=0.0)
ShoppingMall = st.number_input("ShoppingMall", min_value=0.0, value=0.0)
Spa = st.number_input("Spa", min_value=0.0, value=0.0)
VRDeck = st.number_input("VRDeck", min_value=0.0, value=0.0)

Deck = st.selectbox("Deck", ["A","B","C","D","E","F","G", "T", "Unknown"])
Side = st.selectbox("Side", ["P","S", "Unknown"])
Cabin_num = st.number_input("Cabin Number", value=0)

Group_size = st.number_input("Group Size", min_value=1, value=1)
Family_size = st.number_input("Family Size", min_value=1, value=1)

if st.button("Predict"):
    
    # --- FEATURE ENGINEERING (Wajib sama dengan training) ---
    TotalSpending = RoomService + FoodCourt + ShoppingMall + Spa + VRDeck
    HasSpending = 1 if TotalSpending > 0 else 0
    NoSpending = 1 if TotalSpending == 0 else 0
    Age_missing = 0
    CryoSleep_missing = 0
    Solo = 1 if Group_size == 1 else 0

    if Age <= 12: Age_group = "Child"
    elif Age <= 18: Age_group = "Teen"
    elif Age <= 30: Age_group = "Young_Adult"
    elif Age <= 50: Age_group = "Adult"
    else: Age_group = "Senior"

    # Perhitungan Ratio
    RoomService_ratio = RoomService / (TotalSpending + 1)
    FoodCourt_ratio = FoodCourt / (TotalSpending + 1)
    ShoppingMall_ratio = ShoppingMall / (TotalSpending + 1)
    Spa_ratio = Spa / (TotalSpending + 1)
    VRDeck_ratio = VRDeck / (TotalSpending + 1)

    # --- PEMBUATAN DATAFRAME ---
    data = pd.DataFrame([{
        "HomePlanet": HomePlanet,
        "CryoSleep": CryoSleep,
        "Destination": Destination,
        "VIP": VIP,
        "Deck": Deck,
        "Side": Side,
        "Age_group": Age_group,
        "Age": Age,
        "RoomService": RoomService,
        "FoodCourt": FoodCourt,
        "ShoppingMall": ShoppingMall,
        "Spa": Spa,
        "VRDeck": VRDeck,
        "Cabin_num": Cabin_num,
        "Group_size": Group_size,
        "Solo": Solo,
        "Family_size": Family_size,
        "TotalSpending": TotalSpending,
        "HasSpending": HasSpending,
        "NoSpending": NoSpending,
        "Age_missing": Age_missing,
        "CryoSleep_missing": CryoSleep_missing,
        "RoomService_ratio": RoomService_ratio,
        "FoodCourt_ratio": FoodCourt_ratio,
        "ShoppingMall_ratio": ShoppingMall_ratio,
        "Spa_ratio": Spa_ratio,
        "VRDeck_ratio": VRDeck_ratio
    }])

    
    data = data[model.feature_names]

    
    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Age_group']
    data[cat_cols] = data[cat_cols].astype(str)

    
    prediction = model.predict(data)[0]

    if prediction == 1:
        st.success("Passenger was Transported ")
    else:
        st.error("Passenger was NOT Transported ")