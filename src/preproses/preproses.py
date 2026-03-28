import pandas as pd
import pickle
from sklearn.preprocessing import OrdinalEncoder


def preprocess_data(df, is_train=True):
    """Preprocess the data for modeling"""
    
    df = df.copy()

    categorical_features = [
        'HomePlanet', 'CryoSleep', 'Destination',
        'VIP', 'Deck', 'Side', 'Age_group'
    ]

    numerical_features = [
        'Age', 'RoomService', 'FoodCourt', 'ShoppingMall',
        'Spa', 'VRDeck', 'Cabin_num', 'Group_size',
        'Solo', 'Family_size', 'TotalSpending',
        'HasSpending', 'NoSpending',
        'Age_missing', 'CryoSleep_missing'
    ] + [col for col in df.columns if '_ratio' in col]

    # Fill missing categorical
    for col in categorical_features:
        df[col] = df[col].fillna('Unknown')

    # Fill missing numerical
    for col in numerical_features:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    if is_train:
        df[categorical_features] = encoder.fit_transform(df[categorical_features].astype(str))
        
        #with open("model/preprocessor.pkl", "wb") as f:
        #    pickle.dump(encoder, f)
    else:
        
        #with open("model/preprocessor.pkl", "rb") as f:
        #    encoder = pickle.load(f)
        pass

    feature_columns = categorical_features + numerical_features
    X = df[feature_columns]

    if is_train and 'Transported' in df.columns:
        y = df['Transported'].astype(int)
        return X, y, feature_columns, encoder
    
    return X, feature_columns, encoder
