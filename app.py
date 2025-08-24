from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

# Load data and model
train_data = pd.read_csv('train_data.csv')
neighbors_model = joblib.load('neighbors_model.pkl')
feature_cols = ['Age', 'Income', 'Purchase_Frequency', 'Total_Spending']
category_columns = [
    'Product_Category_Preference_Apparel',
    'Product_Category_Preference_Books',
    'Product_Category_Preference_Electronics',
    'Product_Category_Preference_Health & Beauty',
    'Product_Category_Preference_Home & Kitchen'
]
if 'Top_Category' not in train_data.columns:
    train_data['Top_Category'] = train_data[category_columns].idxmax(axis=1)
app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend():
    req = request.json
    new_user = np.array([
        req["Age"],
        req["Income"],
        req["Purchase_Frequency"],
        req["Total_Spending"]
    ]).reshape(1, -1)
    distances, indices = neighbors_model.kneighbors(new_user)
    similar_users = train_data.iloc[indices[0]]
    recommended_category = similar_users['Top_Category'].mode()[0]

    display_names = {
        "Product_Category_Preference_Apparel": "Apparel",
        "Product_Category_Preference_Books": "Books",
        "Product_Category_Preference_Electronics": "Electronics",
        "Product_Category_Preference_Health & Beauty": "Health & Beauty",
        "Product_Category_Preference_Home & Kitchen": "Home & Kitchen"
    }
    result = display_names.get(recommended_category, recommended_category)
    return jsonify({"recommended_category": result})

if __name__ == '__main__':
    app.run(debug=True)