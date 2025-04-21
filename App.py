import pickle
import streamlit as st
import urllib.request

# ✅ FIXED: Use the raw URL of the pickle file
url = "https://raw.githubusercontent.com/dk8281/Marketing_campaign_/main/best_kmeans_pipeline.pkl"

# Load the model
try:
    with urllib.request.urlopen(url) as response:
        model_data = response.read()  # Read bytes
        model = pickle.loads(model_data)  # Load model from bytes
except Exception as e:
    model = None
    st.error(f"❌ Error loading the model: {e}")

# Streamlit UI
def main():
    st.markdown("# 🛍️ Customer Clustering with KMeans")
    st.write("🔮 Predict the cluster a customer belongs to!")

    # Input fields
    Age = st.number_input("📆 Age (Years)", min_value=0.0, format="%.2f")
    Income = st.number_input("💲 Income (Rs)", min_value=0.0, format="%.2f")
    TotalSpend = st.number_input("💰 Total Spend (Rs)", min_value=0.0, format="%.2f")
    FamilySize = st.number_input("👨‍👩‍👦 Family Size", min_value=0.0, format="%.2f")
    NumWebPurchases = st.number_input("🌐 Number of Web Purchases", min_value=0.0, format="%.2f")
    NumStorePurchases = st.number_input("🏬 Number of Store Purchases", min_value=0.0, format="%.2f")
    NumCatalogPurchases = st.number_input("📦 Number of Catalog Purchases", min_value=0.0, format="%.2f")

    # Prediction button
    if st.button("Predict"):
        if model is not None:
            try:
                # Create input array
                inputs = [[
                    Age,
                    Income,
                    TotalSpend,
                    FamilySize,
                    NumWebPurchases,
                    NumStorePurchases,
                    NumCatalogPurchases
                ]]
                # Predict
                prediction = model.predict(inputs)
                cluster_id = int(prediction[0])

                # Display result
                st.success(f"🎯 Cluster Number: {cluster_id}")

                # Optional: interpret clusters (customize this based on your analysis)
                if cluster_id == 0:
                    st.info("🟢 Cluster 0: Likely high-income, low-purchase segment.")
                elif cluster_id == 1:
                    st.info("🔵 Cluster 1: Likely budget-conscious, frequent shopper.")

            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
        else:
            st.warning("⚠️ Model not loaded. Please check the URL or try again later.")

if __name__ == '__main__':
    main()
