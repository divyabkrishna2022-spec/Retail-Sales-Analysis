import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page Configuration
st.set_page_config(
    page_title="Retail Sales Dashboard",
    layout="wide"
)

# Title
st.title("📊 Retail Sales Analysis Dashboard")

st.write("Welcome to the Retail Sales Analysis and Prediction System.")
st.write("Upload your Superstore Sales dataset to begin analysis.")

# Upload CSV File
uploaded_file = st.file_uploader(
    "Upload Superstore CSV File",
    type=["csv"]
)

# Read Dataset
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding="latin1")

    st.success("Dataset uploaded successfully! ✅")

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.subheader("Dataset Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Average Sales", round(df["Sales"].mean(), 2))
    col4.metric("Average Profit", round(df["Profit"].mean(), 2))

    st.subheader("Dataset Information")

    st.write("Shape of Dataset:")
    st.write(df.shape)

    st.write("Column Names:")
    st.write(df.columns)

    st.write("Data Types:")
    st.dataframe(df.dtypes.astype(str).reset_index().rename(
    columns={"index": "Column", 0: "Data Type"}
    ))

    st.write("Missing Values:")
    st.dataframe(df.isnull().sum().reset_index().rename(
    columns={"index": "Column", 0: "Missing Values"}
    ))

    st.subheader("Sales Distribution")

    fig, ax = plt.subplots(figsize=(5,3))

    sns.histplot(df["Sales"], bins=30, ax=ax)

    ax.set_title("Distribution of Sales")
    ax.set_xlabel("Sales")
    ax.set_ylabel("Count")

    st.pyplot(fig, use_container_width=False)

    st.subheader("Category-wise Sales")

    category_sales = df.groupby("Category")["Sales"].sum()

    st.write("Category-wise Sales")
    st.dataframe(category_sales.reset_index())

    fig, ax = plt.subplots(figsize=(5,3))

    category_sales.plot(kind="bar", ax=ax)

    ax.set_title("Total Sales by Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Sales")

    st.pyplot(fig, use_container_width=False)

    st.subheader("Category-wise Profit")

    category_profit = df.groupby("Category")["Profit"].sum()

    fig, ax = plt.subplots(figsize=(5,3))

    category_profit.plot(kind="bar", ax=ax)

    ax.set_title("Category-wise Profit")
    ax.set_xlabel("Category")
    ax.set_ylabel("Profit")

    st.pyplot(fig, use_container_width=False)


    st.subheader("Top 10 States by Sales")

    state_sales = (
    df.groupby("State")["Sales"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
    )

    fig, ax = plt.subplots(figsize=(4,5))

    state_sales.plot(kind="bar", ax=ax)

    ax.set_title("Top 10 States by Sales")
    ax.set_xlabel("State")
    ax.set_ylabel("Sales")

    plt.xticks(rotation=45)

    st.pyplot(fig, use_container_width=False)


    st.subheader("Monthly Sales Trend")

# Convert Order Date to datetime
    df["Order Date"] = pd.to_datetime(df["Order Date"])

# Create Month-Year column
    df["Month"] = df["Order Date"].dt.to_period("M").astype(str)

# Calculate monthly sales
    monthly_sales = df.groupby("Month")["Sales"].sum()

    fig, ax = plt.subplots(figsize=(3,2))

    monthly_sales.plot(ax=ax)

    ax.set_title("Monthly Sales Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Sales")

    plt.xticks(rotation=90)
    plt.tight_layout()

    st.pyplot(fig, use_container_width=False)


    st.subheader("Correlation Heatmap")

    # Select only important numerical columns
    corr = df[["Sales", "Profit", "Quantity", "Discount"]].corr()

    fig, ax = plt.subplots(figsize=(4.5,3.5))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        square=True,
        cbar=True,
        annot_kws={"size":8},
        ax=ax
    )

    ax.set_title("Correlation Heatmap", fontsize=12)

    plt.tight_layout()

    st.pyplot(fig, use_container_width=False)



    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    st.subheader("Linear Regression Model")

    # Select Features and Target
    X = df[["Quantity", "Discount"]]
    y = df["Sales"]

    # Split Dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # Train Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    st.success("Linear Regression Model Trained Successfully ✅")

    st.subheader("Sales Prediction")

    # Predict Sales
    y_pred = model.predict(X_test)

    # Show first 10 predictions
    prediction_df = pd.DataFrame({
        "Actual Sales": y_test.values,
        "Predicted Sales": y_pred
    })

    st.write("Actual vs Predicted Sales")
    st.dataframe(prediction_df.head(10))


    from sklearn.metrics import r2_score

    st.subheader("Model Accuracy")

    # Calculate R² Score
    score = r2_score(y_test, y_pred)

    st.write("R² Score:", round(score, 4))

    st.info(f"The model explains approximately {round(score*100,2)}% of the variation in Sales.")


    st.subheader("Actual vs Predicted Sales")

    fig, ax = plt.subplots(figsize=(3,2.5))

    ax.scatter(y_test, y_pred)

    ax.set_xlabel("Actual Sales")
    ax.set_ylabel("Predicted Sales")
    ax.set_title("Actual vs Predicted Sales")

    plt.tight_layout()

    st.pyplot(fig, use_container_width=False)



    st.subheader("📌 Key Insights")

    st.success("""
    ✔ Technology generated the highest sales.

    ✔ Technology generated the highest profit.

    ✔ California recorded the highest profit.

    ✔ Texas generated the highest loss.

    ✔ Higher discounts generally reduced profit.

    ✔ Sales showed an overall increasing trend from 2014 to 2017.

    ✔ November recorded the highest sales among all months.
    """)