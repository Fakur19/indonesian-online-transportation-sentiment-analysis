import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================================================
# Page Configuration
# ======================================================================================
st.set_page_config(
    page_title="Indonesian Ride-Hailing Sentiment Analysis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================================================
# Styling (Optional) - Makes the app look cleaner
# ======================================================================================
st.markdown(
    """
<style>
    .block-container {
        padding-top: 2rem;
    }
    .st-emotion-cache-16txtl3 {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        text-align: center;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ======================================================================================
# Data Loading (Efficiently)
# ======================================================================================
# This function caches the data to avoid reloading on every interaction.
@st.cache_data
def load_data():
    """
    Loads pre-processed data required for the dashboard.
    In a real scenario, you would run a separate preprocessing script once
    to generate these lightweight files.
    """
    try:
        # Load the main cleaned dataset
        # For demonstration, we assume a smaller, pre-processed file exists.
        # You should create this file from your notebook.
        # df_cleaned = pd.read_parquet(
        #     "data/app_reviews_cleaned.parquet"
        # )  # Assuming you created this

        bucket_name = "goty-sentiment-analysis"  # IMPORTANT: Use your bucket name
        file_name = "app_reviews_cleaned.parquet"
        s3_path = f"s3://{bucket_name}/{file_name}"

        df_cleaned = pd.read_parquet(
            s3_path,
            storage_options={
                "key": st.secrets["aws"]["aws_access_key_id"],
                "secret": st.secrets["aws"]["aws_secret_access_key"],
            },
        )

        df_cleaned["date"] = pd.to_datetime(df_cleaned["date"])

        # Create df_model_data by removing neutral sentiment
        df_model_data = df_cleaned[df_cleaned["sentiment"] != "Netral"].copy()

        # Prepare data for the aspect plot
        # You would save this pre-calculated df to a file
        aspect_plot_df = pd.read_parquet(
            "data/aspect_plot_df.parquet"
        )  # Assuming you created this

        return df_cleaned, df_model_data, aspect_plot_df
    except FileNotFoundError:
        st.error(
            "Error: Pre-processed data files not found. Please run the preprocessing script first."
        )
        return None, None, None


df_cleaned, df_model_data, aspect_plot_df = load_data()


# ======================================================================================
# Sidebar
# ======================================================================================
if df_model_data is not None:
    st.sidebar.header("Dashboard Navigation")
    st.sidebar.write("Use the filters below to explore the data.")

    # App selection filter
    unique_apps = df_model_data["app_name"].unique()
    selected_apps = st.sidebar.multiselect(
        "Select Applications to Display:", options=unique_apps, default=unique_apps
    )

    # MODIFIED: Added GitHub link at the bottom of the sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="text-align: center;">
            <a href="https://github.com/Fakur19/indonesian-online-transportation-sentiment-analysis" target="_blank">
                View Project on GitHub
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Filter data based on selection
    if selected_apps:
        df_filtered = df_model_data[df_model_data["app_name"].isin(selected_apps)]
        if aspect_plot_df is not None:
            aspect_plot_filtered = aspect_plot_df[
                aspect_plot_df["app_name"].isin(selected_apps)
            ]
        else:
            aspect_plot_filtered = pd.DataFrame()  # Empty df if source is None
    else:
        df_filtered = df_model_data
        aspect_plot_filtered = aspect_plot_df
        st.sidebar.warning("Please select at least one application.")
else:
    # Stop the app if data loading failed
    st.stop()


# ======================================================================================
# Main Page Layout
# ======================================================================================
st.title("Competitive Landscape Analysis of Indonesian Online Transportation Apps")
st.markdown(
    "<h3 style='text-align: center;'>A Sentiment Analysis Based on Google Play Store User Reviews</h3>",
    unsafe_allow_html=True,
)


# --- Create Tabs for different sections ---
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Data Distribution",
        "📈 Time-Series Analysis",
        "🔑 Key Driver Analysis",
        "🧩 Aspect-Based Analysis",
    ]
)


# ======================================================================================
# Tab 1: Data Distribution
# ======================================================================================
with tab1:
    st.header("Review and Sentiment Distribution")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Total Reviews per Application")
        fig, ax = plt.subplots(figsize=(10, 6))
        review_counts = df_filtered["app_name"].value_counts()
        sns.barplot(
            x=review_counts.index, y=review_counts.values, palette="viridis", ax=ax
        )
        ax.set_title("Reviews per App", fontsize=16)
        ax.set_xlabel("Apps", fontsize=12)
        ax.set_ylabel("Reviews", fontsize=12)
        plt.xticks(rotation=0)
        st.pyplot(fig)

    with col2:
        st.subheader("Sentiment Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(
            data=df_filtered,
            x="sentiment",
            order=["Positif", "Negatif"],
            palette={"Positif": "#4CAF50", "Negatif": "#F44336"},
            ax=ax,
        )
        ax.set_title("Sentiment Distribution (Positive vs. Negative)", fontsize=16)
        ax.set_xlabel("Sentiment", fontsize=12)
        ax.set_ylabel("Reviews", fontsize=12)
        st.pyplot(fig)

    st.markdown("""
    ---
    ### Interpretation
    * **Review Volume:** This chart shows the comparison of review counts for the selected applications. It provides context on the dataset size for each app, which can influence the representativeness of the analysis. A balanced volume, as seen between Gojek, Maxim, and Grab, makes direct comparisons more robust.
    * **Sentiment Distribution:** Overall, reviews are dominated by positive sentiment (~80%). However, the significant absolute number of negative reviews serves as a crucial focal point for analysis, offering a rich source of data to identify key areas for service improvement.
    """)

# ======================================================================================
# Tab 2: Time-Series Analysis
# ======================================================================================
with tab2:
    st.header("Sentiment Trend Analysis Over Time")

    st.subheader("Comparative Sentiment Trends per Application")

    if not df_filtered.empty:
        df_time = df_filtered.copy()
        df_time["sentiment_score"] = df_time["sentiment"].map(
            {"Positif": 1, "Negatif": 0}
        )

        fig, ax = plt.subplots(figsize=(16, 8))
        sns.set_palette("tab10")

        for app in df_time["app_name"].unique():
            app_data = df_time[df_time["app_name"] == app].copy()
            app_sentiment_trend = (
                app_data.set_index("date")["sentiment_score"]
                .resample("M")
                .mean()
                .rolling(window=3)
                .mean()
            )
            app_sentiment_trend.plot(ax=ax, label=app.capitalize(), linewidth=2.5)

        ax.set_title(
            "Comparative Sentiment Trends per Application (3-Month Moving Average)",
            fontsize=18,
        )
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Sentiment Average (1 = Very Positive)", fontsize=12)
        ax.axhline(y=0.5, color="grey", linestyle="--", label="Neutral Limit (0.5)")
        ax.legend(title="Applications", fontsize="large")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        st.pyplot(fig)

        st.markdown("""
        ---
        ### Interpretation
        This visualization tracks how sentiment perception for each application has evolved over time.
        -   **Stable & High Trend Line (e.g., Maxim):** Indicates consistent customer satisfaction and a strong, stable brand perception.
        -   **Volatile & Competing Lines (e.g., Gojek & Grab):** Reflects intense market competition, likely influenced by price wars, new feature launches, or viral issues. Their frequent crossovers highlight a dynamic battle for user approval.
        -   **Improving Trend Line (e.g., inDrive post-2024):** Signals that service improvements or strategic changes are being positively perceived by users.
        """)
    else:
        st.warning(
            "Please select at least one application to display the time-series analysis."
        )


# ======================================================================================
# Tab 3: Feature Importance Analysis
# ======================================================================================
with tab3:
    st.header("Most Influential Keyword Analysis")
    st.write(
        "This section reveals the keywords that most strongly drive positive or negative sentiment for each application, extracted from the machine learning model."
    )

    # This part requires pre-calculated dataframes for each app's feature importance
    # Example: feature_importance_gojek.parquet, feature_importance_grab.parquet, etc.

    for app_name in selected_apps:
        st.subheader(f"Analysis for: {app_name.capitalize()}")
        try:
            # Load pre-calculated feature importance data
            fi_df = pd.read_parquet(f"data/feature_importance_{app_name}.parquet")
            top_positive = fi_df[fi_df["sentiment"] == "Positif"].head(10)
            top_negative = fi_df[fi_df["sentiment"] == "Negatif"].head(10)

            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots()
                sns.barplot(
                    x="coefficient",
                    y="word",
                    data=top_positive,
                    palette="Greens_r",
                    ax=ax,
                )
                ax.set_title(f"Top Positive Keywords", fontsize=14)
                ax.set_xlabel("Impact")
                ax.set_ylabel("")
                st.pyplot(fig)

            with col2:
                fig, ax = plt.subplots()
                sns.barplot(
                    x="coefficient",
                    y="word",
                    data=top_negative,
                    palette="Reds_r",
                    ax=ax,
                )
                ax.set_title(f"Top Negative Keywords", fontsize=14)
                ax.set_xlabel("Impact")
                ax.set_ylabel("")
                st.pyplot(fig)

        except FileNotFoundError:
            st.warning(
                f"File feature importance untuk '{app_name}' not found. run pre-processing script"
            )
        st.markdown("---")

    st.markdown("""
    ### Interpretation
    This analysis pinpoints the unique perceptual DNA of each application:
    - **Gojek & Grab:** Praised for functionality and reliability but criticized for technical issues and app performance.
    - **inDrive & Maxim:** Clearly win customer approval through affordable pricing.
    - **Maxim's Unique Edge:** Successfully combines the perception of low price with friendly and patient driver service, creating a powerful competitive advantage.
    """)

# ======================================================================================
# Tab 4: Aspect-Based Sentiment Analysis
# ======================================================================================
with tab4:
    st.header("Aspect-Based Sentiment Analysis")
    st.write(
        "Dissecting sentiment into specific categories like Price, Service, and Driver Performance."
    )

    if not aspect_plot_filtered.empty:
        # Create the final visualization using a faceted bar plot
        g = sns.catplot(
            data=aspect_plot_filtered,
            x="percentage",
            y="app_name",
            hue="sentiment",
            col="aspects",
            kind="bar",
            col_wrap=3,  # Adjust wrapping for better layout
            height=4,
            aspect=1.2,
            palette={"Positif": "#4CAF50", "Negatif": "#F44336"},
            sharey=False,
        )

        g.set_titles("Aspect: {col_name}", size=14)
        g.set_axis_labels("Sentiment Percentage (%)", "Application")
        g.fig.suptitle(
            "Aspect-Based Sentiment Comparison per Application", y=1.03, size=18
        )
        g.set(xlim=(0, 1))

        # Format x-axis ticks as percentages
        for ax in g.axes.flat:
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.0%}"))
            # Rotate y-axis labels if they overlap
            plt.setp(ax.get_yticklabels(), rotation=0)

        st.pyplot(g.fig)

        st.markdown("""
        ---
        ### Interpretation
        This visualization directly maps the competitive strengths and weaknesses of each app.
        -   **Green Dominance (e.g., Maxim in 'Harga'):** Indicates a strong, market-leading competitive advantage in that specific aspect.
        -   **Red Dominance (e.g., all apps in 'Customer Service'):** Signals a universal pain point and a critical area for improvement across the industry.
        -   **Green & Red Balance (e.g., Grab in 'Harga'):** Shows a polarized user perception, where some are satisfied (likely due to promotions) while many others are not. This highlights a key area of strategic challenge.
        """)

    else:
        st.warning(
            "Please select at least one application to display the aspect-based analysis."
        )


# ======================================================================================
# Footer
# ======================================================================================
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Cooked in 2025 🔥 </p>",
    unsafe_allow_html=True,
)
