import pandas as pd
from google_play_scraper import reviews, Sort
from app_store_scraper import AppStore
import time
import json  # Required for handling potential errors from the app store scraper

# --- App Configuration ---
# A dictionary of apps to be scraped, with their respective platform IDs.
# App Store ID can be found in the App Store URL.
# e.g., for Gojek: apps.apple.com/id/app/gojek/id944875099 -> the ID is 944875099
apps_to_scrape = {
    "gojek": {
        "play_store_id": "com.gojek.app",
        "app_store_name": "gojek",
        "app_store_id": "944875099",
    },
    "grab": {
        "play_store_id": "com.grabtaxi.passenger",
        "app_store_name": "grab-makanan-pesan-ojek",
        "app_store_id": "647268330",
    },
    "maxim": {
        "play_store_id": "com.taxsee.taxsee",
        # CORRECTED: Name and ID updated to match the Indonesian App Store URL
        "app_store_name": "maxim-transportasi-delivery",
        "app_store_id": "579985456",
    },
    "indrive": {
        "play_store_id": "sinet.startup.inDriver",
        # CORRECTED: Name updated to match the Indonesian App Store URL
        "app_store_name": "indrive-ojek-delivery",
        "app_store_id": "780125801",
    },
}

# Configure the maximum number of reviews to scrape per app, per platform.
MAX_REVIEWS_PER_APP = 500000

print(
    "Starting the review scraping process from Google Play Store and Apple App Store...\n"
)

# A list to hold all the scraped reviews
all_reviews_list = []

# --- Scraping Process ---
for app_name, app_details in apps_to_scrape.items():
    print("-" * 50)
    print(f"Scraping reviews for app: {app_name.capitalize()}")
    print("-" * 50)

    # 1. Scraping from Google Play Store
    print(f"  -> Fetching from Google Play Store...")
    try:
        # We use 'reviews' instead of 'reviews_all' to limit the results with 'count'
        # and prevent excessively long scraping times.
        play_store_reviews, continuation_token = reviews(
            app_details["play_store_id"],
            lang="id",  # Language: Indonesian
            country="id",  # Country: Indonesia
            sort=Sort.NEWEST,  # Sort by newest reviews
            count=MAX_REVIEWS_PER_APP,
            filter_score_with=None,  # Fetch reviews of all ratings
        )

        # Create a DataFrame and standardize the columns
        df_play = pd.DataFrame(play_store_reviews)
        df_play["platform"] = "Google Play"
        df_play_formatted = df_play[
            ["userName", "content", "score", "at", "platform"]
        ].rename(
            columns={
                "userName": "user_name",
                "content": "review_content",
                "score": "rating",
                "at": "date",
            }
        )
        df_play_formatted["app_name"] = app_name
        all_reviews_list.append(df_play_formatted)
        print(
            f"     Successfully fetched {len(df_play_formatted)} reviews from Google Play Store.\n"
        )
    except Exception as e:
        print(f"     Failed to fetch reviews from Google Play Store. Error: {e}\n")

    # 2. Scraping from Apple App Store
    print(f"  -> Fetching from Apple App Store...")
    try:
        app_store_scraper = AppStore(
            country="id",
            app_name=app_details["app_store_name"],
            app_id=app_details["app_store_id"],
        )
        app_store_scraper.review(how_many=MAX_REVIEWS_PER_APP)
        app_store_reviews = app_store_scraper.reviews

        # ADDED ROBUSTNESS: Check if reviews were actually fetched before processing
        if not app_store_reviews:
            print(
                "     No reviews were fetched from App Store. This might be a temporary issue or an API change. Skipping.\n"
            )
            continue  # Move to the next application

        # Create a DataFrame and standardize the columns
        df_appstore = pd.DataFrame(app_store_reviews)
        df_appstore["platform"] = "App Store"
        df_appstore_formatted = df_appstore[
            ["userName", "review", "rating", "date", "platform"]
        ].rename(
            columns={
                "userName": "user_name",
                "review": "review_content",
                # 'rating' column name is already consistent
                "date": "date",
            }
        )
        df_appstore_formatted["app_name"] = app_name
        all_reviews_list.append(df_appstore_formatted)
        print(
            f"     Successfully fetched {len(df_appstore_formatted)} reviews from App Store.\n"
        )
    except Exception as e:
        # This will now catch any other unexpected errors during the App Store process
        print(f"     Failed to process reviews from App Store. Error: {e}\n")

    # A short delay between scraping different apps to be polite to the servers
    time.sleep(10)

# --- Combine and Save Results ---
if all_reviews_list:
    final_df = pd.concat(all_reviews_list, ignore_index=True)

    # Reorder columns for better readability
    final_df = final_df[
        ["app_name", "platform", "date", "user_name", "rating", "review_content"]
    ]

    # Convert the 'date' column to a consistent datetime format
    final_df["date"] = pd.to_datetime(final_df["date"])

    # Sort the final data by app name and then by the newest date
    final_df.sort_values(by=["app_name", "date"], ascending=[True, False], inplace=True)

    # Save the combined DataFrame to a CSV file
    output_filename = "data/app_reviews.csv"
    final_df.to_csv(output_filename, index=False, encoding="utf-8-sig")

    print("=" * 60)
    print("DATA SCRAPING PROCESS COMPLETED!")
    print(f"All reviews have been combined and saved to: {output_filename}")
    print(f"Total reviews collected: {len(final_df)}")
    print("\nFirst 5 rows of the combined data:")
    print(final_df.head())
    print("=" * 60)
else:
    print("No reviews were collected. The process is stopping.")
