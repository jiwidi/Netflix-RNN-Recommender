import pandas as pd
import os
import numpy as np
from tqdm import tqdm


def main():
    print("Processing netflix combined files")
    # DataFrame to store all imported data
    if not os.path.isfile("data/netflix/data.csv"):
        data = open("data/netflix/data.csv", mode="w")

        files = [
            "data/netflix/combined_data_1.txt",
            "data/netflix/combined_data_2.txt",
            "data/netflix/combined_data_3.txt",
            "data/netflix/combined_data_4.txt",
        ]

        # Remove the line with movie_id: and add a new column of movie_id
        # Combine all data files into a csv file
        for file in files:
            print("Opening file: {}".format(file))
            with open(file) as f:
                for line in f:
                    line = line.strip()
                    if line.endswith(":"):
                        movie_id = line.replace(":", "")
                    else:
                        data.write(movie_id + "," + line)
                        data.write("\n")
        data.close()

    # Read all data into a pd dataframe
    df = pd.read_csv(
        "data/netflix/data.csv",
        names=["movie_id", "user_id", "rating", "date"],
        parse_dates=["date"],
        encoding="utf8",
        engine="python",
    )
    df = df[(df["date"] > "2004-01-01")]
    print("Combining data with movie titles")
    # Read movies to merge and get titles
    movies = pd.read_csv(
        "data/netflix/movie_titles.csv", names=["movie_id", "release_year", "title"]
    )
    df = pd.merge(df, movies, on="movie_id")

    print("Processing data into history per user")
    # Order by transaction date
    df["rank_latest"] = df.groupby(["user_id"])["date"].rank(
        method="first", ascending=False
    )
    print("Before cat codes")
    print(df["movie_id"].max())

    print(df["user_id"].max())
    # Variables to index
    ## Movies
    df.movie_id = pd.Categorical(df.movie_id)
    df.movie_id["movie_id"] = df.movie_id.cat.codes
    ## Users
    df.user_id = pd.Categorical(df.user_id)
    df["user_id"] = df.user_id.cat.codes

    df["movie_id"] = pd.to_numeric(df["movie_id"])
    df["user_id"] = pd.to_numeric(df["user_id"])
    print(df["movie_id"].max())
    print(df["user_id"].max())

    uni_user = len(df["user_id"].unique())
    uni_movie = len(df["movie_id"].unique())
    print(f"Unique users: {uni_user} | Unique movies: {uni_movie}")
    assert len(df[df["user_id"] < uni_user]) == uni_user
    assert len(df[df["movie_id"] < uni_movie]) == uni_movie
    ##Create history of ratings per user
    df = (
        df.sort_values(["rank_latest"], ascending=False)
        .groupby(["user_id"])["movie_id"]
        .apply(list)
        .to_frame()
    )
    df.reset_index(inplace=True)

    # Clearing nans
    df["movie_id"] = df["movie_id"].apply(
        lambda x: x if not (np.isnan(x).any()) else np.nan
    )
    df.dropna(inplace=True)

    sys.exit()
    print("Saving processed datasets")
    # Writing processed datasets
    ctrain = 0
    ctest = 0
    with open("data/netflix/test_processed.csv", "w") as test:
        with open("data/netflix/train_processed.csv", "w") as train:
            for idx, row in tqdm(df.iterrows(), total=len(df)):
                history = list(row.movie_id)
                # Train history
                for i in range(len(history) - 1):
                    past = history[:i]
                    future = history[i : i + 1]
                    line = f"{row.user_id}\t{past[-10:]}\t{future[0]}\n"
                    train.write(line)
                    ctrain += 1
                # Test history
                i = len(history) - 1
                past = history[:i]
                future = history[i : i + 1]
                line = f"{row.user_id}\t{past[-10:]}\t{future[0]}\n"
                test.write(line)
                ctest += 1


if __name__ == "__main__":
    main()
