#!/usr/bin/env python3.9
# coding=utf-8
import zipfile

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def load_data(filename: str) -> pd.DataFrame:
    """Load data from ZIP archive. Create single pandas dataframe.

    Parameters:
        `filename`: ZIP file path.
    """
    headers = ["p1", "p36", "p37", "p2a", "weekday(p2a)", "p2b", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13a",
               "p13b", "p13c", "p14", "p15", "p16", "p17", "p18", "p19", "p20", "p21", "p22", "p23", "p24", "p27", "p28",
               "p34", "p35", "p39", "p44", "p45a", "p47", "p48a", "p49", "p50a", "p50b", "p51", "p52", "p53", "p55a",
               "p57", "p58", "a", "b", "d", "e", "f", "g", "h", "i", "j", "k", "l", "n", "o", "p", "q", "r", "s", "t", "p5a"]
    regions = {
        "PHA": "00",
        "STC": "01",
        "JHC": "02",
        "PLK": "03",
        "ULK": "04",
        "HKK": "05",
        "JHM": "06",
        "MSK": "07",
        "OLK": "14",
        "ZLK": "15",
        "VYS": "16",
        "PAK": "17",
        "LBK": "18",
        "KVK": "19",
    }

    with zipfile.ZipFile(filename, "r") as zip:
        dataframes = []
        for zip_year_filename in zip.namelist():
            # ZIP file containing data for the specific year
            year_data = zip.open(zip_year_filename, "r")
            with zipfile.ZipFile(year_data, "r") as zip_year_data:
                for csv_filename in zip_year_data.namelist():
                    # CSV file contains single region data
                    if csv_filename != "CHODCI.csv":
                        region_data = zip_year_data.open(csv_filename, "r")
                        dataframe = pd.read_csv(
                            region_data,
                            encoding="cp1250",
                            low_memory=False,
                            names=headers,
                            sep=";",
                        )
                        if not dataframe.empty:
                            dataframe["region"] = list(regions)[
                                list(regions.values()).index(csv_filename[:2])
                            ]
                            dataframes.append(dataframe)

        # Concatenate to single DataFrame
        df = pd.concat(dataframes, ignore_index=True)

        return df


def parse_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Format and filter out the dataframe

    Parameters:
        `df`: dataframe containing the data for analysis.
        `verbose`: If `True`, calculate and print the total size of the DataFrame before and after parsing.
    """
    categorical_columns = ["p36", "weekday(p2a)", "p6", "p7", "p8", "p9", "p10", "p11", "p13a", "p13b", "p13c", "p15", "p16",
                           "p17", "p18", "p19", "p20", "p21", "p22", "p23", "p24", "p27", "p28", "p34", "p35", "p39", "p44",
                           "p48a", "p49", "p50a", "p50b", "p51", "p55a", "p57", "p58", "k", "p", "q", "t", "p5a"]

    if verbose:
        print(f"orig_size={df.memory_usage(deep=True).sum() / 10**6  :.1f} MB")

    # Add datetime column
    df["date"] = pd.to_datetime(df["p2a"], cache=True)

    # Cast the appropriate columns to category dtype - optimization
    df[categorical_columns] = df[categorical_columns].astype("category")

    for col_name in ["a", "b", "d", "e", "f", "g"]:
        # Remove bad column data
        df[col_name] = df[col_name].replace(col_name.upper() + ":", np.nan)
        # Cast to dtype float
        df[col_name] = df[col_name].str.replace(",", ".").astype(float)

    # Remove duplicate rows
    df = df.drop_duplicates(subset="p1")

    if verbose:
        print(f"new_size={df.memory_usage(deep=True).sum() / 10**6  :.1f} MB")

    return df


def plot_visibility(
    df: pd.DataFrame, fig_location: str = None, show_figure: bool = False
):
    """Calculate and visualize the number of accidents by visibility in individual regions.

    Parameters:
        `df`: dataframe.
        `fig_location`: Path to save the figure.
        `fig_location`: If `True`, display figure.
    """

    regions = ["JHM", "MSK", "OLK", "ZLK"]

    # Get 4 regions data
    dfreg = df.loc[df["region"].isin(regions)][["p1", "region", "p19"]]

    # Add new column with data for aggregation
    dfreg["visibility"] = dfreg["p19"].map(
        {
            1: "Viditelnost: ve dne - nezhoršená",
            2: "Viditelnost: ve dne - zhoršená",
            3: "Viditelnost: ve dne - zhoršená",
            4: "Viditelnost: v noci - nezhoršená",
            5: "Viditelnost: v noci - zhoršená",
            6: "Viditelnost: v noci - nezhoršená",
            7: "Viditelnost: v noci - zhoršená",
        }
    )

    # Aggregate data by region
    dfagg = (
        dfreg.groupby(["region", "visibility"])
        .agg({"p1": "count"})
        .reset_index()
    )

    # Plot aggregation data
    g = sns.catplot(
        data=dfagg,
        x="region",
        y="p1",
        col="visibility",
        col_wrap=2,
        palette="hls",
        legend=False,
        kind="bar",
        sharex=False,
        sharey=False,
        height=2.8,
        aspect=1.5,
        saturation=1,
    )

    g.set(ylim=(0, 30000)).set_titles("{col_name}").set_axis_labels(
        "Kraj", "Počet nehod"
    )

    g.tight_layout()

    if fig_location:
        g.savefig(fig_location)

    if show_figure:
        plt.show()

    plt.close()


def plot_direction(
    df: pd.DataFrame, fig_location: str = None, show_figure: bool = False
):
    """Calculate and visualize the number of accidents by type of collision in individual regions.

    Parameters:
        `df`: dataframe.
        `fig_location`: Path to save the figure.
        `fig_location`: If `True`, display figure.
    """

    regions = ["JHM", "MSK", "OLK", "ZLK"]

    # Filter out data for visualization
    dfreg = df.loc[(df["p7"] != 0) & (df["region"].isin(regions))][
        ["p1", "region", "date", "p7"]
    ]

    # Add car crash direction data column
    dfreg["direction"] = dfreg["p7"].map(
        {
            1: "čelní",
            2: "boční",
            4: "zezadu",
        }
    )

    # Aggregate data by region and month
    dfagg = (
        dfreg.groupby(["region", dfreg.date.dt.month, "direction"])
        .agg({"p1": "count"})
        .reset_index()
    )

    # Plot aggregated data
    g = sns.catplot(
        data=dfagg,
        x="date",
        y="p1",
        col="region",
        hue="direction",
        kind="bar",
        col_wrap=2,
        saturation=1,
        height=2.8,
        aspect=1.5,
        legend=True,
        legend_out=True,
        sharey=False,
        sharex=False,
    )

    g.set(ylim=(0, 500)).set_titles("Kraj: {col_name}").set_axis_labels(
        "Měsíc", "Počet nehod"
    )

    g.legend.set(title="Druh srážky")
    g.tight_layout()

    if fig_location:
        g.savefig(fig_location)

    if show_figure:
        plt.show()

    plt.close()


if __name__ == "__main__":
    df = load_data("data.zip") # available at ehw.fit.vutbr.cz/izv/data.zip
    dfp = parse_data(df, True)
    plot_visibility(dfp, "03.png")
    plot_direction(dfp, "04.png")
