#!/usr/bin/python3.10
# coding=utf-8
import contextily
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.cluster


def make_geo(df: pd.DataFrame) -> geopandas.GeoDataFrame:
    """Konvertovani dataframe do geopandas.GeoDataFrame se spravnym kodovani"""

    # Odstraneni radku s prazdnymi hodnotami souradnic
    df_fil = df[(~df["d"].isnull()) & (~df["e"].isnull())]

    return geopandas.GeoDataFrame(
        df_fil,
        geometry=geopandas.points_from_xy(df_fil["d"], df_fil["e"]),
        crs="EPSG:5514",  # Coordinate Reference System
    )


def plot_geo(
    gdf: geopandas.GeoDataFrame,
    fig_location: str = None,
    show_figure: bool = False,
):
    """Vykresleni grafu s nehodami s alkoholem pro roky 2018-2021"""

    region = "JHM"
    years = [2018, 2019, 2020, 2021]

    # Convert date column type to `datetime`
    gdf["p2a"] = pd.to_datetime(gdf["p2a"], cache=True)

    # Vyber data pro JHM kraj a roky 2018-2021
    gdf_sel = gdf.loc[
        (gdf["region"] == region) & (gdf["p2a"].dt.year.isin(years))
    ]

    # Vyber data s nehodami, kde byl vyznamne zapojen alkohol/drogy
    gdf_sel["large"] = gdf_sel["p11"] >= 3

    # 2x2 figure
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    pos_plot = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for i, year in enumerate(years):
        # Vykresleni podgrafu pro konkretni rok
        gdf_plot = gdf_sel[gdf_sel["p2a"].dt.year == year]
        gdf_plot[gdf_sel["large"]].plot(
            ax=ax[pos_plot[i]], alpha=0.5, color="tab:red"
        )
        gdf_plot[gdf_sel["large"]].boundary.plot(ax=ax[pos_plot[i]])

        # Vykresleni podkladu
        contextily.add_basemap(
            ax[pos_plot[i]],
            crs=gdf_sel.crs.to_string(),
            source=contextily.providers.Stamen.TonerLite,
            alpha=0.6,
        )

        ax[pos_plot[i]].xaxis.set_visible(False)
        ax[pos_plot[i]].yaxis.set_visible(False)
        ax[pos_plot[i]].set_frame_on(False)
        ax[pos_plot[i]].set_title(f"{region} kraj ({year})")

    if show_figure:
        plt.show()
    if fig_location is not None:
        fig.savefig(fig_location)
    plt.close()


def plot_cluster(
    gdf: geopandas.GeoDataFrame,
    fig_location: str = None,
    show_figure: bool = False,
):
    """Vykresleni grafu s lokalitou vsech nehod v kraji shlukovanych do clusteru"""

    region = "JHM"
    road_type_classes = [1, 2, 3]

    # Vyber data pro JHM kraj a silnice 1., 2., a 3. tridy
    gdf_sel = gdf.loc[
        (gdf["p36"].isin(road_type_classes)) & (gdf["region"] == region)
    ]

    # Vytvor NDArray obsahujici souradnice z GeoDataFrame
    coords = np.dstack([gdf_sel.geometry.x, gdf_sel.geometry.y]).reshape(-1, 2)

    # K-means shlukovani podle useku. 20 clusteru - vhodny pocet pro lepsi prehlednost
    # Unsupervised learning - uceni bez ucitele
    db = sklearn.cluster.MiniBatchKMeans(n_clusters=20).fit(coords)

    # Pridej sloupec `cluster` do GeoDataFrame
    gdf_sel["cluster"] = db.labels_

    # Agregace dat podle jednotlivych clusteru
    gdf_agg = gdf_sel.dissolve(by="cluster", aggfunc={"p1": "count"}).rename(
        columns=dict(p1="cnt")
    )

    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)

    # Vykresleni poctu nehod podle useku na mape
    gdf_agg.plot(ax=ax, column="cnt", markersize=2, legend=True)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)
    ax.set_title("Počet nehod v úseku")

    # Podklad
    contextily.add_basemap(
        ax,
        crs=gdf_sel.crs.to_string(),
        source=contextily.providers.Stamen.Terrain,
    )

    if show_figure:
        plt.show()
    if fig_location is not None:
        fig.savefig(fig_location)
    plt.close()


if __name__ == "__main__":
    gdf = make_geo(pd.read_pickle("accidents.pkl.gz"))
    plot_geo(gdf, "geo1.jpg", False)
    plot_cluster(gdf, "geo2.jpg", False)
