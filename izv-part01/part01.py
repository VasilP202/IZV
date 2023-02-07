#!/usr/bin/env python3
"""
IZV cast1 projektu
Autor: xpopos00

Detailni zadani projektu je v samostatnem projektu e-learningu.
Nezapomente na to, ze python soubory maji dane formatovani.

Muzete pouzit libovolnou vestavenou knihovnu a knihovny predstavene na prednasce
"""


from bs4 import BeautifulSoup, element
import requests
import numpy as np
import matplotlib.pyplot as plt
from typing import List


def integrate(x: np.array, y: np.array) -> float:
    """Integrate using the trapezoidal rule.

    Parameters:
        `x`: Sorted vector containing integration points.
        `y`: Values of the integrated function.
    """
    arr = (x[1:] - x[:-1]) * ((y[:-1] + y[1:]) * 0.5)
    return sum(arr)


def generate_graph(
    a: List[float], show_figure: bool = False, save_path: str | None = None
):
    """Generate function `f(x) = a * x^2` graph with various coefficients given.

    Parameters:
        `a`: List of coefficients.
        `show_figure`: Specifies whether to display a plot figure.
        `save_path`: Optional path to save a plot figure.
    """
    rng = np.arange(-3, 3 + 0.1, 0.1)  # Create range interval
    mtx = np.outer(a, rng * rng)

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot()
    ax.set_xlim(left=-3, right=3.9)
    ax.set_ylim(-20, 20)

    for i in range(len(a)):
        ax.plot(rng, mtx[i], label=f"$\gamma_{{{a[i]}}}{{(x)}}$")
        # Fill the area under the curve
        ax.fill_between(rng, mtx[i], alpha=0.2)
        ax.annotate(f"$\int f_{{{a[i]}}}{{(x)dx}}$", [3, mtx[i][-1]])

    fig.legend(loc="upper center", ncol=len(a))

    plt.xlabel("x")
    plt.ylabel("$f_{a}{(x)}$")

    if show_figure:
        plt.show()
    if save_path is not None:
        fig.savefig(save_path)
    plt.close()


def generate_sinus(show_figure: bool = False, save_path: str | None = None):
    """Visualize sine signals `0.5 · sin(1/50πt)`, `0.25 · sin(π𝑡)` and their sum.

    Parameters:
        `show_figure`: Specifies whether to display a plot figure.
        `save_path`: Optional path to save a plot figure.
    """
    time = np.arange(0, 100 + 0.1, 0.1)  # Generate time interval
    amp1 = 0.5 * np.sin(1 / 50.0 * np.pi * time)  # First amplitude
    amp2 = 0.25 * np.sin(np.pi * time)  # Second amplitude
    amp3 = amp1 + amp2  # Sum of sine amplitudes

    fig, axes = plt.subplots(
        nrows=3,
        figsize=(8, 12),
        constrained_layout=True,
    )

    for ax in axes:
        ax.set_xlim(left=0, right=100)
        ax.set_ylim(-0.8, 0.8)
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.set_xlabel("t")

    (ax1, ax2, ax3) = axes

    ax1.plot(time, amp1)
    ax1.set_ylabel("$f_{1}{(x)}$")

    ax2.plot(time, amp2)
    ax2.set_ylabel("$f_{2}{(x)}$")

    mask = np.ma.masked_greater(amp3, amp1)
    ax3.plot(time, amp3, "g")
    ax3.plot(time, mask, "r")
    ax3.set_ylabel("$f_{1}{(x)} + f_{2}{(x)}$")

    plt.xlim((0, 100))
    plt.ylim((-0.8, 0.8))
    if show_figure:
        plt.show()
    if save_path is not None:
        fig.savefig(save_path)
    plt.close()


def _get_temperature_data(temperature_data: element.ResultSet) -> np.ndarray:
    """Create numpy array containing temperature data for one table row."""
    data = []
    for cell in temperature_data:
        cell_value = cell.text.strip()
        if cell_value:
            data.append(float(cell_value.replace(",", ".")))

    return np.array(data)


def download_data(url="https://ehw.fit.vutbr.cz/izv/temp.html"):
    """Download a process table data.

    Parameters:
        `url`: Data source; defaults to `"https://ehw.fit.vutbr.cz/izv/temp.html"`.
    Returns:
        `results`: List of dicts containing processed data with keys `"year"`, `"month"`, `"temp"`.
    """
    # Fetch and parse data from a source
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")

    results = []

    rows = soup.find_all("tr")  # Find all table rows.
    for row in rows:
        # Find the year and month for a given row
        year_and_month = row.find_all("td", class_=["ce1", "ce2"])

        # Find temperature data for a given row
        temp_row_data = row.find_all("td", class_=["ce3"])

        arr = _get_temperature_data(temp_row_data)

        results.append(
            {
                "year": int(year_and_month[0].text.strip()),
                "month": int(year_and_month[1].text.strip()),
                "temp": arr,
            }
        )

    return results


def get_avg_temp(data, year=None, month=None) -> float:
    """Calculate average temperature value.

    Parameters:
        `data`: Temperature stats.
        `year`: Optional year filter value.
        `month`: Optional month filter value.
    """
    if year:
        data = [row for row in data if row["year"] == year]

    if month:
        data = [row for row in data if row["month"] == month]

    sum_total = 0
    length_total = 0
    for row in data:
        sum_total += sum(row["temp"])
        length_total += len(row["temp"])

    return sum_total / length_total
