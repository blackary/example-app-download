from __future__ import annotations

from datetime import date, datetime
from typing import cast

import altair as alt
import pandas as pd
import pypistats
import streamlit as st

st.set_page_config(page_icon="📥", page_title="Download App")


def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )


PACKAGES = [
    "dash",
    "extra-streamlit-components",
    "gradio",
    "hiplot",
    "hydralit-components",
    "jupyter",
    "keras",
    "numpy",
    "pandas",
    "panel",
    "pollination-streamlit-io",
    "pywebio",
    "scikit-learn",
    "st-btn-select",
    "st-clickable-images",
    "streamlit",
    "streamlit-ace",
    "streamlit-aggrid",
    "streamlit-agraph",
    "streamlit-autorefresh",
    "streamlit-bokeh-events",
    "streamlit-chat",
    "streamlit-cookies-manager",
    "streamlit-cropper",
    "streamlit-disqus",
    "streamlit-drawable-canvas",
    "streamlit-echarts",
    "streamlit-folium",
    "streamlit-labelstudio",
    "streamlit-lottie",
    "streamlit-observable",
    "streamlit-on-Hover-tabs",
    "streamlit-option-menu",
    "streamlit-pandas-profiling",
    "streamlit-player",
    "streamlit-plotly-events",
    "streamlit-quill",
    "streamlit-tags",
    "streamlit-vega-lite",
    "streamlit-webrtc",
    "streamlit-wordcloud",
    "streamlit_vtkjs",
    "tensorflow",
    "torch",
    "voila",
    "streamlit-extras",
]

DEFAULT_PACKAGES = ["streamlit", "dash", "gradio", "panel", "voila"]


@st.cache_data(ttl=24 * 60 * 60)
def _get_package_stats(
    package: str, start_date: date, total: str | None = None
) -> pd.DataFrame:
    data: pd.DataFrame = pypistats.overall(
        package,
        start_date=str(start_date),
        end_date=str(date.today()),
        format="pandas",
        total=total,
        mirrors=True,
    )  # type: ignore
    data["project"] = package
    return data


def get_stats(
    packages: list[str], start_date: date, total: str | None = None
) -> pd.DataFrame:
    all_packages: list[pd.DataFrame] = []
    for package in packages:
        data = _get_package_stats(package, start_date, total)
        all_packages.append(data)

    return pd.concat(all_packages)


def get_downloads(packages: list[str], start_date: date, sum_over: str = "monthly"):
    total = None if sum_over == "weekly" else "monthly"

    df = get_stats(packages, start_date=start_date, total=total)

    if sum_over == "monthly":
        pass
    elif sum_over == "weekly":
        df["date"] = pd.to_datetime(df["date"])
        df = (
            df.groupby("project")
            .resample("W", on="date")
            .sum()
            .reset_index()
            .sort_values(by="date")
        )

    # Percentage difference (between 0-1) of downloads of current vs previous month
    df["delta"] = (df.groupby(["project"])["downloads"].pct_change()).fillna(0)
    # BigQuery returns the date column as type dbdate, which is not supported by Altair/Vegalite
    df["date"] = df["date"].astype("datetime64")

    return df


def monthly_downloads(packages: list[str], start_date: date):
    return get_downloads(packages, start_date, sum_over="monthly")


def weekly_downloads(packages: list[str], start_date: date):
    return get_downloads(packages, start_date, sum_over="weekly")


def plot_all_downloads(
    source, x="date", y="downloads", group="project", axis_scale="linear"
):

    if st.checkbox("View logarithmic scale"):
        axis_scale = "log"

    brush = alt.selection_interval(encodings=["x"], empty="all")

    click = alt.selection_multi(encodings=["color"])

    lines = (
        (
            alt.Chart(source)
            .mark_line(point=True)
            .encode(
                x=x,
                y=alt.Y("downloads", scale=alt.Scale(type=f"{axis_scale}")),
                color=group,
                tooltip=[
                    "date",
                    "project",
                    "downloads",
                    alt.Tooltip("delta", format=".2%"),
                ],
            )
        )
        .add_selection(brush)
        .properties(width=550)
        .transform_filter(click)
    )

    bars = (
        alt.Chart(source)
        .mark_bar()
        .encode(
            y=group,
            color=group,
            x=alt.X("downloads:Q", scale=alt.Scale(type=f"{axis_scale}")),
            tooltip=["date", "downloads", alt.Tooltip("delta", format=".2%")],
        )
        .transform_filter(brush)
        .properties(width=550)
        .add_selection(click)
    )

    return lines & bars


def pandasamlit_downloads(source, x="date", y="downloads"):
    # Create a selection that chooses the nearest point & selects based on x-value
    hover = alt.selection_single(
        fields=[x],
        nearest=True,
        on="mouseover",
        empty="none",
    )

    lines = (
        alt.Chart(source)
        .mark_line(point="transparent")
        .encode(x=x, y=y)
        .transform_calculate(color='datum.delta < 0 ? "red" : "green"')
    )

    # Draw points on the line, highlight based on selection, color based on delta
    points = (
        lines.transform_filter(hover)
        .mark_circle(size=65)
        .encode(color=alt.Color("color:N", scale=None))
    )

    # Draw an invisible rule at the location of the selection
    tooltips = (
        alt.Chart(source)
        .mark_rule(opacity=0)
        .encode(
            x=x,
            y=y,
            tooltip=[x, y, alt.Tooltip("delta", format=".2%")],
        )
        .add_selection(hover)
    )

    return (lines + points + tooltips).interactive()


def main():
    col1, col2 = st.columns(2)

    with col1:
        start_date = cast(
            date,
            st.date_input(
                "Select start date",
                date(2020, 1, 1),
                min_value=datetime.strptime("2020-01-01", "%Y-%m-%d"),
                max_value=datetime.now(),
            ),
        )

    with col2:
        weekly_or_monthly = st.selectbox(
            "Select weekly or monthly downloads", ("weekly", "monthly")
        )

    if weekly_or_monthly == "weekly":
        df = weekly_downloads(["streamlit"], start_date)
    else:
        df = monthly_downloads(["streamlit"], start_date)

    st.header("Streamlit downloads")

    st.altair_chart(pandasamlit_downloads(df), use_container_width=True)

    st.header("Compare other packages")

    instructions = """
    Click and drag line chart to select and pan date interval\n
    Hover over bar chart to view downloads\n
    Click on a bar to highlight that package
    """
    select_packages = st.multiselect(
        "Select Python packages to compare",
        PACKAGES,
        default=DEFAULT_PACKAGES,
        help=instructions,
    )

    if weekly_or_monthly == "weekly":
        selected_data_all = weekly_downloads(select_packages, start_date)
    else:
        selected_data_all = monthly_downloads(select_packages, start_date)

    st.write(selected_data_all)

    select_packages_df = pd.DataFrame(select_packages).rename(columns={0: "project"})

    if not select_packages:
        st.stop()

    filtered_df = selected_data_all[
        selected_data_all["project"].isin(select_packages_df["project"])
    ]

    st.altair_chart(plot_all_downloads(filtered_df), use_container_width=True)


st.title("Downloads")
st.write(
    "Metrics on how often different packages is being downloaded from PyPI (Python's "
    "main package repository, i.e. where `pip install [package]` downloads the package "
    "from)."
)
main()
