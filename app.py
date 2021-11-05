import pandas as pd
import seaborn as sns
import streamlit as st

import matplotlib.pyplot as plt

from glob import glob


@st.cache
def load_data(fns):
    data = []
    for fn in fns:
        data.append(read_and_clean_data(fn))
    joined_data = pd.concat(data)
    return joined_data


def read_and_clean_data(fn):
    """
    This function reads the data from the file, and normalizes column names.
    """
    # read in first sheet with metadata
    data = pd.read_excel(fn, 0)
    column_names = [c.lower() for c in data.columns]
    data.columns = column_names
    data["year"] = data["id"].str.split("-").str.get(0)

    # read second sheet, with abstracts and combine with metadata
    abstracts = pd.read_excel(fn, 1, dtype={"Abstract": str})
    combined = data.merge(abstracts, left_on="id", right_on="ID")
    combined.fillna(value={"Abstract": "NOTHING"})
    combined["lower_abstract"] = [
        a.lower() if type(a) == str else a for a in combined["Abstract"]
    ]
    return combined


def main():
    st.title("ACRL Conference Analysis")

    fns = glob("data_cleaned/*.xlsx")
    data = load_data(fns)
    years = sorted(set(data["year"].tolist()))

    # Sidebar
    st.sidebar.title("Analysis options")
    # selected_year = st.sidebar.selectbox(
    #     "Select the year you're interested in: ", years
    # )

    # Sidebar search box
    search_term = st.sidebar.text_input("Search for a word or phrase: ")

    # Filter dataframe based on search term
    if search_term:
        results = data[data["lower_abstract"].str.contains(search_term, na=False)]
        st.write(results[["year", "id", "title", "Abstract", "lower_abstract"]])

        data_for_chart = (
            results["year"]
            .value_counts()
            .sort_index()
            .reset_index()
            .rename(columns={"index": "year", "year": "count"})
        )

        # create seaborn bar plot from year column
        fig = plt.figure(figsize=(10, 10))
        sns.barplot(x="year", y="count", data=data_for_chart)
        st.pyplot(fig)


if __name__ == "__main__":
    main()
