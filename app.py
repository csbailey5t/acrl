import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

import seaborn as sns
import streamlit as st

from streamlit import components

import matplotlib.pyplot as plt

from collections import defaultdict
from gensim import corpora, models
from glob import glob
from nltk.corpus import stopwords

# TODO
# - [ ] Move cleaning and loading code to a separate Jupyter notebook


@st.cache
def load_data(fns):
    data = []
    for fn in fns:
        data.append(read_and_clean_data(fn))
    joined_data = pd.concat(data)
    joined_data.to_csv("data_cleaned/combined.csv", index=False)
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


def text_search(data, search_term):
    # Filter dataframe based on search term
    if search_term:
        results = data[data["lower_abstract"].str.contains(search_term, na=False)]
        st.write(results[["year", "id", "title", "Abstract"]])

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


def hdp_model(corpus, dictionary):
    hdp = models.HdpModel(corpus, id2word=dictionary)
    hdp_data = gensimvis.prepare(hdp, corpus, dictionary)
    html_string = pyLDAvis.prepared_data_to_html(hdp_data)
    components.v1.html(html_string, width=1280, height=1024)


def lda_model(corpus, dictionary, num_topics):
    lda = models.LdaModel(corpus, id2word=dictionary, num_topics=num_topics)
    lda_data = gensimvis.prepare(lda, corpus, dictionary)
    html_string = pyLDAvis.prepared_data_to_html(lda_data)
    components.v1.html(html_string, width=1280, height=1024)


def main():
    st.title("ACRL Conference Analysis")

    # This was the import used also for cleaning; instead let's load cleaned data
    # fns = glob("data_cleaned/*.xlsx")
    # data = load_data(fns)

    data = pd.read_csv("data_cleaned/combined.csv")
    years = sorted(set(data["year"].tolist()))

    # load gensim corpus and dictionary
    dictionary = corpora.Dictionary.load("acrl.dict")
    corpus = corpora.MmCorpus("acrl.mm")

    # Sidebar
    st.sidebar.title("Analysis options")

    # select page
    page = st.sidebar.selectbox(
        "Select page", ["Text search", "HDP model", "LDA model"]
    )

    # selected_year = st.sidebar.selectbox(
    #     "Select the year you're interested in: ", years
    # )

    # Sidebar search box
    search_term = st.sidebar.text_input("Search for a word or phrase: ")

    # sidebar num topics selector
    num_topics = st.sidebar.slider(
        "Select the number of topics you're interested in: ",
        min_value=10,
        max_value=100,
        step=5,
        value=30,
    )

    # render pages based on page selectbox
    if page == "Text search":
        text_search(data, search_term)
    elif page == "HDP model":
        hdp_model(corpus, dictionary)
    elif page == "LDA model":
        lda_model(corpus, dictionary, num_topics)


if __name__ == "__main__":
    main()
