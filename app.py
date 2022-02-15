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
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
from sklearn.manifold import TSNE

import altair as alt


# TODO
# - [ ] Move cleaning and loading code to a separate Jupyter notebook
# - [ ] Visualize topics over time
# - [x] Vectorize w/ TfidfVectorizer and map with tsne
# - [ ] Add legend with top words for topic
# - [ ] Run value counts on top topic - what's the distribution?
# - [x] Remove all the rows with "no abstract" from model
# - [x] Remove custom stopwords
# - [ ] Audit to see what calculations can be saved, and not rerun

# Add search by poster or presentation in stacked bars in text search; we're missing two years of posts

# Thesis: do posters show trends before papers?
# Need to check numbers of posters and presentations each year; normalize counts by year.
# Get distrubtion of abstract length per year.

# Technology through the years
# staff development/professional development
# different in topics between posts and programs
# topic area changes through the years
# which schools participate
# variety of topics during the same conference

# show topic 5 topics per abstract


@st.cache
def load_data(fns):
    data = []
    for fn in fns:
        data.append(read_and_clean_data(fn))
    joined_data = pd.concat(data)
    joined_data.to_csv("data_cleaned/combined.csv", index=False)
    return joined_data

@st.cache
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

# Remove rows with no abstract in a dataframe
def remove_no_abstract(data):
    data = data[data["lower_abstract"] != "no abstract"]
    data = data[data["lower_abstract"] != "No abstract"]
    data = data[data["lower_abstract"] != "no abstract available"]
    data = data.dropna(subset=["lower_abstract"])
    return data


def text_search(data, search_term):
    cleaned_data = remove_no_abstract(data)
    # Filter dataframe based on search term
    if search_term:
        results = cleaned_data[cleaned_data["lower_abstract"].str.contains(search_term, na=False)]
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


# Define function that takes a dataframe, creates a tf-idf model, and visualizes it with t-SNE
def visualize_tsne_model(data):
    data = remove_no_abstract(data)
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf_vectorizer.fit_transform(data["lower_abstract"])
    tsne = TSNE(n_components=2, verbose=1, random_state=0, n_iter=1000)
    tsne_embedding = tsne.fit_transform(tfidf_matrix)

    # visualize tsne_model with altair scatter plot
    tsne_df = pd.DataFrame(tsne_embedding, columns=["x", "y"])

    st.subheader("Visualization of document abstracts tfidf through t-SNE")
    c = (
        alt.Chart(tsne_df)
        .mark_circle(size=10)
        .encode(
            x="x",
            y="y",
        )
        .interactive()
        .properties(width=1280, height=1024)
    )
    st.altair_chart(c, use_container_width=True)



# Define function that takes a dataframe, vectorizers by count, and visualizes it with t-SNE
def visualize_count_vectors(data):
    data = remove_no_abstract(data)
    count_vectorizer = CountVectorizer(stop_words="english")
    count_matrix = count_vectorizer.fit_transform(data["lower_abstract"])
    tsne = TSNE(n_components=2, verbose=1, random_state=0, n_iter=1000)
    tsne_embedding = tsne.fit_transform(count_matrix)

    # visualize tsne_model with altair scatter plot
    tsne_df = pd.DataFrame(tsne_embedding, columns=["x", "y"])

    st.subheader("Visualization of document abstracts count vectors through t-SNE")
    c = (
        alt.Chart(tsne_df)
        .mark_circle(size=10)
        .encode(
            x="x",
            y="y",
        )
        .interactive()
        .properties(width=1280, height=1024)
    )
    st.altair_chart(c, use_container_width=True)


def hdp_model(corpus, dictionary):
    hdp = models.HdpModel(corpus, id2word=dictionary)
    hdp_data = gensimvis.prepare(hdp, corpus, dictionary)
    html_string = pyLDAvis.prepared_data_to_html(hdp_data)
    components.v1.html(html_string, width=1280, height=1024)


def lda_model(data, corpus, dictionary, num_topics):
    lda = models.LdaModel(corpus, id2word=dictionary, num_topics=num_topics)

    # Visualize topics with pyLDAvis
    lda_data = gensimvis.prepare(lda, corpus, dictionary)
    html_string = pyLDAvis.prepared_data_to_html(lda_data)
    components.v1.html(html_string, width=1280, height=1024)

    # Visualize documents w/ t-SNE
    visualize_topics(data, corpus, lda, num_topics)


def visualize_topics(data_df, corpus, model, num_topics):
    """
    Visualizes documents through topic model & t-SNE
    """
    # drop rows from the dataframe where lower_absract is NA
    data_df = data_df.dropna(subset=["lower_abstract"])
    doc_id = data_df["id"].tolist()
    year = data_df["year"].tolist()
    title = data_df["title"].tolist()

    # Create headers for the DataFrame
    headers = ["doc_id", "year", "title"]
    for i in range(num_topics):
        headers.append(f"topic-{i}")

    # Create a DataFrame with the headers
    df = pd.DataFrame(columns=headers)

    # Generally, building pandas DataFrames row by row is not a best practice, but it makes sense here given the gensim function that gives us the topic distribution for a document
    for i in range(len(doc_id)):
        new_row = [doc_id[i], year[i], title[i]]

        for _, prob in model.get_document_topics(corpus[i], minimum_probability=0):
            new_row.append(prob)
        df.loc[doc_id[i]] = new_row

    st.subheader("Document table with topic proportions")
    st.write(df)

    tsne = TSNE(n_components=2)
    lda_data = df.drop(["doc_id", "year", "title"], axis=1).to_numpy()
    tsne_embedding = tsne.fit_transform(lda_data)

    # We'll turn our two dimensional array into a pandas dataframe for ease of use in visualization
    # We'll also add in a hue column that maps to the most significant topic for each document. `argmax` works here because the columns of the array correspond to the topics in our model
    tsne_df = pd.DataFrame(tsne_embedding, columns=["x", "y"])
    tsne_df["hue"] = lda_data.argmax(axis=1)

    # merge the tsne dataframe with the original dataframe
    merged_df = pd.concat([df[["title", "year"]].reset_index(), tsne_df], axis=1)

    st.subheader("Visualization of documents through t-SNE")
    c = (
        alt.Chart(merged_df)
        .mark_circle(size=10)
        .encode(
            x="x",
            y="y",
            color=alt.Color("year:N", scale=alt.Scale(scheme="dark2")),
            tooltip=["hue", "title", "year"],
        )
        .interactive()
        .properties(width=1280, height=1024)
    )
    st.altair_chart(c, use_container_width=True)

    topic_summaries = []
    print("{:20} {}".format("term", "frequency") + "\n")
    for i in range(num_topics):
        print("Topic " + str(i) + " |---------------------\n")
        tmp = explore_topic(model, topic_number=i, topn=10, output=True)
        topic_summaries += [tmp[:10]]
    st.write(topic_summaries)


# Show topic summaries to help interpret vis
# TODO Integrate top 10 topic words into hover
def explore_topic(lda_model, topic_number, topn, output=True):
    """
    accepts an ldamodel, a topic number and topn terms of interest
    prints a formatted list of the topn terms
    """
    terms = []
    for term, frequency in lda_model.show_topic(topic_number, topn=topn):
        terms += [term]
        if output:
            print("{:20} {:.3f}".format(term, round(frequency, 3)))
    return terms


# Streamlit page to find similar abstracts through sentence similarity
# code pulled from https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1
def find_similar_abstracts(data):
    st.subheader("Find similar abstracts")
    st.write(data[["id", "title", "Abstract"]])
    abstract = st.text_area("Enter abstract to find similar abstracts", "")
    docs = data["lower_abstract"].tolist()

    model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")

    # TODO only run the following code if the user enters a sentence
    # TODO add a submit form 
    doc_emb = model.encode(docs)

    if abstract != "":
        query_emb = model.encode(abstract)

        scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
        doc_score_pairs = list(zip(docs, scores))
        doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

        # TODO return not just abstract and score, but metadata from row
        st.write(doc_score_pairs[:10])


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
        "Select page", ["Text search", "Similarity search", "Count vectors", "Tfidf model", "HDP model", "LDA model"]
    )

    # selected_year = st.sidebar.selectbox(
    #     "Select the year you're interested in: ", years
    # )

    # render pages based on page selectbox
    if page == "Text search":
        # Sidebar search box
        search_term = st.sidebar.text_input("Search abstracts for a word or phrase: ")
        text_search(data, search_term)
    elif page == "Similarity search":
        find_similar_abstracts(data)
    elif page == "Count vectors":
        st.subheader("Vectorize by count and visualize with t-SNE")
        visualize_count_vectors(data)
    elif page == "Tfidf model":
        st.subheader("Vectorize with Tf-idf and visualize with t-SNE")
        visualize_tsne_model(data)
    elif page == "HDP model":
        hdp_model(corpus, dictionary)
    elif page == "LDA model":
        # sidebar num topics selector
        num_topics = st.sidebar.slider(
            "Select the number of topics you're interested in: ",
            min_value=10,
            max_value=100,
            step=5,
            value=30,
        )

        lda_model(data, corpus, dictionary, num_topics)


if __name__ == "__main__":
    main()
