import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("metadata.csv", low_memory=False)
    return df

st.title("📊 CORD-19 Metadata Explorer")
st.write("This app explores the **CORD-19 research dataset** and provides simple visualizations.")

df = load_data()

# -------------------------------
# Data Overview
# -------------------------------
st.header("1. Dataset Overview")
st.write("Shape of dataset:", df.shape)
st.write(df.head())

# -------------------------------
# Data Cleaning
# -------------------------------
st.header("2. Data Cleaning")
st.write("Dropping rows with missing titles and publication dates...")
df = df.dropna(subset=["title", "publish_time"])
st.write("New shape:", df.shape)

# -------------------------------
# Publications Over Time
# -------------------------------
st.header("3. Publications Over Time")
df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")
pubs_per_year = df["publish_time"].dt.year.value_counts().sort_index()

fig, ax = plt.subplots()
sns.lineplot(x=pubs_per_year.index, y=pubs_per_year.values, marker="o", ax=ax)
ax.set_title("Number of Publications per Year")
ax.set_xlabel("Year")
ax.set_ylabel("Publications")
st.pyplot(fig)

# -------------------------------
# Top Journals
# -------------------------------
st.header("4. Top Journals")
top_journals = df["journal"].value_counts().head(10)

fig, ax = plt.subplots()
sns.barplot(y=top_journals.index, x=top_journals.values, ax=ax)
ax.set_title("Top 10 Journals by Publications")
ax.set_xlabel("Number of Papers")
st.pyplot(fig)

# -------------------------------
# Word Cloud of Titles
# -------------------------------
st.header("5. Word Cloud of Titles")
text = " ".join(str(title) for title in df["title"].dropna())
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)

# -------------------------------
# Interactive Search
# -------------------------------
st.header("6. Search Papers by Keyword")
keyword = st.text_input("Enter a keyword (e.g., vaccine, pandemic, transmission):")
if keyword:
    results = df[df["title"].str.contains(keyword, case=False, na=False)][["title", "publish_time", "journal"]]
    st.write(f"Found {len(results)} results")
    st.dataframe(results.head(20))
