import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from flask import Flask, render_template, request

app = Flask(__name__)
app.secret_key = 'super secret key'

df = pd.read_csv('https://raw.githubusercontent.com/Marcelo0479/recommendation_system/main/df_heroku_68.csv', sep=';')

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['genre_parental_guidelines_and_description'])
cosine_sim = cosine_similarity(tfidf_matrix)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == 'GET':
        return render_template('index.html')

    else:
        title_name = request.form.get("title")
        title_option = df.title[df.title.str.upper().str.contains(title_name.upper())]
        return render_template('options.html', options=title_option.values)


@app.route("/options", methods=["GET", "POST"])
def options():
    if request.method == 'POST':
        title_name = request.form.get("title")

        if len(df.title[df.title.str.upper().str.contains(title_name.upper())]) > 0:
            idx = df[df.title == title_name].index[0]

            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:11]
            movie_index = [i[0] for i in sim_scores]

            df_titles_streamming = df[['title', 'streaming', 'average_rating']].iloc[movie_index]

            df_sim_scores = pd.DataFrame(sim_scores).set_index(0)
            df_sim_scores.rename(columns={1: 'sim_score'}, inplace=True)
            recommendation = pd.concat([df_titles_streamming, df_sim_scores], axis=1)
            recommendation.sort_values(by='average_rating', ascending=False, inplace=True)

        return render_template('recommendations.html', recommendat=recommendation)


if __name__ == "__main__":
    app.run(debug=True)
