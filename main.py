import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from flask import Flask, render_template, request

app = Flask(__name__)
app.secret_key = 'super secret key'

df = pd.read_csv('https://raw.githubusercontent.com/Marcelo0479/recommendation_system/main/df_prepared_recomendation_plus.csv', sep=';')

df_little_kids = df[(df.parental_guidelines == 'Little Kids') | (df.parental_guidelines == 'ALL AGES')]
df_older_kids = df[df.parental_guidelines == 'Older Kids']
df_teens = df[df.parental_guidelines == 'Teens']
df_adults = df[(df.parental_guidelines == 'Adults') | (df.parental_guidelines == ' ')]


def correct_index(df_by_parental_guideline):
    df_by_parental_guideline.reset_index(inplace=True)
    df_by_parental_guideline.drop(columns='index', inplace=True)


def cosine_sim(df_by_parental_guideline):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_by_parental_guideline['genre_and_description'])
    cosine_sim = cosine_similarity(tfidf_matrix)
    return cosine_sim


def chosen_dataframe(chose_title):
    id_ = df[df.title == chose_title].index[0]
    if df.parental_guidelines[id_] == 'Little Kids':
        return df_little_kids
    if df.parental_guidelines[id_] == 'Older Kids':
        return df_older_kids
    if df.parental_guidelines[id_] == 'Teens':
        return df_teens
    if df.parental_guidelines[id_] == 'Adults':
        return df_adults


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == 'GET':
        return render_template('index.html')

    else:
        title_name = request.form.get("title")
        if not title_name:
            message = 'Must enter a title'
            code_error = 400
            return render_template('apology.html', code_error=code_error, message=message)
        else:
            title_option = df.title[df.title.str.upper().str.contains(title_name.upper())]
            if len(title_option) == 0:
                message = 'Title not found'
                code_error = 400
                return render_template('apology.html', code_error=code_error, message=message)

        return render_template('options.html', options=title_option.values)


@app.route("/options", methods=["GET", "POST"])
def options():
    if request.method == 'POST':
        title_name = request.form.get("title")

        df_by_parental_guideline = chosen_dataframe(title_name)
        correct_index(df_by_parental_guideline)
        cosine_sim_ = cosine_sim(df_by_parental_guideline)

        idx = df_by_parental_guideline[df_by_parental_guideline.title == title_name].index[0]

        sim_scores = list(enumerate(cosine_sim_[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_index = [i[0] for i in sim_scores]

        df_titles_streamming = df_by_parental_guideline[['title', 'streaming', 'average_rating']].iloc[movie_index]

        df_sim_scores = pd.DataFrame(sim_scores).set_index(0)
        df_sim_scores.rename(columns={1: 'sim_score'}, inplace=True)
        recommendation = pd.concat([df_titles_streamming, df_sim_scores], axis=1)
        recommendation.sort_values(by='average_rating', ascending=False, inplace=True)

    return render_template('recommendations.html', recommendat=recommendation)


@app.route("/explanations")
def explanations():
    return render_template('explanations.html')


if __name__ == "__main__":
    app.run(debug=True)
