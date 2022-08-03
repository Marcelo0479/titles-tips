import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from flask import Flask, render_template, request, redirect

app = Flask(__name__)
app.secret_key = 'super secret key'

df = pd.read_csv('https://raw.githubusercontent.com/Marcelo0479/recommendation_system/main/df_prepared_recomendation_plus.csv', sep=';')

df_little_kids = df[(df.parental_guidelines == 'Little Kids') | (df.parental_guidelines == 'ALL AGES')]
df_older_kids = df[df.parental_guidelines == 'Older Kids']
df_teens = df[df.parental_guidelines == 'Teens']
df_adults = df[(df.parental_guidelines == 'Adults') | (df.parental_guidelines == ' ')]

language = ['English', 'Português']

texts_en = ["Home", "Explanations", "Contact",
            "You subscribe to multiple streaming services and don't know what to watch? I can help you.",
            "Enter the name of a movie or TV show that you liked and I will indicate the 10 most similar titles among the main streaming services.",
            "We are currently checking the titles of the following streaming services:",
            'You must enter a title', 'Title not found']
texts_br = ["Inicio", "Explicações", "Contato",
            "Você assina vários serviços de streaming e não sabe o que ver? Eu posso te ajudar.",
            "Digite o nome de um filme ou programa de TV que você gostou e indicarei os 10 títulos mais parecidos entre os principais serviços de streaming.",
            "Atualmente estamos verificando os títulos dos seguintes serviços de streaming:",
            'Você deve digitar um título', 'Título não encontrado']


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

def recommendation(title_name):
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
    df_sim_scores = np.round(df_sim_scores * 100, 2).astype('str') + '%'
    recommendations = pd.concat([df_titles_streamming, df_sim_scores], axis=1)
    recommendations.sort_values(by='average_rating', ascending=False, inplace=True)

    return recommendations


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == 'GET':
        return render_template('index.html', texts=texts_en)

    else:
        title_name = request.form.get("title").strip().title()
        if not title_name:
            message = texts_en[6]
            return render_template('apology.html', message=message, texts=texts_en)
        else:
            title_option = df.title[df.title.str.contains(title_name)]
            if len(title_option) == 0:
                message = texts_en[7]
                return render_template('apology.html', message=message, texts=texts_en)
            if len(title_option) == 1 and title_option.values[0] == title_name:
                recommendations = recommendation(title_name)
                return render_template('recommendations.html', recommendat=recommendations)

        return render_template('options.html', options=title_option.values, texts=texts_en)


@app.route("/br", methods=["GET", "POST"])
def index_br():
    if request.method == 'GET':
        return render_template('index_br.html', texts=texts_br)

    else:
        title_name = request.form.get("title").strip().title()
        if not title_name:
            message = texts_br[6]
            return render_template('apology_br.html', message=message, texts=texts_br)
        else:
            title_option = df.title[df.title.str.contains(title_name)]
            if len(title_option) == 0:
                message = texts_br[7]
                return render_template('apology_br.html', message=message, texts=texts_br)
            if len(title_option) == 1 and title_option.values[0] == title_name:
                recommendations = recommendation(title_name)
                return render_template('recommendations.html', recommendat=recommendations)

        return render_template('opcoes.html', options=title_option.values, texts=texts_br)


@app.route("/options", methods=["GET", "POST"])
def options():
    if request.method == 'POST':
        title_name = request.form.get("title")
        recommendations = recommendation(title_name)
        return render_template('recommendations.html', recommendat=recommendations, texts=texts_en)


@app.route("/opcoes", methods=["GET", "POST"])
def opcoes():
    if request.method == 'POST':
        title_name = request.form.get("title")
        recommendations = recommendation(title_name)
        return render_template('recomendacoes.html', recommendat=recommendations, texts=texts_br)


@app.route("/explanations")
def explanations():
    return render_template('explanations.html', texts=texts_en)


@app.route("/explicacoes")
def explicacoes():
    return render_template('explicacoes.html', texts=texts_br)


@app.route("/contact")
def contact():
    return render_template('contact.html', texts=texts_en)


@app.route("/contato")
def contato():
    return render_template('contato.html', texts=texts_br)


if __name__ == "__main__":
    app.run(debug=True)
