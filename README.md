# Titles-tips

This project consists in a dymanic site of recommendations of audio-visual cinematograph productions. 
My inspiration for this app is an answer for a kaggle task with a Netflix dataset of movies and tv show.

- [The task](https://www.kaggle.com/datasets/shivamb/netflix-shows)
- [The answer](https://www.kaggle.com/code/tylercranmer/netflix-recommendation-system)

My first step was increase the database to others streaming services and sort the recommendations for average IMDB score of each title.
I obtived this datas with similar datasets in kaggle.

- [Prime database](https://www.kaggle.com/datasets/shivamb/amazon-prime-movies-and-tv-shows)
- [Disney + database](https://www.kaggle.com/datasets/shivamb/disney-movies-and-tv-shows)
- [Hulu database](https://www.kaggle.com/datasets/shivamb/hulu-movies-and-tv-shows)

To obtain the IMDBs scores I have to download the two datasets directly of the IMDBs page.
The title.basics.tsv.gz file to get the titles's id and the title.ratings.tsv.gz to get the titles's scores.

- [IMDBs datasets files](https://www.imdb.com/interfaces/)

Then I place this dataset on a google drive account and create a python notebook. In this notebook I build a database crossing this datasets files, clean the unnecessary datas and applied the tfidfvectorizer algorithm to create a coefficient of similarity between each title and all others titles. This algorithm uses pnl to create that coefficient. In this project I used the follow datas for feed the algorithm, the description, the parental guideline and the genre of the titles. Finaly I create a function that return a list of the ten most similar title sorted by IMBD score based on a title.

- [Python notebook](https://colab.research.google.com/drive/1qOG-FHGHFySotNsKQV6WJQ9Z7oPXq1hy#scrollTo=z46CHtdjS_FD)
