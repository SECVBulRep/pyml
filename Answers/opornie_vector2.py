from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer


newsgroups = datasets.fetch_20newsgroups(
                    subset='all', 
                    categories=['alt.atheism', 'sci.space']
             )


y = newsgroups.target

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(newsgroups.data)
features_names = vectorizer.get_feature_names()


print(vectorizer.vocabulary_)
print(vectorizer.idf_)


p=1