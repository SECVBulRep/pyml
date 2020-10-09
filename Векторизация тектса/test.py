from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]
X = vectorizer.fit_transform(corpus)

features_names = vectorizer.get_feature_names()
analyze = vectorizer.build_analyzer()

print(features_names)
print(X.toarray())

