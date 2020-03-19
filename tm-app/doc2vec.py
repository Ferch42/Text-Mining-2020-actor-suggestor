from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

def train_d2v(data, model_name):

    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data['summary_no_names'])]

    max_epochs = 100
    vec_size = 20
    alpha = 0.025

    model = Doc2Vec(size=vec_size,
                    alpha=alpha,
                    min_alpha=0.00025,
                    min_count=1,
                    dm =0) #changing dm to dm=0 will make it use BoW approach)

    model.build_vocab(tagged_data)

    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        model.alpha -= 0.0002
        model.min_alpha = model.alpha

    model.save(model_name)
    print("Model Saved")


# def try_d2v(data, model_name, plot):
#     model= Doc2Vec.load('models/' + model_name)
#     text = word_tokenize(plot)
# 
#     similar_doc = model.docvecs.most_similar(positive=[model.infer_vector(text)],topn=10)
#     for doc, perc in similar_doc:
#         print(doc, perc)
#         print('movie:', data.iloc[int(doc)]['movie_id'], '   Similarity:', perc)
#         print('summary from imdb: ', data.iloc[int(doc)]['info_json']['Plot'], '\n')

def get_cast_d2v(actors_df, actors_dictionary, model_name, plot, movie_name, actor_mapper):
    model= Doc2Vec.load('models/' + model_name)
    text = word_tokenize(plot)

    similar_doc = model.docvecs.most_similar(positive=[model.infer_vector(text)],topn=10)
    text = ''
    for doc, perc in similar_doc:
        if movie_name not in actors_dictionary[actors_df.iloc[int(doc)]['actor_id']]:
            txt = str('actor name: ' + actor_mapper[actors_df.iloc[int(doc)]['actor_id']] + '\nactor id: ' + str(actors_df.iloc[int(doc)]['actor_id'])  + '\nmovies played in:' +  str(actors_dictionary[actors_df.iloc[int(doc)]['actor_id']]['movies']) +  '\nSimilarity:' +  str(perc)  +  '\n\n')
            text += txt

    return similar_doc, text
