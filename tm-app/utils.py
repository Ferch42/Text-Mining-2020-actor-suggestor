from functools import reduce
import spacy
nlp = spacy.load('en_core_web_sm')


def get_actors_and_movies(df):
    """
    Returns a dict with the ratings and movies per actor
    """

    df['cast_id'] = df['cast'].apply(lambda x: [list(i.keys())[0] for i in eval(x)])

    actor_dict = {}

    def add_to_dict(x):
        movie = x['movie_id']
        rating = eval(x['info_json'])['imdbRating']
        for i in x['cast_id']:
            if i not in actor_dict.keys():
                actor_dict[i] = {'movies': [movie], 'rating' : [rating]}
            else:
                actor_dict[i]['movies'].append(movie)
                actor_dict[i]['rating'].append(rating)

    df = df.apply(lambda x: add_to_dict(x), axis=1)

    # return pd.DataFrame.from_dict(actors_dictionary).T
    return actor_dict

def sum_movies_per_actor(df, names=True):
    """
    sums all movies per actor, so return a dictionary with actor id's as
    key and the concatenated plots of all movies the actor played in as value
    """

    if names:
        names = 'summary_wiki'
    else:
        names = 'summary_no_names'

    df['cast_id'] = df['cast'].apply(lambda x: [list(i.keys())[0] for i in eval(x)])

    actor_dict = {}

    def add_to_dict(x):
        movie = x[names]
        if x[names] not in  ["['NoText']", 'NoSearch']:
            for i in x['cast_id']:
                if i not in actor_dict.keys():
                    actor_dict[i] = movie
                else:
                    actor_dict[i] += ' ' + movie

    df.apply(lambda x: add_to_dict(x), axis=1)

    return actor_dict


def avg_score_actor(df):
    """
    returns a column with the average rating an actor got for all the movies he played in
    """

    if 'rating' in df.columns():

        def get_avg(x):
            correct = [float(i) for i in x if i != 'N/A']
            if len(correct) > 0:
                return reduce(lambda a, b: a + b, correct)/len(correct)
            else:
                return None

        return df['rating'].apply(lambda x:  get_avg(x))
    else:
        print('rating column does not exists, please first run "get_actors_and_movies" first')



def remove_person_entity(text):
    """
    removes all person from the  text
    """

    document = nlp(text)
    ents = [e.text for e in document.ents if e.label_ != 'PERSON']
    return " ".join([item.text for item in document if item.text not in ents])


def get_actor_name(df):
    """
    creates a dictionary that maps actor id to actor name
    """

    name_mapper = {}

    def add_name(x):
        for i in eval(x):
            id_n = list(i.keys())[0]
            name = list(i.values())[0][0]
            if id_n not in name_mapper.values():
                name_mapper[id_n] = name
            else:
                pass

    df['cast'].apply(lambda x: add_name(x))

    return name_mapper
