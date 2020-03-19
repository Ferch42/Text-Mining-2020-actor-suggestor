import requests
# import urllib.request
import time
from bs4 import BeautifulSoup
import re
import pandas as pd
import wikipedia as wiki
wiki.set_lang("en")


base_url_pages = 'https://www.imdb.com/search/title/?genres={}&view=simple&start={}&ref_=adv_nxt'

page_nrs = ['301', '351', '401', '451', '501', '551', '601', '651', '701', '751', '801'] #'1', '51', '151', '201', '251'
genres = ['adventure', 'comedy', 'family',
          'romance', 'sci-fi', 'thriller',
            'mystery', 'western',  'horror',
            'crime', 'war', 'musical']

def scrape_movies(urls, pages, genres):
    """
    Scrapes all movies from imdb from different genres.
    return a dictionary with all movies in it
    """

    movie_dict = {}

    for genre in genres:
        movie_dict.update({genre : {}})
        for page_nr in pages:
            url = urls.format(genre, page_nr)
            response = requests.get(url)

            soup = BeautifulSoup(response.text, 'html.parser')

            for line in soup.findAll("span", {"class": "lister-item-header"}):
                link = re.findall('(?<=a href=").*?(?=">)', str(line))[0]
                name = re.findall("(?<=>).*?(?=</a>)", str(line))[0]
                name = name.replace('&amp;', '&')
                movie_dict[genre].update({name: {'url': link}})
    return movie_dict


def scrape_cast_movie(movie):
    """
    input: IMDB movie id
    Scrape the cast of a certain movie
    """
    url = 'https://www.imdb.com/title/{}'.format(movie)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    actors_in_movie = []
    for i in soup.findAll("table", class_ = "cast_list"):
        actors = set(re.findall('(?<=a href="/name/).*?(?=/")', str(i)))
        for actor in actors:
            role = re.findall('(?<={}/">).*?(?=\n</a>)'.format(actor), str(i))
            actors_in_movie.append({actor:role})
    return actors_in_movie


def get_meta_data(movie_id):
    """
    API call which gets all meta data from omdb (IMDB)
    """
    r = requests.get('http://www.omdbapi.com/?i={}&apikey=8147210c'.format(movie_id))
    return r.json() if r.status_code == 200 else {'api_error':' {}'.format(r.status_code)}



def get_wiki_plot(movie_id):
    """
    scrapes Wikipedia for every movie title and returns the movie plot that is on wikipedia.
    """
    data = get_meta_data(movie_id)
    try:
        search = data['Title'] + ' ' + data['Year'] + ' (film)'
        search_result = wiki.search(search, 1)[0]
    except:
        return ('NoSearch', data)
    found = False

    url = 'https://en.wikipedia.org/wiki/{}'.format('_'.join(search_result.split(' ')))
    response = requests.get(url)
    txt = BeautifulSoup(response.text, 'html.parser').text
    txt_list = txt.split('\n')
    plot_names = ['Premise', 'Plot', 'Plot[edit]', 'Premise[edit]']
    indexes = [txt_list.index(i) for i in plot_names if i in txt.split('\n')]
    if len(indexes) > 0:
        index_end = indexes[0] + txt_list[indexes[0]:].index('')
        text_movie = txt_list[indexes[0]+1: index_end]
        found = True

    return (text_movie, data) if found else (['NoText'], data)
