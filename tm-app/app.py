import os
import doc2vec as d2v
import pandas as pd
import utils as util

import tkinter as tk
from tkinter import *
import os


root = tk.Tk()
root.geometry('1200x800')
root.title("Actor Search")

lm = Label(root, text="Movie").grid(row=0, column=0)

em = Entry(root, width=35, borderwidth=5)
em.grid(row=0, column=1, padx=5, pady=5)

lp = Label(root, text="Movie Plot").grid(row=1, column=0)

ep = Entry(root, width=35, borderwidth=5)
ep.grid(row=1, column=1, padx=5, pady=5)


var = IntVar()
Radiobutton(root, text="Doc2Vec", padx = 5, variable=var, value=1).grid(row=2, column=0)
Radiobutton(root, text="LDA", padx = 20, variable=var, value=2).grid(row=2, column=1)


def process_input():
    #when button is clicked
    movie = em.get()
    movie_plot = ep.get()

    if movie and movie_plot:

        if var.get() == 2:
            txt = 'LDA repsone'

        if var.get() == 1:
            similair, txt = d2v.get_cast_d2v(actors_df, actors_dictionary, 'd2v.model', movie_plot.lower() , 'abcdefghkladj', actor_mapper)
            Label(text=txt).grid(row=6, column=0, rowspan=3)

        else:
            Label(text='Please select a method').grid(row=5, column=0, rowspan=3)
    else:
        pass

btn = Button(root, text="Generate!", command=process_input).grid(row=4, column=0, rowspan=4, columnspan=3)

if __name__ == '__main__':

    df = pd.read_csv('data/train_df.csv')

    train = pd.read_csv('data/train_df.csv')

    actors_dictionary = util.get_actors_and_movies(train)
    actors_movie_sum = util.sum_movies_per_actor(train, True)
    actor_mapper = util.get_actor_name(train)

    actors_data = []
    for i in actors_movie_sum.keys():
        actors_data.append([i, actors_movie_sum[i]])

    actors_df = pd.DataFrame(actors_data, columns=['actor_id', 'movie_plot'])

    root.mainloop()




# plot_test = """Wade Wilson is a dishonorably discharged special forces operative working as a mercenary when he meets Vanessa, a prostitute. They become romantically involved, and a year later she accepts his marriage proposal. Wilson is diagnosed with terminal cancer, and leaves Vanessa without warning so she will not have to watch him die. A mysterious recruiter approaches Wilson, offering an experimental cure for his cancer. He is taken to Ajax and Angel Dust, who inject him with a serum designed to awaken latent mutant genes. They subject Wilson to days of torture to induce stress and trigger any mutation he may have, without success. When Wilson discovers Ajax's real name is Francis and mocks him for it, Ajax leaves Wilson in a hyperbaric chamber that periodically takes him to the verge of asphyxiation over a weekend. This finally activates a superhuman healing ability that cures the cancer but leaves Wilson severely disfigured with burn-like scars over his entire body. He escapes from the chamber and attacks Ajax but relents when told that his disfigurement can be cured. Ajax subdues Wilson and leaves him for dead in the now-burning laboratory. Wilson survives and seeks out Vanessa. He does not reveal to her he is alive fearing her reaction to his new appearance. After consulting with his best friend Weasel, Wilson decides to hunt down Ajax for the cure. He becomes a masked vigilante, adopting the name "Deadpool" (from Weasel picking him in a dead pool), and moves into the home of an elderly blind woman named Al. He questions and murders many of Ajax's men until one, the recruiter, reveals his whereabouts. Deadpool intercepts Ajax and a convoy of armed men on an expressway. He kills everyone but Ajax, and demands the cure from him but the X-Man Colossus and his trainee Negasonic Teenage Warhead interrupt him. Colossus wants Deadpool to mend his ways and join the X-Men. Taking advantage of this distraction, Ajax escapes. He goes to Weasel's bar where he learns of Vanessa. Ajax kidnaps Vanessa and takes her to a decommissioned helicarrier in a scrapyard. Deadpool convinces Colossus and Negasonic to help him. They battle Angel Dust and several soldiers while Deadpool fights his way to Ajax. During the battle, Negasonic accidentally destroys the equipment stabilizing the helicarrier. Deadpool protects Vanessa from the collapsing ship, while Colossus carries Negasonic and Angel Dust to safety. Ajax attacks Deadpool again but is overpowered. He reveals there is no cure after all and, despite Colossus's pleading, Deadpool kills him. He promises to try to be more heroic moving forward. Though Vanessa is angry with Wilson for leaving her, she reconciles with him.""".lower()

# movie_name = ''
