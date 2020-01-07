# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 17:10:10 2019

Scraping FAQ du site Web de CenterPark

@author: clair
"""

################################  Import modules


import sys
import re
import pandas as pd
from urllib.request import urlopen
from urllib.error import HTTPError, URLError
from bs4 import BeautifulSoup, NavigableString


################################ Définition de fonctions


def balise_theme(class_):
    return class_ and class_.startswith("faqTabs faqAccordion-title api_accordion-itemTitle")


def balise_quest_rep(class_):
    return class_ and class_.startswith("faqText faqVisibleFirst")


def remove_control_chart(s):
    return s.replace(u'\xa0', u' ')



################################ Programme principal

## chargement page web et récupération de la partie FAQ

try:
    html = urlopen('https://www.centerparcs.fr/fr-fr/faq')
except (HTTPError, URLError) as e:
    sys.exit(e)

bsObj = BeautifulSoup(html, "lxml")

contenu_faq = bsObj.find("ul", class_="api_accordion_faq faqAccordion")

themes = contenu_faq.find_all("li", class_="faqList api_accordion-item faqAccordion-item")

## récupération des couples questions / réponses

lst_faq = []

for theme in themes:
    txt_theme = remove_control_chart(re.findall(r'(.+) \(\d+\)', theme.find("div", class_=balise_theme).find('span').find('span').get_text())[0])
    lst_content_theme = theme.find("div", class_=balise_quest_rep).find('div').contents    
    index_research = 0
    while index_research < len(lst_content_theme):
        if lst_content_theme[index_research].name == 'h3':
            txt_question = remove_control_chart(re.findall(r'\d\. (.+)', lst_content_theme[index_research].text)[0])
            next_h3_founded = False
            txt_reponse = ''
            while not next_h3_founded and index_research != (len(lst_content_theme)-1):
                index_research += 1
                if lst_content_theme[index_research].name == 'h3':
                    next_h3_founded = True
                    index_research -= 1
                elif type(lst_content_theme[index_research]) != NavigableString and lst_content_theme[index_research].name in ['p', 'div']:
                    txt_reponse += remove_control_chart(lst_content_theme[index_research].text)
            lst_faq.append({'theme': txt_theme,
                            'question': txt_question,
                            'reponse': ''.join(txt_reponse.split('\n')).lstrip().rstrip()
                              })
        index_research += 1

## sortie des données au format pickle
            
pd.DataFrame(lst_faq).to_pickle('faq_centerPark.pkl')
