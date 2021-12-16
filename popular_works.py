import pandas as pd
import nltk

jobs = list(pd.read_csv('works.csv').qualification.dropna()) + list(pd.read_csv('works.csv').jobTitle.dropna())

nltk.download('punkt')
stema_finder = nltk.stem.snowball.SnowballStemmer("russian")

ends = ['ер', 'ир', 'ор', 'ар', 'ец', 'ик', 'ел', 'ист', 'ант', 'ог', 'ож', 'ач', 'ед', 'иц']
errors = ['начальник', 'руководител', 'заместител', 'завед', 'помощник', 'специалист', 'мастер', 'консультант',
          'технолог', 'работник', 'представител', 'делопроизводител', 'отдел', 'товар', 'свер', 'дел', 'категор',
          'издел', 'прибор', 'втор', 'кред', 'территор', 'сектор']

works_dictionary = {}

for job in jobs:
    tokens = [wordpunkt for wordpunkts in [nltk.wordpunct_tokenize(word) for word in nltk.word_tokenize(job)] for
              wordpunkt in wordpunkts if len(wordpunkt) > 3]
    for token in tokens:
        stem_token = stema_finder.stem(token)
        if not any([stem_token.endswith(end) for end in ends]) or stem_token in errors:
            continue
        if stem_token not in works_dictionary:
            works_dictionary[stem_token] = [0, set()]
        works_dictionary[stem_token][0] += 1
        works_dictionary[stem_token][1].add(token)

popular_works = [job for job in sorted(works_dictionary.items(), key=lambda i: i[1][0], reverse=True)][:100]

popular_works = pd.DataFrame({'name': [job[0] for job in popular_works],
                              'count': [job[1][0] for job in popular_works],
                              'words': [', '.join(job[1][1]) for job in popular_works]})
popular_works.to_csv('popular_works.csv', encoding='utf-8', index=False)
