import pandas as pd
import nltk
import seaborn as sns
import matplotlib.pyplot as plt

best_jobs = list(pd.read_csv('popular_works.csv').name)

jobs = pd.read_csv('works.csv')[['qualification', 'jobTitle']].dropna()

stema_finder = nltk.stem.snowball.SnowballStemmer("russian")


def get_tags(attribute):
    tokens = [stema_finder.stem(wordpunkt) for wordpunkts in
              [nltk.wordpunct_tokenize(word) for word in nltk.word_tokenize(attribute)] for
              wordpunkt in wordpunkts]
    return set(tokens).intersection(best_jobs)


init_sample = len(jobs)
final_sample = init_sample
heatmap = pd.DataFrame({job: [0] * 100 for job in best_jobs}, index=best_jobs)
for qlf, job in zip(jobs.qualification, jobs.jobTitle):
    qlf_tags = get_tags(qlf)
    job_tags = get_tags(job)
    if len(qlf_tags) == 0 or len(job_tags) == 0:
        final_sample -= 1
        continue
    for qt in qlf_tags:
        for jt in job_tags:
            heatmap[qt][jt] += 1


def get_top_five(iterable):
    return ''.join([f'    {index + 1}) {item[1]} (пунктов - {item[0]});\n' for index, item in
                    enumerate(sorted(zip(iterable, best_jobs), reverse=True)[:5])])


def get_popular(data):
    return [job[1] for job in sorted([(sum(data[job]), job) for job in best_jobs], reverse=True)[:5]]


print(f'Изначальная выборка без NaN состоит из {init_sample} пунктов.')
print(f'После отбора значений, подходящих под составленный топ профессий в выборке осталось {final_sample} резюме, то есть {round(final_sample / init_sample * 100)}%.')
print(f'Не работают по профессии {100 - round(sum(heatmap[job][job] for job in best_jobs) / final_sample * 100)}% респондентов.')
print(f'Менеджерами чаще всего становятся люди с образованием:')
print(f'{get_top_five(heatmap.transpose()["менеджер"])}')
print(f'Люди с образованием инженера чаще всего становятся:')
print(f'{get_top_five(heatmap["инженер"])}')

pop_qlf = get_popular(heatmap)
pop_job = get_popular(heatmap.transpose())
sns.heatmap(heatmap[[*pop_qlf]].transpose()[[*pop_job]], cmap='Spectral', annot=True, fmt="d")
plt.title('Тепловая карта 5 топовых профессий (слева) и топовых должностей (снизу)', fontsize=10)
plt.show()
