import pandas as pd

works = pd.read_csv('works.csv')

#1) Количество записей
print(len(works))

#2) Количество мужчин и женщин
print(len(works[works.gender == "Мужской"]), len(works[works.gender == "Женский"]))

#3) Сколько не NAN в skills
print(len(works[works.skills.notna()]))

#4) Получаем заполненные скиллы
print(works.skills.dropna())

#5) Вывести зарплату только у тех, у которых в скиллах есть питон
print(works[works.skills.str.contains('Python') == True].salary)

#6) Построить перцентили и разброс по з/п у мужчин и женщин.
print(works[works.gender == "Мужской"].describe()[3:].transpose())
print(works[works.gender == "Женский"].describe()[3:].transpose())

#7) Построить графики распределения по з/п мужчин и женщин в зависимости от высшего образования
