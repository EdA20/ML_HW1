Домашнее задание 1

В рамках дз была построена модель регрессии для предсказания стоимости автомобилей, а также реализован веб-сервис для применения построенной модели на новых данных.

Прежде всего был изучен датасет, некоторые признаки были трансформированы в нужный формат, обработаны пропуски, убраны дублирующиеся объекты, изучены графики зависимостей между признаками и с целевой переменной.

Изначальный результат предсказания на гридсерче выдавал 0.62, итоговая модель на гриде выдала 0.85, на тесте - 0.88. Наибольший буст дало логарифмирование таргета, т.е. модель предсказывала логарифм таргета, а потом уже руками предсказание экспоненцировалось. 

Не получилось придумать доп. фичи из существующих, которые бы давали буст в качестве.