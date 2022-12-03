Домашнее задание 1

В рамках дз была построена модель регрессии для предсказания стоимости автомобилей, а также реализован веб-сервис для применения построенной модели на новых данных.

Прежде всего был изучен датасет, некоторые признаки были трансформированы в нужный формат, обработаны пропуски, убраны дублирующиеся объекты, изучены графики зависимостей между признаками и с целевой переменной.

Изначальный результат предсказания на гридсерче выдавал 0.62, итоговая модель на гриде выдала 0.85, на тесте - 0.88. Наибольший буст дало логарифмирование таргета, т.е. модель предсказывала логарифм таргета, а потом уже руками предсказание экспоненцировалось. 

Не получилось придумать доп. фичи из существующих, которые бы давали буст в качестве.

Добавил файл forfastapi и response - инпут и аутпут сервиса для 2 пункта

фотки работы сервиса
![image](https://user-images.githubusercontent.com/64848449/205437410-b9adfb60-8645-413e-9d5a-8579d2074be6.png)
![image](https://user-images.githubusercontent.com/64848449/205437436-de6cd249-52ba-4976-8378-78e7be875db7.png)
![image](https://user-images.githubusercontent.com/64848449/205437438-e26046cc-f69c-4970-85e1-b7c5390e48d8.png)
![image](https://user-images.githubusercontent.com/64848449/205437443-32109f16-c00d-4d79-b6fc-e38c1899b118.png)
