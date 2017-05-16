# Построитель каскада Хаара

Субмодуль компьютероного зрения, который обучает каскад Хаара и позволяет использовать его для распознавания объектов в видеопотоке.

## Создание каскада

Работа модуля разделяется на два этапа: создание каскада и его использование. В данном разделе описываются методы запуска процедуры создания каскада Хаара.

### Получение негативных изображений

Получить изображения можно с помощью разбиения видеоряда на кадры. Для этого используется скрипт `frames_generator.py`. 


Также допускается использование изображений со сторонних ресурсов. Их загрузку осуществляет метод `frames_extractor`. 

Далее полученные изображения масштабируются (по необходимости) с помощью метода `resize`.

Для дальнейшей работы необходимо создать текстовый файл с путями к полученным изображениям (генерируется с помощью консольной утилиты `find`).

### Получение позитивных изображений

Существует 2 способа получения позитивных изображений: встраивание одного позитивного примитива во все негативные изображения с помощью утилиты `opencv_createsamples` или ручной сбор позитивных примеров и дальнейшее формирование файла, содержащего список позитивов, их количества на кадре и координаты их расположения на изображении.

### Создание вектора позитивных изображений

Для создания вектора позитивных изображений используется утилита `opencv_createsamples`.

### Обучение каскада

Для обучение каскада Хаара используется утилита `opencv_traincascade`.

## Использование полученного каскада

Каскад, созданный на предыдущем этапе, используется для обнаружения позитивных объектов в видеопотоке. Его обработка осуществляется с помощью скрипта `main.py`. 
