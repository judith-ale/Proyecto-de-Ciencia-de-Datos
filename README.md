# Proyecto de Ciencia de Datos: Reconocimiento del abecedario en Lengua de Señas Mexicana mediante Redes Neuronales

![Logo ITESO](https://oci02.img.iteso.mx/Identidades-De-Instancia/ITESO/Logos%20ITESO/Logo-ITESO-Principal.jpg)

**Instituto Tecnológico y de Estudios Superiores de Occidente**

- **Materia:** Proyecto de Ciencia de Datos
- **Profesor:** Cristian Camilo Zapata Zuluaga
- **Integrantes:**
    - Judith Alejandra Hinojosa Rábago
    - Christa Itzel Barrios Martinez
- **Fecha de entrega:** 27 de noviembre de 2023


## Introducción:

El proyecto "Reconocimiento del abecedario en Lengua de Señas Mexicana mediante Redes Neuronales" tiene como objetivo utilizar la ciencia de datos y las redes neuronales para abordar un problema crucial en el campo de la comunicación y accesibilidad. La lengua de señas es una forma vital de comunicación para las personas sordas, y garantizar su reconocimiento y comprensión es esencial para facilitar la inclusión y la igualdad de oportunidades. El reconocimiento automático del abecedario en Lengua de Señas Mexicana a través de redes neuronales puede revolucionar la forma en que las personas sordas interactúan con el mundo digital y mejorar su calidad de vida.


![Foto LSM](https://chihuahua.gob.mx/sites/default/atach2/lenguaje_de_senas.png)

## Antecedentes:

En esta sección, se proporciona información contextual relevante para el proyecto. La lengua de señas es un lenguaje visual-gestual utilizado por la comunidad sorda para comunicarse. Aunque existen aplicaciones y dispositivos que facilitan la comunicación en lengua de señas, la automatización del reconocimiento del abecedario puede hacer que esta forma de comunicación sea más accesible y eficiente. Existen investigaciones previas sobre el reconocimiento de gestos y lengua de señas utilizando técnicas de procesamiento de imágenes y aprendizaje profundo. Sin embargo, el enfoque específico en la Lengua de Señas Mexicana y el abecedario requiere un estudio y desarrollo detallado.

La lengua de señas mexicana es una forma vital de comunicación utilizada por las personas sordas en México. Se basa en movimientos y gestos de las manos, los brazos y las expresiones faciales para transmitir significado. Cada país tiene su propia lengua de señas, y la lengua de señas mexicana tiene su propia gramática y vocabulario específicos.

Se encontró que de parte del Instituto Tecnológico Superior de Misantla se realizo una [investigación (2019)](https://rcs.cic.ipn.mx/2019_148_8/Traduccion%20del%20lenguaje%20de%20senas%20usando%20vision%20por%20computadora.pdf) para generar traducción del lenguaje de señas usando visión por computadora realizando el análisis de redes neuronales multicapa y SVM para el reconocimiento de señas. El set de datos que se uso en este proyecto es elmmismo que se estara utilizando a para nuestro proyecto, el cual consiste en 21 señas estaticas con 300 imágenes cada uno, dando un total de 6300 imágenes totales.

Otro [traductor de señas mexicano](https://repositorio.cinvestav.mx/handle/cinvestav/2314?locale-attribute=en) fue el desarrollado en el Centro de Investigación y de Estudios Avanzados del Instituto Politecnico Nacional por el Ing. Gil Alberto Díaz Balderas en el que se propone un conjunto de datos conformado solo por ́angulos  que  ayudan  a  detectar  las  posiciones  de  las  falanges  de  los  dedos de la mano derecha cuando una seña del alfabeto del lenguaje mexicano de señas. El conjunto se redujo a 51 características las cuales fueron utilizadas para entrenar una red neuronal tipo perceptrón multicapa que pudiera reconocer las 21 señas estáticas del alfabeto del lenguaje mexicano de señas.

## Objetivos:

### Objetivos Generales:

El objetivo general de este proyecto es aplicar los conceptos de ciencia de datos y MLOPS (Machine Learning Operations) para desarrollar un sistema de reconocimiento automático del abecedario estático en Lengua de Señas Mexicana. Se espera que este sistema sea en una herramienta útil y accesible para las personas sordas, facilitando la comunicación en un entorno digital y contribuyendo a su inclusión social.

### Objetivos Específicos:

1. Realizar un análisis exploratorio de datos de gestos en Lengua de Señas Mexicana, recopilando un conjunto de datos representativo.
2. Preprocesar los datos.
3. Entrenar un modelo de redes neuronales convolucionales (CNN) capaz de reconocer y clasificar los gestos del abecedario en Lengua de Señas Mexicana.
4. Crear experimentos con distintos modelos y seleccionar el mejor basado en la metrica accuracy
5. Conteneirizar el servicio de reconocimiento para garantizar su escalabilidad y eficiencia.
6. Desplegar el sistema en la nube para que esté disponible de manera global.

## Planteamiento del problema:

El problema central que abordamos es la dificultad que enfrentan las personas sordas señantes al interactuar con otras personas oyentes o que no tienen conocimiento de la lengua de señas mexicana. La falta de herramientas precisas y accesibles para el reconocimiento del abecedario en Lengua de Señas Mexicana dificulta la comunicación efectiva.

Nuestro proyecto busca resolver este problema al desarrollar un sistema de reconocimiento de gestos del abecedario estático mediante redes neuronales.

### Descripción de los datos:

El conjunto de datos que incluye imágenes que han sido previamente recortadas enfocando únicamente a la mano derecha (de una [investigación anterior (2019)](https://rcs.cic.ipn.mx/2019_148_8/Traduccion%20del%20lenguaje%20de%20senas%20usando%20vision%20por%20computadora.pdf), donde se utilizaron para realizar su propio modelo de clasificación), concentrándose exclusivamente en la representación visual de la señal en sí misma. En este escenario, el objetivo se centra en la predicción de la señal del lenguaje de señas representada en la imagen, sin atender a la tarea de localización de la mano o a ningún otro componente contextual. Este conjunto de datos permitirá la capacitación de modelos destinados a identificar y anticipar las letras y signos del lenguaje de señas mexicano a partir de imágenes aisladas.

## Desarrollo de la solución

### EDA (Análisis Exploratorio de Datos)
Para ver el código y la descripción detallada [clic aquí](/EDA/EDA.ipynb).

**Resumen general:**

- Se aseguró que hubiera la correcta cantidad de imágenes en el dataset, es decir, 300 imágenes por cada una de las 21 señas estáticas del abecedario en Lengua de Señas Mexicana, estando estas balanceadas.
- Se reiteró que las imágenes son de 200 x 200 pixeles a 3 canales (RGB).
- Se visualizó una imagen como ejemplo.

### Data Wrangling
Para ver el código y la descripción detallada [clic aquí](/Manipulación%20de%20datos/data_wrangling.ipynb).

**Resumen general:**

- Se separaron las imágenes en tres directorios: `train` (70% de las imágenes), `validation` (20%) y `test`(10%). Esto para tener las imágenes separadas para el posterior entrenamiento y evaluación.
- La separación se hizo estratificando las imágenes de acuerdo a la letra a la que pertenecen para tener un balance en las clases a la hora de entrenar.

### Dataset final a trabajar

En este caso se utilizaron todas las imágenes para asegurar una mayor cantidad de datos.

### Entrenamiento del modelo con MLflow
Para ver el código y la descripción detallada [clic aquí](/Entrenamiento/training.ipynb).

**Resumen general:**

En esta parte, se detalla cómo se entrenó el modelo de ciencia de datos. Deben utilizar MLflow para registrar los experimentos, incluyendo diferentes configuraciones de modelos y parámetros. Además, deben grabar métricas relevantes para evaluar el rendimiento de los modelos.

Se prepararon las listas con los datos y funciones que fueron introducidas en el pipeline de datos.
- Primero se creó una lista por cada dataset (`train`, `validation` y `test`) donde cada item es un string de 2 valores separados por comas, el primero siendo el path de la imagen y el segundo el id de la letra que está haciendo la mano en la imagen.
- Para poder obtener cada imagen y etiqueta de su respectivo string se creó una función donde se separa dicho string por la coma, a continuación se leyó el archivo correspondiente a la imagen, se decodificó como imagen `jpeg` y se normalizó el arreglo (la imagen) a valores entre 0 y 1 flotantes.
- Para la etiqueta se transformó el string a un entero de 32 bits entre 0 y 21.
- Por último se retornó la imagen y la etiqueta.

Con el listado de strings y la función anterior se construyeron los pipelines para entrenamiento, validación y pruebas.
- Primero se creó un iterador del listado para el cual se mezclaran de forma aleatoria las rutas de las imágene. En cada item de la lista se aplicó la función para obtener la imágen como un arreglo, se indicó el almacen de los datos en cache, se obtuvo por cada lote cierta cantidad de imágenes y se hizo la obtención de cada lote en un tiempo dinámico para que se tuviera listo antes de que terminara el modelo con el lote anterior.
- Para entrenamiento y validación el tamaño del lote tuvo que ser el mismo ya que por cada arreglo (imagen) en el entrenamiento del modelo debe haber uno correspondiente en la validación. En este caso se eligió de 32 imágenes.
- Para el caso de pruebas este dataset no tiene la restricción pasada por lo que se elige de la cantidad de imágenes totales en este lote para obtenerlas de forma fácil para su posterior análisis.

Como métrica de desempeño para los modelos se decidió utilizar `accuracy`.

#### Modelo 1: `InceptionV3_model`

En el primer modelo se colocó como una de sus capas, un modelo pre-entrenado con la arquitectura de `InceptionV3` y los pesos de `imagenet` (que no son entrenables ya que se tiene como objetivo usar en su totalidad el modelo pre-entrenado), pero no se tomó en cuenta la capa final ya que se nuestro modelo se utiliza con otro propósito al del `InceptionV3`, por lo que se une a otras capas de un diseño propio para obtener al final las 21 categorías por medio de una función `softmax` que da como resultado un vector de 21 valores donde cada valor es la probabilidad de que la imagen pertenezca a esa categoría, y la suma de este vector da 1, que es la probabilidad total. Para el entrenamiento se especificaraon 10 épocas.

#### Modelo 2: `new_created_model`

En el segundo modelo únicamente es un diseño propio de una red neuronal convolucional para obtener al final las 21 categorías por medio de la función `softmax` que como se especificó en el modelo anterior da como resultado un vector de 21 valores donde cada valor es la probabilidad de que la imagen pertenezca a esa categoría, y la suma de este vector da 1, que es la probabilidad total. Para el entrenamiento se especificaraon 20 épocas.

### Selección del mejor modelo
Para ver el código y la descripción detallada [clic aquí](/Entrenamiento/training.ipynb#Selección-del-modelo).

**Resumen general:**

De acuerdo con las métricas en entrenamiento, validación y pruebas, el modelo que se elegió para continuar el proyecto es el modelo 1 (`InceptionV3_model`) que tuvo el mejor ajuste en entrenamiento y las mejores predicciones en pruebas.

#### Desempeño en entrenamiento y validación

El accuracy más alto entre ambos modelos tanto en el dataset de entrenamiento como en validación estuvo en `InceptionV3_model`.

![Desempeño modelo](/Imágenes/desempeño_modelos.png)

#### Desempeño en pruebas

El accuracy más alto entre ambos modelos en el dataset de pruebas estuvo en `InceptionV3_model`.

##### Modelo 1: `InceptionV3_model`

![InceptionV3 model](/Imágenes/test_InceptionV3_model.png)

##### Modelo 2: `new_created_model`

![InceptionV3 model](/Imágenes/test_new_created_model.png)

### Servir el modelo (API) con el mejor desempeño
Para ver el código [clic aquí](/API/main.py).

Se utilizó el framework [FastAPI](https://fastapi.tiangolo.com/) para crear la API RESTful en Python del proyecto, con una sola extensión `HTTP` (`/api/v0/jpeg/classify`) en la que se utiliza el modelo seleccionado para realizar la predicción sobre una imagen enviada a través de una petición `GET`, donde el archivo que se recibe debe ser una imagen `JPEG` de no más de 2MB, la cual se transforma en un arreglo con la forma (1, 200, 200, 3) y que representa: 1 imagen de 200x200 pixeles a 3 canales.

Se normalizan los valores del arreglo de 0 a 1 como flotantes. Este arreglo se pasa al modelo elegido que fue descargado desde el experimento `mlflow` en `dagshub`. El modelo retorna un vector con 21 valores entre 0 y 1 que suman 1, del cual se toma el índice con el máximo valor (que es la máxima probabilidad de pertenecer a esa categoría), este número se encuentra entre el 0 y el 20. La función retorna por medio de un diccionario la letra correspondiente al índice con la máxima probabilidad.

Para asegurar la extensión de la API se creó un token de acceso sin el cual no se pueden procesar las peticiones.

### Contenerizar del servicio
Para ver el Dockerfile [clic aquí](/API/Dockerfile).

1. Para poder contenerizar el servicio se creo un archivo `Dockerfile` que da las instrucciones para crear el contenedor que correrá el código de la API, donde no sólo se indica que se debe copiar ese código en la imagen, sino que se señalala la imagen base, que en este caso fue `python:3.10.12-slim-buster`, se expone el puerto por el cual se van a recibir las peticiones y se instala todo lo necesario para correr el código, incluyendo las librerías desde el archivo `requirements.txt`.

2. Se ejecuta el comando `docker build -t nombre-imagen ruta-dockerfile`


**En local**
![docker_build_local.png](/Imágenes/docker_build_local.png) 


**En la nube**
![docker_build_AWS.png](/Imágenes/docker_build_AWS.png) 

3. Se ejecuta el comando `docker run -p puerto-salida:puerto-interior --name nombre-contenedor nombre-imagen`


**En local**
![docker_run_local.png](/Imágenes/docker_run_local.png) 


**En la nube**
![docker_run_AWS.png](/Imágenes/docker_run_AWS.png) 

4. Se prueba el funcionamiento del contenedor, en el caso de la nube se obtiene la IP pública de la instancia a la cual se envía la petición.


**Prueba 1: En la nube**
![prueba_1.png](/Imágenes/prueba_1.png) 



**Prueba 2: En la nube**
![prueba_2.png](/Imágenes/prueba_2.png)

Explica cómo se empaquetó el servicio y se subió a una plataforma de nube, lo que puede incluir el uso de contenedores como Docker.

### Despliegue del servicio en la nube

La instancia `Grupo 2` en la nube se creo desde `AWS`:
- Desde una imagen de Ubuntu.
- El tipo de instancia era t2.micro.
- Con acceso por medio de una llave `pem`.

![creación_instancia_AWS.png](/Imágenes/creación_instancia_AWS.png)

- En las reglas de seguridad se dio acceso `SSH` desde la IP de la red donde se configuró la instancia, también se dio acceso público con el protocolo `HTTP` y `HTTPS`, así como al protocolo `TCP` desde el puerto 8000 que es donde se reciben las peticiones de la API.

![reglas_grupo_seguridad_AWS.png](/Imágenes/reglas_grupo_seguridad_AWS.png)

- Para lograr hacer la conexión desde la máquina local se necesita llave `pem` previamente generada, con ella y el comando `ssh -i "key-grupo2.pem" ubuntu@ecX-XXX-XX-XX-XXX.compute-1.amazonaws.com/` se puede acceder a la instancia de forma remota.

![conexión_instancia_AWS.png](/Imágenes/conexión_instancia_AWS.png)

- En la instancia se descarga el proyecto con el comando `git clone`.

![descarga_proyecto_AWS.png](/Imágenes/descarga_proyecto_AWS.png)

- También se instaló docker en la instancia para poder crear la imagen de la API y el contenedor con permisos de administrador (que se acceden con el comando `sudo su`) y ya instalado se contenerizó el servicio como se explicó en la sección anterior.


## Conclusiones

El proyecto cumplió exitosamente con los objetivos establecidos y el modelo demostró una alta precisión con el dataset utilizado. 

La mayor dificultad se presentó durante el proceso de optimización del modelo para reducir el tiempo de procesamiento sin comprometer la precisión. Así como el implementar la API de mlflow para guardar el modelo en la nube.

Como recomendación para futuras mejoras, se sugiere explorar métodos avanzados de optimización para reducir aún más la complejidad computacional sin afectar la calidad del reconocimiento de gestos. También, implementar un método más seguro para el acceso a la API. Así como desarrollar un modelo con mayor complejidad, que sea capaz de reconocer más señas, incluso las que necesitan de movimiento, para poder lograr en un futuro realizar un interprete.


## Referencias
1. Díaz Balderas, G. A. (2016). Traductor del lenguaje de señas mexicano a texto (Master's thesis, Tesis (MC)--Centro de Investigación y de Estudios Avanzados del IPN Departamento de Computación). https://repositorio.cinvestav.mx/handle/cinvestav/2314?locale-attribute=en
1. Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo,
Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis,
Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow,
Andrew Harp, Geoffrey Irving, Michael Isard, Rafal Jozefowicz, Yangqing Jia,
Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Mike Schuster,
Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Jonathon Shlens,
Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker,
Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas,
Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke,
Yuan Yu, and Xiaoqiang Zheng.
TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. Software available from tensorflow.org
2. mlflow. Documentation Python API mlflow.tensorflow. https://mlflow.org/docs/latest/python_api/mlflow.tensorflow.html
3. Morales, Eduardo & Aparicio, Oswaldo & Arguijo, Pedro & Melendez-Armenta, Roberto & López, José Antonio. (2019). Traducción del lenguaje de señas usando visión por computadora. Research in Computing Science. 148. 79-89. [10.13053/rcs-148-8-6](https://rcs.cic.ipn.mx/2019_148_8/Traduccion%20del%20lenguaje%20de%20senas%20usando%20vision%20por%20computadora.pdf).
4. Ramírez, S. FastAPI. https://fastapi.tiangolo.com

