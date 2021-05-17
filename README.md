# Multimodal Speech Emotion Recognition and Ambiguity Resolution

## Abstract
Este trabajo se basó en el [paper](https://arxiv.org/abs/1904.06022) el cual se desarrolló en la universidad de Waterloo, Canadá bajo la tutela del [Prof. Richard Mann](https://cs.uwaterloo.ca/~mannr/) de la materia "Computational Audio".
La idea principal consiste en asemejar el desempeño de modelos de machine learning considerados "resource intensive" mediante modelos más "tradicionales" del área. Particularmente se utilizaron los siguientes modelos "clásicos":
* Logistic Regression
* Multinomial Bayes
* Random Forests
* XGBoosted Decision Trees
* Multi Perceptron

Los cuales fueron comparados con el modelo más reciente:
* Long Short Term Memory (una dirección)
* Long Short Term Memory (bidireccional)


## Base de Datos
Utiliza como base de datos el set de [IEMOCAP](https://link.springer.com/content/pdf/10.1007%2Fs10579-008-9076-6.pdf) el cual consiste en una serie de grabaciones de expresiones faciales, del habla y transcripciones en texto. El [paper](https://arxiv.org/abs/1904.06022) explica a detalle del set.


## Requisitos
El repositorio cuenta con un archivo `.yaml` que contiene todo lo necesario para crear un entorno en Anaconda.

## Observaciones
Para correr los modelos y hacer pruebas se utilizó la siguiente GPU:
* `NVIDIA GEFORCE GTX 1060 Max-Q`
* `6 GB.`
Si se desea procesar la base de datos desde 0 es necesario tener más de ~60GB de espacio libre en disco duro:
* ~40GB BDD
* ~10GB word embeddings
* ~10GB audio vectorizado

## Citation
If you find this work useful, please cite:

```
@article{sahu2019multimodal,
  title={Multimodal Speech Emotion Recognition and Ambiguity Resolution},
  author={Sahu, Gaurav},
  journal={arXiv preprint arXiv:1904.06022},
  year={2019}
}
```
