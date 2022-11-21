# ML-goodreads

- Abordagem utilizada: NLP - Natural Language processing
- Abordagem no site Autotrain (Hugging Face Hub) para classificação (Task): Text Classification (Multi-class)
- Libraries utilizadas: Transformers, Tokenizers, Pytorch, Numpy and Pandas


## Refs:
 - https://huggingface.co/docs/transformers/model_doc/auto
 - https://huggingface.co/docs/transformers/main_classes/trainer
 - https://realpython.com/pandas-dataframe/
 - https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
 - https://developer.oracle.com/pt-BR/learn/technical-articles/1481879246306-134-what-is-pytorch
 - https://ui.autotrain.huggingface.co/21711/trainings
 - https://huggingface.co/fernanda-dionello/autotrain-goodreads_without_bookid-2171169884?text=I+love+AutoTrain+%F0%9F%A4%97
 - https://developers.google.com/machine-learning/crash-course/classification/video-lecture


Link Dataset via Autotrain:
https://huggingface.co/datasets/fernanda-dionello/autotrain-data-goodreads_without_bookid

<img width="806" alt="image" src="https://user-images.githubusercontent.com/74319133/203107625-a91b99d2-e900-4dfb-901c-7e6095749c57.png">
<img width="794" alt="image" src="https://user-images.githubusercontent.com/74319133/203107800-f536214a-c1c5-4cf7-997b-905d23575d99.png">
 
## Métricas dos models treinados no site AutoTrain:
<img width="1150" alt="model_metrics" src="https://user-images.githubusercontent.com/74319133/203105021-9d10d664-d2eb-44b8-88df-2fa866b092f6.png">

### Conceitos relacionados as métricas:

- Loss = é a penalidade para uma previsão ruim. Ou seja, loss é um número que indica quão ruim foi a previsão do modelo em um único exemplo. Se a previsão do modelo for perfeita, loss é zero; caso contrário, loss é maior. O objetivo de treinar um modelo é encontrar um conjunto de pesos e vieses que tenham baixo loss, em média, em todos os exemplos.

- Accuracy = é o cálculo de Número de predições corretas dividido pelo total de predições.
- Precision = A precisão tenta responder à seguinte pergunta: "Qual a proporção de identificações positivas estava correta?"
    - é o cálculo de Número de verdadeiros positivos (TP) dividido pelo número total de verdadeiros positivos (TP) e falsos positivos (FP).
    - Um verdadeiro positivo é um resultado em que o modelo prevê corretamente a classe positiva. Da mesma forma, um verdadeiro negativo é um resultado em que o modelo corretamente prevê a classe negativo.
    - Um falso positivo é um resultado em que o modelo prevê incorretamente a classe positiva. E um falso negativo é um resultado em que o modelo prevê incorretamente a classe negativa.

![image](https://user-images.githubusercontent.com/74319133/203144024-6cd49432-955b-4d86-a033-a7e2a326ce09.png)


- Recall = O recall tenta responder à seguinte pergunta: "Qual proporção de positivos verdadeiros foi identificada corretamente?""
    - é o cálculo de Número de verdadeiros positivos (TP) dividido pelo número total de verdadeiros positivos (TP) e falsos negativos (FN).

- F1-Score = é a média harmônica de precision e recall.  A pontuação F1 varia entre 0 e 1. Quanto mais próximo de 1, melhor o modelo.

* No caso de classificação multiclasse, adotamos métodos de média para cálculo das pontuações, resultando em um conjunto de diferentes pontuações médias (macro, ponderada, micro) no relatório de classificação.
    - Macro - Calcula a métrica para cada classe e calcula a média não ponderada
    - Micro - Calcula a métrica globalmente contando o total de verdadeiros positivos, falsos negativos e falsos positivos (independentes das classes).
    - Ponderada - Calcula a métrica para cada classe e usa a média ponderada com base no número de amostras por classe.
