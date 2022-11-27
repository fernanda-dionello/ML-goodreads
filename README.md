# ML-goodreads

## O problema:
A ML deverá predizer qual o rating (de 0 a 5) que um usuário vai fornecer para um livro com base no review que o mesmo escreveu para a leitura na plataforma GoodReads.

- **Abordagem utilizada: NLP - Natural Language processing**
  - O processamento de linguagem natural (NLP) é um ramo da inteligência artificial (IA) que permite aos computadores compreender, gerar e manipular a linguagem humana. O processamento de linguagem natural tem a capacidade de interrogar os dados com texto ou voz de linguagem natural. Isso também é chamado de "entrada de linguagem".  
- **Abordagem no site Autotrain (Hugging Face Hub) para classificação (Task): Text Classification (Multi-class)**
  - Multi-class text classification é uma tarefa de classificação de texto com mais de duas classes/categorias. Cada amostra de dados pode ser classificada em uma das classes. No entanto, uma amostra de dados não pode pertencer a mais de uma classe simultaneamente
- **Libraries utilizadas: Transformers, Tokenizers, Pytorch, Numpy and Pandas**

## Resolução do problema (Etapas efetuadas):
1. Escolhemos o site Hugging Face que possui o serviço chamado Autotrain, o qual possibilita criar modelos de IA de forma gratuita (com limite de 5 modelos por projeto)
2. Antes de iniciar a criação dos modelos, no arquivo de treinamento (goodreads_train_without_bookid.csv) foram realizadas as seguintes tratativas:
    - Deleção de todas colunas, exceto as de `rating` e `review_text`.
        - `rating`: é a classificação que o usuário forneceu para o livro em que o valor pode variar de 0 a 5;
        - `review_text`: é o texto/review que o usuário escreveu sobre o livro
    - Remoção das seguintes pontuações dos reviews: ', "", ~
3. Após, no site, foi realizado o upload do arquivo csv de treinamento (goodreads_train_without_bookid.csv) com 2950 reviews para análise e foi escolhida a abordagem de Text Classification (Multi-class), sendo associada a coluna `rating` como sendo o target da IA e o `review_text` como o texto a ser analisado pela IA.
4. O autoTrain dividiu o arquivo de treino numa proporção de 80% (2358 reviews) para dados de treino dos modelos (o qual ele utilizaria para treinar a ML que ele criaria) e 20% (592 reviews) para validação (ou seja, testar a eficácia dos modelos criados).
5. No modelo gratuito, o autoTrain gerou os 5 modelos abaixo com os seguintes % de assertividade:
//////colocar o print dos modelos
6. Dessa forma, para cada um dos 5 modelos gerados, realizamos um teste de assertividade dos modelos através do código contido no arquivo main.py. O código criado executa o modelo que estamos passando para ele através de um link do Autrotrain, e tenta advinhar qual é o rating para cada review que está no arquivo (goodreads_test_without_bookid.csv), o qual contém 100 reviews. Depois que a ML gera os 100 resultados, é realizado uma comparação com os reais valores de rating desses reviews de teste que estão contidos no arquivo (goodreads_test_validation.csv). Por fim, conseguimos verificar o percentual de assertividade de cada modelo para o nosso arquivo de teste, apresentando os seguintes resultados:
Modelo (#2171169884 - flowery-raccoon): 83%
Modelo (#2171169881 - worse-caribou): 76%
Modelo (#2171169880 - large-llama): 87%
Modelo (#2171169882 - uneven-cobra): 66%
Modelo (#2171169883 - hospitable-cormorant): 78%
7. Por fim, conseguimos concluir que o modelo que melhor performou para o arquivo de teste enviado foi o (#2171169880 - large-llama), embora o site Hugging Face tenha indicado inicialmente que o (#2171169884 - flowery-raccoon) teve maior assertividade nos treinamentos realizados na plataforma deles. Não é possível de fato concluir o porquê essa discordância ocorreu, mas nota-se que ambos modelos tiveram percentuais de assertividade muito próximos no site Autotrain.

## Refs:
 - https://huggingface.co/docs/transformers/model_doc/auto
 - https://huggingface.co/docs/transformers/main_classes/trainer
 - https://realpython.com/pandas-dataframe/
 - https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
 - https://developer.oracle.com/pt-BR/learn/technical-articles/1481879246306-134-what-is-pytorch
 - https://ui.autotrain.huggingface.co/21711/trainings
 - https://huggingface.co/fernanda-dionello/autotrain-goodreads_without_bookid-2171169884?text=I+love+AutoTrain+%F0%9F%A4%97
 - https://developers.google.com/machine-learning/crash-course/classification/video-lecture
 - https://huggingface.co/docs/transformers/tasks/sequence_classification
 - https://huggingface.co/tasks/text-classification
 - https://huggingface.co/course/chapter3/2?fw=pt
 - https://huggingface.co/course/chapter3/3?fw=pt


**Link Dataset via Autotrain:**
https://huggingface.co/datasets/fernanda-dionello/autotrain-data-goodreads_without_bookid

<img width="806" alt="image" src="https://user-images.githubusercontent.com/74319133/203107625-a91b99d2-e900-4dfb-901c-7e6095749c57.png">
<img width="794" alt="image" src="https://user-images.githubusercontent.com/74319133/203107800-f536214a-c1c5-4cf7-997b-905d23575d99.png">
 
## Métricas dos models treinados no site AutoTrain:
<img width="1150" alt="model_metrics" src="https://user-images.githubusercontent.com/74319133/203105021-9d10d664-d2eb-44b8-88df-2fa866b092f6.png">

### Conceitos relacionados as métricas:

- **Loss** = é a penalidade para uma previsão ruim. Ou seja, loss é um número que indica quão ruim foi a previsão do modelo em um único exemplo. Se a previsão do modelo for perfeita, loss é zero; caso contrário, loss é maior. O objetivo de treinar um modelo é encontrar um conjunto de pesos e vieses que tenham baixo loss, em média, em todos os exemplos.

- **Accuracy** = é o cálculo de Número de predições corretas dividido pelo total de predições.
- **Precision** = A precisão tenta responder à seguinte pergunta: "Qual a proporção de identificações positivas estava correta?"
    - é o cálculo de Número de verdadeiros positivos (TP) dividido pelo número total de verdadeiros positivos (TP) e falsos positivos (FP).
    - Um verdadeiro positivo é um resultado em que o modelo prevê corretamente a classe positiva. Da mesma forma, um verdadeiro negativo é um resultado em que o modelo corretamente prevê a classe negativo.
    - Um falso positivo é um resultado em que o modelo prevê incorretamente a classe positiva. E um falso negativo é um resultado em que o modelo prevê incorretamente a classe negativa.

![image](https://user-images.githubusercontent.com/74319133/203144024-6cd49432-955b-4d86-a033-a7e2a326ce09.png)


- **Recall** = O recall tenta responder à seguinte pergunta: "Qual proporção de positivos verdadeiros foi identificada corretamente?""
    - é o cálculo de Número de verdadeiros positivos (TP) dividido pelo número total de verdadeiros positivos (TP) e falsos negativos (FN).

- **F1-Score** = é a média harmônica de precision e recall. A pontuação F1 varia entre 0 e 1. Quanto mais próximo de 1, melhor o modelo.

* No caso de classificação multiclasse, adotamos métodos de média para cálculo das pontuações, resultando em um conjunto de diferentes pontuações médias (macro, ponderada, micro) no relatório de classificação.
    - **Macro** - Calcula a métrica para cada classe e calcula a média não ponderada
    - **Micro** - Calcula a métrica globalmente contando o total de verdadeiros positivos, falsos negativos e falsos positivos (independentes das classes).
    - **Ponderada** - Calcula a métrica para cada classe e usa a média ponderada com base no número de amostras por classe.
