import csv
import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer


class RatingClassificationDataset:
    def __init__(self, data, tokenizer):
        self.data = data #representa os dados da nossa tabela de treino
        self.tokenizer = tokenizer #é responsável por preparar as entradas para um modelo.

    def __len__(self):
        return len(self.data) # função que retorna a quantidade de dados presentes na tabela de treino

    def __getitem__(self, item):
        text = str(self.data["review_text"].values[item]) #para cada item da tabela armazenamos na variável text o texto de review do usuário
        target = int(self.data["rating"].values[item]) #para cada item da tabela armazenamos na variável target o valor de rating para aquele dado

        # aqui estamos delimitando que os inputs de texto (review do usuário) terá tamanho máximo de 512 caracteres,
        # depois disso nós truncamos o texto, ou seja cortamos a palavra mesmo que ela não esteja completa.
        inputs = self.tokenizer(
            text,
            max_length=512, 
            padding="max_length", 
            truncation=True,
        )

        # O tokenizer retorna um dicionário com todos os argumentos necessários para que seu modelo correspondente
        # funcione corretamente. Os índices de token estão sob a chave “input_ids”.
        ids = inputs["input_ids"]

        # o attention_mask é utilizado ao agrupar sequências em lote. Esse argumento indica ao modelo quais tokens devem ser atendidos e quais não devem.
        # Por exemplo, como temos tamanhos diferentes de texto não podemos colocá-los no tensor da forma que estão. Precisamos uniformizá-los, no nosso caoso, 
        # preenchemos os textos menores até 512 caracteres e truncamos textos muito grandes.
        # Isso pode então ser convertido em um tensor em PyTorch (conforme o return abaixo). A attention_mask é um tensor binário que indica a posição dos índices
        # preenchidos para que o modelo não os ignore. Esta máscara de atenção está no dicionário retornado pelo tokenizer sob a chave “attention_mask” (conforme return abaixo).
        mask = inputs["attention_mask"]

        # O torch.tensor é uma matriz multidimensional contendo elementos de um único tipo de dados.
        # Os tensores PyTorch são variáveis indexadas (arrays) multidimensionais usadas como base para todas
        # as operações avançadas. Ao contrário dos tipos numéricos padrão, os tensores podem ser atribuídos para
        # usar sua CPU ou GPU para acelerar as operações.
        # Existem 5 tipos de tensores, e aqui estamos utilizando o LongTensor: 64-bit int
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(target, dtype=torch.long),
        }

## df é o nosso DataFrame e estamos populando ele através de um arquivo csv.

## Um DataFrame é uma estrutura que contém dados bidimensionais e seus rótulos correspondentes.
# Um DataFrame é semelhante a uma tabela SQL uma planilha do Excel. Em muitos casos, os DataFrames
# são mais rápidos, fáceis de usar e mais poderosos do que tabelas ou planilhas porque são parte integrante
# dos ecossistemas Python e NumPy.
df = pd.read_csv("goodreads_test_without_bookid.csv")

## model é o nosso modelo, que geramos no site do autotrain e que vamos utilizar para testar nossos dados
# O AutoModelForSequenceClassification é uma classe de modelo genérica que será instanciada com um 
# cabeçalho de classificação de sequência quando criada com o método de classe from_pretrained()

## O método from_pretrained cria uma instância da classe AutoTokenizer a partir de um vocabulário de modelo pré-treinado que é passado por parâmetro.
# Dessa forma, estamos passando o modelo que treinamos no site autotrain.

model = AutoModelForSequenceClassification.from_pretrained("fernanda-dionello/autotrain-goodreads_without_bookid-2171169883")

# Link autotrain Modelo (#2171169884 - flowery-raccoon): fernanda-dionello/autotrain-goodreads_without_bookid-2171169884
# Link autotrain Modelo (#2171169881 - worse-caribou): fernanda-dionello/autotrain-goodreads_without_bookid-2171169881
# Link autotrain Modelo (#2171169880 - large-llama): fernanda-dionello/autotrain-goodreads_without_bookid-2171169880
# Link autotrain Modelo (#2171169882 - uneven-cobra): fernanda-dionello/autotrain-goodreads_without_bookid-2171169882
# Link autotrain Modelo (#2171169883 - hospitable-cormorant) fernanda-dionello/autotrain-goodreads_without_bookid-2171169883

## AutoTokenizer é uma classe tokenizer genérica que é instanciada ao ser chamada pelo método de classe [AutoTokenizer.from_pretrained].

## Além disso, estamos aplicando a propriedade use_fast=true, pois o Tokenizer permite uma implementação completa em
# python ou uma implementação “rápida” baseada nos tokenizadores da biblioteca Rust. Nesse caso, optamos pela
# rápida pois permite uma aceleração significativa em particular ao fazer tokenização em lote (batched tokenization)
# além disso, possui métodos específicos para mapeamento de strings e caracteres, sendo o ideal para o nosso caso em
# que utilizamos classificação por texto.

tokenizer = AutoTokenizer.from_pretrained("fernanda-dionello/autotrain-goodreads_without_bookid-2171169883", use_fast=True) 

# estamos adicionando o valor 0 na coluna rating para todos os nossos dados de testes iniciais no DataFrame
df.loc[:, "rating"] = 0

# instanciamos nossa classe RatingClassificationDataset passando o DataFrame criado através dos dados de treino
# e o tokenizer com base no pre-treino realizado na plataforma autotrain
dataset = RatingClassificationDataset(df, tokenizer)

# A classe Trainer fornece uma API para treinamento completo de recursos no PyTorch, contém o loop de treinamento básico.
trainer = Trainer(model)
# O método predict retorna previsões (com métricas se rótulos estiverem disponíveis) em um conjunto de teste (dataset).
preds = trainer.predict(dataset).predictions

# O método argmax do numpy retorna os índices dos valores máximos ao longo de um eixo (axis).
# Aqui como os dados gerados nas predições vem em formato de matriz (2D), queremos pegar o índice
# do maior valor de cada lista (ou seja, de cada linha) da matriz usando axis=1. 
preds = np.argmax(preds, axis=1)

# Nesse momento tendo acesso aos índices dos valores das predições, vamos salvar os valores dos ratings que o ML gerou num array.
test_ratings = []
for rating in preds:
    test_ratings.append(str(rating))

## Medimos acurácia da nossa ML para o atual modelo, validamos se os ratings que a ML trouxe
# para cada caso de teste está correto (verificando as notas corretas no arquivo goodreads_test_validation)
def ratingValidations():
    # criação de um array que vai salvar os ratings corretos de cada dado de teste oriundos do arquivo goodreads_test_validation.csv
    validation_ratings = []
    # variável que vai salvar a quantidade de acertos
    rightValue = 0
    # salvando os ratings corretos de cada caso de teste no array validation_ratings
    with open('goodreads_test_validation.csv', encoding='utf-8') as validation:
        table = csv.reader(validation, delimiter=';')
        for line in table:
            validation_ratings.append(line[0])
        del validation_ratings[0]

    # nesse momentos fazemos as comparações dos ratings corretos com aqueles que a ML gerou
    for rating in range(len(validation_ratings)):
        if(validation_ratings[rating] == test_ratings[rating]):
            rightValue += 1
    
    print(f"Quantity of right values: {rightValue}%")


ratingValidations()