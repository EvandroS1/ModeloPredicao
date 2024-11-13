# Projeto de Aprendizagem de Máquina: Previsão de Acidentes nas BRs do Brasil

Este projeto utiliza um algoritmo de machine learning, mais especificamente uma árvore de decisão, para analisar uma base de dados de acidentes nas rodovias federais (BRs) do Brasil. O objetivo é prever, com base em entradas de dados como o dia da semana e a BR específica, se ocorrerá um acidente com vítima fatal.

## Integrantes do Trabalho

- Henrique Brito
- Thiago Corrêa
- Evandro Gomes

## Professor Responsável

- Bruno Zolotareff dos Santos (Aprendizagem de Máquina)

## Sumário

- [Introdução](#introdução)
- [Descrição dos Dados](#descrição-dos-dados)
- [Pré-processamento dos Dados](#pré-processamento-dos-dados)
- [Algoritmo de Machine Learning](#algoritmo-de-machine-learning)
- [Treinamento e Avaliação](#treinamento-e-avaliação)
- [Como Utilizar](#como-utilizar)
- [Conclusão](#conclusão)
- [Material de Referência](#material-de-referência)

## Introdução

Acidentes de trânsito são uma das principais causas de mortalidade no Brasil, especialmente nas rodovias federais (BRs). Este projeto visa utilizar técnicas de machine learning para analisar dados históricos de acidentes e prever a ocorrência de acidentes fatais com base em características específicas, como o dia da semana e a BR.

## Descrição dos Dados

A base de dados utilizada neste projeto contém informações detalhadas sobre acidentes de trânsito ocorridos nas BRs do Brasil. As principais colunas da base de dados são:

- `data_inversa`: Data do acidente
- `dia_semana`: Dia da semana em que o acidente ocorreu
- `uf`: Unidade federativa onde o acidente ocorreu
- `br`: Rodovia federal onde o acidente ocorreu
- `causa_acidente`: Causa do acidente
- `ano`: Ano do acidente
- `mortos`: Número de vítimas fatais no acidente

## Pré-processamento dos Dados

Antes de treinar o modelo de machine learning, foi realizado um pré-processamento dos dados, que incluiu:

- Tratamento de valores faltantes
- Conversão de variáveis categóricas em numéricas
- Normalização dos dados
- Divisão dos dados em conjuntos de treino e teste

## Algoritmo de Machine Learning

O algoritmo escolhido para este projeto foi a árvore de decisão. Este algoritmo é conhecido por sua interpretabilidade e capacidade de lidar com variáveis categóricas e numéricas. Foi utilizado o pacote `scikit-learn` para a implementação do modelo.

## Treinamento e Avaliação

O modelo foi treinado utilizando os dados de treino e avaliado com os dados de teste. As métricas de avaliação incluíram acurácia, precisão, recall e F1-score. O desempenho do modelo foi analisado para garantir sua eficácia na previsão de acidentes fatais.

## Como Utilizar

Para utilizar o modelo e prever a ocorrência de acidentes fatais, siga os passos abaixo:

1. Clone este repositório: `git clone https://github.com/seu_usuario/seu_repositorio.git`
2. Instale as dependências: `npm install`
3. Execute o script de previsão: `node ArvoreDecisão.js"`

## Conclusão

Este projeto demonstra a aplicação de técnicas de machine learning para a análise de dados de acidentes de trânsito e a previsão de ocorrências fatais. A utilização de árvores de decisão se mostrou eficaz e pode ser uma ferramenta valiosa para autoridades e órgãos responsáveis pela segurança no trânsito.

## Material de Referência

A base de dados utilizada neste projeto pode ser encontrada no seguinte link: [Acidentes Rodovias Federais Brasil - Jan07 a Jul19](https://www.kaggle.com/datasets/equeiroz/acidentes-rodovias-federais-brasil-jan07-a-jul19/code)

---

Para mais informações, entre em contato com Evandro.
