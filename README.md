# README

<div>
    <button onclick="document.getElementById('pt').style.display='block';document.getElementById('en').style.display='none'">Português</button>
    <button onclick="document.getElementById('en').style.display='block';document.getElementById('pt').style.display='none'">English</button>
</div>

<div id="pt" style="display:block">

## Introdução
O objetivo desse projeto é o de criar uma LLM (Large Language Model) utilizando a tecnologia GPT que possa ser executada em uma máquina comum. Apesar da proposta de ser uma IA generativa esse código precisará de um amplo poder computacional para de fato ser capaz de gerar textos relevantes.

## Dependências
O projeto faz uso do Pytorch e apesar de ter sido disponibilizado um arquivo de requerimentos recomenda-se que utilize as instruções de instalação do Pytorch presentes em sua homepage. Além dele também pode ser necessário instalar o pickle.

## Executando
Para que o código possa ser executado será necessário que se crie uma pasta com o nome ‘_dataset’ ou altere o código para procurar em uma pasta de sua preferência que o sistema irá pesquisar os arquivos necessários. Será necessário criar um arquivo de treinamento e um de validação para assegurar a assertividade do modelo. Também é importante frisar que o modelo pode ser escalonado utilizando o arquivo hyperparamenters para configurá-lo da maneira que achar melhor, a configuração que deixei é muito precária para que o sistema gere textos que façam algum sentido.

## Sobre o Dataset
Utilizei o OpenWebText2 para treinar o modelo por ser o mais indicado e também curado para isso, existem outros repositórios de textos disponíveis para utilizar, lembrando bem de ser sempre um único arquivo txt.
</div>

<div id="en" style="display:none">
## Introduction
The goal of this project is to create a LLM (Large Language Model) using GPT technology that can be run on a regular machine. Despite being a generative AI, this code will require significant computational power to actually generate relevant texts.

## Dependencies
The project uses Pytorch, and although a requirements file has been provided, it is recommended to use the installation instructions available on the Pytorch homepage. Additionally, you might need to install pickle.

## Running
To run the code, you'll need to create a folder named ‘_dataset’ or modify the code to search in a folder of your preference where the system will look for the necessary files. A training file and a validation file will need to be created to ensure the model's accuracy. It is also important to note that the model can be scaled using the hyperparameters file to configure it as you see fit. The configuration I left is very basic for the system to generate texts that make any sense.

## About the Dataset
I used OpenWebText2 to train the model as it is the most recommended and curated for this purpose. There are other text repositories available for use, always remembering to have a single txt file.
</div>
