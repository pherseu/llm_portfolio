# LLM Project / Projeto LLM

<button onclick="toggleLanguage()">Switch to English / Mudar para Português</button>

<div id="portugues">
## Introdução
O objetivo desse projeto é o de criar uma LLM (Large Language Model) utilizando a tecnologia GPT que possa ser executada em uma máquina comum. Apesar da proposta de ser uma IA generativa, esse código precisará de um amplo poder computacional para de fato ser capaz de gerar textos relevantes.

## Executando
Para que o código possa ser executado, será necessário que se crie uma pasta com o nome ‘_dataset’ ou altere o código para procurar em uma pasta de sua preferência. O sistema irá pesquisar os arquivos necessários.
Será necessário criar um arquivo de treinamento e um de validação para assegurar a assertividade do modelo.
Também é importante frisar que o modelo pode ser escalonado utilizando o arquivo `hyperparameters` para configurá-lo da maneira que achar melhor. A configuração que deixei é muito precária para que o sistema gere textos que façam algum sentido.

## Sobre o Dataset
Utilizei o OpenWebText2 para treinar o modelo por ser o mais indicado e também curado para isso. Existem outros repositórios de textos disponíveis para utilizar, lembrando bem de ser sempre um único arquivo txt.
</div>

<div id="ingles" style="display:none;">
## Introduction
The goal of this project is to create an LLM (Large Language Model) using GPT technology that can be run on a standard machine. Although the proposal is to develop a generative AI, this code will require significant computational power to effectively generate relevant texts.

## Running
To execute the code, it will be necessary to create a folder named '_dataset' or modify the code to search in a folder of your preference where the system will look for the necessary files.
It will be necessary to create a training file and a validation file to ensure the model's accuracy.
It is also important to emphasize that the model can be scaled using the `hyperparameters` file to configure it as you see fit. The configuration I left is very rudimentary for the system to generate texts that make any sense.

## About the Dataset
I used OpenWebText2 to train the model as it is the most recommended and curated for this purpose. There are other text repositories available to use, always remembering to use a single txt file.
</div>

<script>
function toggleLanguage() {
  var portugues = document.getElementById("portugues");
  var ingles = document.getElementById("ingles");
  if (portugues.style.display === "none") {
    portugues.style.display = "block";
    ingles.style.display = "none";
  } else {
    portugues.style.display = "none";
    ingles.style.display = "block";
  }
}
</script>