const csv = require('csvtojson');
const math = require('mathjs');

// Função Sigmoid
function sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
}

// Função para carregar o CSV e transformar em um formato chave-valor
async function loadCSV(filePath) {
    try {
        const jsonArray = await csv({ delimiter: ';' }).fromFile(filePath);
        return jsonArray;
    } catch (error) {
        console.error(`Erro ao carregar o arquivo CSV: ${error}`);
    }
}

// Função para transformar os dados do CSV em um formato chave-valor
async function transformCSVtoKeyValue(filePath) {
    const data = await loadCSV(filePath);
    if (!data) {
        console.log("Nenhum dado carregado.");
        return;
    }

    // Transformar os dados em chave-valor
    const keyValueData = data.map(row => {
        return {
            dia_semana: row['dia_semana'],
            br: row['br'],
            mortos: row['mortos']
        };
    });

    return keyValueData;
}

// Função para treinar a regressão logística
function trainLogisticRegression(data, learningRate = 0.01, iterations = 1000) {
    let weights = {};
    let totalRows = data.length;
    let features = Object.keys(data[0]).filter(attr => attr !== 'mortos'); // Apenas 'dia_semana' e 'br'

    // Inicializar os pesos
    features.forEach(attr => {
        weights[attr] = Math.random();
    });

    // Treinamento
    for (let i = 0; i < iterations; i++) {
        let gradients = {};
        features.forEach(attr => gradients[attr] = 0);

        data.forEach(row => {
            let z = 0;
            features.forEach(attr => {
                z += weights[attr] * parseFloat(row[attr]);
            });

            let prediction = sigmoid(z);
            let error = prediction - (row['mortos'] > 1 ? 1 : 0);

            features.forEach(attr => {
                gradients[attr] += error * parseFloat(row[attr]);
            });
        });

        features.forEach(attr => {
            weights[attr] -= (learningRate / totalRows) * gradients[attr];
        });
    }

    return weights;
}

// Função para realizar a predição com o modelo treinado
function predictLogisticRegression(weights, sample) {
    let z = 0;
    Object.keys(sample).forEach(attr => {
        if (weights[attr]) {
            z += weights[attr] * parseFloat(sample[attr]);
        }
    });
    return sigmoid(z) >= 0.5 ? 'Acidente com mortos' : 'Sem mortos';
}

// Função principal para rodar o modelo
async function runLogisticRegression() {
    let filePath = './data/acidentes.csv';  // Caminho para o arquivo CSV
    let keyValueData = await transformCSVtoKeyValue(filePath);

    if (!keyValueData) {
        console.log("Nenhum dado carregado.");
        return;
    }

    // Treinar o modelo com os dados transformados
    let weights = trainLogisticRegression(keyValueData);

    // Obter uma lista única de BRs presentes no CSV
    const uniqueBrs = [...new Set(keyValueData.map(row => row.br))];

    // Gerar e testar 10 amostras aleatórias
    for (let i = 0; i < 10; i++) {
        // Escolher aleatoriamente uma BR e um dia da semana
        const randomBr = uniqueBrs[Math.floor(Math.random() * uniqueBrs.length)];
        const filteredDataByBr = keyValueData.filter(row => row.br === randomBr);
        const randomRow = filteredDataByBr[Math.floor(Math.random() * filteredDataByBr.length)];

        const sample = {
            'dia_semana': randomRow.dia_semana, // Escolher aleatoriamente o dia da semana da BR selecionada
            'br': randomRow.br                  // BR selecionada aleatoriamente
        };

        let prediction = predictLogisticRegression(weights, sample);
        console.log(`Amostra ${i + 1}:`);
        console.log(`Entrada: Dia da semana = ${sample['dia_semana']}, BR = ${sample['br']}`);
        console.log(`Predição: ${prediction}`);
        console.log('---');
    }
}

runLogisticRegression();
