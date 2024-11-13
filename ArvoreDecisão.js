const { DecisionTreeClassifier } = require('ml-cart');
const csv = require('csvtojson');

// Função para carregar dados do CSV
async function loadCSV(filePath) {
    try {
        const jsonArray = await csv({ delimiter: ';' }).fromFile(filePath);
        return jsonArray;
    } catch (error) {
        console.error(`Erro ao carregar o arquivo CSV: ${error}`);
    }
}

// Função para transformar o CSV para chave-valor
async function transformCSVtoKeyValue(filePath) {
    const data = await loadCSV(filePath);
    if (!data) {
        console.log("Nenhum dado carregado.");
        return;
    }

    const keyValueData = data.map(row => {
        return {
            dia_semana: row['dia_semana'],
            uf: row['uf'],
            br: row['br'],
            causa_acidente: row['causa_acidente'],
            mortos: row['mortos']
        };
    });

    return keyValueData;
}

// Função para realizar One-Hot Encoding manualmente
function oneHotEncode(values) {
    const uniqueValues = Array.from(new Set(values));
    return values.map(value => {
        const encoding = new Array(uniqueValues.length).fill(0);
        encoding[uniqueValues.indexOf(value)] = 1;
        return encoding;
    });
}

// Função para preparar os dados para o modelo
function prepareData(data) {
    const diaSemana = data.map(row => row['dia_semana']);
    const br = data.map(row => row['br']);
    
    const diaSemanaEncoded = oneHotEncode(diaSemana);
    const brEncoded = oneHotEncode(br);

    const features = diaSemanaEncoded.map((encoding, index) => encoding.concat(brEncoded[index]));
    const labels = data.map(row => (row['mortos'] == 1 ? 1 : 0)); // 1 para sem mortos, 0 para com mortos

    console.log('Distribuição de mortos: ', data.map(row => row['mortos']));

    return { features, labels, uniqueDiaSemana: Array.from(new Set(diaSemana)), uniqueBr: Array.from(new Set(br)) };
}

async function runDecisionTree() {
    const filePath = './data/acidentes.csv';
    const keyValueData = await transformCSVtoKeyValue(filePath);

    if (!keyValueData) {
        console.log("Nenhum dado carregado.");
        return;
    }

    // Preparar os dados
    const { features, labels, uniqueDiaSemana, uniqueBr } = prepareData(keyValueData);

    // Verificando a distribuição dos rótulos
    const mortosCount = labels.filter(label => label >= 1).length;
    const semMortosCount = labels.length - mortosCount;
    console.log(`Distribuição dos dados: Com mortos = ${mortosCount}, Sem mortos = ${semMortosCount}`);

    // Configuração da Árvore de Decisão
    const options = {
        gainFunction: 'gini',
        maxDepth: 10,
        minNumSamples: 3
    };

    const decisionTree = new DecisionTreeClassifier(options);

    // Treinando o modelo
    decisionTree.train(features, labels);

    // Gerando e testando 10 amostras aleatórias
    for (let i = 0; i < 10; i++) {
        // Escolher aleatoriamente uma BR e um dia da semana
        const randomBr = uniqueBr[Math.floor(Math.random() * uniqueBr.length)];
        const filteredDataByBr = keyValueData.filter(row => row.br === randomBr);
        const randomRow = filteredDataByBr[Math.floor(Math.random() * filteredDataByBr.length)];

        const sample = {
            'dia_semana': randomRow.dia_semana, // Dia da semana da BR aleatória
            'br': randomRow.br                  // BR aleatória
        };

        const sampleEncoded = oneHotEncode([sample['dia_semana']]).concat(oneHotEncode([sample['br']]));
        const numericSample = sampleEncoded.flat();

        const prediction = decisionTree.predict([numericSample]);

        console.log(`Amostra ${i + 1}:`);
        console.log(`Entrada: Dia da semana = ${sample['dia_semana']}, BR = ${sample['br']}`);
        console.log(`Predição: ${prediction[0] === 1 ? 'Acidente com mortos' : 'Sem mortos'}`);
        console.log('---');
    }
}

runDecisionTree();
