const fs = require('fs');
const csv = require('csvtojson');
const { DecisionTreeClassifier } = require('ml-cart');

async function loadCSV(filePath) {
    try {
        const jsonArray = await csv().fromFile(filePath);
        return jsonArray;
    } catch (error) {
        console.error(`Erro ao carregar o arquivo CSV: ${error}`);
    }
}

function prepareData(data) {
    const features = [];
    const labels = [];

    data.forEach(row => {
        const feature = [
            row['dia_semana'],           // Dia da semana
            row['horario'],              // Horário
            row['uf'],                   // Estado
            row['causa_acidente'],       // Causa do acidente
            row['tipo_acidente'],        // Tipo de acidente
            row['fase_dia'],             // Fase do dia
            row['condicao_metereologica'] // Condições meteorológicas
        ];

        // Convertendo os valores para números ou categorias indexadas, com verificação de existência
        const numericFeature = feature.map(val => {
            if (val === undefined || val === null) return 0; // valor padrão caso esteja indefinido
            return isNaN(parseFloat(val)) ? val.toString().charCodeAt(0) : parseFloat(val);
        });

        features.push(numericFeature);
        labels.push(row['mortos'] > 0 ? 1 : 0); // 1 se houver mortos, 0 caso contrário
    });

    return { features, labels };
}

async function runDecisionTree() {
    const filePath = './data/acidentes.csv';

    // Carregando e preparando os dados
    const data = await loadCSV(filePath);
    if (!data) {
        console.log("Nenhum dado carregado.");
        return;
    }

    const { features, labels } = prepareData(data);

    // Configuração da Árvore de Decisão
    const options = {
        gainFunction: 'gini',
        maxDepth: 10,
        minNumSamples: 3
    };

    const decisionTree = new DecisionTreeClassifier(options);

    // Treinando o modelo
    decisionTree.train(features, labels);

    // Testando uma amostra de entrada
    const sample = [
        'Terça',         // Dia da semana
        '02:10:00',       // Horário
        'MA',             // Estado
        'Animais na Pista', // Causa do acidente
        'Atropelamento de animal', // Tipo de acidente
        'Plena noite',          // Fase do dia
        'Ceu Claro'       // Condições meteorológicas
    ];

    const numericSample = sample.map(val => {
        if (val === undefined || val === null) return 0;
        return isNaN(parseFloat(val)) ? val.toString().charCodeAt(0) : parseFloat(val);
    });

    const prediction = decisionTree.predict([numericSample]);

    console.log(`Predição: ${prediction[0] === 1 ? 'Acidente com mortos' : 'Sem mortos'}`);
}

runDecisionTree();
