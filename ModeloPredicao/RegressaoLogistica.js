const csv = require('csvtojson');
const math = require('mathjs');

function sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
}

async function loadCSV(filePath) {
    const jsonArray = await csv().fromFile(filePath);
    return jsonArray;
}

function trainLogisticRegression(data, learningRate = 0.01, iterations = 1000) {
    let weights = {};
    let totalRows = data.length;
    let features = Object.keys(data[0]).filter(attr => attr !== 'mortos');

    features.forEach(attr => {
        weights[attr] = Math.random();
    });

    for (let i = 0; i < iterations; i++) {
        let gradients = {};
        features.forEach(attr => gradients[attr] = 0);

        data.forEach(row => {
            let z = 0;
            features.forEach(attr => {
                z += weights[attr] * parseFloat(row[attr]);
            });

            let prediction = sigmoid(z);
            let error = prediction - (row['mortos'] > 0 ? 1 : 0);

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

function predictLogisticRegression(weights, sample) {
    let z = 0;
    Object.keys(sample).forEach(attr => {
        if (weights[attr]) {
            z += weights[attr] * parseFloat(sample[attr]);
        }
    });
    return sigmoid(z) >= 0.5 ? 'Acidente com mortos' : 'Sem mortos';
}

async function runLogisticRegression() {
    let filePath = './data/acidentes.csv';
    let data = await loadCSV(filePath);

    let weights = trainLogisticRegression(data);

    let sample = {
        'dia_semana': 'QUARTA',
        'horario': '05:30:00',
        'uf': 'CE',
        'br': '101',
        'km': 45,
        'municipio': 'CAUCAIA',
        'causa_acidente': 'Falta de atenção',
        'tipo_acidente': 'Colisão frontal',
        'classificacao_acidente': 'Grave',
        'fase_dia': 'Tarde',
        'sentido_via': 'Centro - Zona Leste',
        'condicao_metereologica': 'Céu limpo',
        'tipo_pista': 'Pavimentada',
        'tracado_via': 'Reta',
        'uso_solo': 'Residencial',
    };

    let prediction = predictLogisticRegression(weights, sample);
    console.log(`Predição: ${prediction}`);
}

runLogisticRegression();
