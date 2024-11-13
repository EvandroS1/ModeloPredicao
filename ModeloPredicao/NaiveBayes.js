const csv = require('csvtojson');
const math = require('mathjs');

async function loadCSV(filePath) {
    const jsonArray = await csv().fromFile(filePath);
    return jsonArray;
}

function trainNaiveBayesImproved(data) {
    let classes = {};
    let totalRows = data.length;

    data.forEach(row => {
        let result = row['mortos'] > 0 ? 1 : 0;
        if (!classes[result]) {
            classes[result] = { total: 0, attributes: {}, means: {}, stds: {} };
        }
        classes[result].total += 1;

        for (let attribute in row) {
            if (attribute === 'mortos') continue;

            if (['km', 'ano', 'pessoas', 'feridos', 'veiculos'].includes(attribute)) {
                if (!classes[result].attributes[attribute]) {
                    classes[result].attributes[attribute] = [];
                }
                classes[result].attributes[attribute].push(parseFloat(row[attribute]));
            } else {
                if (!classes[result].attributes[attribute]) {
                    classes[result].attributes[attribute] = {};
                }
                let value = row[attribute];
                if (!classes[result].attributes[attribute][value]) {
                    classes[result].attributes[attribute][value] = 0;
                }
                classes[result].attributes[attribute][value] += 1;
            }
        }
    });

    for (let result in classes) {
        classes[result].priorProbability = classes[result].total / totalRows;
        for (let attribute in classes[result].attributes) {
            if (Array.isArray(classes[result].attributes[attribute])) {
                classes[result].means[attribute] = math.mean(classes[result].attributes[attribute]);
                classes[result].stds[attribute] = math.std(classes[result].attributes[attribute]);
            } else {
                for (let value in classes[result].attributes[attribute]) {
                    classes[result].attributes[attribute][value] /= classes[result].total;
                }
            }
        }
    }

    return classes;
}

function gaussianProbability(x, mean, std) {
    return (1 / (Math.sqrt(2 * Math.PI) * std)) * Math.exp(-((x - mean) ** 2) / (2 * std ** 2));
}

function predictNaiveBayesImproved(model, sample) {
    let maxProbability = -1;
    let bestClass = null;

    for (let result in model) {
        let probability = model[result].priorProbability;

        for (let attribute in sample) {
            if (model[result].means[attribute] !== undefined) {
                let mean = model[result].means[attribute];
                let std = model[result].stds[attribute];
                let value = parseFloat(sample[attribute]);
                let prob = gaussianProbability(value, mean, std);
                probability *= prob;
            } else if (model[result].attributes[attribute] && model[result].attributes[attribute][sample[attribute]]) {
                probability *= model[result].attributes[attribute][sample[attribute]];
            } else {
                probability *= 0.01;
            }
        }

        if (probability > maxProbability) {
            maxProbability = probability;
            bestClass = result;
        }
    }

    return bestClass;
}

async function runNaiveBayes() {
    let filePath = './data/acidentes.csv';
    let data = await loadCSV(filePath);

    let model = trainNaiveBayesImproved(data);

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
    

    let prediction = predictNaiveBayesImproved(model, sample);
    console.log(`Predição: ${prediction === 1 ? 'Acidente com mortos' : 'Sem mortos'}`);
}

runNaiveBayes();
