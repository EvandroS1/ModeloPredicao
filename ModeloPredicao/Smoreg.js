const csv = require('csvtojson');
const math = require('mathjs');

async function loadCSV(filePath) {
    const jsonArray = await csv().fromFile(filePath);
    return jsonArray;
}

function kernelLinear(x1, x2) {
    return math.dot(Object.values(x1), Object.values(x2));
}

function trainSMOReg(data, C = 1.0, tol = 0.001, maxPasses = 5) {
    let alphas = Array(data.length).fill(0);
    let b = 0;
    let passes = 0;
    let features = Object.keys(data[0]).filter(attr => attr !== 'mortos');

    const getY = row => (row['mortos'] > 0 ? 1 : -1);

    while (passes < maxPasses) {
        let alphaChanged = 0;

        for (let i = 0; i < data.length; i++) {
            let x_i = features.reduce((obj, attr) => {
                obj[attr] = parseFloat(data[i][attr]);
                return obj;
            }, {});
            let y_i = getY(data[i]);

            let E_i = predictSMOReg(features, alphas, b, x_i, data, getY) - y_i;

            if ((y_i * E_i < -tol && alphas[i] < C) || (y_i * E_i > tol && alphas[i] > 0)) {
                let j = i;
                while (j === i) {
                    j = Math.floor(Math.random() * data.length);
                }

                let x_j = features.reduce((obj, attr) => {
                    obj[attr] = parseFloat(data[j][attr]);
                    return obj;
                }, {});
                let y_j = getY(data[j]);

                let E_j = predictSMOReg(features, alphas, b, x_j, data, getY) - y_j;

                let alpha_i_old = alphas[i];
                let alpha_j_old = alphas[j];

                let L, H;
                if (y_i !== y_j) {
                    L = Math.max(0, alphas[j] - alphas[i]);
                    H = Math.min(C, C + alphas[j] - alphas[i]);
                } else {
                    L = Math.max(0, alphas[i] + alphas[j] - C);
                    H = Math.min(C, alphas[i] + alphas[j]);
                }

                if (L === H) continue;

                let eta = 2 * kernelLinear(x_i, x_j) - kernelLinear(x_i, x_i) - kernelLinear(x_j, x_j);
                if (eta >= 0) continue;

                alphas[j] -= (y_j * (E_i - E_j)) / eta;
                alphas[j] = Math.min(H, Math.max(L, alphas[j]));

                if (Math.abs(alphas[j] - alpha_j_old) < tol) continue;

                alphas[i] += y_i * y_j * (alpha_j_old - alphas[j]);

                let b1 = b - E_i - y_i * (alphas[i] - alpha_i_old) * kernelLinear(x_i, x_i) - y_j * (alphas[j] - alpha_j_old) * kernelLinear(x_i, x_j);
                let b2 = b - E_j - y_i * (alphas[i] - alpha_i_old) * kernelLinear(x_i, x_j) - y_j * (alphas[j] - alpha_j_old) * kernelLinear(x_j, x_j);

                b = (b1 + b2) / 2;
                alphaChanged++;
            }
        }

        passes = alphaChanged === 0 ? passes + 1 : 0;
    }

    return { alphas, b };
}

function predictSMOReg(features, alphas, b, sample, data, getY) {
    let prediction = 0;

    for (let i = 0; i < data.length; i++) {
        let x_i = features.reduce((obj, attr) => {
            obj[attr] = parseFloat(data[i][attr]);
            return obj;
        }, {});

        prediction += alphas[i] * getY(data[i]) * kernelLinear(x_i, sample);
    }

    return prediction + b;
}

async function runSMOReg() {
    let filePath = './data/acidentes.csv';
    let data = await loadCSV(filePath);

    let { alphas, b } = trainSMOReg(data);

    let sample = {
        'dia_semana': 4,
        'horario': '05:30:00',
        'uf': 'CE',
        'br': 101,
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

    let prediction = predictSMOReg(Object.keys(sample), alphas, b, sample, data, row => (row['mortos'] > 0 ? 1 : -1));
    console.log(`Predição: ${prediction >= 0 ? 'Acidente com mortos' : 'Sem mortos'}`);
}

runSMOReg();
