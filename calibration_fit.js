const fs = require('fs').promises;
const tf = require('@tensorflow/tfjs-node');
const Papa = require('papaparse');

const MODEL_PATH = 'model.json';
const SCALER_PATH = 'scaler.json';
const CALIB_CSV = 'calib_samples.csv';
const CALIB_JSON = 'calib.json';

async function loadCSV(filepath) {
    const content = await fs.readFile(filepath, 'utf8');
    return new Promise((resolve, reject) => {
        Papa.parse(content, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            complete: (results) => resolve(results.data),
            error: (err) => reject(err)
        });
    });
}

function linearRegression(X, y) {
    // Simple linear regression: y = a*X + b
    // X is 1D array, y is 1D array
    const n = X.length;

    const sumX = X.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = X.reduce((sum, x, i) => sum + x * y[i], 0);
    const sumX2 = X.reduce((sum, x) => sum + x * x, 0);

    const meanX = sumX / n;
    const meanY = sumY / n;

    // Calculate slope (a) and intercept (b)
    const a = (sumXY - n * meanX * meanY) / (sumX2 - n * meanX * meanX);
    const b = meanY - a * meanX;

    return { a, b };
}

async function main() {
    console.log('Loading calibration data...');
    const df = await loadCSV(CALIB_CSV);

    const features = ['Bedrooms', 'Square Footage', 'Coverage A', 'Age of Home'];
    const X = df.map(row => features.map(f => row[f]));
    const yTrue = df.map(row => row['ActualPremium']);

    console.log('Loading model and scaler...');
    const model = await tf.loadLayersModel(`file://${MODEL_PATH}`);
    const scalerData = await fs.readFile(SCALER_PATH, 'utf8');
    const scaler = JSON.parse(scalerData);

    // Scale features
    console.log('Scaling features...');
    const XScaled = X.map(row =>
        row.map((val, i) => (val - scaler.mean[i]) / scaler.scale[i])
    );

    // Predict
    console.log('Making predictions...');
    const XTensor = tf.tensor2d(XScaled);
    const predTensor = model.predict(XTensor);
    const yPred = await predTensor.data();

    // Fit linear regression: yTrue = a * yPred + b
    console.log('Fitting calibration...');
    const { a, b } = linearRegression(Array.from(yPred), yTrue);

    // Apply shrinkage for stability with only 3 points
    const alpha = 0.3;
    const aFinal = (1 - alpha) * a + alpha * 1.0;
    const bFinal = (1 - alpha) * b + alpha * 0.0;

    const calibration = { a: aFinal, b: bFinal };

    // Save calibration
    console.log('Saving calibration...');
    await fs.writeFile(CALIB_JSON, JSON.stringify(calibration, null, 2));

    console.log('Calibration results:', calibration);

    // Cleanup
    XTensor.dispose();
    predTensor.dispose();
}

main().catch(err => {
    console.error('Error:', err);
    process.exit(1);
});