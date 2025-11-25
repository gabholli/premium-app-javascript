// =============================
// train_and_serialize.js
// =============================
/**
 * Trains a simple linear regression model on your simulated dataset and writes:
 *   - model.json (TensorFlow.js model)
 *   - scaler.json (standardization parameters)
 * 
 * Usage:
 *   node train_and_serialize.js --csv simulated_home_insurance_quotes.csv
 */

const fs = require('fs').promises;
const tf = require('@tensorflow/tfjs-node');
const Papa = require('papaparse');

// Parse command line arguments
function parseArgs() {
    const args = process.argv.slice(2);
    const parsed = {
        csv: 'simulated_home_insurance_quotes.csv',
        model: 'model.json',
        scaler: 'scaler.json'
    };

    for (let i = 0; i < args.length; i++) {
        if (args[i] === '--csv' && args[i + 1]) {
            parsed.csv = args[i + 1];
            i++;
        } else if (args[i] === '--model' && args[i + 1]) {
            parsed.model = args[i + 1];
            i++;
        } else if (args[i] === '--scaler' && args[i + 1]) {
            parsed.scaler = args[i + 1];
            i++;
        }
    }

    return parsed;
}

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

function preprocessData(df) {
    // Remove unwanted columns
    const dropCols = ['ZIP Code', 'Year'];
    df = df.map(row => {
        const newRow = { ...row };
        dropCols.forEach(col => delete newRow[col]);
        return newRow;
    });

    // Create Age of Home from Year Built
    df = df.map(row => {
        if ('Year Built' in row && !('Age of Home' in row)) {
            row['Age of Home'] = 2025 - row['Year Built'];
            delete row['Year Built'];
        }
        return row;
    });

    // Convert Yes/No columns to 0/1
    const ynCols = [
        'Has Swimming Pool',
        'Security System Installed',
        'Has Garage',
        'Has Basement'
    ];

    df = df.map(row => {
        ynCols.forEach(col => {
            if (col in row && typeof row[col] === 'string') {
                row[col] = row[col].trim().toLowerCase() === 'yes' ? 1 : 0;
            }
        });
        return row;
    });

    return df;
}

function trainTestSplit(X, y, testSize = 0.33, randomState = 42) {
    // Simple deterministic shuffle based on seed
    const indices = X.map((_, i) => i);

    // Fisher-Yates shuffle with seeded random
    let rng = randomState;
    const seededRandom = () => {
        rng = (rng * 9301 + 49297) % 233280;
        return rng / 233280;
    };

    for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(seededRandom() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
    }

    const splitIdx = Math.floor(X.length * (1 - testSize));
    const trainIndices = indices.slice(0, splitIdx);
    const testIndices = indices.slice(splitIdx);

    return {
        X_train: trainIndices.map(i => X[i]),
        X_test: testIndices.map(i => X[i]),
        y_train: trainIndices.map(i => y[i]),
        y_test: testIndices.map(i => y[i])
    };
}

function standardScaler(X_train) {
    const numFeatures = X_train[0].length;
    const mean = new Array(numFeatures).fill(0);
    const std = new Array(numFeatures).fill(0);

    // Calculate mean
    for (let j = 0; j < numFeatures; j++) {
        let sum = 0;
        for (let i = 0; i < X_train.length; i++) {
            sum += X_train[i][j];
        }
        mean[j] = sum / X_train.length;
    }

    // Calculate std
    for (let j = 0; j < numFeatures; j++) {
        let sumSq = 0;
        for (let i = 0; i < X_train.length; i++) {
            sumSq += Math.pow(X_train[i][j] - mean[j], 2);
        }
        std[j] = Math.sqrt(sumSq / X_train.length);
    }

    const transform = (X) => {
        return X.map(row =>
            row.map((val, j) => (val - mean[j]) / std[j])
        );
    };

    return { mean, scale: std, transform };
}

function calculateMetrics(yTrue, yPred) {
    const n = yTrue.length;

    // MAE
    let mae = 0;
    for (let i = 0; i < n; i++) {
        mae += Math.abs(yTrue[i] - yPred[i]);
    }
    mae /= n;

    // RMSE
    let mse = 0;
    for (let i = 0; i < n; i++) {
        mse += Math.pow(yTrue[i] - yPred[i], 2);
    }
    const rmse = Math.sqrt(mse / n);

    // R2
    const yMean = yTrue.reduce((a, b) => a + b, 0) / n;
    let ssTot = 0;
    let ssRes = 0;
    for (let i = 0; i < n; i++) {
        ssTot += Math.pow(yTrue[i] - yMean, 2);
        ssRes += Math.pow(yTrue[i] - yPred[i], 2);
    }
    const r2 = 1 - (ssRes / ssTot);

    return { MAE: mae, RMSE: rmse, R2: r2 };
}

async function main(csvPath, outModel, outScaler) {
    console.log(`Loading CSV: ${csvPath}`);
    let df = await loadCSV(csvPath);

    console.log(`Preprocessing data...`);
    df = preprocessData(df);

    // Ensure Coverage A exists
    if (!df[0].hasOwnProperty('Coverage A')) {
        if (df[0].hasOwnProperty('Property Value')) {
            df = df.map(row => {
                row['Coverage A'] = row['Property Value'];
                return row;
            });
        } else {
            throw new Error('CSV must include "Coverage A" or "Property Value"');
        }
    }

    // Select features
    const features = ['Bedrooms', 'Square Footage', 'Coverage A', 'Age of Home'];
    const missingFeats = features.filter(f => !df[0].hasOwnProperty(f));

    if (missingFeats.length > 0) {
        throw new Error(`CSV is missing required columns: ${missingFeats.join(', ')}`);
    }

    // Extract X and y
    const X = df.map(row => features.map(f => row[f]));
    const y = df.map(row => row['Premiums Per Policy']);

    console.log(`Splitting data...`);
    const { X_train, X_test, y_train, y_test } = trainTestSplit(X, y);

    console.log(`Scaling features...`);
    const scaler = standardScaler(X_train);
    const X_train_scaled = scaler.transform(X_train);
    const X_test_scaled = scaler.transform(X_test);

    // Build and train model
    console.log(`Training linear regression model...`);
    const model = tf.sequential({
        layers: [
            tf.layers.dense({
                inputShape: [features.length],
                units: 1,
                useBias: true,
                kernelInitializer: 'zeros',
                biasInitializer: 'zeros'
            })
        ]
    });

    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'meanSquaredError'
    });

    const X_train_tensor = tf.tensor2d(X_train_scaled);
    const y_train_tensor = tf.tensor2d(y_train, [y_train.length, 1]);

    await model.fit(X_train_tensor, y_train_tensor, {
        epochs: 100,
        verbose: 0,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                if ((epoch + 1) % 20 === 0) {
                    console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}`);
                }
            }
        }
    });

    // Evaluate
    console.log(`Evaluating model...`);
    const X_test_tensor = tf.tensor2d(X_test_scaled);
    const predictions = model.predict(X_test_tensor);
    const yPred = await predictions.data();

    const metrics = calculateMetrics(y_test, Array.from(yPred));
    console.log(JSON.stringify(metrics, null, 2));

    // Save model
    console.log(`Saving model to ${outModel}...`);
    await model.save(`file://./${outModel.replace('.json', '')}`);

    // Save scaler
    console.log(`Saving scaler to ${outScaler}...`);
    await fs.writeFile(outScaler, JSON.stringify({
        mean: scaler.mean,
        scale: scaler.scale
    }, null, 2));

    console.log(`Done! Saved: ${outModel}, ${outScaler}`);

    // Cleanup
    X_train_tensor.dispose();
    y_train_tensor.dispose();
    X_test_tensor.dispose();
    predictions.dispose();
}

// Run
const args = parseArgs();
main(args.csv, args.model, args.scaler).catch(err => {
    console.error('Error:', err);
    process.exit(1);
});