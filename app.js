const express = require('express');
const path = require('path');
const fs = require('fs').promises;
const tf = require('@tensorflow/tfjs-node');
require('@tensorflow/tfjs-backend-cpu');

// -------------------------
// App setup
// -------------------------
const app = express();
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use('/static', express.static('static'));
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'templates'));

// Artifact paths (env overrides allowed)
const MODEL_PATH = process.env.MODEL_PATH || 'model.json';
const SCALER_PATH = process.env.SCALER_PATH || 'scaler.json';
const CALIB_PATH = process.env.CALIB_PATH || 'calib.json';

// Lazy-loaded artifacts
let _model = null;
let _scaler = null;
let _calib = null;

async function loadModel() {
    if (_model === null) {
        // Create custom IOHandler for loading
        const modelDir = MODEL_PATH.replace('.json', '');

        const loadHandler = {
            load: async () => {
                const modelJSON = JSON.parse(
                    await fs.readFile(path.join(modelDir, 'model.json'), 'utf8')
                );
                const weightData = await fs.readFile(path.join(modelDir, 'weights.bin'));

                return {
                    modelTopology: modelJSON.modelTopology,
                    weightSpecs: modelJSON.weightsManifest[0].weights,
                    weightData: weightData.buffer
                };
            }
        };

        _model = await tf.loadLayersModel(loadHandler);
    }
    return _model;
}

async function loadScaler() {
    if (_scaler === null) {
        const data = await fs.readFile(SCALER_PATH, 'utf8');
        _scaler = JSON.parse(data);
    }
    return _scaler;
}

async function loadCalibration() {
    if (_calib === null) {
        try {
            const data = await fs.readFile(CALIB_PATH, 'utf8');
            _calib = JSON.parse(data);
        } catch (err) {
            _calib = null; // File doesn't exist, that's okay
        }
    }
    return _calib;
}

function scaleFeatures(X, scaler) {
    // X is [bedrooms, sqft, coverageA, age]
    // Apply standardization: (X - mean) / std
    const scaled = X.map((val, i) => {
        return (val - scaler.mean[i]) / scaler.scale[i];
    });
    return scaled;
}

// -------------------------
// Routes
// -------------------------
app.get('/health', (req, res) => {
    res.json({ status: 'ok' });
});

app.get('/', (req, res) => {
    res.render('index');
});

app.post('/predict', async (req, res) => {
    try {
        await loadModel();
        await loadScaler();
        await loadCalibration();

        const data = req.body;

        // Prefer Coverage A; fall back to Property Value if provided
        const covA = data['Coverage A'] || data['Property Value'];

        const requiredBase = ['Bedrooms', 'Square Footage', 'Age of Home'];
        const missing = requiredBase.filter(k => !(k in data));
        if (covA === undefined) {
            missing.push('Coverage A (or Property Value)');
        }

        if (missing.length > 0) {
            return res.status(400).json({
                error: `Missing fields: ${missing.join(', ')}`
            });
        }

        // Parse numbers safely
        let bedrooms, sqft, coverageA, age;
        try {
            bedrooms = parseFloat(data['Bedrooms']);
            sqft = parseFloat(data['Square Footage']);
            coverageA = parseFloat(covA);
            age = parseFloat(data['Age of Home']);

            if (isNaN(bedrooms) || isNaN(sqft) || isNaN(coverageA) || isNaN(age)) {
                throw new Error('Invalid number format');
            }
        } catch (e) {
            return res.status(400).json({
                error: `Invalid input types: ${e.message}`
            });
        }

        // Feature order must match training
        const X = [bedrooms, sqft, coverageA, age];

        // Scale features
        const Xs = scaleFeatures(X, _scaler);

        // Predict using TensorFlow.js
        const inputTensor = tf.tensor2d([Xs]);
        const predTensor = _model.predict(inputTensor);
        const predArray = await predTensor.data();
        let pred = predArray[0];

        // Clean up tensors
        inputTensor.dispose();
        predTensor.dispose();

        // Optional calibration: y_cal = a*y + b
        let predCal = pred;
        if (_calib && 'a' in _calib && 'b' in _calib) {
            predCal = _calib.a * pred + _calib.b;
            predCal = Math.max(0.0, predCal); // keep sane
        }

        res.json({
            predicted_premium: Math.round(predCal * 100) / 100
            // Uncomment to debug:
            // base_prediction: Math.round(pred * 100) / 100,
            // calibration: _calib
        });

    } catch (error) {
        console.error('Prediction error:', error);
        res.status(500).json({
            error: 'Internal server error during prediction'
        });
    }
});

// -------------------------
// Server startup
// -------------------------
const PORT = process.env.PORT || 5000;
app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on http://0.0.0.0:${PORT}`);
});