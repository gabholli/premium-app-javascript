
# Home Insurance Premium Predictor - JavaScript Version

This is a complete JavaScript/Node.js conversion of the Python-based premium prediction app. It uses TensorFlow.js for machine learning and Express for the web server.

## Prerequisites

- Node.js 16+ 
- npm or yarn

## Installation

```bash
npm install
```

## Project Structure

```
premium-app-js/
├── app.js                          # Main Express server
├── train_and_serialize.js          # Training script
├── calibration_fit.js              # Calibration script
├── package.json                    # Dependencies
├── simulated_home_insurance_quotes.csv
├── calib_samples.csv
├── model.json/                     # Generated TensorFlow.js model
├── scaler.json                     # Generated scaler parameters
├── calib.json                      # Generated calibration parameters
├── templates/
│   └── index.ejs                   # HTML template (convert from .html)
└── static/                         # Static assets (CSS, JS)
```

## Usage

### 1. Train the Model

Train the linear regression model on your CSV data:

```bash
npm run train
# or
node train_and_serialize.js --csv simulated_home_insurance_quotes.csv
```

This creates:
- `model.json/` - TensorFlow.js model directory
- `scaler.json` - Feature scaling parameters

### 2. Calibrate (Optional)

If you have calibration samples:

```bash
npm run calibrate
# or
node calibration_fit.js
```

This creates:
- `calib.json` - Calibration parameters (a, b)

### 3. Run the Server

Start the Express server:

```bash
npm start
# or for development with auto-reload:
npm run dev
```

Server runs on `http://localhost:5000`

## API Endpoints

### GET `/health`
Health check endpoint
```json
{ "status": "ok" }
```

### GET `/`
Serves the main HTML interface

### POST `/predict`
Predict insurance premium

**Request Body (JSON or form-encoded):**
```json
{
  "Bedrooms": 3,
  "Square Footage": 2000,
  "Coverage A": 250000,
  "Age of Home": 15
}
```

**Response:**
```json
{
  "predicted_premium": 1234.56
}
```

## Key Differences from Python Version

### 1. **Model Format**
- Python: Uses scikit-learn's `LinearRegression` saved with `joblib` (.pkl files)
- JavaScript: Uses TensorFlow.js with a single dense layer (.json model)

### 2. **Template Engine**
- Python: Uses Jinja2 (Flask default)
- JavaScript: Uses EJS (Express compatible)
  - Convert `index.html` to `index.ejs` (usually no changes needed for simple templates)

### 3. **CSV Parsing**
- Python: Uses pandas
- JavaScript: Uses PapaParse library

### 4. **Standard Scaler**
- Python: scikit-learn's `StandardScaler`
- JavaScript: Custom implementation (saves mean/scale to JSON)

### 5. **Dependencies**
- Install with: `npm install`
- Main deps: express, @tensorflow/tfjs-node, ejs, papaparse

## Environment Variables

```bash
MODEL_PATH=model.json      # Path to TensorFlow.js model
SCALER_PATH=scaler.json    # Path to scaler parameters
CALIB_PATH=calib.json      # Path to calibration parameters
PORT=5000                  # Server port
```

## Converting Your Template

If your `templates/index.html` uses Jinja2 syntax, convert it to EJS:

**Jinja2 → EJS**
```html
<!-- Jinja2 -->
{{ variable }}
{% for item in items %}
{% endfor %}

<!-- EJS -->
<%= variable %>
<% for (let item of items) { %>
<% } %>
```

For a simple HTML form with no template variables, just rename it to `.ejs`.

## Performance Notes

- TensorFlow.js models are slightly larger than scikit-learn models
- Prediction speed is comparable for small models
- Consider using `@tensorflow/tfjs` (browser version) for client-side predictions

## Troubleshooting

**Error: Cannot find module '@tensorflow/tfjs-node'**
```bash
npm install @tensorflow/tfjs-node
```

**Model loading fails:**
- Ensure `model.json` exists and the corresponding weight files are in `model.json/`
- Check file paths are correct

**Different predictions than Python:**
- TensorFlow.js uses different initialization and optimization
- Results should be very close but may not be identical
- Re-train if you need exact reproducibility

## License

Same as original Python version