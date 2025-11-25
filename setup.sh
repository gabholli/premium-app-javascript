#!/bin/bash

# Quick setup script for premium-app-js
# This creates all the necessary files

echo "Creating premium-app-js directory structure..."
mkdir -p premium-app-js/{templates,static}
cd premium-app-js

# Create package.json
cat > package.json << 'EOF'
{
  "name": "premium-app-js",
  "version": "1.0.0",
  "description": "Home insurance premium prediction app - JavaScript version",
  "main": "app.js",
  "scripts": {
    "start": "node app.js",
    "train": "node train_and_serialize.js --csv simulated_home_insurance_quotes.csv",
    "calibrate": "node calibration_fit.js",
    "dev": "nodemon app.js"
  },
  "keywords": [
    "insurance",
    "machine-learning",
    "tensorflow",
    "express"
  ],
  "author": "",
  "license": "ISC",
  "dependencies": {
    "@tensorflow/tfjs-node": "^4.11.0",
    "express": "^4.18.2",
    "ejs": "^3.1.9",
    "papaparse": "^5.4.1"
  },
  "devDependencies": {
    "nodemon": "^3.0.1"
  }
}
EOF

# Create .gitignore
cat > .gitignore << 'EOF'
node_modules/
*.pkl
model.json/
scaler.json
calib.json
.env
*.log
EOF

echo "Files created! Next steps:"
echo "1. Copy app.js, train_and_serialize.js, and calibration_fit.js from the artifacts above"
echo "2. Copy your CSV files to this directory"
echo "3. Rename templates/index.html to templates/index.ejs"
echo "4. Run: npm install"
echo "5. Run: npm run train"
echo "6. Run: npm start"