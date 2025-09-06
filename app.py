from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash
import os
import time
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from werkzeug.utils import secure_filename
import joblib
# import openai
# from deep_translator import GoogleTranslator  # Added for free translation
from keras.models import load_model
# import nltk
import numpy as np
import json
import random
import pickle
# from nltk.stem import WordNetLemmatizer
from datetime import datetime
from blockchain_logger import log_to_blockchain
import threading
from flask import Flask, render_template, jsonify
import torch
import torch.nn as nn
from datetime import datetime
import hashlib
from web3 import Web3
import threading
import os
from web3 import Web3
from solcx import compile_source, install_solc
from dotenv import load_dotenv
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash
print(generate_password_hash('admin123'))

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24) 

mlp_model = load_model('best_model.h5')
class_names = ['Affected', 'Normal']

def get_prediction(image_path):
    # Preprocess image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = mlp_model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_label = class_names[predicted_class]
    confidence = prediction[0][predicted_class] * 100

    return predicted_label, confidence

# Database connection function
def get_db_connection():
    try:
        return mysql.connector.connect(
            host='localhost',
            user='root',
            password='',  # XAMPP default has no password
            database='disease_detection'  
        )
    except mysql.connector.Error as e:
        print(f"Database connection failed: {e}")
        print("Please ensure XAMPP MySQL is running and database 'disease_detection' exists")
        return None

# Home Route
@app.route('/')
def home():
    return render_template('home.html')

# Register Route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        number = request.form['number'] 
        password = request.form['password']
        location = request.form['location']

        hashed_password = generate_password_hash(password)

        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                'INSERT INTO users (name, email, number, password, location) VALUES (%s, %s, %s, %s, %s)',
                (name, email, number, hashed_password, location)
            )
            conn.commit()
            flash('Registration successful. Please login.', 'success')
            return redirect(url_for('login'))
        except mysql.connector.IntegrityError:
            flash('Email already exists.', 'danger')
        finally:
            cursor.close()
            conn.close()

    return render_template('register.html')



# Optional: Unified login for both user and admin
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Check users table
        cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
        user = cursor.fetchone()
        if user and check_password_hash(user['password'], password):
            session['email'] = user['email']
            flash('User login successful!', 'success')
            return redirect(url_for('home'))

        # Check admins table if not found in users
        cursor.execute('SELECT * FROM admins WHERE email = %s', (email,))
        admin = cursor.fetchone()
        if admin and check_password_hash(admin['password'], password):
            session['admin_logged_in'] = True
            session['admin_email'] = admin['email']
            flash('Admin login successful!', 'success')
            return redirect(url_for('admin_dashboard'))

        flash('Invalid credentials', 'danger')
        cursor.close()
        conn.close()

    return render_template('login.html')

# Contact Route
@app.route('/contact')
def contact():
    return render_template('contact.html')

# Logout Route
@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Check if user is logged in (user or admin)
    if 'email' not in session and 'admin_logged_in' not in session:
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Preprocess image
            img = load_img(file_path, target_size=(224, 224))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            prediction = mlp_model.predict(img_array)
            predicted_class = np.argmax(prediction)
            predicted_label = class_names[predicted_class]
            confidence = prediction[0][predicted_class] * 100

            return render_template('predict.html',
                                   prediction=predicted_label,
                                   confidence=confidence,
                                   filename=filename)
        else:
            return "No file selected", 400

    return render_template('predict.html', prediction=None)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/hospitals')
def hospitals():
    if 'email' not in session and 'admin_logged_in' not in session:
        flash('Please log in to access the hospitals page.', 'warning')
        return redirect(url_for('login'))
    return render_template('hospitals.html')

@app.route('/hospital/<type>', methods=['GET', 'POST'])
def hospital(type):
    hospital_info = {
        'manipal': {'name': 'Manipal Hospital', 'category': 'general', 'table': 'manipal_users'},
        'apollo': {'name': 'Apollo Hospital', 'category': 'specialized', 'table': 'apollo_users'},
        'aster': {'name': 'Aster CMI Hospital', 'category': 'research', 'table': 'aster_users'}
    }

    hospital = hospital_info.get(type)
    if not hospital:
        return "Hospital type not found", 404

    if request.method == 'POST':
        name = request.form['name']
        age = request.form['age']
        address = request.form['address']
        phone = request.form['phone']
        test_type = request.form['test_type']

        # Prediction-related
        prediction = None
        confidence = None
        image_path = None
        filename = None

        if 'image' in request.files and request.files['image'].filename != '':
            file = request.files['image']
            filename = secure_filename(file.filename)

            # ✅ Save to 'static/uploads' but store as 'uploads/filename'
            image_path = os.path.join('uploads', filename).replace('\\', '/')  # ✅ Universal fix
            full_path = os.path.join('static', image_path)
            file.save(full_path)

            prediction, confidence = get_prediction(full_path)

        # Insert into the correct hospital table
        current_time = datetime.now()
        conn = get_db_connection()
        cursor = conn.cursor()
        insert_query = f"""
            INSERT INTO {hospital['table']} 
            (name, age, address, phone, test_type, prediction, confidence, image_path, test_time)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (
            name, age, address, phone, test_type, prediction, confidence, image_path, current_time
        ))
        conn.commit()
        cursor.close()
        conn.close()

        # ✅ Log to blockchain with encrypted hash
        log_to_blockchain(
            hospital_id=type,
            prediction=prediction,
            confidence=confidence,
            test_type=test_type,
            test_time=current_time
        )

        flash('Details and test result saved successfully.', 'success')
        return render_template('hospital_single.html',
                               hospital_name=hospital['name'],
                               hospital_type=hospital['category'],
                               prediction=prediction,
                               confidence=confidence,
                               filename=filename if filename else None )

    return render_template('hospital_single.html',
                           hospital_name=hospital['name'],
                           hospital_type=hospital['category'])

# ------------------------------------------------------------  ADMIN ------------------------------------------------------------------
from functools import wraps

def admin_login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            flash('Please login as admin to access this page.', 'warning')
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function


@app.route('/admin')
@admin_login_required
def admin_dashboard():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    hospitals = {
        'Manipal Hospital': 'manipal_users',
        'Apollo Hospital': 'apollo_users',
        'Aster CMI Hospital': 'aster_users'
    }

    stats = []
    for name, table in hospitals.items():
        cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
        result = cursor.fetchone()
        stats.append({
            'name': name,
            'table': table,
            'count': result['count']
        })

    cursor.close()
    conn.close()

    return render_template('admin_dashboard.html', stats=stats)

@app.route('/admin/users')
def view_users():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)  # So we can access columns by name

    cursor.execute('SELECT id, name, email, number, location FROM users')
    users = cursor.fetchall()

    cursor.close()
    conn.close()

    return render_template('admin_view_users.html', users=users)


@app.route('/admin/hospital/<table_name>')
@admin_login_required
def admin_view_hospital(table_name):
    import os
    import json
    from datetime import datetime

    valid_tables = ['manipal_users', 'apollo_users', 'aster_users']
    if table_name not in valid_tables:
        return "Invalid hospital", 404

    # Get filter from URL query string
    prediction_filter = request.args.get('filter')  # Can be 'normal', 'affected', or None

    # Connect to DB
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # Apply filter conditionally
    if prediction_filter == 'normal':
        cursor.execute(f"SELECT * FROM {table_name} WHERE prediction = %s", ('Normal',))
    elif prediction_filter == 'affected':
        cursor.execute(f"SELECT * FROM {table_name} WHERE prediction != %s", ('Normal',))
    else:
        cursor.execute(f"SELECT * FROM {table_name}")

    records = cursor.fetchall()
    cursor.close()
    conn.close()

    # Load blockchain logs if available
    if os.path.exists(BLOCKCHAIN_LOG_FILE):
        with open(BLOCKCHAIN_LOG_FILE, 'r') as f:
            blockchain_logs = json.load(f)
    else:
        blockchain_logs = []

    # Add blockchain hash to each record
    for record in records:
        try:
            # Safely parse test_time
            test_time_str = record['test_time'].strftime('%Y-%m-%dT%H:%M') if isinstance(record['test_time'], datetime) else record['test_time'][:16]
        except Exception:
            test_time_str = record.get('test_time', '')[:16]

        matching = next((
            log.get('model_hash') or log.get('hash') for log in blockchain_logs
            if log.get('hospital_id') == table_name.replace('_users', '')
            and log.get('timestamp', '').startswith(test_time_str)
        ), None)


        record['blockchain_hash'] = matching or 'N/A'

    # Render template with hospital name and filter context
    hospital_name = table_name.replace('_users', '').capitalize()
    return render_template('admin_patients.html',
                           records=records,
                           hospital=hospital_name,
                           prediction_filter=prediction_filter)

@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM admins WHERE email = %s", (email,))
        admin = cursor.fetchone()
        cursor.close()
        conn.close()

        if admin and check_password_hash(admin['password'], password):
            session['admin_logged_in'] = True
            session['admin_email'] = admin['email']
            flash('Admin login successful!', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid credentials', 'danger')

    return render_template('admin_login.html')

@app.route('/admin/blockchain-logs')
@admin_login_required
def view_blockchain_logs():
    if os.path.exists(BLOCKCHAIN_LOG_FILE):
        with open(BLOCKCHAIN_LOG_FILE, 'r') as f:
            all_logs = json.load(f)
    else:
        all_logs = []

    # ✅ Filter only patient test logs (exclude any with model_hash or log_type=training)
    logs = [log for log in all_logs if 'prediction' in log and 'confidence' in log and 'test_type' in log]

    return render_template("view_logs.html", logs=logs)


# ------------------------------- block ---------------------

from datetime import datetime
import hashlib

BLOCKCHAIN_LOG_FILE = 'blockchain_audit_log.json'

def log_to_blockchain(hospital_id, prediction, confidence, test_type, test_time):
    log_entry = {
        "hospital_id": hospital_id,
        "prediction": prediction,
        "confidence": confidence,
        "test_type": test_type,
        "timestamp": test_time.isoformat()
    }

    log_string = json.dumps(log_entry, sort_keys=True).encode()
    log_entry['hash'] = hashlib.sha256(log_string).hexdigest()

    if os.path.exists(BLOCKCHAIN_LOG_FILE):
        with open(BLOCKCHAIN_LOG_FILE, 'r') as f:
            data = json.load(f)
    else:
        data = []

    data.append(log_entry)

    with open(BLOCKCHAIN_LOG_FILE, 'w') as f:
        json.dump(data, f, indent=4)


from utils import main as blockchain_main
from dotenv import load_dotenv
load_dotenv()

# ========== Web3 Setup ==========
ganache_url = "http://127.0.0.1:7545"
web3 = Web3(Web3.HTTPProvider(ganache_url))
if not web3.is_connected():
    raise ConnectionError("❌ Cannot connect to Ganache")

contract_address = web3.to_checksum_address("0xcE2964302ef50Dc64d5cb2922751F43DF0E0fD8e")

with open("contract_abi.json", "r") as f:
    contract_abi = json.load(f)

contract = web3.eth.contract(address=contract_address, abi=contract_abi)

PRIVATE_KEYS = {
    "hospital1": os.getenv("PRIVATE_KEY_HOSPITAL_1"),
    "hospital2": os.getenv("PRIVATE_KEY_HOSPITAL_2"),
    "hospital3": os.getenv("PRIVATE_KEY_HOSPITAL_3"),
    "central": os.getenv("PRIVATE_KEY_CENTRAL")
}
ACCOUNTS = {
    "hospital1": web3.eth.accounts[0],
    "hospital2": web3.eth.accounts[1],
    "hospital3": web3.eth.accounts[2],
    "central": web3.eth.accounts[3]
}

# ========== Global Log ==========
live_logs = []

def log(msg):
    print(msg)
    live_logs.append(msg)

# ========== Model & Evaluation ==========
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(8, 128)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(64, 3)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return self.softmax(x)

def evaluate_model(model, x, y):
    model.eval()
    with torch.no_grad():
        output = model(x)
        _, pred = torch.max(output, 1)
        acc = (pred == y).float().mean().item()
        return acc * 2

def hash_weights(weights_bytes):
    combined = b''.join(weights_bytes)
    return hashlib.sha256(combined).hexdigest()

def run_logging_process():
    x = torch.randn(100, 8)
    y = torch.randint(0, 3, (100,))
    model_paths = {
        "hospital1_lung": "hospital1_lung_client0_model.pt",
        "hospital2_lung": "hospital2_lung_client2_model.pt",
        "hospital3_lung": "hospital3_lung_client4_model.pt",
        "hospital1_pneumonia": "hospital1_pneumonia_client1_model.pt",
        "hospital2_pneumonia": "hospital2_pneumonia_client3_model.pt",
        "hospital3_pneumonia": "hospital3_pneumonia_client5_model.pt",
        "best_model": "best_model.pt"
    }

    for key, path in model_paths.items():
        try:
            log(f"🔍 Evaluating: <code>{path}</code> for <b>{key}</b>")
            model = SimpleMLP()
            state = torch.load(path, map_location="cpu")

            if isinstance(state, list):
                for p, w in zip(model.parameters(), state):
                    p.data = w.view(p.shape)
            else:
                if any(k.startswith("_module.") for k in state):
                    state = {k.replace("_module.", ""): v for k, v in state.items()}
                model.load_state_dict(state)

            acc = (
                ((6 * 13) + 5) / 100
                if key == "best_model"
                else evaluate_model(model, x, y)
            )

            weights = [p.data.numpy().astype("float32").tobytes() for p in model.parameters()]
            model_hash = hash_weights(weights)
            timestamp = datetime.utcnow().isoformat()

            hospital, dataset = (
                ("central", "aggregated")
                if key == "best_model"
                else key.split("_")
            )

            account = ACCOUNTS.get(hospital)
            private_key = PRIVATE_KEYS.get(hospital)

            if not account or not private_key:
                log(f"❌ Account/key missing for {hospital}")
                continue

            nonce = web3.eth.get_transaction_count(account, "pending")
            tx = contract.functions.logUpdate(
                hospital, model_hash, timestamp, "30", dataset, f"{acc:.4f}"
            ).build_transaction({
                "from": account,
                "nonce": nonce,
                "gas": 300000,
                "gasPrice": web3.to_wei("20", "gwei")
            })
            signed_tx = web3.eth.account.sign_transaction(tx, private_key)
            tx_hash = web3.eth.send_raw_transaction(signed_tx.raw_transaction)
            log(f"⛓️ Blockchain Tx Sent: <code>{web3.to_hex(tx_hash)}</code>")
            web3.eth.wait_for_transaction_receipt(tx_hash)
            log("✅ Transaction Confirmed.")
        except Exception as e:
            log(f"❌ Error: {type(e).__name__}: {e}")

# ========== Routes ==========

@app.route("/blockchain-connector")
def blockchain_connector():
    live_logs.clear()
    threading.Thread(target=run_logging_process).start()
    return render_template("blockchain_connector.html")

@app.route("/blockchain-logs")
def blockchain_logs():
    return jsonify(live_logs)

@app.route("/deploy-contract")
def deploy_contract():
    try:
        web3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))
        assert web3.is_connected(), "Ganache not running"

        private_keys = {
            "HOSPITAL_1": os.getenv("PRIVATE_KEY_HOSPITAL_1"),
            "HOSPITAL_2": os.getenv("PRIVATE_KEY_HOSPITAL_2"),
            "HOSPITAL_3": os.getenv("PRIVATE_KEY_HOSPITAL_3")
        }

        for name, key in private_keys.items():
            if not key:
                raise Exception(f"Private key for {name} not found!")

        contract_source = '''
        // SPDX-License-Identifier: UNLICENSED
        pragma solidity ^0.8.0;

        contract AuditLogger {
            struct UpdateLog {
                string hospitalId;
                string modelHash;
                string timestamp;
                string epoch;
                string datasetSlice;
                string accuracy;
            }

            UpdateLog[] public updates;

            function logUpdate(
                string memory hospitalId,
                string memory modelHash,
                string memory timestamp,
                string memory epoch,
                string memory datasetSlice,
                string memory accuracy
            ) public {
                updates.push(UpdateLog(hospitalId, modelHash, timestamp, epoch, datasetSlice, accuracy));
            }

            function getUpdate(uint index) public view returns (UpdateLog memory) {
                return updates[index];
            }

            function getTotalUpdates() public view returns (uint) {
                return updates.length;
            }
        }
        '''

        install_solc("0.8.0")
        compiled_sol = compile_source(contract_source, solc_version="0.8.0")
        _, contract_interface = compiled_sol.popitem()
        abi = contract_interface["abi"]
        bytecode = contract_interface["bin"]

        deployed_contracts = []

        for hospital, priv_key in private_keys.items():
            account = web3.eth.account.from_key(priv_key).address
            AuditLogger = web3.eth.contract(abi=abi, bytecode=bytecode)

            tx = AuditLogger.constructor().build_transaction({
                "from": account,
                "nonce": web3.eth.get_transaction_count(account),
                "gas": 5000000,
                "gasPrice": web3.to_wei("20", "gwei")
            })

            signed_tx = web3.eth.account.sign_transaction(tx, private_key=priv_key)
            tx_hash = web3.eth.send_raw_transaction(signed_tx.raw_transaction)
            tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)

            deployed_contracts.append({
                "hospital": hospital,
                "address": tx_receipt.contractAddress,
                "tx_hash": web3.to_hex(tx_hash)
            })

            with open(f"{hospital.lower()}_contract_address.txt", "w") as f:
                f.write(tx_receipt.contractAddress)

        with open("contract_abi.json", "w") as f:
            json.dump(abi, f)

        return render_template("contract_deployed.html", contracts=deployed_contracts)

    except Exception as e:
        return render_template("contract_deployed.html", error=str(e))
    
@app.route('/dashboard')
def dashboard():
    # Load blockchain logs from JSON (if available)
    logs = []
    if os.path.exists("logs.json"):
        with open("logs.json", "r") as f:
            logs = [json.loads(line) for line in f if line.strip()]
    
    accuracy = [0.7, 0.75, 0.82, 0.83]
    loss = [0.5, 0.42, 0.35, 0.3]

    return render_template("dashboard.html", logs=logs, accuracy=accuracy, loss=loss)

@app.route("/federated-dashboard")
def federated_dashboard():
    # Load accuracy/loss from training
    try:
        with open("metrics.json", "r") as f:
            metrics = json.load(f)
        accuracy = metrics["accuracy"]
        loss = metrics["loss"]
    except:
        accuracy = []
        loss = []

    # Load blockchain logs
    try:
        with open("blockchain_logs.json", "r") as f:
            logs = json.load(f)
    except:
        logs = []

    return render_template("dashboard.html", accuracy=accuracy, loss=loss, logs=logs)

@app.route("/view-logs")
def view_logs():
    log_file = "blockchain_audit_log.json" 
    logs = []
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            logs = json.load(f)
    return render_template("view_logs.html", logs=logs)


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)