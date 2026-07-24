# =============================================================
# app.py — AuditSmart v1.0
# Multi-Class Document Classification & Contract Audit System
# Author      : Ruhorimbere Fred (2305001581)
# Supervisor  : Dr. Eustach Uwimana
# University  : University of Kigali — BBIT 2026
# =============================================================

import os
import re
import jwt
import json
import hashlib
import datetime
import numpy as np
import pandas as pd

from flask import (
    Flask, request, jsonify,
    render_template, redirect, url_for, session, send_file
)
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Document reading
import PyPDF2
from docx import Document as DocxDocument

# ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV

# Database
import mysql.connector

# =============================================================
# APP SETUP
# =============================================================
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'auditsmart_secret_2026')
CORS(app)

UPLOAD_FOLDER = '/tmp/uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create uploads folder on startup
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =============================================================
# DATABASE
# =============================================================
from urllib.parse import urlparse

_mysql_url = os.environ.get('MYSQL_URL', '')
if _mysql_url:
    _parsed = urlparse(_mysql_url)
    DB_CONFIG = dict(
        host=_parsed.hostname,
        port=_parsed.port or 3306,
        user=_parsed.username,
        password=_parsed.password,
        database=_parsed.path.lstrip('/')
    )
else:
    DB_CONFIG = dict(
        host='localhost',
        user='root',
        password='',
        database='auditsmart_db'
    )

def get_db():
    return mysql.connector.connect(**DB_CONFIG)

# =============================================================
# DOCUMENT TYPES
# =============================================================
DOCUMENT_TYPES = [
    'NDA',
    'Employment Contract',
    'Invoice',
    'Service Agreement',
    'Purchase Order',
    'Lease Agreement',
    'Partnership Agreement',
]

# =============================================================
# COMPLIANCE RULES
# =============================================================
COMPLIANCE_RULES = {
    'NDA': {
        'Parties Named':        ['between', 'party', 'undersigned'],
        'Confidentiality':      ['confidential', 'secret', 'disclose'],
        'Duration':             ['year', 'month', 'period', 'term', 'duration'],
        'Governing Law':        ['governed', 'jurisdiction', 'law of'],
        'Signature Required':   ['signed', 'signature', 'executed'],
    },
    'Employment Contract': {
        'Parties Named':        ['employer', 'employee', 'between'],
        'Job Title':            ['position', 'title', 'role', 'job'],
        'Salary Mentioned':     ['salary', 'wage', 'compensation', 'pay'],
        'Start Date':           ['start date', 'commencement', 'effective date'],
        'Termination Clause':   ['terminate', 'termination', 'notice period'],
        'Governing Law':        ['governed', 'jurisdiction', 'law'],
    },
    'Invoice': {
        'Invoice Number':       ['invoice no', 'invoice number', 'inv#'],
        'Date':                 ['date', 'dated'],
        'Amount':               ['total', 'amount', 'rwf', 'usd', 'price'],
        'Seller Details':       ['from', 'seller', 'vendor', 'company'],
        'Buyer Details':        ['to', 'buyer', 'client', 'customer'],
        'Payment Terms':        ['payment', 'due', 'days', 'bank'],
    },
    'Service Agreement': {
        'Parties Named':        ['client', 'service provider', 'between'],
        'Scope of Work':        ['scope', 'services', 'deliverable'],
        'Payment Terms':        ['payment', 'fee', 'amount', 'rwf'],
        'Duration':             ['term', 'period', 'duration', 'month'],
        'Termination Clause':   ['terminate', 'termination', 'cancel'],
        'Governing Law':        ['governed', 'jurisdiction', 'law'],
    },
    'Purchase Order': {
        'PO Number':            ['po number', 'purchase order no', 'order no'],
        'Vendor Details':       ['vendor', 'supplier', 'from'],
        'Item Description':     ['item', 'description', 'product', 'goods'],
        'Quantity':             ['quantity', 'qty', 'units'],
        'Price':                ['price', 'amount', 'total', 'rwf'],
        'Delivery Date':        ['delivery', 'deliver by', 'ship date'],
    },
    'Lease Agreement': {
        'Parties Named':        ['landlord', 'tenant', 'lessor', 'lessee'],
        'Property Address':     ['property', 'premises', 'address', 'located'],
        'Rent Amount':          ['rent', 'monthly', 'amount', 'rwf'],
        'Lease Duration':       ['term', 'period', 'month', 'year'],
        'Deposit':              ['deposit', 'security', 'advance'],
        'Governing Law':        ['governed', 'jurisdiction', 'law'],
    },
    'Partnership Agreement': {
        'Partners Named':       ['partner', 'parties', 'between'],
        'Business Purpose':     ['purpose', 'business', 'objective'],
        'Profit Sharing':       ['profit', 'share', 'distribution', 'percent'],
        'Duration':             ['term', 'period', 'duration'],
        'Termination Clause':   ['terminate', 'dissolution', 'dissolved', 'wind up'],
        'Governing Law':        ['governed', 'jurisdiction', 'law'],
    },
}

# =============================================================
# TRAINING DATA
# =============================================================
TRAINING_DATA = {
    'NDA': [
        "This non-disclosure agreement is entered into between the parties. All confidential information shall remain secret and shall not be disclosed to any third party without prior written consent.",
        "The undersigned parties agree to keep all disclosed information confidential. Duration of this NDA is two years. Signed by both parties.",
        "This NDA governs the confidential relationship between the company and the recipient. Confidential information must not be disclosed to third parties.",
        "Both parties agree not to disclose any secret information shared during the term of this agreement. Jurisdiction is Kigali Rwanda.",
        "Non-disclosure agreement between TechCorp and the employee. All proprietary information is confidential. This agreement is valid for three years.",
        "This confidentiality agreement is signed between two undersigned parties. All secret business information exchanged shall not be disclosed. Governed by laws of Rwanda.",
        "The parties hereby agree that all confidential data trade secrets and proprietary information shall remain protected. This NDA is valid for five years and signed by both parties.",
        "Non disclosure agreement entered between ABC Ltd and XYZ consultant. Confidential information includes financial data technical plans and business strategies. Duration three years. Jurisdiction Rwanda.",
        "This agreement protects confidential information shared between the disclosing party and the receiving party. The receiving party shall not disclose any secret information. Period of confidentiality is two years.",
        "Both undersigned parties agree to maintain strict confidentiality of all proprietary information. Breach of this NDA shall result in legal action. Governed by Rwandan law. Signed and executed.",
        "Confidentiality and non disclosure agreement between employer and contractor. All technical information financial data and business plans are confidential and secret. Term of agreement is three years.",
        "This non disclosure agreement protects all confidential information exchanged between parties. Neither party shall disclose secret information to third parties. Signed by authorized representatives. Governed by law.",
        "The parties agree that confidential information including trade secrets business plans and financial data shall not be disclosed. This agreement is governed by the laws of Rwanda. Duration is four years.",
        "Non disclosure and confidentiality agreement. The undersigned agree to keep all information secret and confidential. Violation of this agreement shall be subject to legal penalties. Signed by both parties.",
        "This NDA is between the company and the recipient. All confidential information shared during meetings discussions and correspondence shall remain secret. Term is two years. Jurisdiction Kigali Rwanda.",
    ],
    'Employment Contract': [
        "This employment contract is between the employer and the employee. The position is Software Engineer. Salary is 500000 RWF per month. Start date is January 2026.",
        "The employee agrees to work for the employer in the role of accountant. Monthly compensation is 400000 RWF. Notice period for termination is one month.",
        "Employment agreement between University of Kigali and the staff member. Job title is Lecturer. Salary and benefits are outlined herein. Governed by Rwandan labor law.",
        "This contract of employment sets out the terms for the position of manager. The commencement date is March 1 2026. Termination requires 30 days notice.",
        "The employer hereby employs the employee as a data analyst. Wages shall be paid monthly. This contract is governed by the laws of Rwanda.",
        "Employment contract between Kigali Hospital and nurse. Position is senior nurse. Monthly salary is 600000 RWF. Start date 1 July 2026. Termination notice 30 days. Governed by Rwandan labor law.",
        "This contract of service is between the employer company and the employee. The role is project manager. Compensation is 900000 RWF monthly. Commencement date August 2026. Either party may terminate with notice.",
        "The employer engages the employee as marketing director. Monthly wage is 1200000 RWF. Job title is Director of Marketing. Start date September 2026. Termination clause requires 60 days written notice.",
        "Employment agreement for the position of teacher at Green Hills Academy. The employee salary is 450000 RWF per month. Commencement is January 2026. Termination by either party requires one month notice. Governed by Rwanda law.",
        "This employment contract hires the employee as finance officer. The monthly salary and benefits package is 750000 RWF. The position commences on 1 August 2026. Termination notice period is 30 days. Jurisdiction Rwanda.",
        "Contract of employment between Rwanda Revenue Authority and tax officer. Position title is Senior Tax Analyst. Monthly compensation 850000 RWF. Start date 1 March 2026. Notice for termination 60 days. Governed by Rwandan law.",
        "The employer offers the employee the position of IT administrator. Monthly salary is 700000 RWF. Employment commences on 15 July 2026. This contract may be terminated with 30 days written notice. Governed by Rwanda labor laws.",
        "Employment contract for position of human resources manager. Employer is Kigali Business Center. Employee monthly wage is 950000 RWF. Commencement date is 1 September 2026. Termination requires 45 days notice.",
        "This agreement employs the candidate as operations supervisor. Job title Operations Supervisor. Monthly salary 650000 RWF. Start date October 2026. Contract governed by Rwandan employment law. Termination notice 30 days.",
        "Employment offer letter and contract between the company and new employee. Position is business development officer. Salary 550000 RWF monthly. Commencement January 2026. Termination clause with 30 days notice. Rwanda jurisdiction.",
    ],
    'Invoice': [
        "Invoice No 001. Date June 2026. From ABC Company. To XYZ Client. Total amount 1500000 RWF. Payment due within 30 days.",
        "Invoice number INV-2026-045. Seller Tech Solutions Ltd. Buyer Government of Rwanda. Amount 2000000 RWF. Bank transfer required.",
        "Tax invoice dated 10 July 2026. Vendor Office Supplies Co. Customer Kigali Hospital. Total 350000 RWF. Due date 30 July 2026.",
        "Invoice from supplier to client. Description of goods delivered. Total price 800000 RWF. Payment terms 15 days from invoice date.",
        "INV 2026-101. From Printing Services. To University of Kigali. Items 500 books. Amount due 600000 RWF.",
        "Tax invoice number INV-001-2026. Date 1 July 2026. From Rwanda Supplies Ltd to Kigali City Council. Description office furniture. Total amount 4500000 RWF. Payment due 30 days. Bank of Kigali account.",
        "Invoice INV-2026-089. Seller name ABC Electronics. Buyer name Ministry of Education Rwanda. Item description 50 desktop computers. Total price 15000000 RWF. Payment terms net 30 days from invoice date.",
        "Commercial invoice number 2026-567. From vendor Tech Hardware Ltd to buyer Kigali University. Date 5 July 2026. Items laptops and accessories. Amount 8000000 RWF. Payment due within 14 days.",
        "Invoice from cleaning services company to client hospital. Invoice number CLN-2026-023. Date July 2026. Amount 500000 RWF. Payment terms within 7 days. Bank details provided.",
        "Tax invoice INV-2026-112. Seller Rwanda Printing Press. Buyer Government Procurement Agency. Description 10000 booklets. Total amount 2000000 RWF. Payment due 45 days from invoice date.",
        "Invoice number 2026-034 from supplier to buyer. Date 10 June 2026. Description of services rendered. Total amount due 1800000 RWF. Payment terms 30 days. Bank transfer to account number provided.",
        "Commercial invoice from seller ABC Trading to buyer XYZ Corporation. Invoice no 2026-078. Date July 2026. Goods description agricultural equipment. Total price 6000000 RWF. Payment within 30 days.",
        "Invoice INV-RW-2026-099. From IT services provider to client company. Date 1 July 2026. Services rendered network installation and support. Amount 3500000 RWF. Payment terms net 15 days.",
        "Tax invoice number 2026-145. Vendor name medical supplies company. Customer Kigali Central Hospital. Items surgical equipment and medicines. Total 7500000 RWF. Due date 30 days from invoice date.",
        "Invoice from construction company to real estate developer. Invoice number CON-2026-056. Date July 2026. Description construction materials delivered. Amount 12000000 RWF. Payment due within 30 days of receipt.",
    ],
    'Service Agreement': [
        "This service agreement is between the client and the service provider. Scope of services includes software development. Fee is 3000000 RWF. Term is 12 months.",
        "Service contract between ABC Ltd and XYZ Consultants. Deliverables include monthly reports. Payment of 500000 RWF per month. Termination requires 30 days notice.",
        "Agreement for provision of cleaning services. Service provider shall deliver weekly cleaning. Total fee is 200000 RWF per month. Governed by laws of Rwanda.",
        "This agreement covers IT support services. The client shall pay 1000000 RWF quarterly. Duration is one year. Either party may terminate with notice.",
        "Consulting service agreement. Scope includes strategic planning and implementation. Compensation is 2500000 RWF. Jurisdiction is Rwanda.",
        "Service agreement between client Rwanda Telecom and provider Digital Solutions Ltd. Scope of work includes network maintenance and technical support. Monthly fee 1500000 RWF. Term 24 months. Termination 30 days notice. Governed by Rwanda law.",
        "This agreement is for provision of security services between the client and service provider. Scope of services is 24 hour security guard services. Monthly payment 800000 RWF. Duration 12 months. Either party may terminate with 30 days notice.",
        "Professional services agreement between consulting firm and client company. Scope of work includes financial audit and advisory services. Fee 4000000 RWF per quarter. Term 12 months. Termination clause 60 days. Jurisdiction Rwanda.",
        "Service contract for provision of catering services. Client is University of Kigali. Service provider is Fresh Foods Ltd. Scope includes daily meal provision. Monthly fee 3000000 RWF. Duration one academic year. Termination 30 days notice.",
        "Agreement for software development services between client and developer. Scope of work includes design development testing and deployment. Total fee 10000000 RWF. Project duration 6 months. Termination with 14 days notice. Governed by Rwanda law.",
        "This service agreement covers maintenance and repair services for medical equipment. Client is Kigali Hospital. Provider is MedTech Services. Scope of services quarterly maintenance. Annual fee 2400000 RWF. Term 2 years. Governed by Rwandan law.",
        "Marketing services agreement between client and agency. Scope of work includes digital marketing social media and advertising. Monthly retainer fee 1200000 RWF. Term 12 months. Either party may terminate with 30 days written notice.",
        "Service agreement for legal advisory services. Client is Rwanda Development Board. Provider is Legal Associates Ltd. Scope includes contract review and legal advice. Monthly fee 2000000 RWF. Duration 12 months. Termination 60 days notice.",
        "Agreement for transportation and logistics services. Scope of work includes daily pickup and delivery. Client pays 600000 RWF monthly. Service duration 12 months. Either party may terminate with 30 days notice. Governed by Rwanda law.",
        "This agreement covers training and capacity building services. Client organization and training provider agree on scope of work including workshops and seminars. Fee 5000000 RWF. Term 6 months. Termination 30 days. Rwanda jurisdiction.",
    ],
    'Purchase Order': [
        "Purchase Order No PO-2026-001. Vendor Office Depot. Item Office chairs. Quantity 50 units. Price 75000 RWF each. Deliver by August 2026.",
        "PO Number 2026-045. Supplier Tech Hardware Ltd. Product Laptops. Qty 20. Total amount 4000000 RWF. Ship date July 2026.",
        "Order No ORD-789. From Kigali School. To Book Supplier. Description Textbooks for 2026. Quantity 1000. Price 5000 RWF each.",
        "Purchase order for medical supplies. Vendor Pharma Ltd. Items surgical gloves and masks. Quantity as listed. Delivery within 14 days.",
        "PO-2026-112. Company purchasing 10 servers from vendor. Unit price 800000 RWF. Total 8000000 RWF. Expected delivery date September 2026.",
        "Purchase order number PO-RW-2026-034. Issued by University of Kigali to vendor office supplies company. Item description desks and chairs. Quantity 200 units. Price per unit 45000 RWF. Total 9000000 RWF. Deliver by 31 July 2026.",
        "PO number 2026-078. Buyer Rwanda Ministry of Health. Supplier pharmaceutical company. Product description medicines and medical supplies. Quantity as per attached list. Total amount 25000000 RWF. Delivery date August 2026.",
        "Purchase order PO-2026-056. From Kigali City Council to construction materials supplier. Items cement sand and steel. Quantity 500 tons. Unit price 150000 RWF. Total 75000000 RWF. Ship date September 2026.",
        "Order number ORD-2026-089. Purchasing organization Rwanda Revenue Authority. Vendor IT equipment supplier. Description computers printers and accessories. Quantity 100 units. Price 400000 RWF each. Deliver by October 2026.",
        "Purchase order PO-2026-023. Issued by hospital to medical equipment vendor. Items surgical instruments and hospital beds. Quantity 50 units. Total amount 15000000 RWF. Expected delivery within 30 days.",
        "PO number KGL-2026-067. Buyer Kigali International Airport. Seller cleaning equipment supplier. Product description industrial cleaning machines. Quantity 10 units. Unit price 2000000 RWF. Total 20000000 RWF. Ship date August 2026.",
        "Purchase order 2026-145 from Rwanda Broadcasting Corporation to electronics vendor. Items cameras microphones and broadcasting equipment. Quantity as specified. Total price 35000000 RWF. Delivery date November 2026.",
        "PO-2026-034. Procurement from stationery supplier. Buyer Rwanda National Police. Items pens paper folders and office supplies. Quantity 1000 units each. Total amount 5000000 RWF. Deliver by 15 August 2026.",
        "Purchase order number 2026-112. Issued by Kigali Convention Center to furniture supplier. Description conference tables and chairs. Quantity 300 units. Price 80000 RWF per unit. Total 24000000 RWF. Ship date September 2026.",
        "PO-RW-2026-099. Buyer Rwanda Agriculture Board. Vendor agricultural equipment company. Items tractors irrigation equipment and seeds. Quantity as per specification. Total 50000000 RWF. Delivery October 2026.",
    ],
    'Lease Agreement': [
        "Lease agreement between landlord and tenant. Property located at KG 5 Ave Kigali. Monthly rent is 300000 RWF. Lease term is 12 months. Security deposit required.",
        "This rental agreement is between the lessor and lessee. Premises at Muhoza Musanze. Rent of 150000 RWF per month. Duration two years. Deposit of two months rent.",
        "Commercial lease between property owner and business tenant. Address KN 3 Road Kigali. Monthly amount 800000 RWF. Term 3 years. Governed by Rwandan law.",
        "Residential tenancy agreement. Landlord rents property to tenant. Location Nyamirambo Kigali. Rent 200000 RWF monthly. Deposit 400000 RWF. Notice period 30 days.",
        "Office space lease. Lessor grants lessee rights to occupy premises at Kimironko. Monthly rent 500000 RWF. Lease period 24 months. Security deposit paid.",
        "Lease agreement between property owner as landlord and business company as tenant. Premises located at Kacyiru Kigali. Monthly rent 1200000 RWF. Lease term 3 years. Security deposit 2400000 RWF. Governed by Rwandan property law.",
        "Residential lease between lessor and lessee. Property address Remera Kigali Rwanda. Monthly rental amount 250000 RWF. Lease duration 12 months. Security deposit one month rent. Either party may terminate with 30 days notice.",
        "Commercial property lease agreement. Landlord rents shop premises to tenant at Nyabugogo market Kigali. Monthly rent 400000 RWF. Lease period 2 years. Deposit 800000 RWF. Governed by laws of Rwanda.",
        "Tenancy agreement for office space between lessor company and lessee business. Premises at KG 9 Avenue Kigali. Monthly rent 900000 RWF. Lease term 24 months. Security deposit 1800000 RWF. Termination 60 days notice.",
        "Lease of agricultural land between landowner and farmer. Property located in Musanze Northern Province Rwanda. Annual rent 500000 RWF. Lease duration 5 years. Deposit required. Governed by Rwandan land law.",
        "Warehouse lease agreement between landlord and logistics company. Property address Masaka Kigali. Monthly rent 1500000 RWF. Lease term 3 years. Security deposit 3000000 RWF. Governed by laws of Rwanda.",
        "Residential apartment lease between landlord and tenant. Premises located at Gisozi Kigali. Monthly rental 350000 RWF. Duration 12 months. Security deposit 700000 RWF. Notice for termination 30 days.",
        "Commercial lease for restaurant premises. Lessor rents property to lessee at Kimihurura Kigali. Monthly rent 2000000 RWF. Lease period 5 years. Deposit 4000000 RWF. Governed by Rwandan commercial law.",
        "Office lease agreement between property developer and tech company. Premises KN 5 Road Kigali. Monthly rent 3000000 RWF. Lease term 36 months. Security deposit 6000000 RWF. Either party terminates with 60 days notice.",
        "Lease contract between landlord and tenant for residential property. Address Kabeza Kigali Rwanda. Monthly rent 180000 RWF. Lease duration 24 months. Deposit one month. Governed by Rwanda housing regulations.",
    ],
    'Partnership Agreement': [
        "Partnership agreement between two parties to carry on business together. Purpose is to operate a restaurant. Profits shall be shared equally. Term is five years.",
        "This agreement establishes a business partnership. Partners agree to share profits 60 percent and 40 percent. Business objective is real estate development. Dissolution requires mutual consent.",
        "Partnership contract between three partners. Business purpose is agricultural trading. Profit distribution based on capital contribution. Governed by Rwanda company law.",
        "Joint venture agreement between companies. Partners contribute equally to capital. Business goal is technology services. Termination by written notice of 90 days.",
        "Partnership deed between individuals. Parties agree to conduct business together. Profit and loss sharing ratio defined herein. Duration of partnership is three years.",
        "Business partnership agreement between two entrepreneurs. Partners agree on business purpose of operating a retail shop. Profit sharing ratio 50 50. Capital contribution equal. Duration 5 years. Dissolution by mutual agreement. Governed by Rwanda law.",
        "Partnership agreement between three companies for joint venture in construction business. Business objective building residential houses. Profit distribution 40 35 25 percent. Term 10 years. Termination requires written notice 90 days.",
        "This deed of partnership is entered between partners for the purpose of operating an agricultural cooperative. Partners share profits and losses equally. Business duration 5 years. Termination by unanimous consent. Governed by Rwandan cooperative law.",
        "Joint venture partnership agreement between foreign investor and local company. Business purpose technology park development. Profit sharing 60 percent foreign partner 40 percent local partner. Term 15 years. Governed by Rwanda investment law.",
        "Partnership agreement for medical clinic business. Partners are two doctors. Business objective provide healthcare services. Profit sharing equally between partners. Duration 10 years. Dissolution requires 6 months notice. Rwanda jurisdiction.",
        "Business partnership deed between four entrepreneurs. Purpose operating transport and logistics company. Profit distribution based on capital contribution percentage. Term 5 years. Termination with 90 days written notice. Governed by Rwandan company law.",
        "Partnership agreement between NGO and local community organization. Business purpose implement development projects. Resources and profits shared equally. Duration 3 years renewable. Termination 60 days notice. Governed by Rwanda NGO law.",
        "Joint venture agreement between two banks for financial services business. Partners share profits 50 50. Business objective provide microfinance services. Duration 5 years. Dissolution by board resolution. Governed by Rwanda banking law.",
        "Partnership contract between hotel operator and property owner. Business purpose operate boutique hotel. Profit sharing 70 operator 30 property owner. Term 10 years. Termination 6 months notice. Governed by Rwandan hospitality law.",
        "This partnership agreement is between two software companies. Business objective develop and market software products. Profit and loss shared equally between partners. Duration 5 years. Termination 90 days written notice. Rwanda jurisdiction.",
    ],
}

# =============================================================
# ML MODEL
# =============================================================
_classifier = None
_vectorizer = None

def train_model():
    global _classifier, _vectorizer
    texts, labels = [], []
    for label, samples in TRAINING_DATA.items():
        for text in samples:
            texts.append(text)
            labels.append(label)
            texts.append(text.lower())
            labels.append(label)
            texts.append(text.upper())
            labels.append(label)

    _vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 3),
        stop_words='english',
        sublinear_tf=True,
        min_df=1,
        analyzer='word',
    )
    X = _vectorizer.fit_transform(texts)
    svc = LinearSVC(C=1.0, max_iter=5000)
    _classifier = CalibratedClassifierCV(svc, cv=3)
    _classifier.fit(X, labels)
    print('✅ AuditSmart ML model trained successfully!')

# =============================================================
# HELPERS
# =============================================================
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text(filepath):
    ext = filepath.rsplit('.', 1)[1].lower()
    text = ''
    try:
        if ext == 'pdf':
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ''
        elif ext == 'docx':
            doc = DocxDocument(filepath)
            for para in doc.paragraphs:
                text += para.text + '\n'
        elif ext == 'txt':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
    except Exception as e:
        print(f'Text extraction error: {e}')
    return text.strip()

def classify_document(text):
    if _classifier is None or _vectorizer is None:
        train_model()
    X = _vectorizer.transform([text])
    predicted  = _classifier.predict(X)[0]
    proba      = _classifier.predict_proba(X)[0]
    classes    = _classifier.classes_
    confidence = round(float(max(proba)) * 100, 1)
    top3 = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)[:3]
    return predicted, confidence, top3

def audit_compliance(text, doc_type):
    rules      = COMPLIANCE_RULES.get(doc_type, {})
    results    = {}
    text_lower = text.lower()
    passed     = 0
    for clause, keywords in rules.items():
        found = any(kw in text_lower for kw in keywords)
        results[clause] = found
        if found:
            passed += 1
    total = len(rules)
    score = round((passed / total) * 100) if total > 0 else 0
    return results, score, passed, total

def hash_password(pwd):
    return hashlib.sha256(pwd.encode()).hexdigest()

def generate_token(user_id, email):
    return jwt.encode({
        'user_id': user_id,
        'email':   email,
        'exp':     datetime.datetime.utcnow() + datetime.timedelta(days=30)
    }, app.secret_key, algorithm='HS256')

# =============================================================
# DATABASE INIT
# =============================================================
@app.route('/admin/init_db', methods=['POST'])
def init_db():
    try:
        db  = get_db()
        cur = db.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS users (
            user_id    INT AUTO_INCREMENT PRIMARY KEY,
            full_name  VARCHAR(100) NOT NULL,
            email      VARCHAR(100) UNIQUE NOT NULL,
            password   VARCHAR(255) NOT NULL,
            company    VARCHAR(100),
            role       VARCHAR(20) DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        cur.execute('''CREATE TABLE IF NOT EXISTS documents (
            doc_id           INT AUTO_INCREMENT PRIMARY KEY,
            user_id          INT NOT NULL,
            filename         VARCHAR(255),
            original_name    VARCHAR(255),
            doc_type         VARCHAR(50),
            confidence       FLOAT,
            compliance_score INT,
            clauses_passed   INT,
            clauses_total    INT,
            text_preview     TEXT,
            uploaded_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )''')
        cur.execute('''CREATE TABLE IF NOT EXISTS audit_results (
            result_id INT AUTO_INCREMENT PRIMARY KEY,
            doc_id    INT NOT NULL,
            clause    VARCHAR(100),
            status    BOOLEAN,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        )''')
        cur.execute('''CREATE TABLE IF NOT EXISTS feedback (
            feedback_id INT AUTO_INCREMENT PRIMARY KEY,
            user_id     INT NOT NULL,
            rating      INT NOT NULL,
            category    VARCHAR(50),
            message     TEXT,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        db.commit()
        cur.close(); db.close()
        return jsonify({'status': 'success', 'message': 'Database initialised!'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# =============================================================
# AUTH ROUTES
# =============================================================
@app.route('/api/register', methods=['POST'])
def register():
    try:
        d       = request.get_json() or {}
        name    = d.get('full_name', '').strip()
        email   = d.get('email', '').strip().lower()
        pwd     = d.get('password', '')
        company = d.get('company', '').strip()
        if not name or not email or not pwd:
            return jsonify({'status': 'error', 'message': 'All fields required'}), 400
        db  = get_db()
        cur = db.cursor()
        cur.execute(
            'INSERT INTO users (full_name, email, password, company) VALUES (%s,%s,%s,%s)',
            (name, email, hash_password(pwd), company)
        )
        db.commit()
        user_id = cur.lastrowid
        cur.close(); db.close()
        token = generate_token(user_id, email)
        return jsonify({
            'status':  'success',
            'message': 'Account created!',
            'token':   token,
            'user':    {'user_id': user_id, 'full_name': name,
                        'email': email, 'company': company}
        })
    except mysql.connector.IntegrityError:
        return jsonify({'status': 'error', 'message': 'Email already registered'}), 409
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        d     = request.get_json() or {}
        email = d.get('email', '').strip().lower()
        pwd   = d.get('password', '')
        if not email or not pwd:
            return jsonify({'status': 'error',
                'message': 'Email and password required'}), 400
        db  = get_db()
        cur = db.cursor(dictionary=True)
        cur.execute(
            'SELECT * FROM users WHERE email=%s AND password=%s',
            (email, hash_password(pwd))
        )
        user = cur.fetchone()
        cur.close(); db.close()
        if user:
            token = generate_token(user['user_id'], user['email'])
            return jsonify({
                'status': 'success',
                'token':  token,
                'user': {
                    'user_id':   user['user_id'],
                    'full_name': user['full_name'],
                    'email':     user['email'],
                    'company':   user['company'],
                    'role':      user['role'],
                }
            })
        return jsonify({'status': 'error',
            'message': 'Invalid email or password'}), 401
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# =============================================================
# DOCUMENT UPLOAD & CLASSIFY
# =============================================================
@app.route('/api/upload', methods=['POST'])
def upload_document():
    try:
        auth     = request.headers.get('Authorization', '')
        token    = auth.replace('Bearer ', '')
        payload  = jwt.decode(token, app.secret_key, algorithms=['HS256'])
        user_id  = payload['user_id']

        if 'file' not in request.files:
            return jsonify({'status': 'error',
                'message': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'status': 'error',
                'message': 'Invalid file type. Use PDF, DOCX or TXT'}), 400

        filename   = secure_filename(file.filename)
        saved_name = f"{user_id}_{int(datetime.datetime.now().timestamp())}_{filename}"
        filepath   = os.path.join(app.config['UPLOAD_FOLDER'], saved_name)
        
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)

        text = extract_text(filepath)
        if not text or len(text) < 20:
            return jsonify({'status': 'error',
                'message': 'Could not read document text'}), 400

        doc_type, confidence, top3 = classify_document(text)
        clauses, score, passed, total = audit_compliance(text, doc_type)

        db  = get_db()
        cur = db.cursor()
        cur.execute('''INSERT INTO documents
            (user_id, filename, original_name, doc_type, confidence,
             compliance_score, clauses_passed, clauses_total, text_preview)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)''',
            (user_id, saved_name, filename, doc_type, confidence,
             score, passed, total, text[:500])
        )
        doc_id = cur.lastrowid
        for clause, status in clauses.items():
            cur.execute(
                'INSERT INTO audit_results (doc_id, clause, status) VALUES (%s,%s,%s)',
                (doc_id, clause, status)
            )
        db.commit()
        cur.close(); db.close()

        return jsonify({
            'status':           'success',
            'doc_id':           doc_id,
            'filename':         filename,
            'doc_type':         doc_type,
            'confidence':       confidence,
            'compliance_score': score,
            'clauses_passed':   passed,
            'clauses_total':    total,
            'clauses':          clauses,
            'top3':             [(t, round(float(p)*100,1)) for t,p in top3],
            'text_preview':     text[:300],
        })
    except jwt.ExpiredSignatureError:
        return jsonify({'status': 'error', 'message': 'Session expired'}), 401
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# =============================================================
# DOCUMENT HISTORY
# =============================================================
@app.route('/api/documents', methods=['GET'])
def get_documents():
    try:
        auth    = request.headers.get('Authorization', '')
        token   = auth.replace('Bearer ', '')
        payload = jwt.decode(token, app.secret_key, algorithms=['HS256'])
        user_id = payload['user_id']

        db  = get_db()
        cur = db.cursor(dictionary=True)
        cur.execute('''SELECT doc_id, original_name, doc_type, confidence,
                       compliance_score, clauses_passed, clauses_total, uploaded_at
                       FROM documents WHERE user_id=%s
                       ORDER BY uploaded_at DESC''', (user_id,))
        docs = cur.fetchall()
        cur.close(); db.close()

        for d in docs:
            if isinstance(d.get('uploaded_at'), datetime.datetime):
                d['uploaded_at'] = d['uploaded_at'].strftime('%Y-%m-%d %H:%M')

        return jsonify({'status': 'success', 'documents': docs})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/documents/<int:doc_id>', methods=['GET'])
def get_document_detail(doc_id):
    try:
        db  = get_db()
        cur = db.cursor(dictionary=True)
        cur.execute('SELECT * FROM documents WHERE doc_id=%s', (doc_id,))
        doc = cur.fetchone()
        cur.execute(
            'SELECT clause, status FROM audit_results WHERE doc_id=%s', (doc_id,))
        clauses = cur.fetchall()
        cur.close(); db.close()

        if not doc:
            return jsonify({'status': 'error',
                'message': 'Document not found'}), 404

        if isinstance(doc.get('uploaded_at'), datetime.datetime):
            doc['uploaded_at'] = doc['uploaded_at'].strftime('%Y-%m-%d %H:%M')

        return jsonify({'status': 'success', 'document': doc, 'clauses': clauses})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# =============================================================
# STATS
# =============================================================
@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        auth    = request.headers.get('Authorization', '')
        token   = auth.replace('Bearer ', '')
        payload = jwt.decode(token, app.secret_key, algorithms=['HS256'])
        user_id = payload['user_id']

        db  = get_db()
        cur = db.cursor(dictionary=True)
        cur.execute(
            'SELECT COUNT(*) as total FROM documents WHERE user_id=%s', (user_id,))
        total = cur.fetchone()['total']
        cur.execute(
            'SELECT COUNT(*) as compliant FROM documents WHERE user_id=%s AND compliance_score>=80',
            (user_id,))
        compliant = cur.fetchone()['compliant']
        cur.execute(
            'SELECT doc_type, COUNT(*) as count FROM documents WHERE user_id=%s GROUP BY doc_type',
            (user_id,))
        by_type = cur.fetchall()
        cur.close(); db.close()

        return jsonify({
            'status':         'success',
            'total_docs':     total,
            'compliant_docs': compliant,
            'issues_found':   total - compliant,
            'by_type':        by_type,
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# =============================================================
# PDF REPORT
# =============================================================
@app.route('/api/report/<int:doc_id>', methods=['GET'])
def generate_report(doc_id):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (SimpleDocTemplate, Paragraph,
                                         Spacer, Table, TableStyle, HRFlowable)
        from reportlab.lib.units import cm
        import io

        db  = get_db()
        cur = db.cursor(dictionary=True)
        cur.execute('SELECT * FROM documents WHERE doc_id=%s', (doc_id,))
        doc = cur.fetchone()
        cur.execute(
            'SELECT clause, status FROM audit_results WHERE doc_id=%s', (doc_id,))
        clauses = cur.fetchall()
        cur.execute(
            'SELECT full_name, email, company FROM users WHERE user_id=%s',
            (doc['user_id'],))
        user = cur.fetchone()
        cur.close(); db.close()

        if not doc:
            return jsonify({'status': 'error', 'message': 'Not found'}), 404

        buffer = io.BytesIO()
        pdf    = SimpleDocTemplate(buffer, pagesize=A4,
                    rightMargin=2*cm, leftMargin=2*cm,
                    topMargin=2*cm, bottomMargin=2*cm)
        story  = []

       header_style = ParagraphStyle('H', fontSize=22,
    fontName='Helvetica-Bold',
    textColor=colors.HexColor('#1a3c5e'),
    spaceAfter=8, spaceBefore=4)
sub_style = ParagraphStyle('S', fontSize=10,
    fontName='Helvetica',
    textColor=colors.HexColor('#888888'),
    spaceAfter=6)
        section_style = ParagraphStyle('Sec', fontSize=13,
            fontName='Helvetica-Bold',
            textColor=colors.HexColor('#1a3c5e'),
            spaceBefore=16, spaceAfter=8)

        story.append(Paragraph('AuditSmart', header_style))
        story.append(Paragraph(
            'AI-Powered Document Classification & Contract Audit System', sub_style))
        story.append(Paragraph('University of Kigali · BBIT 2026 · Rwanda', sub_style))
        story.append(HRFlowable(width="100%", thickness=2,
            color=colors.HexColor('#1a3c5e'), spaceAfter=16))

        title_style = ParagraphStyle('T', fontSize=16,
            fontName='Helvetica-Bold',
            textColor=colors.white, alignment=1)
        title_table = Table(
            [[Paragraph('DOCUMENT AUDIT REPORT', title_style)]],
            colWidths=[17*cm])
        title_table.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,-1), colors.HexColor('#1a3c5e')),
            ('TOPPADDING',    (0,0), (-1,-1), 14),
            ('BOTTOMPADDING', (0,0), (-1,-1), 14),
        ]))
        story.append(title_table)
        story.append(Spacer(1, 16))

        story.append(Paragraph('Document Information', section_style))
        uploaded_at = doc['uploaded_at']
        if isinstance(uploaded_at, datetime.datetime):
            uploaded_at = uploaded_at.strftime('%Y-%m-%d %H:%M')

        info_data = [
            ['Document Name',    doc['original_name']],
            ['Document Type',    doc['doc_type']],
            ['AI Confidence',    f"{doc['confidence']}%"],
            ['Uploaded By',      user['full_name']],
            ['Company',          user['company'] or 'N/A'],
            ['Email',            user['email']],
            ['Upload Date',      str(uploaded_at)],
            ['Report Generated', datetime.datetime.now().strftime('%Y-%m-%d %H:%M')],
        ]
        info_table = Table(info_data, colWidths=[5*cm, 12*cm])
        info_table.setStyle(TableStyle([
            ('FONTNAME',      (0,0), (0,-1), 'Helvetica-Bold'),
            ('FONTNAME',      (1,0), (1,-1), 'Helvetica'),
            ('FONTSIZE',      (0,0), (-1,-1), 10),
            ('TEXTCOLOR',     (0,0), (0,-1), colors.HexColor('#1a3c5e')),
            ('TEXTCOLOR',     (1,0), (1,-1), colors.HexColor('#333333')),
            ('ROWBACKGROUNDS',(0,0), (-1,-1),
                [colors.HexColor('#f8fbff'), colors.white]),
            ('TOPPADDING',    (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ('LEFTPADDING',   (0,0), (-1,-1), 10),
            ('GRID',          (0,0), (-1,-1), 0.5, colors.HexColor('#e0e0e0')),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 16))

        story.append(Paragraph('Compliance Score', section_style))
        score  = doc['compliance_score']
        passed = doc['clauses_passed']
        total  = doc['clauses_total']
        clr    = colors.HexColor(
            '#2e7d32' if score >= 80 else
            '#e65100' if score >= 50 else '#c62828')
        status_txt = (
            'FULLY COMPLIANT'     if score >= 80 else
            'PARTIALLY COMPLIANT' if score >= 50 else
            'NON COMPLIANT')

        score_table = Table(
            [[f'{score}%', f'{passed}/{total}', status_txt],
             ['Compliance Score', 'Clauses Passed', 'Overall Status']],
            colWidths=[5.6*cm, 5.6*cm, 5.6*cm])
        score_table.setStyle(TableStyle([
            ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',      (0,0), (-1,0), 22),
            ('FONTNAME',      (0,1), (-1,1), 'Helvetica'),
            ('FONTSIZE',      (0,1), (-1,1), 9),
            ('TEXTCOLOR',     (0,0), (-1,0), clr),
            ('TEXTCOLOR',     (0,1), (-1,1), colors.HexColor('#888888')),
            ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
            ('BACKGROUND',    (0,0), (-1,-1), colors.HexColor('#f8fbff')),
            ('TOPPADDING',    (0,0), (-1,-1), 14),
            ('BOTTOMPADDING', (0,0), (-1,-1), 14),
            ('GRID',          (0,0), (-1,-1), 0.5, colors.HexColor('#e0e0e0')),
        ]))
        story.append(score_table)
        story.append(Spacer(1, 16))

        story.append(Paragraph('Clause Audit Results', section_style))
        clause_data = [['Clause', 'Status', 'Result']]
        for c in clauses:
            ok = bool(c['status'])
            clause_data.append([
                c['clause'],
                'FOUND'   if ok else 'MISSING',
                'Compliant' if ok else 'Needs Attention'
            ])

        clause_table = Table(clause_data, colWidths=[8*cm, 4*cm, 5*cm])
        clause_table.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,0), colors.HexColor('#1a3c5e')),
            ('TEXTCOLOR',     (0,0), (-1,0), colors.white),
            ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',      (0,0), (-1,0), 10),
            ('ALIGN',         (0,0), (-1,0), 'CENTER'),
            ('FONTNAME',      (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE',      (0,1), (-1,-1), 10),
            ('ROWBACKGROUNDS',(0,1), (-1,-1),
                [colors.HexColor('#f8fbff'), colors.white]),
            ('TOPPADDING',    (0,0), (-1,-1), 9),
            ('BOTTOMPADDING', (0,0), (-1,-1), 9),
            ('LEFTPADDING',   (0,0), (-1,-1), 10),
            ('GRID',          (0,0), (-1,-1), 0.5, colors.HexColor('#e0e0e0')),
        ]))
        for i, c in enumerate(clauses, start=1):
            ok  = bool(c['status'])
            clr = colors.HexColor('#2e7d32' if ok else '#c62828')
            clause_table.setStyle(TableStyle([
                ('TEXTCOLOR', (1,i), (2,i), clr),
                ('FONTNAME',  (1,i), (2,i), 'Helvetica-Bold'),
            ]))
        story.append(clause_table)
        story.append(Spacer(1, 16))

        if doc.get('text_preview'):
            story.append(Paragraph('Document Text Preview', section_style))
            preview_style = ParagraphStyle('P', fontSize=9,
                fontName='Helvetica',
                textColor=colors.HexColor('#555555'),
                leading=14, backColor=colors.HexColor('#f8fbff'),
                borderPadding=10)
            preview_text = doc['text_preview'][:400].replace('\n', '<br/>')
            story.append(Paragraph(preview_text, preview_style))
            story.append(Spacer(1, 16))

        story.append(Paragraph('Recommendations', section_style))
        missing = [c['clause'] for c in clauses if not bool(c['status'])]
        rec_text = (
            'This document is fully compliant. All required clauses have been '
            'found and verified. No action is required at this time.'
            if not missing else
            f'This document is missing {len(missing)} required clause(s): '
            f'{", ".join(missing)}. It is recommended to review and update '
            'the document before signing or submitting for approval.'
        )
        rec_style = ParagraphStyle('R', fontSize=10, fontName='Helvetica',
            textColor=colors.HexColor('#333333'), leading=16,
            backColor=colors.HexColor('#e8f5e9' if not missing else '#fff3e0'),
            borderPadding=12)
        story.append(Paragraph(rec_text, rec_style))
        story.append(Spacer(1, 24))

        story.append(HRFlowable(width="100%", thickness=1,
            color=colors.HexColor('#e0e0e0'), spaceAfter=10))
        footer_style = ParagraphStyle('F', fontSize=8, fontName='Helvetica',
            textColor=colors.HexColor('#aaaaaa'), alignment=1)
        story.append(Paragraph(
            'AuditSmart v1.0  ·  Ruhorimbere Fred (2305001581)  ·  '
            'Supervisor: Dr. Eustach Uwimana  ·  University of Kigali  ·  BBIT 2026',
            footer_style))
        story.append(Paragraph(
            f'Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}  ·  '
            'AI-Powered Document Audit System  ·  Rwanda',
            footer_style))

        pdf.build(story)
        buffer.seek(0)
        return send_file(buffer, mimetype='application/pdf',
            as_attachment=True,
            download_name=f'AuditSmart_Report_{doc["doc_type"].replace(" ","_")}_{doc_id}.pdf')

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# =============================================================
# ADMIN ROUTES
# =============================================================
@app.route('/api/admin/users', methods=['GET'])
def admin_get_users():
    try:
        auth    = request.headers.get('Authorization', '')
        token   = auth.replace('Bearer ', '')
        payload = jwt.decode(token, app.secret_key, algorithms=['HS256'])
        db  = get_db()
        cur = db.cursor(dictionary=True)
        cur.execute('SELECT role FROM users WHERE user_id=%s', (payload['user_id'],))
        user = cur.fetchone()
        if not user or user['role'] != 'admin':
            return jsonify({'status': 'error',
                'message': 'Admin access required'}), 403
        cur.execute('''SELECT u.user_id, u.full_name, u.email, u.company, u.role,
                       u.created_at, COUNT(d.doc_id) as total_docs
                       FROM users u
                       LEFT JOIN documents d ON u.user_id = d.user_id
                       GROUP BY u.user_id
                       ORDER BY u.created_at DESC''')
        users = cur.fetchall()
        cur.close(); db.close()
        for u in users:
            if isinstance(u.get('created_at'), datetime.datetime):
                u['created_at'] = u['created_at'].strftime('%Y-%m-%d %H:%M')
        return jsonify({'status': 'success', 'users': users})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/admin/documents', methods=['GET'])
def admin_get_documents():
    try:
        auth    = request.headers.get('Authorization', '')
        token   = auth.replace('Bearer ', '')
        payload = jwt.decode(token, app.secret_key, algorithms=['HS256'])
        db  = get_db()
        cur = db.cursor(dictionary=True)
        cur.execute('SELECT role FROM users WHERE user_id=%s', (payload['user_id'],))
        user = cur.fetchone()
        if not user or user['role'] != 'admin':
            return jsonify({'status': 'error',
                'message': 'Admin access required'}), 403
        cur.execute('''SELECT d.doc_id, d.original_name, d.doc_type,
                       d.confidence, d.compliance_score, d.clauses_passed,
                       d.clauses_total, d.uploaded_at, u.full_name, u.email
                       FROM documents d
                       JOIN users u ON d.user_id = u.user_id
                       ORDER BY d.uploaded_at DESC''')
        docs = cur.fetchall()
        cur.close(); db.close()
        for d in docs:
            if isinstance(d.get('uploaded_at'), datetime.datetime):
                d['uploaded_at'] = d['uploaded_at'].strftime('%Y-%m-%d %H:%M')
        return jsonify({'status': 'success', 'documents': docs})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/admin/stats', methods=['GET'])
def admin_stats():
    try:
        auth    = request.headers.get('Authorization', '')
        token   = auth.replace('Bearer ', '')
        payload = jwt.decode(token, app.secret_key, algorithms=['HS256'])
        db  = get_db()
        cur = db.cursor(dictionary=True)
        cur.execute('SELECT role FROM users WHERE user_id=%s', (payload['user_id'],))
        user = cur.fetchone()
        if not user or user['role'] != 'admin':
            return jsonify({'status': 'error',
                'message': 'Admin access required'}), 403
        cur.execute('SELECT COUNT(*) as total FROM users')
        total_users = cur.fetchone()['total']
        cur.execute('SELECT COUNT(*) as total FROM documents')
        total_docs = cur.fetchone()['total']
        cur.execute(
            'SELECT COUNT(*) as total FROM documents WHERE compliance_score >= 80')
        compliant = cur.fetchone()['total']
        cur.execute(
            'SELECT doc_type, COUNT(*) as count FROM documents GROUP BY doc_type')
        by_type = cur.fetchall()
        cur.close(); db.close()
        return jsonify({
            'status':      'success',
            'total_users': total_users,
            'total_docs':  total_docs,
            'compliant':   compliant,
            'issues':      total_docs - compliant,
            'by_type':     by_type,
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# =============================================================
# MODEL EVALUATION
# =============================================================
@app.route('/api/evaluation', methods=['GET'])
def get_evaluation():
    try:
        from sklearn.metrics import precision_score, recall_score, f1_score
        from sklearn.model_selection import cross_val_predict

        texts, labels = [], []
        for label, samples in TRAINING_DATA.items():
            for text in samples:
                texts.append(text)
                labels.append(label)

        X     = _vectorizer.transform(texts)
        preds = cross_val_predict(
            CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=5000), cv=3),
            X, labels, cv=3
        )

        precision = round(
            precision_score(labels, preds, average='weighted') * 100, 2)
        recall    = round(
            recall_score(labels, preds, average='weighted') * 100, 2)
        f1        = round(
            f1_score(labels, preds, average='weighted') * 100, 2)

        classes = sorted(set(labels))
        p_each  = precision_score(labels, preds, average=None, labels=classes)
        r_each  = recall_score(labels, preds, average=None, labels=classes)
        f_each  = f1_score(labels, preds, average=None, labels=classes)

        per_class = [
            {
                'class':     cls,
                'precision': round(float(p_each[i]) * 100, 2),
                'recall':    round(float(r_each[i]) * 100, 2),
                'f1':        round(float(f_each[i]) * 100, 2),
            }
            for i, cls in enumerate(classes)
        ]

        return jsonify({
            'status':  'success',
            'overall': {
                'precision': precision,
                'recall':    recall,
                'f1_score':  f1,
                'accuracy':  round((precision + recall + f1) / 3, 2),
            },
            'per_class':      per_class,
            'total_samples':  len(texts),
            'total_classes':  len(classes),
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# =============================================================
# USER FEEDBACK
# =============================================================
@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    try:
        auth    = request.headers.get('Authorization', '')
        token   = auth.replace('Bearer ', '')
        payload = jwt.decode(token, app.secret_key, algorithms=['HS256'])
        user_id = payload['user_id']
        d       = request.get_json() or {}
        rating  = d.get('rating', 0)
        category= d.get('category', '')
        message = d.get('message', '').strip()
        if not message or not rating:
            return jsonify({'status': 'error',
                'message': 'Rating and message required'}), 400
        db  = get_db()
        cur = db.cursor()
        cur.execute(
            'INSERT INTO feedback (user_id, rating, category, message) VALUES (%s,%s,%s,%s)',
            (user_id, rating, category, message)
        )
        db.commit()
        cur.close(); db.close()
        return jsonify({'status': 'success',
            'message': 'Feedback submitted successfully!'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/feedback', methods=['GET'])
def get_feedback():
    try:
        auth    = request.headers.get('Authorization', '')
        token   = auth.replace('Bearer ', '')
        payload = jwt.decode(token, app.secret_key, algorithms=['HS256'])
        db  = get_db()
        cur = db.cursor(dictionary=True)
        cur.execute('SELECT role FROM users WHERE user_id=%s', (payload['user_id'],))
        user = cur.fetchone()
        if not user or user['role'] != 'admin':
            return jsonify({'status': 'error',
                'message': 'Admin access required'}), 403
        cur.execute('''SELECT f.feedback_id, f.rating, f.category,
                       f.message, f.created_at, u.full_name, u.email
                       FROM feedback f
                       JOIN users u ON f.user_id = u.user_id
                       ORDER BY f.created_at DESC''')
        feedbacks = cur.fetchall()
        cur.close(); db.close()
        for f in feedbacks:
            if isinstance(f.get('created_at'), datetime.datetime):
                f['created_at'] = f['created_at'].strftime('%Y-%m-%d %H:%M')
        avg = round(sum(f['rating'] for f in feedbacks) / len(feedbacks), 1) \
            if feedbacks else 0
        return jsonify({
            'status':     'success',
            'feedbacks':  feedbacks,
            'total':      len(feedbacks),
            'avg_rating': avg,
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# =============================================================
# STATUS & PAGES
# =============================================================
@app.route('/api/status')
def status():
    return jsonify({
        'status':    'running',
        'system':    'AuditSmart',
        'version':   '1.0',
        'author':    'Ruhorimbere Fred (2305001581)',
        'doc_types': DOCUMENT_TYPES,
    })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/admin')
def admin_page():
    return render_template('admin.html')

# =============================================================
# STARTUP
# =============================================================
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
train_model()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)