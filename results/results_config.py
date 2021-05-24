import os

script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, 'data')
img_dir = os.path.join(script_dir, 'images')
data_file = os.path.join(data_dir, 'bert-base-time.csv')

psql_creds = {
    'dbname': os.getenv('BERT_DB_NAME'),
    'user': os.getenv('BERT_DB_USER'),
    'password': os.getenv('BERT_DB_PASS'),
    'host': os.getenv('BERT_DB_HOST'),
    'sslmode': 'verify-ca',
    'sslrootcert': os.getenv('BERT_DB_CERT_PATH')
}
