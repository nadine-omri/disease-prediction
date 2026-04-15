import requests

def download_dataset(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
            print(f'Downloaded {filename}')
    else:
        print(f'Failed to download {filename}')

def main():
    datasets = {
        'Heart Disease': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/heart-disease.data',
        'Diabetes': 'https://archive.ics.uci.edu/ml/machine-learning-databases/diabetes/diabetes.csv',
        'Breast Cancer': 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
    }

    for name, url in datasets.items():
        download_dataset(url, f'{name.replace(' ', '_').lower()}.csv')

if __name__ == '__main__':
    main()