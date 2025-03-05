import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def clean_dataframe(df):
    """
    Sütun isimlerindeki ve hücrelerdeki gereksiz boşlukları temizler,
    tamamen boş satır ve sütunları kaldırır.
    """
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
    df.dropna(how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    return df

def load_and_integrate_data(file_list, common_key, delimiter=','):
    """
    Verilen CSV dosyalarını okuyup temizler ve ortak bir anahtar üzerinden birleştirir.
    :param file_list: Birleştirilecek dosyaların yolu listesi.
    :param common_key: Dosyalar arasında birleştirme için ortak sütun adı (örneğin "durak_id").
    :param delimiter: CSV dosyalarında kullanılan ayırıcı (örneğin, ',' veya ';').
    :return: Birleştirilmiş DataFrame.
    """
    dataframes = []
    for file in file_list:
        df = pd.read_csv(file, delimiter=delimiter)
        df = clean_dataframe(df)
        print(f"{file} sütunları: {df.columns.tolist()}")
        dataframes.append(df)
    
    if common_key is None:
        raise KeyError("Birleştirme için ortak sütun bulunamadı. Lütfen 'common_key' parametresini belirtin.")
    
    merged_df = dataframes[0]
    for df in dataframes[1:]:
        merged_df = pd.merge(merged_df, df, on=common_key, how="left")
    return merged_df

def create_sample_data():
    """
    Örnek veri seti oluşturur ve 'model/sample_data.csv' olarak kaydeder.
    """
    np.random.seed(42)
    n_samples = 1000
    data = {
        'durak_id': np.arange(1, n_samples+1),  # Örnek olarak durak_id oluşturduk
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples),
        'target': np.random.normal(0, 1, n_samples)
    }
    df = pd.DataFrame(data)
    df.to_csv('model/sample_data.csv', index=False)
    print("Örnek veri seti oluşturuldu: model/sample_data.csv")

if __name__ == '__main__':
    # Dosya yollarını ayarlayın (örneğin, C:\surdurulebilirlik klasöründeyse)
    files = [r"C:\surdurulebilirlik\guzergahlar.csv", r"C:\surdurulebilirlik\duraklar.csv"]
    
    # Ortak sütun olarak "durak_id" kullanıyoruz. Eğer dosyalarınızda farklı bir sütun varsa, burayı değiştirin.
    try:
        merged_df = load_and_integrate_data(files, common_key="durak_id", delimiter=',')
        print("Birleştirilmiş veri seti örneği:")
        print(merged_df.head())
    except KeyError as e:
        print(e)
    
    # Örnek veri seti oluşturmak isterseniz:
    create_sample_data()
