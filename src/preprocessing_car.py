import pandas as pd
from datetime import datetime
import hydra
from omegaconf import DictConfig

def load_raw_data(path):
    return pd.read_excel(path)

def clean_data(df, cfg):
    df.rename(columns={'Price (INR Lakhs)': 'Price'}, inplace=True)
    df.dropna(subset=['Engine', 'Power', 'Seats'], inplace=True)

    for r in cfg.preprocessing.mileage_replace:
        df['Mileage'] = df['Mileage'].str.replace(r,'')
    df['Power'] = df['Power'].str.replace(cfg.preprocessing.power_replace,'')
    df['Engine'] = df['Engine'].str.replace(cfg.preprocessing.engine_replace,'')

    df = df[df['Power']!='null'].copy()
    df['Mileage'] = df['Mileage'].astype(float)
    df['Power'] = df['Power'].astype(float)
    df['Engine'] = df['Engine'].astype(float)

    df['Car_Age'] = cfg.preprocessing.current_year - df['Year']
    df['km/year'] = df['Kilometers_Driven'] / df['Car_Age']

    Q1 = df['Price'].quantile(0.25)
    Q3 = df['Price'].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df = df[(df['Price'] >= lower_bound) & (df['Price'] <= upper_bound)].copy()

    df = df[df['km/year'] < cfg.preprocessing.max_km_per_year].copy()
    df = df[df['Mileage'] != 0].copy()
    df = df[df['Power'] <= cfg.preprocessing.max_power].copy()
    
    # Replace brand temporarily
    df['Brand_Model'] = df['Brand_Model'].str.replace(cfg.preprocessing.brand_temp_replace.original,
                                                      cfg.preprocessing.brand_temp_replace.temp)
    
    # Split by first space
    df[['Brand', 'Model']] = df['Brand_Model'].str.split(' ', n=1, expand=True)
    
    # Restore brand formatting
    df['Brand'] = df['Brand'].str.replace(cfg.preprocessing.brand_temp_replace.temp,
                                          cfg.preprocessing.brand_temp_replace.original)

    df.drop('Brand_Model', axis=1, inplace=True)

    return df

def save_cleaned_data(df, path):
    df.to_csv(path, index=False)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    df_raw = load_raw_data(cfg.dataset.raw_path)
    df_clean = clean_data(df_raw, cfg)
    save_cleaned_data(df_clean, cfg.dataset.processed_path)
    print(f"Data cleaning complete. Saved to {cfg.dataset.processed_path}")

if __name__ == "__main__":
    main()
