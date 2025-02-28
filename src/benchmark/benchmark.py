import numpy as np
from tqdm import tqdm
import torch
import pandas as pd
import pickle
from pathlib import Path

def build_predictions(model, dataset_to_benchmark, device, path):
    selected_windows = dataset_to_benchmark['window_id'].unique()

    columns_to_keep = ["log_return_DlyClose", "log_return_DlyLow", "log_return_DlyHigh",
                       "log_return_DlyBid", "log_return_DlyAsk", "volume_normalized"]

    n_iterations = 100
    size_per_iteration = 6130
    available_windows = set(selected_windows)

    all_results = []

    for iteration in tqdm(range(n_iterations), desc="Bootstrap Progress"):
        current_available = np.array(list(available_windows))
        random_windows = np.random.choice(current_available, size=size_per_iteration, replace=False)

        selected_stocks = dataset_to_benchmark[dataset_to_benchmark["window_id"].isin(random_windows)]
        available_windows -= set(random_windows)

        windows_data = []
        window_info = []

        for window_id in random_windows:
            window_data = selected_stocks[selected_stocks["window_id"] == window_id]
            if not window_data.empty:
                windows_data.append(window_data[columns_to_keep].values)
                window_info.append({
                    'permno': window_data.iloc[0]["PERMNO"],
                    'date': window_data.iloc[-1]["DlyCalDt"],
                    'window_id': window_id
                })

        # Conversion en tenseur
        try:
            all_windows_tensor = torch.FloatTensor(np.array(windows_data)).transpose(1, 2).to(device)
        except:
            continue

        # Prédiction en une seule fois
        with torch.no_grad():
            try:
                reconstructed = model(all_windows_tensor)
                # Moyenne des pertes par fenêtre
                losses = torch.stack([
                    torch.nn.MSELoss()(reconstructed[i], all_windows_tensor[i])
                    for i in range(len(all_windows_tensor))
                ]).cpu().numpy()
            except Exception as e:
                print(f"Erreur lors de la prédiction: {e}")

        iteration_results = pd.DataFrame({
            'window_id': [info['window_id'] for info in window_info],
            'loss': losses,
            'dates': [info['date'] for info in window_info],
            'permno': [info['permno'] for info in window_info]
        })

        all_results.append(iteration_results)

    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv(path + "predictions_results.csv", index=False)
    return results_df

def build_quantiles(results_df, quantiles, dsf, cache_file='variations_cache.pkl'):
    # Charger les variations déjà calculées
    cache_path = Path(cache_file)
    if cache_path.exists():
        try:
            with open(cache_file, 'rb') as f:
                already_fetched_variations = pickle.load(f)
        except:
            already_fetched_variations = {}
    else:
        already_fetched_variations = {}

    results = {}
    for quantile in quantiles:
        q99 = results_df['loss'].quantile(quantile)
        results_df_quantile = results_df.loc[results_df['loss'] >= q99]

        unique_permnos = results_df_quantile["permno"].unique()
        subset = None if dsf is None else dsf[dsf["PERMNO"].isin(unique_permnos)]
        results_price = []
        dates_save = []
        window_save = []

        for index, line in tqdm(results_df_quantile.iterrows(), total=len(results_df_quantile)):
            try:
                results_price.append(already_fetched_variations[line["window_id"]])
                dates_save.append(line["dates"])
                window_save.append(line["window_id"])
                continue
            except:
                pass

            to_analyse = subset[subset["PERMNO"] == line["permno"]].copy()
            target_date = pd.to_datetime(line["dates"])
            to_analyse["DlyCalDt"] = pd.to_datetime(to_analyse["DlyCalDt"])

            future_date_min = target_date + pd.Timedelta(days=10)
            future_date_max = target_date + pd.Timedelta(days=25)

            mask = (to_analyse["DlyCalDt"] >= future_date_min) & (to_analyse["DlyCalDt"] <= future_date_max)
            current_price = to_analyse[to_analyse["DlyCalDt"] == line["dates"]]["DlyClose"]

            dates_in_range = to_analyse[mask].copy()
            if len(dates_in_range) > 0 and len(current_price) > 0 and current_price.iloc[0] != 0:
                ideal_date = target_date + pd.Timedelta(days=20)
                dates_in_range["date_diff"] = abs(dates_in_range["DlyCalDt"] - ideal_date)

                closest_date = dates_in_range.loc[dates_in_range["date_diff"].idxmin()]
                future_price = closest_date["DlyClose"]
                current_price = current_price.iloc[0]
                result = min(((future_price - current_price) / current_price) * 100, 100)
                results_price.append(result)
                already_fetched_variations[line["window_id"]] = result
                dates_save.append(line["dates"])
                window_save.append(line["window_id"])

        results[quantile] = {"variations": results_price, "threshold": q99, "dates": dates_save, "window_id": window_save}

        # Sauvegarder les variations après chaque quantile
        with open(cache_file, 'wb') as f:
            pickle.dump(already_fetched_variations, f)

    return results
