import json
import os
import pandas as pd

# Ordner mit JSON-Dateien der Objektive
json_folder = "baslerWebExtractor/lensSpecs"

# JSON-Datei mit Preisen für Objektive (Pfad anpassen)
price_file = "baslerWebExtractor/prizes_lens.json"

# Die gewünschten Felder in der Tabelle
fields = [
    "ordernumber",
    "model_name",
    "focal_length",
    "mount",
    "sensor_format",
    "min_object_distance",
    "price"
]

def load_price_map(price_file):
    with open(price_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        # Mapping von ordernumber → amount
        return {str(entry["ordernumber"]): entry["amount"] for entry in data.get("prices", [])}

def extract_lens_info(lens, price_map):
    ordernumber = str(lens.get("ordernumber"))
    model_name = lens.get("model_name")
    focal_length = lens.get("focal_length")
    mount = lens.get("mount", {}).get("name")

    sensor_format = None
    min_object_distance = None

    for spec in lens.get("specs", []):
        if spec.get("type") == "lens_sensor_format":
            sensor_format = spec.get("value", {}).get("label")
        elif spec.get("type") == "minobjectdistance":
            min_object_distance = spec.get("value", {}).get("min_value")

    price = price_map.get(ordernumber)

    return {
        "ordernumber": ordernumber,
        "model_name": model_name,
        "focal_length": focal_length,
        "mount": mount,
        "sensor_format": sensor_format,
        "min_object_distance": min_object_distance,
        "price": price
    }

def load_all_lenses_from_json_folder(folder, price_map):
    lenses = []
    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            filepath = os.path.join(folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                for val in data.values():
                    if isinstance(val, list):
                        for lens in val:
                            info = extract_lens_info(lens, price_map)
                            lenses.append(info)
                    elif isinstance(val, dict):
                        info = extract_lens_info(val, price_map)
                        lenses.append(info)
    return lenses

if __name__ == "__main__":
    price_map = load_price_map(price_file)
    all_lenses = load_all_lenses_from_json_folder(json_folder, price_map)
    df = pd.DataFrame(all_lenses, columns=fields)
    df = df.drop_duplicates(subset="ordernumber", keep="first").reset_index(drop=True)
    print(df)

    # Optional: CSV speichern
    df.to_csv("baslerWebExtractor/basler_lenses.csv", index=True, sep=";", decimal=",")
