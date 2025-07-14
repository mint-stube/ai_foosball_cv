import json
import os
import pandas as pd

# Ordner mit JSON-Dateien (hier anpassen)
json_folder = "selector\camSpecs"

price_file = "selector\prizes_cam.json"

# Die gewünschten Felder in der Tabelle
fields = [
    "ordernumber",
    "series_name",
    "model_name",
    "product_line",
    "shutter",
    "monocolor",
    "sensor_optical_size",
    "resolution_horizontal",
    "resolution_vertical",
    "pixel_size_horizontal",
    "pixel_size_vertical",
    "sensor_name",
    "housing_name",
    "interface",
    "sensor_framerate_min",
    "sensor_framerate_max",
    "mount",
    "price"
]

def load_price_map(price_file):
    with open(price_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        # Mapping von ordernumber → amount
        return {entry["ordernumber"]: entry["amount"] for entry in data.get("prices", [])}

def extract_camera_info(camera, shutter, price_map):
    # Hilfsfunktion zum Extrahieren der Infos aus einem Kamera-JSON-Objekt
    # Manche Felder liegen verschachtelt vor, daher aufpassen und mit .get() arbeiten

    ordernumber = camera.get("ordernumber")
    series_name = camera.get("series", {}).get("name")
    model_name = camera.get("model_name")
    product_line = camera.get("product_line")
    monocolor = camera.get("monocolor")
    sensor_optical_size = camera.get("sensor", {}).get("optical_size", {}).get("value")
    resolution_horizontal = camera.get("sensor", {}).get("resolution", {}).get("horizontal")
    resolution_vertical = camera.get("sensor", {}).get("resolution", {}).get("vertical")
    pixel_size_horizontal = camera.get("sensor", {}).get("pixel_size", {}).get("horizontal")
    pixel_size_vertical = camera.get("sensor", {}).get("pixel_size", {}).get("vertical")
    sensor_name = camera.get("sensor", {}).get("name")
    housing_name = camera.get("housing", {}).get("name")
    interface = camera.get("interface")
    if "CS-Mount" in model_name:
        mount = "CS-Mount"
    elif "No-Mount" in model_name:
        mount = "No-Mount"
    elif "S-Mount" in model_name:
        mount = "S-Mount"
    elif "Board" in housing_name:
        mount = "No-Mount"
    else:
        mount = "C-Mount"

    # sensor_framerate steht in "specs", suchen wir nach dem Eintrag mit "type" == "sensor_framerate"
    sensor_framerate = None
    for spec in camera.get("specs", []):
        if spec.get("type") == "sensor_framerate":
            sensor_framerate_min = spec.get("value", {}).get("min_value")
            sensor_framerate_max = spec.get("value", {}).get("max_value")
            break
    
    price = price_map.get(ordernumber)

    return {
        "ordernumber": ordernumber,
        "series_name": series_name,
        "model_name": model_name,
        "product_line": product_line,
        "shutter": shutter,
        "monocolor": monocolor,
        "sensor_optical_size": sensor_optical_size,
        "resolution_horizontal": resolution_horizontal,
        "resolution_vertical": resolution_vertical,
        "pixel_size_horizontal": pixel_size_horizontal,
        "pixel_size_vertical": pixel_size_vertical,
        "sensor_name": sensor_name,
        "housing_name": housing_name,
        "interface": interface,
        "sensor_framerate_min": sensor_framerate_min,
        "sensor_framerate_max": sensor_framerate_max,
        "mount": mount,
        "price": price
    }

def load_all_cameras_from_json_folder(folder, price_map):
    cameras = []
    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            filepath = os.path.join(folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                # In deinem Beispiel sind die Kameras unter "items"
                shutter = str(filename).split("_")[0] 
                items = data.get("items", [])
                for cam in items:
                    info = extract_camera_info(cam, shutter, price_map)
                    cameras.append(info)
    return cameras

if __name__ == "__main__":
    price_map = load_price_map(price_file)
    all_cameras = load_all_cameras_from_json_folder(json_folder, price_map)
    df = pd.DataFrame(all_cameras, columns=fields)
    df = df.drop_duplicates(subset="ordernumber", keep="first").reset_index(drop=True)
    print(df)

    # Optional: CSV speichern
    df.to_csv("selector/basler_cameras.csv", index=True, sep=";", decimal=",")
