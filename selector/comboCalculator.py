import pandas as pd

def calcFOV(f, sensor_w, sensor_h, d):
    fov_h = round(sensor_w * d / f, 4)
    fov_v = round(sensor_h * d / f, 4)
    return fov_h, fov_v

def calcOpticalSize(s):
    s = s.replace("\"","").replace("''","")
    s = s.replace(",", ".")
    if "/" in s:
        return round(float(s.split("/")[0]) / float(s.split("/")[1]),4)
    else:
        try:
            return float(s)
        except:
            return float(0)
        
def getScoreFromLUT(val, lut, reverse=False):
    s = 0
    for i, threshold in enumerate(lut):
        if (reverse and val <= threshold) or (not reverse and val >= threshold):
            s = i
    return s

def calcMaxScore(d_inc, f, pitch_x, pitch_y, w_px_max, h_px_max, base_fps, price, fov_x_add = 0, fov_y_add = 0, fps_factor=0.5):
    best_score = 0
    best_d = 15000
    best_res = 1000
    best_fps = 0

    best_score_fps = 0
    best_score_d = 0
    best_score_res = 0
    best_score_price = 0

    for d in range(d_min, d_max+d_inc, d_inc):
        res_x = round((pitch_x * d) / (f * 1000),4)
        res_y = round((pitch_y * d) / (f * 1000),4)
        res = max(res_x, res_y)

        w_px = round(((fov_h_min + fov_x_add) * f * 1000) / (pitch_x * d),4)    
        h_px = round(((fov_v_min + fov_y_add) * f * 1000) / (pitch_y * d),4)
        if w_px > w_px_max or h_px > h_px_max:
            continue
        else:
            fps = round(base_fps +  base_fps * fps_factor * (h_px_max-h_px) / h_px_max, 4)

        
        price_score = getScoreFromLUT(price, lut_price, True)
        fps_score = getScoreFromLUT(fps, lut_fps, False)
        res_score = getScoreFromLUT(res, lut_res, True)
        d_score = getScoreFromLUT(d, lut_d, True)
        
        score = price_score * k_price + fps_score * k_fps + res_score * k_res + d_score * k_d
        if score > best_score:
            best_d = d
            best_res = res
            best_score = score
            best_fps = fps
            best_score_fps = fps_score
            best_score_d = d_score
            best_score_price = price_score
            best_score_res = res_score

    return best_score_fps, best_score_d, best_score_res, best_score_price, best_d, best_res, best_fps, best_score

# Limits
cost_max = 600
fps_min = 75

d_min = 600
d_max = 1500
fov_h_min = 1180
fov_v_min = 680

res_min = 12.5

def genLinearLUT(leftVal, rightVal, steps):
    stepsize = (rightVal-leftVal)/ (steps-1)
    return [round(leftVal + i * stepsize, 4) for i in range(steps)]
# Bewertung
lut_price = [500, 470, 440, 410, 380, 350, 320, 290, 260, 230]
lut_fps = [100, 115, 130, 145, 160, 175, 190, 205, 220, 235]
lut_res = [4.6, 4.2, 3.8, 3.4, 3.0, 2.6, 2.2, 1.8, 1.4, 1.0]
lut_d = [1500, 1410, 1320, 1230, 1140, 1050, 960, 870, 780, 690]

lut_price = genLinearLUT(600, 230, 5)
lut_fps = genLinearLUT(120, 200, 5)
lut_res = genLinearLUT(8.0, 1.0, 5)
lut_d = genLinearLUT(1300, 640, 5)

#lut_price = genLinearLUT(600, 230, 20)
#lut_fps = genLinearLUT(120, 200, 20)
#lut_res = genLinearLUT(4.0, .3, 20)
#lut_d = genLinearLUT(1400, 600, 20)


print(lut_fps)
print(lut_d)
print(lut_res)
print(lut_price)
k_price = 3
k_fps = 1
k_res = 4
k_d = 3

# Pfade der csv-Dateien
camera_csv_path = "selector/basler_cameras.csv"
lens_csv_path = "selector/basler_lenses.csv"
combo_csv_path = "selector/basler_combos.csv"
ok_csv_path = "selector/basler_ok.csv"

# Dateien einlesen
camera_df = pd.read_csv(camera_csv_path, sep=";", decimal=",")
lens_df = pd.read_csv(lens_csv_path, sep=";", decimal=",")

# Daten bereinigen / umwandeln
lens_df = lens_df.dropna(subset=["ordernumber", "focal_length"]) 
lens_df["focal_length"] = lens_df["focal_length"].str.replace("mm","").str.strip().astype(float) 

# Kombinationen bilden
combo_df = pd.merge(camera_df, lens_df, how="cross", suffixes=("_cam", "_lens"))


print(f"Bilanz nach Merge: {len(combo_df)} Kombinationen")

# Neue Informationen generieren
# Gesamtpreis
combo_df["total_price"] = combo_df["price_cam"] + combo_df["price_lens"]

# Field of View
sensor_w = combo_df["resolution_horizontal"] * combo_df["pixel_size_horizontal"] / 1000
sensor_h = combo_df["resolution_vertical"] * combo_df["pixel_size_vertical"] / 1000
combo_df["max_fov_h"], combo_df["max_fov_v"] = calcFOV(combo_df["focal_length"], sensor_w, sensor_h, d_max)

combo_df["resolution_mm"] = pd.DataFrame({"res_h": round(combo_df["max_fov_h"]/combo_df["resolution_horizontal"],4), "res_v": round(combo_df["max_fov_v"]/combo_df["resolution_vertical"],4)}).max(axis=1)

# KO-Kriterien filtern
# Kosten
ok_df_cost = combo_df[combo_df["total_price"] <= cost_max] 
ko_df_cost = combo_df[combo_df["total_price"] > cost_max] 
ok_df = ok_df_cost
print(f"Bilanz nach Kosten-Kriterium: {len(ok_df)} OK | {len(ko_df_cost)} nach Kosten aussortiert")

# Mount
ok_df_mount = ok_df[(ok_df["mount_cam"] != "No-Mount")]
ko_df_mount_nomount = ok_df[(ok_df["mount_cam"] == "No-Mount")]
ko_df_mount_mismatch = ok_df_mount[(ok_df_mount["mount_cam"].str.upper()=="S-MOUNT") & (ok_df_mount["mount_lens"].str.upper()!="S-MOUNT")]
ok_df_mount = ok_df_mount[~((ok_df_mount["mount_cam"].str.upper()=="S-MOUNT") & (ok_df_mount["mount_lens"].str.upper()!="S-MOUNT"))]
ok_df = ok_df_mount
ko_df_mount = pd.concat([ko_df_mount_mismatch, ko_df_mount_nomount])
print(f"Bilanz nach Mount-Kriterium: {len(ok_df)} OK | {len(ko_df_mount)} nach Mount aussortiert")

# Sensorsize
ok_df_size = ok_df[ok_df["sensor_format"].apply(calcOpticalSize) >= ok_df["sensor_optical_size"].apply(calcOpticalSize)]
ko_df_size = ok_df[ok_df["sensor_format"].apply(calcOpticalSize) < ok_df["sensor_optical_size"].apply(calcOpticalSize)]
ok_df = ok_df_size
print(f"Bilanz nach Size-Kriterium: {len(ok_df)} OK | {len(ko_df_size)} nach Size aussortiert")

# Interface
ok_df_interface = ok_df[(ok_df["interface"] != "BCON for MIPI")]
ko_df_interface = ok_df[(ok_df["interface"] == "BCON for MIPI")]
ok_df = ok_df_interface
print(f"Bilanz nach Interface-Kriterium: {len(ok_df)} OK | {len(ko_df_interface)} nach Interface aussortiert")

# Framerate
ok_df_fps = ok_df[ok_df["sensor_framerate_max"] >= fps_min]
ko_df_fps = ok_df[ok_df["sensor_framerate_max"] < fps_min]
ok_df = ok_df_fps
print(f"Bilanz nach Framerate-Kriterium: {len(ok_df)} OK | {len(ko_df_fps)} nach Framerate aussortiert")

# Maximales FOV
ok_df_fov = ok_df[(ok_df["max_fov_h"] >= fov_h_min) & (ok_df["max_fov_v"] >= fov_v_min)]
ko_df_fov = ok_df[(ok_df["max_fov_h"] < fov_h_min) | (ok_df["max_fov_v"] < fov_v_min)]
ok_df = ok_df_fov
print(f"Bilanz nach FOV-Kriterium: {len(ok_df)} OK | {len(ko_df_fov)} nach FOV aussortiert")

# Resolution bei maximalem FOV
ok_df_res = ok_df[ok_df["resolution_mm"] <= res_min]
ko_df_res = ok_df[ok_df["resolution_mm"] > res_min]
ok_df = ok_df_res
print(f"Bilanz nach Auflösungs-Kriterium: {len(ok_df)} OK | {len(ko_df_res)} nach Auflösungs aussortiert")

# Sortieren
ok_df = ok_df.sort_values(by=["sensor_name","total_price"], ascending=[True,True])

# Zusammenfuegen und speichern
ko_df = pd.concat([ko_df_fov, ko_df_fps, ko_df_interface, ko_df_size, ko_df_mount, ko_df_cost], ignore_index=True)
combo_df = pd.concat([ok_df, ko_df], ignore_index=True)

combo_df.to_csv(combo_csv_path, sep=";", decimal=",")

def apply_calc(row):
    best_score_fps, best_score_d, best_score_res, best_score_price, best_d, best_res, best_fps, best_score = calcMaxScore(
        10,
        row["focal_length"],
        row["pixel_size_horizontal"],
        row["pixel_size_vertical"],
        row["resolution_horizontal"],
        row["resolution_vertical"],
        row["sensor_framerate_max"],
        row["total_price"],
        fov_x_add=100,
        fov_y_add=100,
        fps_factor=0.8
    )
    return pd.Series([best_score_fps, best_score_d, best_score_res, best_score_price, best_d, best_res, best_fps, best_score], index=["fps_score","d_score", "res_score","price_score", "best_d", "best_res", "best_fps", "score"])



ok_df[["fps_score","d_score", "res_score","price_score", "best_d", "best_res", "best_fps", "score"]] = ok_df.apply(apply_calc, axis=1)
ok_df = ok_df.sort_values(by=["score","focal_length", "total_price"], ascending=[False, True, True])

ok_df.to_csv(ok_csv_path, sep=";", decimal=",")
print("Done")