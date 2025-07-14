import pandas as pd

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

def genLinearLUT(leftVal, rightVal, steps):
    stepsize = (rightVal-leftVal)/ (steps-1)
    return [round(leftVal + i * stepsize, 4) for i in range(steps)]

d_min = 600
d_max = 1500
fov_h_min = 1180
fov_v_min = 680

lut_price = genLinearLUT(600, 230, 5)
lut_fps = genLinearLUT(120, 200, 5)
lut_res = genLinearLUT(4.0, 1.0, 5)
lut_d = genLinearLUT(1400, 600, 5)

k_price = 3
k_fps = 1
k_res = 3
k_d = 4

ok_csv_path = "baslerWebExtractor/basler_ok.csv"
final_csv_path = ""
