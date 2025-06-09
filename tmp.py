import cv2
import numpy as np
import math
import random

# 1. 画像の読み込みと前処理 (変更なし)
def load_and_prepare_image(image_path, target_long_edge=512):
    original_img_bgr = cv2.imread(image_path)
    if original_img_bgr is None:
        print(f"エラー: 画像が見つかりません - {image_path}")
        return None, None

    height, width, _ = original_img_bgr.shape
    if height == 0 or width == 0:
        print(f"エラー: 画像サイズが無効です - {image_path}")
        return None, None

    if height > width:
        new_height = target_long_edge
        new_width = int(width * (target_long_edge / height))
    else:
        new_width = target_long_edge
        new_height = int(height * (target_long_edge / width))
    
    if new_width == 0 or new_height == 0:
        print(f"エラー: リサイズ後の画像サイズが無効になります。元画像サイズ: ({width},{height})")
        return None, None

    resized_img_bgr = cv2.resize(original_img_bgr, (new_width, new_height))
    gray_img = cv2.cvtColor(resized_img_bgr, cv2.COLOR_BGR2GRAY)
    return gray_img, resized_img_bgr

# 2. ピンの配置 (変更なし)
def place_pins(image_shape, num_pins, radius_ratio=0.9):
    center_y, center_x = image_shape[0] // 2, image_shape[1] // 2
    radius = min(center_y, center_x) * radius_ratio 
    
    pins = []
    for i in range(num_pins):
        angle = 2 * math.pi * i / num_pins
        x = int(center_x + radius * math.cos(angle))
        y = int(center_y + radius * math.sin(angle))
        pins.append((x, y))
    return pins

# 3. 目標通過本数マップの作成 (変更なし)
def create_target_coverage_map(image, max_lines_per_pixel, gamma=1.0):
    image_float = image.astype(np.float32) / 255.0
    if gamma != 1.0:
        image_gamma_corrected_float = np.power(image_float, gamma)
        image_gamma_corrected_float = np.clip(image_gamma_corrected_float, 0.0, 1.0)
        image_for_map = image_gamma_corrected_float * 255.0
    else:
        image_for_map = image.astype(np.float32)

    target_map = max_lines_per_pixel * ((255.0 - image_for_map) / 255.0)
    return target_map

# 4. 線が通過するピクセル座標の取得 (★★★ line_thickness 引数を追加 ★★★)
def get_line_pixels(p1_coord, p2_coord, image_shape, line_thickness=1):
    temp_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    pt1 = (int(round(p1_coord[0])), int(round(p1_coord[1])))
    pt2 = (int(round(p2_coord[0])), int(round(p2_coord[1])))
    cv2.line(temp_mask, pt1, pt2, 255, line_thickness) # ★★★ここで太さを指定
    line_pixel_coords = np.argwhere(temp_mask == 255)
    return [(coord[0], coord[1]) for coord in line_pixel_coords]

# 5. コスト削減量の計算 (変更なし)
def calculate_cost_reduction(line_pixels, target_map, current_coverage_map):
    delta_cost = 0
    for r, c in line_pixels:
        delta_cost += (2 * (target_map[r, c] - current_coverage_map[r, c]) - 1)
    return delta_cost

# 6. 最良の候補線の探索 (★★★ line_thickness 引数を追加 ★★★)
def find_best_line_candidate(pins, target_map, current_coverage_map, num_candidate_lines_per_step, image_shape, line_thickness=1):
    best_line_info = None
    max_cost_reduction = 0 

    num_pins = len(pins)
    if num_pins < 2:
        return None

    for _ in range(num_candidate_lines_per_step):
        idx1, idx2 = random.sample(range(num_pins), 2)
        pin1_coord = pins[idx1]
        pin2_coord = pins[idx2]

        # ★★★ line_thickness を get_line_pixels に渡す
        line_pixels = get_line_pixels(pin1_coord, pin2_coord, image_shape, line_thickness)
        if not line_pixels:
            continue
            
        current_reduction = calculate_cost_reduction(line_pixels, target_map, current_coverage_map)

        if current_reduction > max_cost_reduction:
            max_cost_reduction = current_reduction
            best_line_info = {
                "pin_indices": (idx1, idx2),
                "coords": (pin1_coord, pin2_coord),
                "pixels": line_pixels,
                "reduction": current_reduction
            }
            
    return best_line_info

# 8. メイン処理 (★★★ line_thickness 引数を追加 ★★★)
def generate_string_art_new_logic(
    image_path,
    num_pins=200,
    max_lines_per_pixel=10,
    total_num_lines=3000,
    num_candidate_lines_per_step=200,
    image_target_long_edge=512,
    ink_amount_per_line=0.02,
    gamma=1.0,
    line_thickness=1 # ★★★追加: デフォルトの線の太さ
):
    print("1. 画像の読み込みと準備...")
    gray_image, resized_color_image = load_and_prepare_image(image_path, image_target_long_edge)
    if gray_image is None:
        return None, None, None
    
    image_shape = gray_image.shape
    print(f"画像サイズ: {image_shape}")

    print("2. ピンの配置...")
    pins = place_pins(image_shape, num_pins)

    print("3. 目標通過本数マップの作成...")
    target_map = create_target_coverage_map(gray_image, max_lines_per_pixel, gamma)

    current_coverage_map = np.zeros_like(target_map, dtype=np.float32)
    float_canvas = np.ones(image_shape[:2], dtype=np.float32)

    generated_lines_indices = []

    print(f"4. 最大 {total_num_lines} 本の線を生成中 (コスト削減が見込めなくなったら中断)...")
    actual_lines_drawn = 0
    for i in range(total_num_lines):
        best_line_info = find_best_line_candidate(
            pins,
            target_map,
            current_coverage_map,
            num_candidate_lines_per_step,
            image_shape,
            line_thickness # ★★★ line_thickness を渡す
        )

        if best_line_info: 
            for r, c in best_line_info["pixels"]:
                float_canvas[r, c] = max(0.0, float_canvas[r, c] - ink_amount_per_line)
                current_coverage_map[r, c] += 1
            
            generated_lines_indices.append(best_line_info["pin_indices"])
            actual_lines_drawn +=1

            if (i + 1) % 100 == 0:
                print(f"  ステップ {i+1} (描画 {actual_lines_drawn}本目): 最良線のコスト削減量 {best_line_info['reduction']:.2f}")
        else:
            print(f"  ステップ {i+1}: コストを正に削減できる線が見つからなかったため、処理を中断します。")
            break

    print("5. 描画完了。")
    final_art_gray_uint8 = np.clip(float_canvas * 255.0, 0, 255).astype(np.uint8)
    final_art_bgr_display = cv2.cvtColor(final_art_gray_uint8, cv2.COLOR_GRAY2BGR)
    
    return final_art_bgr_display, generated_lines_indices, resized_color_image

if __name__ == "__main__":
    input_image_path = "washimimizuku.jpg"  # ここに入力画像のパスを指定してください
    
    params = {
        "num_pins": 180,
        "max_lines_per_pixel": 20,
        "total_num_lines": 8000,
        "num_candidate_lines_per_step": 200,
        "image_target_long_edge": 512,
        "ink_amount_per_line": 0.05,
        "gamma": 0.8, # ★★★ガンマ補正値 
        "line_thickness": 1 # ★★★追加: 線の太さ (1, 2, 3 などで試す)
    }
    
    # 例: 少し太い線で試す場合
    # params["line_thickness"] = 2
    # params["ink_amount_per_line"] = 0.005 # 太くした分、インク量を少し減らすとバランスが取れるかも

    final_art_image, lines_list, original_resized_image = generate_string_art_new_logic(
        input_image_path, **params
    )

    if final_art_image is not None:
        print(f"\n生成された線の総数: {len(lines_list)}")

        cv2.imshow("Original Resized Image", original_resized_image)
        cv2.imshow(f"Generated String Art (Ink Model, Thickness {params['line_thickness']})", final_art_image) # ウィンドウタイトルに太さ情報追加
        
        if original_resized_image is not None and final_art_image is not None and \
           original_resized_image.shape == final_art_image.shape:
            combined_image = np.hstack((original_resized_image, final_art_image))
            cv2.imshow("Original vs Generated", combined_image)
            # 画像保存
            cv2.imwrite("combined_original_vs_generated.jpg", combined_image)
        
        # 線リストをファイルに保存する例
        output_filename = f"string_art_lines_t{params['line_thickness']}.txt"
        with open(output_filename, "w") as f:
            f.write(f"# Image Source: {input_image_path}\n")
            f.write(f"# Total Pins: {params['num_pins']}\n")
            f.write(f"# Lines Drawn: {len(lines_list)}\n")
            f.write(f"# Max Lines Per Pixel: {params['max_lines_per_pixel']}\n")
            f.write(f"# Ink Amount Per Line: {params['ink_amount_per_line']}\n")
            f.write(f"# Gamma: {params['gamma']}\n")
            f.write(f"# Line Thickness for Eval: {params['line_thickness']}\n")
            f.write("# Pin numbering is 0-indexed, typically counter-clockwise starting from the right.\n")
            f.write("# Format: start_pin_index,end_pin_index\n")
            for line_indices in lines_list:
                f.write(f"{line_indices[0]},{line_indices[1]}\n")
        print(f"線リストを {output_filename} に保存しました。")

        cv2.waitKey(0)
        cv2.destroyAllWindows()