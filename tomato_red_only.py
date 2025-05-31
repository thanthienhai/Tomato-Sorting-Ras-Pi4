import cv2
import numpy as np

# --- Cấu hình ---
LOWER_RED_HSV1 = np.array((0, 120, 70))
UPPER_RED_HSV1 = np.array((8, 255, 255))
LOWER_RED_HSV2 = np.array((165, 120, 70))
UPPER_RED_HSV2 = np.array((180, 255, 255))

# --- NGƯỠNG CHO MÀU MỤC TIÊU (VÀNG) - ĐIỀU CHỈNH CHO LINH HOẠT HƠN ---
LOWER_TARGET_COLOR_HSV = np.array((18, 40, 80))  # H_min, S_min, V_min giảm để bắt rộng hơn
UPPER_TARGET_COLOR_HSV = np.array((38, 255, 255))  # H_max nới rộng, S_max, V_max giữ nguyên

MEDIAN_BLUR_KERNEL_SIZE = 5
OPENING_KERNEL_SIZE = 5  # Có thể tăng lên 7 nếu mask ban đầu bị nhiễu nhiều
OPENING_ITERATIONS = 1
DILATE_ITERATIONS_AFTER_OPENING = 7  # Tăng lại một chút để nối liền contour nếu mask ban đầu tốt hơn

MIN_CONTOUR_AREA = 1000  # Điều chỉnh nếu vật thể nhỏ hơn/lớn hơn
MIN_SOLIDITY = 0.75  # Giảm nhẹ nếu vật thể không hoàn toàn lồi
MIN_ASPECT_RATIO = 0.60  # Nới lỏng một chút
MAX_ASPECT_RATIO = 1.40  # Nới lỏng một chút
MIN_EXTENT = 0.45  # Giảm nhẹ
MIN_CIRCULARITY_RELAXED = 0.70  # Giảm nhẹ, vì mask ban đầu có thể chưa hoàn hảo

ENABLE_POST_FILTER_COLOR_CLASSIFICATION = True
# --- Ngưỡng phân loại màu ---
CLASSIFY_RED_MIN_HUE1 = 0
CLASSIFY_RED_MAX_HUE1 = 8
CLASSIFY_RED_MIN_HUE2 = 165
CLASSIFY_RED_MAX_HUE2 = 180
CLASSIFY_RED_MIN_SATURATION = 100
CLASSIFY_RED_MIN_VALUE = 70

# --- NGƯỠNG PHÂN LOẠI CHO MÀU MỤC TIÊU (VÀNG) - ĐIỀU CHỈNH ---
CLASSIFY_TARGET_MIN_HUE = 18  # Đồng bộ với LOWER_TARGET_COLOR_HSV
CLASSIFY_TARGET_MAX_HUE = 40  # Hơi rộng hơn UPPER_TARGET_COLOR_HSV
CLASSIFY_TARGET_MIN_SATURATION = 50  # Giảm để linh hoạt hơn
CLASSIFY_TARGET_MIN_VALUE = 90  # Giảm để linh hoạt hơn

DOMINANT_PIXEL_RATIO_THRESHOLD = 0.35  # Giữ nguyên hoặc giảm nhẹ nếu màu không quá đồng nhất
SIGNIFICANT_PIXEL_RATIO_THRESHOLD = 0.20

RADIUS_AT_KNOWN_DISTANCE_PX = 26
KNOWN_DISTANCE_M = 0.4
RADIUS_TO_METERS_CALIB = RADIUS_AT_KNOWN_DISTANCE_PX * KNOWN_DISTANCE_M
OBJECT_REAL_WIDTH_M = 0.05
PIX_TO_METERS_CALIB = OBJECT_REAL_WIDTH_M / (RADIUS_AT_KNOWN_DISTANCE_PX * 2)
ASSUMED_Z_HEIGHT_M = 0.35

SHOW_PROCESSING_STEPS = True
SHOW_FINAL_RESULT_IMAGE = True
ONLY_DETECT_LARGEST_CONTOUR = True


def get_dominant_color_in_roi(hsv_image_roi, original_mask_roi):
    total_pixels_in_contour = cv2.countNonZero(original_mask_roi)
    if total_pixels_in_contour == 0:
        return "unknown", 0, 0, 0

    mean_hsv_roi_val = cv2.mean(hsv_image_roi, mask=original_mask_roi)
    hue_roi, saturation_roi, value_roi = mean_hsv_roi_val[0], mean_hsv_roi_val[1], mean_hsv_roi_val[2]

    is_definitely_red = False
    is_red_by_mean = ((CLASSIFY_RED_MIN_HUE1 <= hue_roi <= CLASSIFY_RED_MAX_HUE1 or
                       CLASSIFY_RED_MIN_HUE2 <= hue_roi <= CLASSIFY_RED_MAX_HUE2) and
                      saturation_roi >= CLASSIFY_RED_MIN_SATURATION and
                      value_roi >= CLASSIFY_RED_MIN_VALUE)
    red_ratio = 0  # Khởi tạo

    if is_red_by_mean:
        temp_mask_red1 = cv2.inRange(hsv_image_roi,
                                     np.array(
                                         [CLASSIFY_RED_MIN_HUE1, CLASSIFY_RED_MIN_SATURATION, CLASSIFY_RED_MIN_VALUE]),
                                     np.array([CLASSIFY_RED_MAX_HUE1, 255, 255]))
        temp_mask_red2 = cv2.inRange(hsv_image_roi,
                                     np.array(
                                         [CLASSIFY_RED_MIN_HUE2, CLASSIFY_RED_MIN_SATURATION, CLASSIFY_RED_MIN_VALUE]),
                                     np.array([CLASSIFY_RED_MAX_HUE2, 255, 255]))
        mask_red_pixels_in_roi = cv2.bitwise_or(temp_mask_red1, temp_mask_red2)
        mask_red_pixels_in_roi = cv2.bitwise_and(mask_red_pixels_in_roi, mask_red_pixels_in_roi, mask=original_mask_roi)
        red_pixel_count = cv2.countNonZero(mask_red_pixels_in_roi)
        red_ratio = red_pixel_count / total_pixels_in_contour

        if red_ratio >= DOMINANT_PIXEL_RATIO_THRESHOLD:
            is_target_by_mean_check = (CLASSIFY_TARGET_MIN_HUE <= hue_roi <= CLASSIFY_TARGET_MAX_HUE and
                                       saturation_roi >= CLASSIFY_TARGET_MIN_SATURATION and
                                       value_roi >= CLASSIFY_TARGET_MIN_VALUE)
            if is_target_by_mean_check:
                temp_mask_target = cv2.inRange(hsv_image_roi,
                                               np.array([CLASSIFY_TARGET_MIN_HUE, CLASSIFY_TARGET_MIN_SATURATION,
                                                         CLASSIFY_TARGET_MIN_VALUE]),
                                               np.array([CLASSIFY_TARGET_MAX_HUE, 255, 255]))
                mask_target_pixels_in_roi_check = cv2.bitwise_and(temp_mask_target, temp_mask_target,
                                                                  mask=original_mask_roi)
                target_pixel_count_check = cv2.countNonZero(mask_target_pixels_in_roi_check)
                target_ratio_check = target_pixel_count_check / total_pixels_in_contour
                if red_ratio > target_ratio_check * 1.5: is_definitely_red = True
            else:
                is_definitely_red = True
        elif red_ratio >= SIGNIFICANT_PIXEL_RATIO_THRESHOLD and not \
                (CLASSIFY_TARGET_MIN_HUE <= hue_roi <= CLASSIFY_TARGET_MAX_HUE and \
                 saturation_roi >= CLASSIFY_TARGET_MIN_SATURATION and \
                 value_roi >= CLASSIFY_TARGET_MIN_VALUE):
            is_definitely_red = True

    if is_definitely_red:
        if SHOW_PROCESSING_STEPS: print(
            f"    ROI Classified as RED: Mean(H:{hue_roi:.1f} S:{saturation_roi:.1f} V:{value_roi:.1f}), RedRatio:{red_ratio:.2f}")
        return "red", hue_roi, saturation_roi, value_roi
    else:
        is_target_by_mean_info = (CLASSIFY_TARGET_MIN_HUE <= hue_roi <= CLASSIFY_TARGET_MAX_HUE and
                                  saturation_roi >= CLASSIFY_TARGET_MIN_SATURATION and
                                  value_roi >= CLASSIFY_TARGET_MIN_VALUE)

        temp_mask_target_color = cv2.inRange(hsv_image_roi,
                                             np.array([CLASSIFY_TARGET_MIN_HUE, CLASSIFY_TARGET_MIN_SATURATION,
                                                       CLASSIFY_TARGET_MIN_VALUE]),
                                             np.array([CLASSIFY_TARGET_MAX_HUE, 255, 255]))
        mask_target_color_pixels_in_roi = cv2.bitwise_and(temp_mask_target_color, temp_mask_target_color,
                                                          mask=original_mask_roi)
        target_color_pixel_count = cv2.countNonZero(mask_target_color_pixels_in_roi)
        target_color_ratio = target_color_pixel_count / total_pixels_in_contour

        if SHOW_PROCESSING_STEPS:
            print(
                f"    ROI Defaulted to TARGET_COLOR (Not Red): Mean(H:{hue_roi:.1f} S:{saturation_roi:.1f} V:{value_roi:.1f})")
            print(
                f"      TargetMatch (for info): {is_target_by_mean_info}, TargetRatio (for info): {target_color_ratio:.2f}")
            print(f"      RedMatch (for info): {is_red_by_mean}, RedRatio (for info): {red_ratio:.2f}")
        return "target_color", hue_roi, saturation_roi, value_roi


def detect_objects_contour_features(image_path):
    img = cv2.imread(image_path)
    if img is None: print(f"Lỗi đọc ảnh: '{image_path}'"); return None, None

    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)

    original_img_display = img.copy()
    height, width, _ = img.shape

    blur = cv2.medianBlur(img, MEDIAN_BLUR_KERNEL_SIZE)
    hsv_image = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    if SHOW_PROCESSING_STEPS:
        cv2.imshow("1. Blur", blur)
        cv2.imshow("2. HSV Image", hsv_image)

    mask_r1 = cv2.inRange(hsv_image, LOWER_RED_HSV1, UPPER_RED_HSV1)
    mask_r2 = cv2.inRange(hsv_image, LOWER_RED_HSV2, UPPER_RED_HSV2)
    mask_red_initial = cv2.bitwise_or(mask_r1, mask_r2)

    mask_target_color_initial = cv2.inRange(hsv_image, LOWER_TARGET_COLOR_HSV, UPPER_TARGET_COLOR_HSV)

    # QUAN TRỌNG: Kết hợp cả hai mask để tìm contour ban đầu
    combined_mask = cv2.bitwise_or(mask_red_initial, mask_target_color_initial)

    if SHOW_PROCESSING_STEPS:
        cv2.imshow("Debug Target Color Mask (Initial)", mask_target_color_initial)
        cv2.imshow("Debug Red Mask (Initial)", mask_red_initial)
        cv2.imshow("3. Combined Raw Mask (For Contours)", combined_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPENING_KERNEL_SIZE, OPENING_KERNEL_SIZE))
    opened_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=OPENING_ITERATIONS)
    if SHOW_PROCESSING_STEPS: cv2.imshow("4. Opened Mask", opened_mask)

    dilated_mask = cv2.dilate(opened_mask, None, iterations=DILATE_ITERATIONS_AFTER_OPENING)
    if SHOW_PROCESSING_STEPS: cv2.imshow("5. Dilated Mask (Post-Opening)", dilated_mask)

    contours, hierarchy = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    valid_contours_data = []

    if contours:
        if SHOW_PROCESSING_STEPS:
            feature_debug_img = original_img_display.copy()
            cv2.drawContours(feature_debug_img, contours, -1, (255, 0, 255), 1)

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < MIN_CONTOUR_AREA: continue

            x_b, y_b, w_b, h_b = cv2.boundingRect(cnt)
            if w_b == 0 or h_b == 0: continue
            aspect_ratio = float(w_b) / h_b
            if not (MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO): continue
            rect_area = w_b * h_b
            extent = float(area) / rect_area if rect_area > 0 else 0
            if extent < MIN_EXTENT: continue
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            if solidity < MIN_SOLIDITY: continue
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0
            if circularity < MIN_CIRCULARITY_RELAXED: continue

            if SHOW_PROCESSING_STEPS:
                print(
                    f"Contour {i} PASSED SHAPE. Area:{area:.0f}, AR:{aspect_ratio:.2f}, Ext:{extent:.2f}, Sol:{solidity:.2f}, Circ:{circularity:.2f}")
                cv2.drawContours(feature_debug_img, [cnt], -1, (0, 255, 255), 2)

            (x_mc, y_mc), radius_mc = cv2.minEnclosingCircle(cnt)
            center_2d_mc = (int(x_mc), int(y_mc))
            radius_px_mc = int(radius_mc)
            detected_color = "generic_shape"

            if ENABLE_POST_FILTER_COLOR_CLASSIFICATION:
                contour_roi_mask = np.zeros(dilated_mask.shape, dtype="uint8")
                cv2.drawContours(contour_roi_mask, [cnt], -1, 255, -1)
                if cv2.countNonZero(contour_roi_mask) == 0:  # Kiểm tra contour_roi_mask không rỗng
                    if SHOW_PROCESSING_STEPS: print(f"  Contour {i} has empty ROI mask. Skipping color classification.")
                    continue  # Bỏ qua nếu mask ROI rỗng

                hsv_image_roi_pixels = cv2.bitwise_and(hsv_image, hsv_image, mask=contour_roi_mask)

                detected_color, hue_roi, saturation_roi, value_roi = get_dominant_color_in_roi(hsv_image_roi_pixels,
                                                                                               contour_roi_mask)

                if SHOW_PROCESSING_STEPS:
                    print(
                        f"  Contour {i} at {center_2d_mc} ROUGH_R={radius_px_mc} CLASSIFIED as {detected_color} (Mean H:{hue_roi:.1f} S:{saturation_roi:.1f} V:{value_roi:.1f}).")

            if detected_color != "unknown":  # Chỉ thêm nếu phân loại thành công (red hoặc target_color)
                valid_contours_data.append({
                    'contour': cnt, 'area': area,
                    'center_2d': center_2d_mc,
                    'radius_px': radius_px_mc,
                    'type': detected_color,
                    'bbox': (x_b, y_b, w_b, h_b)
                })
            elif SHOW_PROCESSING_STEPS:
                print(
                    f"  Contour {i} at {center_2d_mc} ROUGH_R={radius_px_mc} resulted in UNKNOWN post-classification (should be rare).")
                cv2.drawContours(feature_debug_img, [cnt], -1, (0, 0, 0), 2)

        if SHOW_PROCESSING_STEPS: cv2.imshow("6. Feature & Color Filtering Debug", feature_debug_img)

    detected_objects_final = []
    if valid_contours_data:
        if ONLY_DETECT_LARGEST_CONTOUR:
            valid_contours_data.sort(key=lambda x: x['area'], reverse=True)
            if valid_contours_data: detected_objects_final.append(valid_contours_data[0])
        else:
            detected_objects_final = valid_contours_data

        for data in detected_objects_final:
            center_2d = data['center_2d']
            radius_px = data['radius_px']
            detected_color = data['type']

            draw_color = (128, 128, 128)
            if detected_color == "red":
                draw_color = (0, 0, 255)
            elif detected_color == "target_color":
                draw_color = (0, 255, 255)  # Vàng BGR cho target color (vàng)
            elif detected_color == "generic_shape":
                draw_color = (0, 128, 128)

            cv2.circle(original_img_display, center_2d, radius_px, draw_color, 2)
            cv2.circle(original_img_display, center_2d, 3, (255, 0, 0), -1)

            estimated_distance_m = 0
            if radius_px > 0: estimated_distance_m = RADIUS_TO_METERS_CALIB / radius_px
            image_center_x_px = width / 2.0
            current_x_mc = center_2d[0]
            estimated_offset_y_m = PIX_TO_METERS_CALIB * (image_center_x_px - current_x_mc)
            estimated_pos_z_m = ASSUMED_Z_HEIGHT_M

            object_info = {'center_2d': center_2d, 'radius_px': radius_px, 'type': detected_color,
                           'estimated_distance_m': round(estimated_distance_m, 3),
                           'estimated_offset_y_m': round(estimated_offset_y_m, 3),
                           'estimated_pos_z_m': estimated_pos_z_m}
            for i_final, item_final in enumerate(detected_objects_final):
                if item_final['center_2d'] == center_2d and item_final['radius_px'] == radius_px:
                    detected_objects_final[i_final].update(object_info);
                    break
            info_text = f"{detected_color.replace('_', ' ').capitalize()} D:{estimated_distance_m:.2f}m R:{radius_px}px"
            cv2.putText(original_img_display, info_text,
                        (center_2d[0] - radius_px, center_2d[1] - radius_px - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 2)

    if SHOW_PROCESSING_STEPS:
        print("\n--- Kết thúc xử lý ---")
        final_display_title = "7. Final Detections (Contour Features)"
        if ONLY_DETECT_LARGEST_CONTOUR and detected_objects_final: final_display_title += " (Largest)"
        cv2.imshow(final_display_title, original_img_display)
        key = cv2.waitKey(0)
        if key == ord('q'): cv2.destroyAllWindows()
    elif SHOW_FINAL_RESULT_IMAGE and original_img_display is not None:
        cv2.imshow("Final Result", original_img_display)
        cv2.waitKey(0);
        cv2.destroyAllWindows()

    return detected_objects_final, original_img_display


if __name__ == "__main__":
    print("Chương trình phát hiện đối tượng. Logic: Nếu không phải Đỏ (chín) thì là Target Color (xanh/vàng).")
    print("Ngưỡng HSV ban đầu cho màu mục tiêu (vàng):")
    print(f"  LOWER: H={LOWER_TARGET_COLOR_HSV[0]}, S={LOWER_TARGET_COLOR_HSV[1]}, V={LOWER_TARGET_COLOR_HSV[2]}")
    print(f"  UPPER: H={UPPER_TARGET_COLOR_HSV[0]}, S={UPPER_TARGET_COLOR_HSV[1]}, V={UPPER_TARGET_COLOR_HSV[2]}")
    print("-" * 30)
    input_image_path = "images_tomato/z6658190211590_1306174bc6061d722861c1844e08eed8.jpg"  # << THAY ĐỔI ĐƯỜNG DẪN

    results, result_image = detect_objects_contour_features(input_image_path)

    if results:
        print(f"\nĐã phát hiện {len(results)} đối tượng trong ảnh '{input_image_path}':")
        for i, obj in enumerate(results):
            type_display = obj['type'].replace('_', ' ').capitalize()
            if type_display == "Target color":
                type_display = "Vật thể vàng"  # Thay đổi tên hiển thị
            elif type_display == "Red":
                type_display = "Vật thể đỏ"

            print(f"  Đối tượng {i + 1}: Loại: {type_display}, Tâm: {obj['center_2d']}, R: {obj['radius_px']}px")
        if not SHOW_PROCESSING_STEPS and SHOW_FINAL_RESULT_IMAGE and result_image is not None:
            title = f"Phát hiện - {input_image_path}"
            if ONLY_DETECT_LARGEST_CONTOUR and len(results) > 0: title += " (Chỉ lớn nhất)"
            cv2.imshow(title, result_image);
            cv2.waitKey(0);
            cv2.destroyAllWindows()
    elif results is not None:
        print(f"Không phát hiện được đối tượng nào trong ảnh '{input_image_path}'.")
        if not SHOW_PROCESSING_STEPS and SHOW_FINAL_RESULT_IMAGE and result_image is not None:
            cv2.imshow(f"Không phát hiện - {input_image_path}", result_image);
            cv2.waitKey(0);
            cv2.destroyAllWindows()