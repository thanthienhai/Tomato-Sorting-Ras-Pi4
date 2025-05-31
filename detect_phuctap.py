import cv2
import numpy as np

# --- Cấu hình (CẦN TINH CHỈNH KỸ LƯỠNG) ---
LOWER_RED_HSV1 = np.array((0, 120, 70))
UPPER_RED_HSV1 = np.array((15, 255, 255))
LOWER_RED_HSV2 = np.array((165, 120, 70))
UPPER_RED_HSV2 = np.array((180, 255, 255))

LOWER_GREEN_HSV = np.array((30, 30, 30))
UPPER_GREEN_HSV = np.array((95, 255, 255))

MEDIAN_BLUR_KERNEL_SIZE = 5
OPENING_KERNEL_SIZE = 5
OPENING_ITERATIONS = 1
DILATE_ITERATIONS_AFTER_OPENING = 7

MIN_CONTOUR_AREA = 1000
MIN_SOLIDITY = 0.80
MIN_ASPECT_RATIO = 0.65
MAX_ASPECT_RATIO = 1.35
MIN_EXTENT = 0.5
MIN_CIRCULARITY_RELAXED = 0.55

ENABLE_POST_FILTER_COLOR_CLASSIFICATION = True
# --- Ngưỡng phân loại màu (ĐÃ NỚI LỎNG HƠN - CẦN KIỂM TRA LẠI VỚI OUTPUT CỦA BẠN) ---
CLASSIFY_RED_MIN_HUE1 = 0;
CLASSIFY_RED_MAX_HUE1 = 18  # Nới HUE đỏ
CLASSIFY_RED_MIN_HUE2 = 160;
CLASSIFY_RED_MAX_HUE2 = 180
CLASSIFY_RED_MIN_SATURATION = 90  # Giảm S đáng kể cho đỏ
CLASSIFY_RED_MIN_VALUE = 50  # Giảm V cho đỏ

CLASSIFY_GREEN_MIN_HUE = 20;
CLASSIFY_GREEN_MAX_HUE = 100  # Nới HUE xanh
CLASSIFY_GREEN_MIN_SATURATION = 20  # Giảm S đáng kể cho xanh
CLASSIFY_GREEN_MIN_VALUE = 20  # Giảm V cho xanh

# Ngưỡng tỷ lệ pixel trong ROI để xác định màu chủ đạo (ĐÃ GIẢM)
DOMINANT_PIXEL_RATIO_THRESHOLD = 0.35  # Chỉ cần 35% pixel là màu đó (và mean color khớp)
SIGNIFICANT_PIXEL_RATIO_THRESHOLD = 0.20  # Ít nhất 20% pixel thuộc màu đó để được xem xét

RADIUS_AT_KNOWN_DISTANCE_PX = 26;
KNOWN_DISTANCE_M = 0.4
RADIUS_TO_METERS_CALIB = RADIUS_AT_KNOWN_DISTANCE_PX * KNOWN_DISTANCE_M
TOMATO_REAL_WIDTH_M = 0.05
PIX_TO_METERS_CALIB = TOMATO_REAL_WIDTH_M / (RADIUS_AT_KNOWN_DISTANCE_PX * 2)
ASSUMED_Z_HEIGHT_M = 0.35

SHOW_PROCESSING_STEPS = True
SHOW_FINAL_RESULT_IMAGE = True
ONLY_DETECT_LARGEST_CONTOUR = True


def get_dominant_color_in_roi(hsv_image_roi, original_mask_roi):
    # Tạo mask cho màu đỏ trong ROI dựa trên ngưỡng CLASSIFY
    # (Lưu ý: hsv_image_roi là phần ảnh HSV đã được crop theo contour_roi_mask)
    temp_mask_red1 = cv2.inRange(hsv_image_roi,
                                 np.array([CLASSIFY_RED_MIN_HUE1, CLASSIFY_RED_MIN_SATURATION, CLASSIFY_RED_MIN_VALUE]),
                                 np.array([CLASSIFY_RED_MAX_HUE1, 255, 255]))
    temp_mask_red2 = cv2.inRange(hsv_image_roi,
                                 np.array([CLASSIFY_RED_MIN_HUE2, CLASSIFY_RED_MIN_SATURATION, CLASSIFY_RED_MIN_VALUE]),
                                 np.array([CLASSIFY_RED_MAX_HUE2, 255, 255]))
    mask_red_pixels_in_roi = cv2.bitwise_or(temp_mask_red1, temp_mask_red2)
    mask_red_pixels_in_roi = cv2.bitwise_and(mask_red_pixels_in_roi, mask_red_pixels_in_roi, mask=original_mask_roi)

    # Tạo mask cho màu xanh trong ROI dựa trên ngưỡng CLASSIFY
    temp_mask_green = cv2.inRange(hsv_image_roi,
                                  np.array([CLASSIFY_GREEN_MIN_HUE, CLASSIFY_GREEN_MIN_SATURATION,
                                            CLASSIFY_GREEN_MIN_VALUE]),
                                  np.array([CLASSIFY_GREEN_MAX_HUE, 255, 255]))
    mask_green_pixels_in_roi = cv2.bitwise_and(temp_mask_green, temp_mask_green, mask=original_mask_roi)

    red_pixel_count = cv2.countNonZero(mask_red_pixels_in_roi)
    green_pixel_count = cv2.countNonZero(mask_green_pixels_in_roi)
    total_pixels_in_contour = cv2.countNonZero(original_mask_roi)

    if total_pixels_in_contour == 0:
        return "unknown", 0, 0, 0

    mean_hsv_roi_val = cv2.mean(hsv_image_roi, mask=original_mask_roi)
    hue_roi, saturation_roi, value_roi = mean_hsv_roi_val[0], mean_hsv_roi_val[1], mean_hsv_roi_val[2]

    # Điều kiện màu trung bình khớp (sử dụng lại ngưỡng CLASSIFY)
    is_red_by_mean = ((CLASSIFY_RED_MIN_HUE1 <= hue_roi <= CLASSIFY_RED_MAX_HUE1 or
                       CLASSIFY_RED_MIN_HUE2 <= hue_roi <= CLASSIFY_RED_MAX_HUE2) and
                      saturation_roi >= CLASSIFY_RED_MIN_SATURATION and
                      value_roi >= CLASSIFY_RED_MIN_VALUE)

    is_green_by_mean = (CLASSIFY_GREEN_MIN_HUE <= hue_roi <= CLASSIFY_GREEN_MAX_HUE and
                        saturation_roi >= CLASSIFY_GREEN_MIN_SATURATION and
                        value_roi >= CLASSIFY_GREEN_MIN_VALUE)

    # Tỷ lệ pixel thực tế
    red_ratio = red_pixel_count / total_pixels_in_contour if total_pixels_in_contour > 0 else 0
    green_ratio = green_pixel_count / total_pixels_in_contour if total_pixels_in_contour > 0 else 0

    # --- Logic phân loại màu cải tiến ---
    # Ưu tiên 1: Màu chiếm đa số pixel VÀ màu trung bình cũng khớp mạnh mẽ
    if is_red_by_mean and red_ratio >= DOMINANT_PIXEL_RATIO_THRESHOLD:
        # Nếu có vẻ xanh đáng kể, kiểm tra xem đỏ có trội hơn hẳn không
        if is_green_by_mean and green_ratio >= SIGNIFICANT_PIXEL_RATIO_THRESHOLD:
            if red_ratio > green_ratio * 1.5:  # Đỏ phải trội hơn xanh ít nhất 1.5 lần
                return "red", hue_roi, saturation_roi, value_roi
        else:  # Không có vẻ xanh đáng kể
            return "red", hue_roi, saturation_roi, value_roi

    if is_green_by_mean and green_ratio >= DOMINANT_PIXEL_RATIO_THRESHOLD:
        if is_red_by_mean and red_ratio >= SIGNIFICANT_PIXEL_RATIO_THRESHOLD:
            if green_ratio > red_ratio * 1.5:  # Xanh phải trội hơn đỏ ít nhất 1.5 lần
                return "green", hue_roi, saturation_roi, value_roi
        else:  # Không có vẻ đỏ đáng kể
            return "green", hue_roi, saturation_roi, value_roi

    # Ưu tiên 2: Nếu không có màu nào chiếm đa số rõ ràng, nhưng màu trung bình chỉ khớp MỘT màu
    if is_red_by_mean and not is_green_by_mean and red_ratio >= SIGNIFICANT_PIXEL_RATIO_THRESHOLD:  # Có vẻ là đỏ và không có vẻ là xanh
        return "red", hue_roi, saturation_roi, value_roi

    if is_green_by_mean and not is_red_by_mean and green_ratio >= SIGNIFICANT_PIXEL_RATIO_THRESHOLD:  # Có vẻ là xanh và không có vẻ là đỏ
        return "green", hue_roi, saturation_roi, value_roi

    # Fallback cuối cùng: Nếu cả hai màu đều có vẻ hiện diện theo màu trung bình (is_red_by_mean AND is_green_by_mean)
    # hoặc không màu nào có đủ pixel đáng kể, thì ưu tiên màu nào có nhiều pixel hơn một chút
    # (ít tin cậy hơn)
    if is_red_by_mean and is_green_by_mean:  # Cả hai màu đều có vẻ khớp mean
        if red_ratio > green_ratio + 0.1:  # Đỏ nhiều hơn xanh một chút
            return "red", hue_roi, saturation_roi, value_roi
        if green_ratio > red_ratio + 0.1:  # Xanh nhiều hơn đỏ một chút
            return "green", hue_roi, saturation_roi, value_roi

    if SHOW_PROCESSING_STEPS:
        print(
            f"    ROI Color Analysis: Mean(H:{hue_roi:.1f} S:{saturation_roi:.1f} V:{value_roi:.1f}), RedMatch:{is_red_by_mean}, GreenMatch:{is_green_by_mean}, RedRatio:{red_ratio:.2f}, GreenRatio:{green_ratio:.2f}")

    return "unknown", hue_roi, saturation_roi, value_roi


def detect_tomatoes_contour_features(image_path):
    img = cv2.imread(image_path)
    if img is None: print(f"Lỗi đọc ảnh: '{image_path}'"); return None, None

    original_img_display = img.copy()
    height, width, _ = img.shape

    blur = cv2.medianBlur(img, MEDIAN_BLUR_KERNEL_SIZE)
    hsv_image = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)  # Dùng cho phân tích màu cuối

    if SHOW_PROCESSING_STEPS:
        cv2.imshow("1. Blur", blur)
        cv2.imshow("2. HSV Image", hsv_image)

    mask_r1 = cv2.inRange(hsv_image, LOWER_RED_HSV1, UPPER_RED_HSV1)
    mask_r2 = cv2.inRange(hsv_image, LOWER_RED_HSV2, UPPER_RED_HSV2)
    mask_red = cv2.bitwise_or(mask_r1, mask_r2)
    mask_green = cv2.inRange(hsv_image, LOWER_GREEN_HSV, UPPER_GREEN_HSV)
    combined_mask = cv2.bitwise_or(mask_red, mask_green)
    if SHOW_PROCESSING_STEPS: cv2.imshow("3. Combined Raw Mask", combined_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPENING_KERNEL_SIZE, OPENING_KERNEL_SIZE))
    opened_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=OPENING_ITERATIONS)
    if SHOW_PROCESSING_STEPS: cv2.imshow("4. Opened Mask", opened_mask)

    dilated_mask = cv2.dilate(opened_mask, None, iterations=DILATE_ITERATIONS_AFTER_OPENING)
    if SHOW_PROCESSING_STEPS: cv2.imshow("5. Dilated Mask (Post-Opening)", dilated_mask)

    contours, hierarchy = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

    valid_contours_data = []  # Các contour qua lọc hình dạng và màu sắc

    if contours:
        if SHOW_PROCESSING_STEPS:
            feature_debug_img = original_img_display.copy()  # Dùng ảnh gốc để vẽ debug
            cv2.drawContours(feature_debug_img, contours, -1, (255, 0, 255), 1)  # Vẽ tất cả contour ban đầu (tím)

        for i, cnt in enumerate(contours):
            # --- Lọc hình dạng ---
            area = cv2.contourArea(cnt)
            if area < MIN_CONTOUR_AREA:
                if SHOW_PROCESSING_STEPS: print(f"Contour {i} REJ by AREA ({area:.0f} < {MIN_CONTOUR_AREA})")
                continue

            x_b, y_b, w_b, h_b = cv2.boundingRect(cnt)
            if w_b == 0 or h_b == 0: continue

            aspect_ratio = float(w_b) / h_b
            if not (MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO):
                if SHOW_PROCESSING_STEPS: print(f"Contour {i} REJ by ASPECT_RATIO ({aspect_ratio:.2f})")
                continue

            rect_area = w_b * h_b
            extent = float(area) / rect_area if rect_area > 0 else 0
            if extent < MIN_EXTENT:
                if SHOW_PROCESSING_STEPS: print(f"Contour {i} REJ by EXTENT ({extent:.2f} < {MIN_EXTENT})")
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            if solidity < MIN_SOLIDITY:
                if SHOW_PROCESSING_STEPS: print(f"Contour {i} REJ by SOLIDITY ({solidity:.2f} < {MIN_SOLIDITY})")
                continue

            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0
            if circularity < MIN_CIRCULARITY_RELAXED:
                if SHOW_PROCESSING_STEPS: print(
                    f"Contour {i} REJ by CIRCULARITY ({circularity:.2f} < {MIN_CIRCULARITY_RELAXED})")
                continue

            if SHOW_PROCESSING_STEPS:
                print(
                    f"Contour {i} PASSED SHAPE. Area:{area:.0f}, AR:{aspect_ratio:.2f}, Ext:{extent:.2f}, Sol:{solidity:.2f}, Circ:{circularity:.2f}")
                cv2.drawContours(feature_debug_img, [cnt], -1, (0, 255, 255), 2)  # Vàng cho contour qua lọc hình dạng

            # --- Phân loại màu cho contour đã qua lọc hình dạng ---
            (x_mc, y_mc), radius_mc = cv2.minEnclosingCircle(cnt)  # Vẫn cần để ước tính khoảng cách
            center_2d_mc = (int(x_mc), int(y_mc))
            radius_px_mc = int(radius_mc)
            detected_color = "unknown"

            if ENABLE_POST_FILTER_COLOR_CLASSIFICATION:
                contour_roi_mask = np.zeros(dilated_mask.shape, dtype="uint8")
                cv2.drawContours(contour_roi_mask, [cnt], -1, 255, -1)
                # Lấy phần ảnh HSV gốc (đã blur) tương ứng với ROI của contour
                hsv_image_roi_pixels = cv2.bitwise_and(hsv_image, hsv_image, mask=contour_roi_mask)

                detected_color, hue_roi, saturation_roi, value_roi = get_dominant_color_in_roi(hsv_image_roi_pixels,
                                                                                               contour_roi_mask)

                if detected_color == "unknown":
                    if SHOW_PROCESSING_STEPS: print(
                        f"  Contour {i} at {center_2d_mc} ROUGH_R={radius_px_mc} FAILED COLOR CLASSIFICATION.")
                    cv2.drawContours(feature_debug_img, [cnt], -1, (0, 0, 0), 2)  # Đen nếu fail color
                    continue  # Bỏ qua nếu không phân loại được màu
                elif SHOW_PROCESSING_STEPS:
                    print(f"  Contour {i} at {center_2d_mc} ROUGH_R={radius_px_mc} CLASSIFIED as {detected_color}.")

            valid_contours_data.append({
                'contour': cnt, 'area': area,
                'center_2d': center_2d_mc,
                'radius_px': radius_px_mc,  # Bán kính từ minEnclosingCircle để ước lượng khoảng cách
                'type': detected_color if ENABLE_POST_FILTER_COLOR_CLASSIFICATION else "generic_shape",
                'bbox': (x_b, y_b, w_b, h_b)
            })
        if SHOW_PROCESSING_STEPS: cv2.imshow("6. Feature & Color Filtering Debug", feature_debug_img)

    # Xử lý cuối cùng (lớn nhất hoặc tất cả)
    detected_tomatoes_final = []
    if valid_contours_data:
        if ONLY_DETECT_LARGEST_CONTOUR:
            valid_contours_data.sort(key=lambda x: x['area'], reverse=True)
            if valid_contours_data: detected_tomatoes_final.append(valid_contours_data[0])
        else:
            detected_tomatoes_final = valid_contours_data

        for data in detected_tomatoes_final:
            center_2d = data['center_2d']
            radius_px = data['radius_px']
            detected_color = data['type']

            draw_color = (0, 0, 0)
            if detected_color == "red":
                draw_color = (0, 0, 255)
            elif detected_color == "green":
                draw_color = (0, 255, 0)
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

            tomato_info = {
                'center_2d': center_2d, 'radius_px': radius_px, 'type': detected_color,
                'estimated_distance_m': round(estimated_distance_m, 3),
                'estimated_offset_y_m': round(estimated_offset_y_m, 3),
                'estimated_pos_z_m': estimated_pos_z_m
            }
            # Cập nhật lại thông tin trong list (vì detected_tomatoes_final có thể chỉ có 1 phần tử)
            for i_final, item_final in enumerate(detected_tomatoes_final):
                if item_final['center_2d'] == center_2d and item_final['radius_px'] == radius_px:
                    detected_tomatoes_final[i_final].update(tomato_info)
                    break

            info_text = f"{detected_color.capitalize()} D:{estimated_distance_m:.2f}m R:{radius_px}px"
            cv2.putText(original_img_display, info_text,
                        (center_2d[0] - radius_px, center_2d[1] - radius_px - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 2)

    if SHOW_PROCESSING_STEPS:
        print("\n--- Kết thúc xử lý ---")
        final_display_title = "7. Final Detections (Contour Features)"
        if ONLY_DETECT_LARGEST_CONTOUR and detected_tomatoes_final: final_display_title += " (Largest)"
        cv2.imshow(final_display_title, original_img_display)
        if cv2.waitKey(0) & 0xFF == ord('q'): pass
        cv2.destroyAllWindows()

    return detected_tomatoes_final, original_img_display


if __name__ == "__main__":
    input_image_path = "images_tomato/bad3_jpeg.rf.98ce1eb4500e2dc373adb1daa9a469af.jpg"  # << THAY ĐỔI ĐƯỜNG DẪN

    results, result_image = detect_tomatoes_contour_features(input_image_path)

    if results:
        print(f"\nĐã phát hiện {len(results)} cà chua (Contour Features) trong ảnh '{input_image_path}':")
        for i, tomato in enumerate(results):
            print(
                f"  Cà chua {i + 1}: Loại: {tomato['type'].capitalize()}, Tâm: {tomato['center_2d']}, R: {tomato['radius_px']}px, Dist: {tomato.get('estimated_distance_m', 'N/A')}m")

        if SHOW_FINAL_RESULT_IMAGE and result_image is not None:
            title = f"Phát hiện Contour Feat. - {input_image_path}"
            if ONLY_DETECT_LARGEST_CONTOUR and len(results) > 0:
                title += " (Chỉ lớn nhất)"
            cv2.imshow(title, result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    elif results is not None:
        print(f"Không phát hiện được cà chua nào (Contour Features) trong ảnh '{input_image_path}'.")
        if SHOW_FINAL_RESULT_IMAGE and result_image is not None:
            cv2.imshow(f"Không phát hiện Contour Feat. - {input_image_path}", result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()