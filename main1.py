import cv2
import numpy as np
import logging
from datetime import datetime

# --- Cấu hình Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'object_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

# --- Cấu hình Màu sắc và Xử lý Ảnh ---
LOWER_RED_HSV1 = np.array((0, 100, 80))
UPPER_RED_HSV1 = np.array((10, 255, 255))
LOWER_RED_HSV2 = np.array((160, 100, 80))
UPPER_RED_HSV2 = np.array((180, 255, 255))

LOWER_TARGET_COLOR_HSV = np.array((18, 50, 80))  # Target color (xanh/vàng)
UPPER_TARGET_COLOR_HSV = np.array((38, 255, 255))

MEDIAN_BLUR_KERNEL_SIZE = 5
OPENING_KERNEL_SIZE = 5
OPENING_ITERATIONS = 1
DILATE_ITERATIONS_AFTER_OPENING = 5

MIN_CONTOUR_AREA = 800 # Áp dụng cho ảnh đã resize 640x480
MIN_SOLIDITY = 0.70
MIN_ASPECT_RATIO = 0.55
MAX_ASPECT_RATIO = 1.45
MIN_EXTENT = 0.40
MIN_CIRCULARITY_RELAXED = 0.60

ENABLE_POST_FILTER_COLOR_CLASSIFICATION = True

CLASSIFY_RED_MIN_HUE1 = 0
CLASSIFY_RED_MAX_HUE1 = 12
CLASSIFY_RED_MIN_HUE2 = 158
CLASSIFY_RED_MAX_HUE2 = 180
CLASSIFY_RED_MIN_SATURATION = 80
CLASSIFY_RED_MIN_VALUE = 70

CLASSIFY_TARGET_MIN_HUE = 18
CLASSIFY_TARGET_MAX_HUE = 40
CLASSIFY_TARGET_MIN_SATURATION = 50
CLASSIFY_TARGET_MIN_VALUE = 80

DOMINANT_PIXEL_RATIO_THRESHOLD = 0.40
SIGNIFICANT_PIXEL_RATIO_THRESHOLD = 0.25

ONLY_DETECT_LARGEST_CONTOUR = True # Nếu False, sẽ xử lý và vẽ tất cả các contour hợp lệ

# Kích thước chuẩn để xử lý ảnh
RESIZED_WIDTH = 640
RESIZED_HEIGHT = 480


def get_dominant_color_in_roi(hsv_image_roi, original_mask_roi):
    total_pixels_in_contour = cv2.countNonZero(original_mask_roi)
    if total_pixels_in_contour == 0: return "unknown", 0, 0, 0

    mean_hsv_roi_val = cv2.mean(hsv_image_roi, mask=original_mask_roi)
    hue_roi, saturation_roi, value_roi = mean_hsv_roi_val[0], mean_hsv_roi_val[1], mean_hsv_roi_val[2]

    is_red_by_mean = ((CLASSIFY_RED_MIN_HUE1 <= hue_roi <= CLASSIFY_RED_MAX_HUE1 or
                       CLASSIFY_RED_MIN_HUE2 <= hue_roi <= CLASSIFY_RED_MAX_HUE2) and
                      saturation_roi >= CLASSIFY_RED_MIN_SATURATION and
                      value_roi >= CLASSIFY_RED_MIN_VALUE)
    temp_mask_red1 = cv2.inRange(hsv_image_roi,
                                 np.array([CLASSIFY_RED_MIN_HUE1, CLASSIFY_RED_MIN_SATURATION, CLASSIFY_RED_MIN_VALUE]),
                                 np.array([CLASSIFY_RED_MAX_HUE1, 255, 255]))
    temp_mask_red2 = cv2.inRange(hsv_image_roi,
                                 np.array([CLASSIFY_RED_MIN_HUE2, CLASSIFY_RED_MIN_SATURATION, CLASSIFY_RED_MIN_VALUE]),
                                 np.array([CLASSIFY_RED_MAX_HUE2, 255, 255]))
    mask_red_pixels_in_roi = cv2.bitwise_or(temp_mask_red1, temp_mask_red2)
    mask_red_pixels_in_roi = cv2.bitwise_and(mask_red_pixels_in_roi, mask_red_pixels_in_roi, mask=original_mask_roi)
    red_pixel_count = cv2.countNonZero(mask_red_pixels_in_roi)
    red_ratio = red_pixel_count / total_pixels_in_contour if total_pixels_in_contour > 0 else 0

    is_target_by_mean = (CLASSIFY_TARGET_MIN_HUE <= hue_roi <= CLASSIFY_TARGET_MAX_HUE and
                         saturation_roi >= CLASSIFY_TARGET_MIN_SATURATION and
                         value_roi >= CLASSIFY_TARGET_MIN_VALUE)
    temp_mask_target = cv2.inRange(hsv_image_roi, np.array(
        [CLASSIFY_TARGET_MIN_HUE, CLASSIFY_TARGET_MIN_SATURATION, CLASSIFY_TARGET_MIN_VALUE]),
                                   np.array([CLASSIFY_TARGET_MAX_HUE, 255, 255]))
    mask_target_pixels_in_roi = cv2.bitwise_and(temp_mask_target, temp_mask_target, mask=original_mask_roi)
    target_pixel_count = cv2.countNonZero(mask_target_pixels_in_roi)
    target_ratio = target_pixel_count / total_pixels_in_contour if total_pixels_in_contour > 0 else 0

    logging.debug(f"    ROI Analysis - Mean HSV: H={hue_roi:.1f} S={saturation_roi:.1f} V={value_roi:.1f}")
    logging.debug(f"      Red: is_mean_match={is_red_by_mean}, ratio={red_ratio:.2f}")
    logging.debug(f"      Target: is_mean_match={is_target_by_mean}, ratio={target_ratio:.2f}")

    if is_red_by_mean and red_ratio >= DOMINANT_PIXEL_RATIO_THRESHOLD:
        if is_target_by_mean and target_ratio >= SIGNIFICANT_PIXEL_RATIO_THRESHOLD and red_ratio <= target_ratio * 1.5: # Ambiguity
            pass # Fall through to check target dominance or other conditions
        else:
            logging.debug("    Decision: Red (dominant)")
            return "red", hue_roi, saturation_roi, value_roi
    if is_target_by_mean and target_ratio >= DOMINANT_PIXEL_RATIO_THRESHOLD:
        if is_red_by_mean and red_ratio >= SIGNIFICANT_PIXEL_RATIO_THRESHOLD and target_ratio <= red_ratio * 1.5: # Ambiguity
            pass # Fall through to check other conditions
        else:
            logging.debug("    Decision: Target Color (dominant)")
            return "target_color", hue_roi, saturation_roi, value_roi

    # Fallbacks for significant ratios if not dominant
    if is_red_by_mean and not is_target_by_mean and red_ratio >= SIGNIFICANT_PIXEL_RATIO_THRESHOLD:
        logging.debug("    Decision: Red (significant, no target mean)")
        return "red", hue_roi, saturation_roi, value_roi
    if is_target_by_mean and not is_red_by_mean and target_ratio >= SIGNIFICANT_PIXEL_RATIO_THRESHOLD:
        logging.debug("    Decision: Target Color (significant, no red mean)")
        return "target_color", hue_roi, saturation_roi, value_roi

    # Fallbacks if both means match but ratios are only significant (more complex ambiguity)
    if is_red_by_mean and red_ratio >= SIGNIFICANT_PIXEL_RATIO_THRESHOLD: # Red is significant
        if is_target_by_mean and target_ratio > red_ratio * 0.8 : # Target is also significant and comparable
             logging.debug("    Decision: Unknown (ambiguous fallback, both significant & comparable)")
             return "unknown", hue_roi, saturation_roi, value_roi
        logging.debug("    Decision: Red (fallback, significant red, target not comparable or not present)")
        return "red", hue_roi, saturation_roi, value_roi
    if is_target_by_mean and target_ratio >= SIGNIFICANT_PIXEL_RATIO_THRESHOLD: # Target is significant (red was not dominant or significant enough before)
        logging.debug("    Decision: Target Color (fallback, significant target)")
        return "target_color", hue_roi, saturation_roi, value_roi

    logging.debug("    Decision: Unknown (final fallback)")
    return "unknown", hue_roi, saturation_roi, value_roi


def process_image_frame(frame_to_process):
    if frame_to_process is None:
        logging.error("Frame đầu vào cho process_image_frame là None.")
        return [] # Trả về danh sách rỗng nếu không có frame

    # Resize ảnh về kích thước chuẩn để xử lý
    img_resized = cv2.resize(frame_to_process, (RESIZED_WIDTH, RESIZED_HEIGHT), interpolation=cv2.INTER_AREA)

    blur = cv2.medianBlur(img_resized, MEDIAN_BLUR_KERNEL_SIZE)
    hsv_image = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    mask_r1 = cv2.inRange(hsv_image, LOWER_RED_HSV1, UPPER_RED_HSV1)
    mask_r2 = cv2.inRange(hsv_image, LOWER_RED_HSV2, UPPER_RED_HSV2)
    mask_red_initial = cv2.bitwise_or(mask_r1, mask_r2)
    mask_target_color_initial = cv2.inRange(hsv_image, LOWER_TARGET_COLOR_HSV, UPPER_TARGET_COLOR_HSV)
    combined_mask = cv2.bitwise_or(mask_red_initial, mask_target_color_initial)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPENING_KERNEL_SIZE, OPENING_KERNEL_SIZE))
    opened_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=OPENING_ITERATIONS)
    dilated_mask = cv2.dilate(opened_mask, None, iterations=DILATE_ITERATIONS_AFTER_OPENING)

    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours_data = []

    if contours:
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
            if perimeter == 0: continue # Tránh lỗi chia cho 0
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity < MIN_CIRCULARITY_RELAXED: continue

            logging.debug(f"Contour {i} PASSED SHAPE (on resized). Area:{area:.0f}, AR:{aspect_ratio:.2f}, Circularity:{circularity:.2f}")
            (x_mc, y_mc), radius_mc = cv2.minEnclosingCircle(cnt)
            center_2d_resized = (int(x_mc), int(y_mc)) # Tọa độ trên ảnh đã resize
            radius_px_resized = int(radius_mc)      # Bán kính trên ảnh đã resize
            detected_color_type = "generic_shape" # Mặc định nếu không phân loại màu

            if ENABLE_POST_FILTER_COLOR_CLASSIFICATION:
                contour_roi_mask = np.zeros(dilated_mask.shape, dtype="uint8")
                cv2.drawContours(contour_roi_mask, [cnt], -1, 255, -1)
                if cv2.countNonZero(contour_roi_mask) == 0:
                    logging.debug(f"  Contour {i} SKIPPED color classification (empty ROI mask).")
                    continue
                hsv_image_roi_pixels = cv2.bitwise_and(hsv_image, hsv_image, mask=contour_roi_mask)
                detected_color_type, _, _, _ = get_dominant_color_in_roi(hsv_image_roi_pixels, contour_roi_mask)
                logging.debug(f"  Contour {i} CLASSIFIED as {detected_color_type}")

            if detected_color_type != "unknown":
                valid_contours_data.append({
                    'area_resized': area,
                    'center_2d_resized': center_2d_resized,
                    'radius_px_resized': radius_px_resized,
                    'type': detected_color_type
                })
            else:
                logging.debug(f"  Contour {i} resulted in UNKNOWN post-classification, not added to valid list.")

    detected_objects_info = []
    if valid_contours_data:
        if ONLY_DETECT_LARGEST_CONTOUR:
            valid_contours_data.sort(key=lambda x: x['area_resized'], reverse=True)
            if valid_contours_data: detected_objects_info.append(valid_contours_data[0])
        else:
            detected_objects_info = valid_contours_data # Lấy tất cả nếu không chỉ lấy lớn nhất

        for data in detected_objects_info: # Log thông tin đối tượng đã được chọn
            logging.info(f"Selected '{data['type']}' object (on resized) at {data['center_2d_resized']}, R={data['radius_px_resized']}")

    if not detected_objects_info:
        logging.info("No valid objects selected after processing (resized) frame.")

    return detected_objects_info # Trả về thông tin đối tượng với tọa độ trên ảnh resized


def process_image_from_file(image_path):
    logging.info(f"Đang xử lý ảnh từ: {image_path}")
    original_frame = cv2.imread(image_path)

    if original_frame is None:
        logging.error(f"Không thể đọc ảnh từ đường dẫn: {image_path}")
        return

    original_height, original_width = original_frame.shape[:2]
    # Tạo một bản sao của ảnh gốc để vẽ lên
    output_image_with_detections = original_frame.copy()

    try:
        # Xử lý ảnh (hàm process_image_frame sẽ resize ảnh này)
        # Kết quả trả về là thông tin đối tượng với tọa độ trên ảnh đã resize
        detected_objects = process_image_frame(original_frame.copy()) # Truyền bản sao để đảm bảo an toàn

        if detected_objects:
            logging.info(f"Tìm thấy {len(detected_objects)} đối tượng hợp lệ. Đang quy đổi tọa độ và vẽ lên ảnh gốc.")
            # Tính toán tỷ lệ co giãn
            scale_x = original_width / RESIZED_WIDTH
            scale_y = original_height / RESIZED_HEIGHT

            for obj_data in detected_objects:
                center_resized = obj_data['center_2d_resized']
                radius_resized = obj_data['radius_px_resized']
                obj_type = obj_data['type']

                # Quy đổi tọa độ và bán kính về kích thước ảnh gốc
                center_original = (int(center_resized[0] * scale_x), int(center_resized[1] * scale_y))
                # Scale bán kính, có thể dùng trung bình của scale_x và scale_y nếu tỷ lệ khung hình thay đổi nhiều
                # Hoặc chọn một trong hai (ví dụ scale_x) nếu muốn giữ hình dạng tròn trên ảnh gốc
                # Ở đây dùng trung bình để cân bằng
                radius_original = int(radius_resized * (scale_x + scale_y) / 2.0)
                radius_original = max(1, radius_original) # Đảm bảo bán kính ít nhất là 1

                draw_color = (128, 128, 128)  # Xám cho unknown/generic
                display_name = "Unknown"
                # Độ dày nét vẽ tỷ lệ với kích thước ảnh, đảm bảo ít nhất là 1
                line_thickness = max(1, int(min(original_width, original_height) / 300))


                if obj_type == "red":
                    draw_color = (0, 0, 255)  # Đỏ BGR
                    display_name = "Do"
                elif obj_type == "target_color":
                    draw_color = (0, 255, 255)  # Vàng BGR (cho target)
                    display_name = "Xanh/Vang"

                logging.info(f"  Vẽ đối tượng '{display_name}' trên ảnh gốc tại {center_original}, R_gốc={radius_original}")

                # Vẽ lên output_image_with_detections (bản sao của ảnh gốc)
                cv2.circle(output_image_with_detections, center_original, radius_original, draw_color, line_thickness)
                # Vẽ tâm vòng tròn nhỏ hơn, màu khác để dễ thấy
                center_dot_radius = max(1, int(line_thickness / 2) +1)
                cv2.circle(output_image_with_detections, center_original, center_dot_radius, (255, 0, 0), -1) # Tâm vòng tròn màu xanh dương

                info_text = f"{display_name} R:{radius_original}"
                # Kích thước font tỷ lệ, đảm bảo không quá nhỏ hoặc quá lớn
                font_scale = max(0.4, min(2.0, original_width / 1000.0))
                text_thickness = max(1, int(line_thickness / 1.5)) # Độ dày text có thể mỏng hơn đường viền

                # Tính toán vị trí đặt text để không bị che khuất hoặc ra ngoài ảnh
                text_size, _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
                text_w, text_h = text_size

                text_x = center_original[0] - radius_original
                text_y = center_original[1] - radius_original - text_h # Dịch lên trên một chút so với đỉnh vòng tròn

                # Điều chỉnh nếu text bị ra ngoài lề
                if text_y < text_h: # Nếu quá gần mép trên
                    text_y = center_original[1] + radius_original + text_h + int(5 * font_scale) # Dịch xuống dưới
                if text_x < 0: # Nếu quá gần mép trái
                    text_x = 5
                if text_x + text_w > original_width: # Nếu quá gần mép phải
                    text_x = original_width - text_w - 5


                cv2.putText(output_image_with_detections, info_text,
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, draw_color, text_thickness)
        else:
            logging.info("Không có đối tượng nào được phát hiện để vẽ lên ảnh gốc.")

        cv2.imshow("Object Detection (on Original)", output_image_with_detections)
        logging.info("Hiển thị ảnh kết quả. Nhấn phím bất kỳ để đóng.")

        output_filename = f"result_on_original_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        try:
            cv2.imwrite(output_filename, output_image_with_detections)
            logging.info(f"Đã lưu ảnh kết quả: {output_filename}")
        except Exception as e_save:
            logging.error(f"Không thể lưu ảnh kết quả {output_filename}: {e_save}")


        cv2.waitKey(0)

    except Exception as e:
        logging.error(f"Lỗi khi xử lý ảnh {image_path}: {e}", exc_info=True)
    finally:
        cv2.destroyAllWindows()
        logging.info("Đã đóng tất cả cửa sổ hiển thị.")


if __name__ == "__main__":
    logging.info("Khởi chạy chương trình phát hiện đối tượng từ file ảnh (vẽ trên ảnh gốc).")
    # Để debug chi tiết, đổi logging.INFO ở đầu thành logging.DEBUG
    # logging.getLogger().setLevel(logging.DEBUG) # Bỏ comment dòng này để debug

    input_image_path = "images_tomato/img_2.png" # Đường dẫn tới file ảnh của bạn

    # Tạo một ảnh test nếu file input_image_path không tồn tại
    try:
        test_img_read = cv2.imread(input_image_path)
        if test_img_read is None:
            logging.info(f"File {input_image_path} không tồn tại. Đang tạo ảnh test...")
            # Ảnh test với kích thước lớn hơn kích thước xử lý chuẩn
            test_img_height, test_img_width = 960, 1280 # Ví dụ: 1280x960
            test_img = np.zeros((test_img_height, test_img_width, 3), dtype=np.uint8)
            cv2.rectangle(test_img, (0,0), (test_img_width-1, test_img_height-1), (50,50,50), -1) # Nền xám tối

            # Vật thể màu đỏ
            cv2.circle(test_img, (int(test_img_width * 0.25), int(test_img_height * 0.3)),
                       int(min(test_img_width, test_img_height) * 0.08), (0, 0, 255), -1) # BGR: Đỏ
            # Vật thể màu mục tiêu (vàng)
            cv2.circle(test_img, (int(test_img_width * 0.6), int(test_img_height * 0.6)),
                       int(min(test_img_width, test_img_height) * 0.1), (0, 220, 220), -1) # BGR: Vàng
            # Một vật thể nhỏ khác màu đỏ để test ONLY_DETECT_LARGEST_CONTOUR
            cv2.circle(test_img, (int(test_img_width * 0.8), int(test_img_height * 0.25)),
                       int(min(test_img_width, test_img_height) * 0.04), (0, 0, 255), -1) # BGR: Đỏ nhỏ
            # Một vật thể màu xanh lá cây (không phải target, không phải đỏ)
            cv2.circle(test_img, (int(test_img_width * 0.4), int(test_img_height * 0.8)),
                       int(min(test_img_width, test_img_height) * 0.06), (0, 255, 0), -1) # BGR: Xanh lá


            cv2.imwrite(input_image_path, test_img)
            logging.info(f"Đã tạo ảnh test tại: {input_image_path} ({test_img_width}x{test_img_height})")
    except Exception as e_create_test:
        logging.warning(f"Không thể tạo ảnh test tự động: {e_create_test}. Hãy đảm bảo file '{input_image_path}' tồn tại hoặc bạn có quyền ghi vào thư mục hiện tại.")


    process_image_from_file(input_image_path)
    logging.info("Kết thúc chương trình.")