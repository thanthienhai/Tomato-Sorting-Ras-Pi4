import cv2
import numpy as np
import time
import serial
import logging
from datetime import datetime

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,  # Đặt INFO để thấy cả debug nếu cần, hoặc DEBUG để thấy mặc định
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # In ra console
        logging.FileHandler(f'tomato_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')  # Lưu vào file
    ]
)

# Cấu hình UART (điều chỉnh nếu cần)
UART_PORT = '/dev/ttyUSB0'
UART_BAUDRATE = 115200
UART_TIMEOUT = 1

# --- Cấu hình Màu sắc và Xử lý Ảnh ---
# Ngưỡng HSV ban đầu cho MÀU ĐỎ
LOWER_RED_HSV1 = np.array((0, 100, 80))  # S_min, V_min có thể cần điều chỉnh tùy ánh sáng
UPPER_RED_HSV1 = np.array((10, 255, 255))
LOWER_RED_HSV2 = np.array((160, 100, 80))  # S_min, V_min có thể cần điều chỉnh tùy ánh sáng
UPPER_RED_HSV2 = np.array((180, 255, 255))

# Ngưỡng HSV ban đầu cho MÀU VÀNG (TARGET_COLOR)
LOWER_TARGET_COLOR_HSV = np.array((18, 50, 80))  # H_min, S_min, V_min
UPPER_TARGET_COLOR_HSV = np.array((38, 255, 255))  # H_max

# Thông số xử lý hình thái học
MEDIAN_BLUR_KERNEL_SIZE = 5
OPENING_KERNEL_SIZE = 5
OPENING_ITERATIONS = 1
DILATE_ITERATIONS_AFTER_OPENING = 7

# Ngưỡng lọc contour theo hình dạng
MIN_CONTOUR_AREA = 800  # Giảm nếu vật thể nhỏ
MIN_SOLIDITY = 0.70  # Độ "đặc" của contour
MIN_ASPECT_RATIO = 0.55  # Tỷ lệ rộng/cao
MAX_ASPECT_RATIO = 1.45
MIN_EXTENT = 0.40  # Tỷ lệ diện tích contour / diện tích bounding box
MIN_CIRCULARITY_RELAXED = 0.60  # Độ tròn (gần 1 là tròn hoàn hảo)

ENABLE_POST_FILTER_COLOR_CLASSIFICATION = True

# Ngưỡng phân loại màu chi tiết cho MÀU ĐỎ
CLASSIFY_RED_MIN_HUE1 = 0
CLASSIFY_RED_MAX_HUE1 = 12
CLASSIFY_RED_MIN_HUE2 = 158
CLASSIFY_RED_MAX_HUE2 = 180
CLASSIFY_RED_MIN_SATURATION = 70  # Yêu cầu Saturation cao cho màu đỏ
CLASSIFY_RED_MIN_VALUE = 60  # Yêu cầu Value tương đối cao cho màu đỏ

# Ngưỡng phân loại màu chi tiết cho MÀU VÀNG (TARGET_COLOR)
CLASSIFY_TARGET_MIN_HUE = 18
CLASSIFY_TARGET_MAX_HUE = 40
CLASSIFY_TARGET_MIN_SATURATION = 50
CLASSIFY_TARGET_MIN_VALUE = 80

# Ngưỡng tỷ lệ pixel để xác định màu chủ đạo
DOMINANT_PIXEL_RATIO_THRESHOLD = 0.35
SIGNIFICANT_PIXEL_RATIO_THRESHOLD = 0.20

# Các hằng số hiệu chuẩn (giữ nguyên hoặc điều chỉnh nếu bạn có hệ thống đo lường)
RADIUS_AT_KNOWN_DISTANCE_PX = 26
KNOWN_DISTANCE_M = 0.4
RADIUS_TO_METERS_CALIB = RADIUS_AT_KNOWN_DISTANCE_PX * KNOWN_DISTANCE_M
OBJECT_REAL_WIDTH_M = 0.05
PIX_TO_METERS_CALIB = OBJECT_REAL_WIDTH_M / (RADIUS_AT_KNOWN_DISTANCE_PX * 2)
ASSUMED_Z_HEIGHT_M = 0.35

# Cấu hình hiển thị
SHOW_PROCESSING_STEPS = True  # Đặt False để tăng tốc độ khi không cần debug từng bước
SHOW_FINAL_RESULT_IMAGE = True
ONLY_DETECT_LARGEST_CONTOUR = True  # Chỉ xử lý contour lớn nhất


def init_uart():
    try:
        ser = serial.Serial(UART_PORT, UART_BAUDRATE, timeout=UART_TIMEOUT)
        if ser.is_open:
            logging.info(f"Đã kết nối UART thành công - Port: {UART_PORT}, Baudrate: {UART_BAUDRATE}")
            return ser
        logging.error(f"Không thể mở cổng UART: {UART_PORT}")
        return None
    except serial.SerialException as e:
        logging.error(f"Lỗi kết nối UART: {e}")
        return None


def send_tomato_data(ser, color_code, center_x, center_y, radius):
    if ser and ser.is_open:
        data = f"{color_code},{center_x},{center_y},{radius}\n"
        try:
            ser.write(data.encode())
            logging.debug(f"Đã gửi dữ liệu UART: {data.strip()}")
        except serial.SerialException as e:
            logging.error(f"Lỗi gửi dữ liệu UART: {e}")


def get_dominant_color_in_roi(hsv_image_roi, original_mask_roi):
    total_pixels_in_contour = cv2.countNonZero(original_mask_roi)
    if total_pixels_in_contour == 0:
        return "unknown", 0, 0, 0

    mean_hsv_roi_val = cv2.mean(hsv_image_roi, mask=original_mask_roi)
    hue_roi, saturation_roi, value_roi = mean_hsv_roi_val[0], mean_hsv_roi_val[1], mean_hsv_roi_val[2]

    # Tính toán các thông số cho MÀU ĐỎ
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

    # Tính toán các thông số cho MÀU TARGET (VÀNG)
    is_target_by_mean = (CLASSIFY_TARGET_MIN_HUE <= hue_roi <= CLASSIFY_TARGET_MAX_HUE and
                         saturation_roi >= CLASSIFY_TARGET_MIN_SATURATION and
                         value_roi >= CLASSIFY_TARGET_MIN_VALUE)

    temp_mask_target = cv2.inRange(hsv_image_roi,
                                   np.array([CLASSIFY_TARGET_MIN_HUE, CLASSIFY_TARGET_MIN_SATURATION,
                                             CLASSIFY_TARGET_MIN_VALUE]),
                                   np.array([CLASSIFY_TARGET_MAX_HUE, 255, 255]))
    mask_target_pixels_in_roi = cv2.bitwise_and(temp_mask_target, temp_mask_target, mask=original_mask_roi)
    target_pixel_count = cv2.countNonZero(mask_target_pixels_in_roi)
    target_ratio = target_pixel_count / total_pixels_in_contour if total_pixels_in_contour > 0 else 0

    # Gỡ lỗi chi tiết
    logging.debug(f"    ROI Analysis - Mean HSV: H={hue_roi:.1f} S={saturation_roi:.1f} V={value_roi:.1f}")
    logging.debug(
        f"      Red: is_mean_match={is_red_by_mean}, ratio={red_ratio:.2f} (Thresh S_min={CLASSIFY_RED_MIN_SATURATION}, V_min={CLASSIFY_RED_MIN_VALUE})")
    logging.debug(
        f"      Target: is_mean_match={is_target_by_mean}, ratio={target_ratio:.2f} (Thresh S_min={CLASSIFY_TARGET_MIN_SATURATION}, V_min={CLASSIFY_TARGET_MIN_VALUE})")

    # Logic phân loại:
    # Ưu tiên 1: Nếu chắc chắn là đỏ và không bị nhầm lẫn nhiều với target
    if is_red_by_mean and red_ratio >= DOMINANT_PIXEL_RATIO_THRESHOLD:
        if is_target_by_mean and target_ratio >= SIGNIFICANT_PIXEL_RATIO_THRESHOLD:
            if red_ratio > target_ratio * 1.5:
                logging.debug("    Decision: Red (dominant, stronger than target)")
                return "red", hue_roi, saturation_roi, value_roi
        else:
            logging.debug("    Decision: Red (dominant, no significant target match)")
            return "red", hue_roi, saturation_roi, value_roi

    # Ưu tiên 2: Nếu chắc chắn là target và không bị nhầm lẫn nhiều với đỏ
    if is_target_by_mean and target_ratio >= DOMINANT_PIXEL_RATIO_THRESHOLD:
        if is_red_by_mean and red_ratio >= SIGNIFICANT_PIXEL_RATIO_THRESHOLD:
            if target_ratio > red_ratio * 1.5:
                logging.debug("    Decision: Target Color (dominant, stronger than red)")
                return "target_color", hue_roi, saturation_roi, value_roi
        else:
            logging.debug("    Decision: Target Color (dominant, no significant red match)")
            return "target_color", hue_roi, saturation_roi, value_roi

    # Ưu tiên 3: Nếu màu trung bình chỉ khớp MỘT MÀU và tỷ lệ pixel đáng kể
    if is_red_by_mean and not is_target_by_mean and red_ratio >= SIGNIFICANT_PIXEL_RATIO_THRESHOLD:
        logging.debug("    Decision: Red (mean match, significant ratio, no target mean match)")
        return "red", hue_roi, saturation_roi, value_roi

    if is_target_by_mean and not is_red_by_mean and target_ratio >= SIGNIFICANT_PIXEL_RATIO_THRESHOLD:
        logging.debug("    Decision: Target Color (mean match, significant ratio, no red mean match)")
        return "target_color", hue_roi, saturation_roi, value_roi

    # Logic mặc định: Nếu không phải Đỏ (chín) thì là Target Color (xanh/vàng)
    # Điều kiện này được áp dụng nếu các điều kiện trên không đủ mạnh để quyết định.
    # Tuy nhiên, chúng ta vẫn muốn ưu tiên Đỏ nếu nó có dấu hiệu mạnh hơn một chút.
    if is_red_by_mean and red_ratio > target_ratio + 0.05:  # Đỏ có vẻ trội hơn một chút
        logging.debug("    Decision: Red (fallback, slightly stronger red signal)")
        return "red", hue_roi, saturation_roi, value_roi
    # Nếu không thì mặc định là target_color (nếu có bất kỳ dấu hiệu nào của target hoặc không rõ ràng)
    # Hoặc nếu is_target_by_mean và target_ratio có vẻ ổn
    if is_target_by_mean and target_ratio >= (
            SIGNIFICANT_PIXEL_RATIO_THRESHOLD * 0.5):  # Yêu cầu target có ít nhất 1/2 ngưỡng significant
        logging.debug("    Decision: Target Color (defaulted as not clearly red, or target has some presence)")
        return "target_color", hue_roi, saturation_roi, value_roi

    # Cuối cùng, nếu không thể xác định
    logging.debug("    Decision: Unknown (no strong conditions met, and target presence very weak or ambiguous)")
    return "unknown", hue_roi, saturation_roi, value_roi


def process_video_frame(frame, ser_instance):  # Thêm ser_instance để truyền vào
    if frame is None:
        return None, None

    img = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
    original_img_display = img.copy()
    height, width, _ = img.shape

    blur = cv2.medianBlur(img, MEDIAN_BLUR_KERNEL_SIZE)
    hsv_image = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    if SHOW_PROCESSING_STEPS: cv2.imshow("1. Blur", blur)
    if SHOW_PROCESSING_STEPS: cv2.imshow("2. HSV Image", hsv_image)

    mask_r1 = cv2.inRange(hsv_image, LOWER_RED_HSV1, UPPER_RED_HSV1)
    mask_r2 = cv2.inRange(hsv_image, LOWER_RED_HSV2, UPPER_RED_HSV2)
    mask_red_initial = cv2.bitwise_or(mask_r1, mask_r2)

    mask_target_color_initial = cv2.inRange(hsv_image, LOWER_TARGET_COLOR_HSV, UPPER_TARGET_COLOR_HSV)
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

    contours, hierarchy = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours_data = []
    feature_debug_img = None  # Khởi tạo
    if SHOW_PROCESSING_STEPS: feature_debug_img = original_img_display.copy()

    if contours:
        if SHOW_PROCESSING_STEPS and feature_debug_img is not None:
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

            if SHOW_PROCESSING_STEPS and feature_debug_img is not None:
                logging.debug(
                    f"Contour {i} PASSED SHAPE. Area:{area:.0f}, AR:{aspect_ratio:.2f}, Ext:{extent:.2f}, Sol:{solidity:.2f}, Circ:{circularity:.2f}")
                cv2.drawContours(feature_debug_img, [cnt], -1, (0, 255, 255), 2)

            (x_mc, y_mc), radius_mc = cv2.minEnclosingCircle(cnt)
            center_2d_mc = (int(x_mc), int(y_mc))
            radius_px_mc = int(radius_mc)
            detected_color = "generic_shape"

            if ENABLE_POST_FILTER_COLOR_CLASSIFICATION:
                contour_roi_mask = np.zeros(dilated_mask.shape, dtype="uint8")
                cv2.drawContours(contour_roi_mask, [cnt], -1, 255, -1)
                if cv2.countNonZero(contour_roi_mask) == 0:
                    logging.debug(f"  Contour {i} has empty ROI mask. Skipping color classification.")
                    continue

                hsv_image_roi_pixels = cv2.bitwise_and(hsv_image, hsv_image, mask=contour_roi_mask)
                detected_color, hue_roi, saturation_roi, value_roi = get_dominant_color_in_roi(hsv_image_roi_pixels,
                                                                                               contour_roi_mask)

                logging.debug(
                    f"  Contour {i} at {center_2d_mc} ROUGH_R={radius_px_mc} CLASSIFIED as {detected_color} (Mean H:{hue_roi:.1f} S:{saturation_roi:.1f} V:{value_roi:.1f}).")

            if detected_color != "unknown":
                valid_contours_data.append({
                    'contour': cnt, 'area': area,
                    'center_2d': center_2d_mc,
                    'radius_px': radius_px_mc,
                    'type': detected_color,
                    'bbox': (x_b, y_b, w_b, h_b)
                })
            elif SHOW_PROCESSING_STEPS and feature_debug_img is not None:
                logging.debug(
                    f"  Contour {i} at {center_2d_mc} ROUGH_R={radius_px_mc} resulted in UNKNOWN post-classification.")
                cv2.drawContours(feature_debug_img, [cnt], -1, (0, 0, 0), 2)  # Vẽ màu đen nếu unknown

    if SHOW_PROCESSING_STEPS and feature_debug_img is not None: cv2.imshow("6. Feature & Color Filtering Debug",
                                                                           feature_debug_img)

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
            detected_color_type = data['type']  # Đổi tên biến để tránh nhầm lẫn

            draw_color = (128, 128, 128)  # Xám mặc định
            if detected_color_type == "red":
                draw_color = (0, 0, 255)  # Đỏ
            elif detected_color_type == "target_color":
                draw_color = (0, 255, 255)  # Vàng
            elif detected_color_type == "generic_shape":
                draw_color = (0, 128, 128)  # Teal

            cv2.circle(original_img_display, center_2d, radius_px, draw_color, 2)
            cv2.circle(original_img_display, center_2d, 3, (255, 0, 0), -1)  # Tâm màu xanh

            estimated_distance_m = 0
            if radius_px > 0: estimated_distance_m = RADIUS_TO_METERS_CALIB / radius_px
            # ... (Tính toán offset và pos_z giữ nguyên) ...

            object_info = data.copy()  # Sao chép dữ liệu gốc
            object_info.update({  # Cập nhật thêm thông tin ước lượng
                'estimated_distance_m': round(estimated_distance_m, 3),
                # 'estimated_offset_y_m': round(estimated_offset_y_m, 3), # Bỏ nếu không dùng
                # 'estimated_pos_z_m': estimated_pos_z_m # Bỏ nếu không dùng
            })
            # Cập nhật lại trong list detected_objects_final (quan trọng nếu ONLY_DETECT_LARGEST_CONTOUR=False)
            for i_final, item_final in enumerate(detected_objects_final):
                if item_final['contour'] is data['contour']:  # So sánh contour object
                    detected_objects_final[i_final] = object_info
                    break

            info_text = f"{detected_color_type.replace('_', ' ').capitalize()} R:{radius_px}"
            if 'estimated_distance_m' in object_info: info_text += f" D:{object_info['estimated_distance_m']:.2f}m"

            cv2.putText(original_img_display, info_text,
                        (center_2d[0] - radius_px, center_2d[1] - radius_px - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 2)

            # Gửi dữ liệu UART ngay sau khi xử lý mỗi đối tượng trong final list
            if ser_instance and ser_instance.is_open:
                color_code = 1 if detected_color_type == "red" else 0  # 1 cho đỏ, 0 cho target (vàng)
                send_tomato_data(ser_instance, color_code, center_2d[0], center_2d[1], radius_px)

    if SHOW_PROCESSING_STEPS:
        final_display_title = "7. Final Detections"
        if ONLY_DETECT_LARGEST_CONTOUR and detected_objects_final: final_display_title += " (Largest)"
        cv2.imshow(final_display_title, original_img_display)
        # Không gọi waitKey(0) ở đây nữa nếu chạy trong vòng lặp video

    return detected_objects_final, original_img_display


def run_camera_detection(camera_id=0, target_fps=10, use_uart=True, max_retries=3):
    retry_count = 0
    ser_instance = None  # Khởi tạo UART instance

    while retry_count <= max_retries:
        if use_uart and (ser_instance is None or not ser_instance.is_open):
            ser_instance = init_uart()
            if ser_instance is None and use_uart:  # Nếu không khởi tạo được UART và vẫn muốn dùng
                logging.warning("Không thể khởi tạo UART, sẽ thử lại sau.")
                time.sleep(5)  # Chờ trước khi thử lại vòng lặp chính
                retry_count += 1
                continue

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logging.error(f"Không thể mở camera {camera_id}")
            retry_count += 1
            if retry_count <= max_retries:
                logging.info(f"Đang thử lại sau 5 giây... (Lần {retry_count}/{max_retries})")
                if ser_instance and ser_instance.is_open: ser_instance.close()  # Đóng UART trước khi thử lại
                ser_instance = None
                time.sleep(5)
                continue
            else:
                logging.error("Đã vượt quá số lần thử lại cho phép để mở camera.")
                return

        logging.info(f"Bắt đầu nhận diện từ camera {camera_id}. Nhấn 'q' để thoát.")
        frame_count = 0
        start_time = time.time()

        consecutive_read_errors = 0
        max_consecutive_read_errors = 20  # Số lỗi đọc frame liên tiếp tối đa trước khi reset camera

        while True:
            ret, frame = cap.read()
            if not ret:
                consecutive_read_errors += 1
                logging.warning(
                    f"Không đọc được frame từ camera (Lỗi liên tiếp: {consecutive_read_errors}/{max_consecutive_read_errors})")
                if consecutive_read_errors >= max_consecutive_read_errors:
                    logging.error("Quá nhiều lỗi đọc frame, khởi động lại kết nối camera...")
                    break  # Thoát vòng lặp trong để thử lại kết nối camera
                time.sleep(0.05)  # Đợi một chút
                continue

            consecutive_read_errors = 0  # Reset khi đọc thành công
            frame_count += 1

            try:
                # Truyền ser_instance vào process_video_frame
                results, result_image = process_video_frame(frame, ser_instance if use_uart else None)

                if result_image is not None:
                    # Hiển thị FPS (tính toán đơn giản)
                    if frame_count % target_fps == 0:  # Cập nhật FPS mỗi giây (ước lượng)
                        elapsed_time_fps = time.time() - start_time
                        current_fps_val = frame_count / elapsed_time_fps if elapsed_time_fps > 0 else 0
                        logging.info(f"FPS hiện tại (ước lượng): {current_fps_val:.1f}")

                    cv2.imshow("Tomato Detection", result_image)

                # Gửi dữ liệu UART đã được chuyển vào trong process_video_frame

            except Exception as e:
                logging.error(f"Lỗi khi xử lý frame {frame_count}: {e}", exc_info=True)  # Thêm exc_info để có traceback

            key = cv2.waitKey(1) & 0xFF  # Chờ 1ms
            if key == ord('q'):
                logging.info("Người dùng yêu cầu thoát.")
                if cap.isOpened(): cap.release()
                if ser_instance and ser_instance.is_open: ser_instance.close()
                cv2.destroyAllWindows()
                return  # Thoát hoàn toàn
            elif key == ord('s') and result_image is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"detection_{timestamp}.jpg"
                cv2.imwrite(filename, result_image)
                logging.info(f"Đã lưu ảnh: {filename}")

        # Nếu thoát vòng lặp trong do lỗi camera
        if cap.isOpened(): cap.release()
        # Không đóng UART ở đây nếu vẫn còn retry
        retry_count += 1
        if retry_count <= max_retries:
            logging.info(f"Khởi động lại camera sau lỗi. Lần thử {retry_count}/{max_retries}")
            if ser_instance and ser_instance.is_open: ser_instance.close()  # Đóng UART trước khi thử lại
            ser_instance = None
            time.sleep(2)  # Chờ một chút trước khi thử lại
        else:
            logging.error("Đã vượt quá số lần thử lại cho camera.")

    # Dọn dẹp cuối cùng nếu vòng lặp retry kết thúc
    if 'cap' in locals() and cap.isOpened(): cap.release()
    if ser_instance and ser_instance.is_open: ser_instance.close()
    cv2.destroyAllWindows()
    total_elapsed_time = time.time() - start_time
    logging.info(f"Kết thúc phiên làm việc. Đã xử lý {frame_count} frames.")
    if frame_count > 0 and total_elapsed_time > 0:
        logging.info(f"FPS trung bình cuối cùng: {frame_count / total_elapsed_time:.1f}")


if __name__ == "__main__":
    logging.info("Chương trình phát hiện đối tượng từ camera.")
    # Để debug chi tiết hơn, có thể đặt logging level thành DEBUG
    # logging.getLogger().setLevel(logging.DEBUG)

    # Ví dụ chạy với camera ID 0, FPS mục tiêu 10, bật UART
    run_camera_detection(camera_id=0, target_fps=10, use_uart=True, max_retries=3)