import cv2
import numpy as np
import time
import serial  # Bỏ nếu không dùng UART
import logging
from datetime import datetime

# --- Cấu hình Logging ---
logging.basicConfig(
    level=logging.INFO,  # Đổi thành logging.DEBUG để xem log chi tiết hơn
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'object_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

# --- Cấu hình UART ---
USE_UART = True  # Đặt True nếu muốn dùng UART
UART_PORT = '/dev/ttyUSB0'  # Thay đổi nếu cần
UART_BAUDRATE = 115200
UART_TIMEOUT = 1
ser_instance_global = None

# --- Cấu hình Màu sắc và Xử lý Ảnh ---
LOWER_RED_HSV1 = np.array((0, 100, 80))
UPPER_RED_HSV1 = np.array((10, 255, 255))
LOWER_RED_HSV2 = np.array((160, 100, 80))
UPPER_RED_HSV2 = np.array((180, 255, 255))

LOWER_TARGET_COLOR_HSV = np.array((18, 50, 80))  # Target color (xanh/vàng)
UPPER_TARGET_COLOR_HSV = np.array((38, 255, 255))

# --- THÊM MÀU GREEN-2 ---
# RGB(69, 141, 80) -> HSV khoảng (70, 122, 141)
LOWER_GREEN2_HSV = np.array((65, 80, 70))  # Hơi nới lỏng S và V
UPPER_GREEN2_HSV = np.array((85, 255, 220))  # Hơi nới lỏng S và V
# --------------------------

MEDIAN_BLUR_KERNEL_SIZE = 5
OPENING_KERNEL_SIZE = 5
OPENING_ITERATIONS = 1
DILATE_ITERATIONS_AFTER_OPENING = 5

MIN_CONTOUR_AREA = 800
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

# --- THÊM THAM SỐ PHÂN LOẠI CHO GREEN-2 ---
CLASSIFY_GREEN2_MIN_HUE = 65
CLASSIFY_GREEN2_MAX_HUE = 85
CLASSIFY_GREEN2_MIN_SATURATION = 80
CLASSIFY_GREEN2_MIN_VALUE = 70
# -----------------------------------------

DOMINANT_PIXEL_RATIO_THRESHOLD = 0.40
SIGNIFICANT_PIXEL_RATIO_THRESHOLD = 0.25

SHOW_FINAL_RESULT_IMAGE_ONLY = True
ONLY_DETECT_LARGEST_CONTOUR = True


def init_uart_if_needed():
    global ser_instance_global
    if USE_UART and (ser_instance_global is None or not ser_instance_global.is_open):
        try:
            ser_instance_global = serial.Serial(UART_PORT, UART_BAUDRATE, timeout=UART_TIMEOUT)
            if ser_instance_global.is_open:
                logging.info(f"Đã kết nối UART: {UART_PORT}")
            else:
                logging.error(f"Không thể mở UART: {UART_PORT}")
                ser_instance_global = None
        except serial.SerialException as e:
            logging.error(f"Lỗi UART: {e}")
            ser_instance_global = None
    return ser_instance_global


def send_object_data_uart(color_code, center_x, center_y, radius):
    global ser_instance_global
    if not USE_UART: return

    if ser_instance_global and ser_instance_global.is_open:
        data = f"{color_code},{center_x},{center_y},{radius}\n"
        try:
            ser_instance_global.write(data.encode())
            if color_code == 0 and center_x == 0 and center_y == 0 and radius == 0:
                logging.debug(f"Sent UART (No Detection): {data.strip()}")
            else:
                logging.info(f"Sent UART (Detection): {data.strip()}")
        except serial.SerialException as e:
            logging.error(f"Lỗi gửi UART: {e}")
    else:
        if not ser_instance_global:
            logging.warning("UART send: ser_instance_global is None.")
        elif not ser_instance_global.is_open:
            logging.warning("UART send: ser_instance_global is not open.")


# CẬP NHẬT HÀM get_dominant_color_in_roi ĐỂ XỬ LÝ 3 MÀU
def get_dominant_color_in_roi(hsv_image_roi, original_mask_roi):
    total_pixels_in_contour = cv2.countNonZero(original_mask_roi)
    if total_pixels_in_contour == 0: return "unknown", 0, 0, 0

    mean_hsv_roi_val = cv2.mean(hsv_image_roi, mask=original_mask_roi)
    hue_roi, saturation_roi, value_roi = mean_hsv_roi_val[0], mean_hsv_roi_val[1], mean_hsv_roi_val[2]

    # Phân tích màu Đỏ
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

    # Phân tích màu Target (Xanh/Vàng)
    is_target_by_mean = (CLASSIFY_TARGET_MIN_HUE <= hue_roi <= CLASSIFY_TARGET_MAX_HUE and
                         saturation_roi >= CLASSIFY_TARGET_MIN_SATURATION and
                         value_roi >= CLASSIFY_TARGET_MIN_VALUE)
    temp_mask_target = cv2.inRange(hsv_image_roi, np.array(
        [CLASSIFY_TARGET_MIN_HUE, CLASSIFY_TARGET_MIN_SATURATION, CLASSIFY_TARGET_MIN_VALUE]),
                                   np.array([CLASSIFY_TARGET_MAX_HUE, 255, 255]))
    mask_target_pixels_in_roi = cv2.bitwise_and(temp_mask_target, temp_mask_target, mask=original_mask_roi)
    target_pixel_count = cv2.countNonZero(mask_target_pixels_in_roi)
    target_ratio = target_pixel_count / total_pixels_in_contour if total_pixels_in_contour > 0 else 0

    # --- THÊM PHÂN TÍCH MÀU GREEN-2 ---
    is_green2_by_mean = (CLASSIFY_GREEN2_MIN_HUE <= hue_roi <= CLASSIFY_GREEN2_MAX_HUE and
                         saturation_roi >= CLASSIFY_GREEN2_MIN_SATURATION and
                         value_roi >= CLASSIFY_GREEN2_MIN_VALUE)
    temp_mask_green2 = cv2.inRange(hsv_image_roi, np.array(
        [CLASSIFY_GREEN2_MIN_HUE, CLASSIFY_GREEN2_MIN_SATURATION, CLASSIFY_GREEN2_MIN_VALUE]),
                                   np.array([CLASSIFY_GREEN2_MAX_HUE, 255, 255]))
    mask_green2_pixels_in_roi = cv2.bitwise_and(temp_mask_green2, temp_mask_green2, mask=original_mask_roi)
    green2_pixel_count = cv2.countNonZero(mask_green2_pixels_in_roi)
    green2_ratio = green2_pixel_count / total_pixels_in_contour if total_pixels_in_contour > 0 else 0
    # ----------------------------------

    logging.debug(f"    ROI Analysis - Mean HSV: H={hue_roi:.1f} S={saturation_roi:.1f} V={value_roi:.1f}")
    logging.debug(f"      Red: is_mean_match={is_red_by_mean}, ratio={red_ratio:.2f}")
    logging.debug(f"      Target: is_mean_match={is_target_by_mean}, ratio={target_ratio:.2f}")
    logging.debug(f"      Green-2: is_mean_match={is_green2_by_mean}, ratio={green2_ratio:.2f}")  # Thêm log

    # Logic quyết định màu chủ đạo (đơn giản hóa: ưu tiên màu có tỷ lệ cao nhất nếu đạt ngưỡng)
    # Tạo danh sách các ứng viên màu
    candidates = []
    if is_red_by_mean:
        candidates.append({"name": "red", "ratio": red_ratio})
    if is_target_by_mean:
        candidates.append({"name": "target_color", "ratio": target_ratio})
    if is_green2_by_mean:  # Thêm green-2
        candidates.append({"name": "green-2", "ratio": green2_ratio})

    # Sắp xếp các ứng viên theo tỷ lệ giảm dần
    candidates.sort(key=lambda x: x["ratio"], reverse=True)

    # Kiểm tra ứng viên hàng đầu
    if candidates and candidates[0]["ratio"] >= DOMINANT_PIXEL_RATIO_THRESHOLD:
        logging.debug(f"    Decision: {candidates[0]['name']} (dominant with ratio {candidates[0]['ratio']:.2f})")
        return candidates[0]["name"], hue_roi, saturation_roi, value_roi

    # Nếu không có màu nào dominant, kiểm tra màu significant (ít chặt chẽ hơn)
    # (Logic này có thể cần điều chỉnh thêm để tránh xung đột nếu nhiều màu significant)
    if candidates:  # Kiểm tra theo thứ tự đã sắp xếp
        for cand in candidates:
            if cand["ratio"] >= SIGNIFICANT_PIXEL_RATIO_THRESHOLD:
                logging.debug(
                    f"    Decision: {cand['name']} (significant with ratio {cand['ratio']:.2f}, no clear dominant)")
                return cand["name"], hue_roi, saturation_roi, value_roi

    # Logic cũ (có thể giữ lại một phần nếu logic trên chưa đủ tốt, nhưng sẽ phức tạp)
    # Cân nhắc: Logic cũ khá phức tạp để mở rộng cho 3+ màu. Logic mới ở trên đơn giản hơn.
    # Nếu bạn muốn giữ logic cũ và mở rộng, bạn sẽ cần nhiều câu lệnh if/elif lồng nhau hơn.
    # Ví dụ:
    # if is_red_by_mean and red_ratio >= DOMINANT_PIXEL_RATIO_THRESHOLD:
    #   is_target_also_strong = is_target_by_mean and target_ratio >= SIGNIFICANT_PIXEL_RATIO_THRESHOLD
    #   is_green2_also_strong = is_green2_by_mean and green2_ratio >= SIGNIFICANT_PIXEL_RATIO_THRESHOLD
    #   if is_target_also_strong and target_ratio > red_ratio * 0.8: # Nếu target cũng mạnh
    #       pass # Bỏ qua, để target kiểm tra
    #   elif is_green2_also_strong and green2_ratio > red_ratio * 0.8: # Nếu green2 cũng mạnh
    #       pass # Bỏ qua, để green2 kiểm tra
    #   else:
    #       return "red", hue_roi, saturation_roi, value_roi
    # ... và tương tự cho target_color và green-2, rất phức tạp.

    logging.debug("    Decision: Unknown (final fallback)")
    return "unknown", hue_roi, saturation_roi, value_roi


def process_video_frame(frame):
    if frame is None: return None, None

    img = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
    original_img_display = img.copy()

    blur = cv2.medianBlur(img, MEDIAN_BLUR_KERNEL_SIZE)
    hsv_image = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    mask_r1 = cv2.inRange(hsv_image, LOWER_RED_HSV1, UPPER_RED_HSV1)
    mask_r2 = cv2.inRange(hsv_image, LOWER_RED_HSV2, UPPER_RED_HSV2)
    mask_red_initial = cv2.bitwise_or(mask_r1, mask_r2)
    mask_target_color_initial = cv2.inRange(hsv_image, LOWER_TARGET_COLOR_HSV, UPPER_TARGET_COLOR_HSV)

    # --- THÊM MASK CHO GREEN-2 ---
    mask_green2_initial = cv2.inRange(hsv_image, LOWER_GREEN2_HSV, UPPER_GREEN2_HSV)
    # ----------------------------

    # Kết hợp các mask lại
    combined_mask_rg = cv2.bitwise_or(mask_red_initial, mask_target_color_initial)
    combined_mask = cv2.bitwise_or(combined_mask_rg, mask_green2_initial)  # Thêm mask green-2

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
            circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0
            if circularity < MIN_CIRCULARITY_RELAXED: continue

            logging.debug(f"Contour {i} PASSED SHAPE. Area:{area:.0f}, AR:{aspect_ratio:.2f}")
            (x_mc, y_mc), radius_mc = cv2.minEnclosingCircle(cnt)
            center_2d_mc = (int(x_mc), int(y_mc))
            radius_px_mc = int(radius_mc)
            detected_color_type = "generic_shape"  # Mặc định

            if ENABLE_POST_FILTER_COLOR_CLASSIFICATION:
                contour_roi_mask = np.zeros(dilated_mask.shape, dtype="uint8")
                cv2.drawContours(contour_roi_mask, [cnt], -1, 255, -1)
                if cv2.countNonZero(contour_roi_mask) == 0:
                    logging.debug(f"  Contour {i} has zero pixels in ROI mask, skipping color classification.")
                    continue
                hsv_image_roi_pixels = cv2.bitwise_and(hsv_image, hsv_image, mask=contour_roi_mask)
                # Hàm get_dominant_color_in_roi đã được cập nhật
                detected_color_type, _, _, _ = get_dominant_color_in_roi(hsv_image_roi_pixels, contour_roi_mask)
                logging.debug(f"  Contour {i} CLASSIFIED as {detected_color_type}")

            if detected_color_type != "unknown" and detected_color_type != "generic_shape":
                valid_contours_data.append({
                    'contour': cnt, 'area': area, 'center_2d': center_2d_mc,
                    'radius_px': radius_px_mc, 'type': detected_color_type
                })
            else:
                logging.debug(f"  Contour {i} resulted in UNKNOWN/GENERIC post-classification, not added.")

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
            obj_type = data['type']

            draw_color = (128, 128, 128)  # Xám cho unknown/generic
            color_code_to_send = 0
            display_name = "Unknown"

            if obj_type == "red":
                draw_color = (0, 0, 255)  # Đỏ (BGR)
                color_code_to_send = 2
                display_name = "Do (2)"
            elif obj_type == "target_color":
                draw_color = (0, 255, 255)  # Vàng (BGR)
                color_code_to_send = 1
                display_name = "Xanh/Vang (1)"
            # --- THÊM XỬ LÝ CHO GREEN-2 ---
            elif obj_type == "green-2":
                draw_color = (0, 255, 0)  # Xanh lá cây (BGR)
                color_code_to_send = 3  # Mã UART là 3
                display_name = "Xanh-2 (3)"  # Tên hiển thị
            # --------------------------------

            cv2.circle(original_img_display, center_2d, radius_px, draw_color, 2)
            cv2.circle(original_img_display, center_2d, 3, (255, 0, 0), -1)

            info_text = f"{display_name} R:{radius_px}"
            cv2.putText(original_img_display, info_text,
                        (center_2d[0] - radius_px, center_2d[1] - radius_px - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 2)

            if USE_UART:
                send_object_data_uart(color_code_to_send, center_2d[0], center_2d[1], radius_px)

    if not detected_objects_final and USE_UART:
        send_object_data_uart(0, 0, 0, 0)

    return detected_objects_final, original_img_display


def run_camera_detection(camera_id=0, target_fps=10, max_retries=3):
    global ser_instance_global
    retry_count = 0

    if USE_UART: init_uart_if_needed()

    while retry_count <= max_retries:
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logging.error(f"Không thể mở camera {camera_id}")
            retry_count += 1
            if retry_count <= max_retries:
                logging.info(f"Thử lại sau 5s... ({retry_count}/{max_retries})")
                time.sleep(5)
                continue
            else:
                logging.error("Vượt quá số lần thử lại camera.")
                return

        logging.info(f"Bắt đầu nhận diện từ camera {camera_id}. Nhấn 'q' để thoát.")
        frame_count_interval = 0
        start_time_fps_calc = time.time()
        consecutive_read_errors = 0
        max_consecutive_read_errors = 30

        while True:
            ret, frame = cap.read()
            if not ret:
                consecutive_read_errors += 1
                logging.warning(f"Không đọc được frame (Lỗi liên tiếp: {consecutive_read_errors})")
                if consecutive_read_errors >= max_consecutive_read_errors:
                    logging.error("Quá nhiều lỗi đọc frame, reset camera.")
                    break
                time.sleep(0.05)
                continue
            consecutive_read_errors = 0

            try:
                results, result_image = process_video_frame(frame)

                if result_image is not None and SHOW_FINAL_RESULT_IMAGE_ONLY:
                    cv2.imshow("Object Detection", result_image)

                frame_count_interval += 1
                if frame_count_interval >= target_fps * 2:  # Log FPS mỗi 2s
                    elapsed_time = time.time() - start_time_fps_calc
                    if elapsed_time > 0:
                        current_fps_val = frame_count_interval / elapsed_time
                        logging.info(
                            f"FPS (ước lượng): {current_fps_val:.1f} ({frame_count_interval} frames / {elapsed_time:.2f}s)")
                    frame_count_interval = 0
                    start_time_fps_calc = time.time()

            except Exception as e:
                logging.error(f"Lỗi xử lý frame: {e}", exc_info=True)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logging.info("Người dùng thoát.")
                if cap.isOpened(): cap.release()
                if USE_UART and ser_instance_global and ser_instance_global.is_open: ser_instance_global.close()
                cv2.destroyAllWindows()
                return
            elif key == ord('s') and result_image is not None:
                fname = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
                cv2.imwrite(fname, result_image)
                logging.info(f"Đã lưu ảnh: {fname}")

        if cap.isOpened(): cap.release()
        retry_count += 1
        if retry_count <= max_retries:
            logging.info(f"Reset camera. Thử lại {retry_count}/{max_retries}")
            time.sleep(2)
        else:
            logging.error("Vượt quá số lần thử lại camera.")

    if USE_UART and ser_instance_global and ser_instance_global.is_open: ser_instance_global.close()
    cv2.destroyAllWindows()
    logging.info("Kết thúc chương trình.")


if __name__ == "__main__":
    logging.info("Khởi chạy chương trình phát hiện đối tượng.")
    # Để debug chi tiết, đổi logging.INFO ở đầu thành logging.DEBUG
    # logging.getLogger().setLevel(logging.DEBUG) # Hoặc dùng dòng này

    run_camera_detection(camera_id=0, target_fps=10, max_retries=3)