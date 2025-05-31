import cv2
import numpy as np
import serial
import time
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'tomato_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

UART_PORT = '/dev/ttyUSB0'
UART_BAUDRATE = 115200

def init_uart():
    """Khởi tạo kết nối UART"""
    try:
        ser = serial.Serial(UART_PORT, UART_BAUDRATE, timeout=1)
        logging.info(f"Đã kết nối UART thành công - Port: {UART_PORT}, Baudrate: {UART_BAUDRATE}")
        return ser
    except serial.SerialException as e:
        logging.error(f"Lỗi kết nối UART: {e}")
        return None

def send_tomato_data(ser, color, center_x, center_y, radius):
    """Gửi dữ liệu cà chua qua UART
    Format: "color,x,y,radius\n"
    """
    if ser and ser.is_open:
        data = f"{color},{center_x},{center_y},{radius}\n"
        try:
            ser.write(data.encode())
            logging.debug(f"Đã gửi dữ liệu UART: {data.strip()}")
        except serial.SerialException as e:
            logging.error(f"Lỗi gửi dữ liệu UART: {e}")

# --- Cấu hình (CẦN TINH CHỈNH KỸ LƯỠNG) ---
# Dải màu (quan trọng cho mask ban đầu)
LOWER_RED_HSV1 = np.array((0, 120, 70))
UPPER_RED_HSV1 = np.array((15, 255, 255))
LOWER_RED_HSV2 = np.array((165, 120, 70))
UPPER_RED_HSV2 = np.array((180, 255, 255))

LOWER_GREEN_HSV = np.array((30, 30, 30))
UPPER_GREEN_HSV = np.array((95, 255, 255))

# Tiền xử lý
MEDIAN_BLUR_KERNEL_SIZE = 5

# Phép toán hình thái học cho mask màu (QUAN TRỌNG)
OPENING_KERNEL_SIZE = 5  # Kích thước kernel cho phép Opening (loại bỏ nhiễu nhỏ, cuống mỏng)
OPENING_ITERATIONS = 1  # Số lần lặp Opening
DILATE_ITERATIONS_AFTER_OPENING = 7  # Dilate sau Opening để nối lại phần thân cà chua
# ERODE_FINAL_MASK_ITERATIONS = 1 # (Tùy chọn) Erode nhẹ cuối cùng để làm co lại một chút

# Lọc hình dạng Contour (QUAN TRỌNG)
MIN_CONTOUR_AREA = 1000  # Tăng ngưỡng diện tích
MIN_SOLIDITY = 0.80  # Độ đặc: Diện tích contour / Diện tích bao lồi (cao cho vật thể lồi)
MIN_ASPECT_RATIO = 0.65  # Tỷ lệ W/H của bounding box
MAX_ASPECT_RATIO = 1.35
MIN_EXTENT = 0.5  # Mức độ chiếm đầy bounding box
MIN_CIRCULARITY_RELAXED = 0.55  # Ngưỡng độ tròn nới lỏng hơn

# Phân loại màu sau khi đã có contour tốt (giữ nguyên hoặc tinh chỉnh)
ENABLE_POST_FILTER_COLOR_CLASSIFICATION = True
CLASSIFY_RED_MIN_HUE1 = 0;
CLASSIFY_RED_MAX_HUE1 = 12
CLASSIFY_RED_MIN_HUE2 = 168;
CLASSIFY_RED_MAX_HUE2 = 180
CLASSIFY_RED_MIN_SATURATION = 110  # Giảm S để linh hoạt hơn
CLASSIFY_RED_MIN_VALUE = 60

CLASSIFY_GREEN_MIN_HUE = 25;
CLASSIFY_GREEN_MAX_HUE = 95
CLASSIFY_GREEN_MIN_SATURATION = 25  # Giảm S cho xanh
CLASSIFY_GREEN_MIN_VALUE = 25

# Ước tính 3D
RADIUS_AT_KNOWN_DISTANCE_PX = 26;
KNOWN_DISTANCE_M = 0.4
RADIUS_TO_METERS_CALIB = RADIUS_AT_KNOWN_DISTANCE_PX * KNOWN_DISTANCE_M
TOMATO_REAL_WIDTH_M = 0.05
PIX_TO_METERS_CALIB = TOMATO_REAL_WIDTH_M / (RADIUS_AT_KNOWN_DISTANCE_PX * 2)
ASSUMED_Z_HEIGHT_M = 0.35

SHOW_PROCESSING_STEPS = True
SHOW_FINAL_RESULT_IMAGE = True
ONLY_DETECT_LARGEST_CONTOUR = True


# Hàm get_dominant_color_in_roi (giữ nguyên như phiên bản trước)
def get_dominant_color_in_roi(hsv_image_roi, original_mask_roi):
    # ... (code hàm này giữ nguyên) ...
    mask_red_roi1 = cv2.inRange(hsv_image_roi, CLASSIFY_RED_MIN_HUE1,
                                CLASSIFY_RED_MAX_HUE1)  # Dùng ngưỡng classify ở đây
    mask_red_roi_H2 = cv2.inRange(hsv_image_roi, CLASSIFY_RED_MIN_HUE2, CLASSIFY_RED_MAX_HUE2)
    mask_red_roi = cv2.bitwise_or(mask_red_roi1, mask_red_roi_H2)  # Mask cho phần HUE đỏ
    # Kết hợp với ngưỡng S, V cho đỏ
    s_channel_roi = hsv_image_roi[:, :, 1]
    v_channel_roi = hsv_image_roi[:, :, 2]
    mask_red_sv = cv2.bitwise_and(cv2.compare(s_channel_roi, CLASSIFY_RED_MIN_SATURATION, cv2.CMP_GE),
                                  cv2.compare(v_channel_roi, CLASSIFY_RED_MIN_VALUE, cv2.CMP_GE))
    mask_red_roi_final = cv2.bitwise_and(mask_red_roi, mask_red_sv)
    mask_red_roi_final = cv2.bitwise_and(mask_red_roi_final, mask_red_roi_final, mask=original_mask_roi)

    mask_green_roi_H = cv2.inRange(hsv_image_roi, CLASSIFY_GREEN_MIN_HUE, CLASSIFY_GREEN_MAX_HUE)
    mask_green_sv = cv2.bitwise_and(cv2.compare(s_channel_roi, CLASSIFY_GREEN_MIN_SATURATION, cv2.CMP_GE),
                                    cv2.compare(v_channel_roi, CLASSIFY_GREEN_MIN_VALUE, cv2.CMP_GE))
    mask_green_roi_final = cv2.bitwise_and(mask_green_roi_H, mask_green_sv)
    mask_green_roi_final = cv2.bitwise_and(mask_green_roi_final, mask_green_roi_final, mask=original_mask_roi)

    red_pixel_count = cv2.countNonZero(mask_red_roi_final)
    green_pixel_count = cv2.countNonZero(mask_green_roi_final)
    total_pixels_in_contour = cv2.countNonZero(original_mask_roi)

    if total_pixels_in_contour == 0:
        return "unknown", 0, 0, 0

    mean_hsv_roi_val = cv2.mean(hsv_image_roi, mask=original_mask_roi)
    hue_roi, saturation_roi, value_roi = mean_hsv_roi_val[0], mean_hsv_roi_val[1], mean_hsv_roi_val[2]

    # Ưu tiên dựa trên số pixel và màu trung bình phải khớp ngưỡng classify
    pixel_ratio_threshold = 0.5  # Nếu hơn 50% pixel là màu đó và màu trung bình cũng khớp

    is_red_by_mean = ((CLASSIFY_RED_MIN_HUE1 <= hue_roi <= CLASSIFY_RED_MAX_HUE1 or
                       CLASSIFY_RED_MIN_HUE2 <= hue_roi <= CLASSIFY_RED_MAX_HUE2) and
                      saturation_roi >= CLASSIFY_RED_MIN_SATURATION and
                      value_roi >= CLASSIFY_RED_MIN_VALUE)

    is_green_by_mean = (CLASSIFY_GREEN_MIN_HUE <= hue_roi <= CLASSIFY_GREEN_MAX_HUE and
                        saturation_roi >= CLASSIFY_GREEN_MIN_SATURATION and
                        value_roi >= CLASSIFY_GREEN_MIN_VALUE)

    if red_pixel_count > green_pixel_count and (
            red_pixel_count / total_pixels_in_contour) > pixel_ratio_threshold and is_red_by_mean:
        return "red", hue_roi, saturation_roi, value_roi
    elif green_pixel_count > red_pixel_count and (
            green_pixel_count / total_pixels_in_contour) > pixel_ratio_threshold and is_green_by_mean:
        return "green", hue_roi, saturation_roi, value_roi

    # Fallback nếu pixel count không rõ ràng, chỉ dựa vào mean
    if is_red_by_mean: return "red", hue_roi, saturation_roi, value_roi
    if is_green_by_mean: return "green", hue_roi, saturation_roi, value_roi

    return "unknown", hue_roi, saturation_roi, value_roi


def detect_tomatoes_contour_features(image_path):
    img = cv2.imread(image_path)
    if img is None: print(f"Lỗi đọc ảnh: '{image_path}'"); return None, None

    original_img_display = img.copy()
    height, width, _ = img.shape

    blur = cv2.medianBlur(img, MEDIAN_BLUR_KERNEL_SIZE)
    hsv_image = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    if SHOW_PROCESSING_STEPS:
        cv2.imshow("1. Blur", blur)
        cv2.imshow("2. HSV Image", hsv_image)

    mask_r1 = cv2.inRange(hsv_image, LOWER_RED_HSV1, UPPER_RED_HSV1)
    mask_r2 = cv2.inRange(hsv_image, LOWER_RED_HSV2, UPPER_RED_HSV2)
    mask_red = cv2.bitwise_or(mask_r1, mask_r2)
    mask_green = cv2.inRange(hsv_image, LOWER_GREEN_HSV, UPPER_GREEN_HSV)
    combined_mask = cv2.bitwise_or(mask_red, mask_green)
    if SHOW_PROCESSING_STEPS: cv2.imshow("3. Combined Raw Mask", combined_mask)

    # --- ÁP DỤNG MORPHOLOGICAL OPENING ĐỂ LOẠI BỎ CUỐNG/LÁ NHỎ ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPENING_KERNEL_SIZE, OPENING_KERNEL_SIZE))
    opened_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=OPENING_ITERATIONS)
    if SHOW_PROCESSING_STEPS: cv2.imshow("4. Opened Mask (Removed Small Noise/Stems)", opened_mask)

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
            extent = float(area) / rect_area
            if extent < MIN_EXTENT:
                if SHOW_PROCESSING_STEPS: print(f"Contour {i} REJ by EXTENT ({extent:.2f} < {MIN_EXTENT})")
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = 0
            if hull_area > 0: solidity = float(area) / hull_area
            if solidity < MIN_SOLIDITY:
                if SHOW_PROCESSING_STEPS: print(f"Contour {i} REJ by SOLIDITY ({solidity:.2f} < {MIN_SOLIDITY})")
                continue

            perimeter = cv2.arcLength(cnt, True)
            circularity = 0
            if perimeter > 0: circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity < MIN_CIRCULARITY_RELAXED:
                if SHOW_PROCESSING_STEPS: print(
                    f"Contour {i} REJ by CIRCULARITY ({circularity:.2f} < {MIN_CIRCULARITY_RELAXED})")
                continue

            if SHOW_PROCESSING_STEPS:
                print(
                    f"Contour {i} PASSED SHAPE. Area:{area:.0f}, AR:{aspect_ratio:.2f}, Ext:{extent:.2f}, Sol:{solidity:.2f}, Circ:{circularity:.2f}")
                cv2.drawContours(feature_debug_img, [cnt], -1, (0, 255, 255), 2)

            # Lấy vòng tròn ngoại tiếp để tính bán kính cho ước lượng khoảng cách
            (x_mc, y_mc), radius_mc = cv2.minEnclosingCircle(cnt)

            valid_contours_data.append({
                'contour': cnt, 'area': area,
                'center_2d': (int(x_mc), int(y_mc)),  # Dùng tâm của minEnclosingCircle
                'radius_px': int(radius_mc),  # Dùng bán kính của minEnclosingCircle
                'bbox': (x_b, y_b, w_b, h_b)
            })
        if SHOW_PROCESSING_STEPS: cv2.imshow("6. Feature Filtering Debug", feature_debug_img)

    # Phân loại màu cho các contour đã qua lọc hình dạng
    classified_tomatoes_data = []
    if valid_contours_data:
        for data in valid_contours_data:
            cnt = data['contour']
            center_2d = data['center_2d']  # Tâm từ minEnclosingCircle
            radius_px = data['radius_px']  # Bán kính từ minEnclosingCircle
            detected_color = "unknown"

            if ENABLE_POST_FILTER_COLOR_CLASSIFICATION:
                contour_roi_mask = np.zeros(dilated_mask.shape, dtype="uint8")
                cv2.drawContours(contour_roi_mask, [cnt], -1, 255, -1)
                hsv_image_roi_pixels = cv2.bitwise_and(hsv_image, hsv_image, mask=contour_roi_mask)
                detected_color, hue_roi, saturation_roi, value_roi = get_dominant_color_in_roi(hsv_image_roi_pixels,
                                                                                               contour_roi_mask)

                if detected_color == "unknown":
                    if SHOW_PROCESSING_STEPS: print(
                        f"Contour at {center_2d} ROUGH_R={radius_px} REJ by COLOR. Mean HSV: H={hue_roi:.1f} S={saturation_roi:.1f} V={value_roi:.1f}")
                    continue
                elif SHOW_PROCESSING_STEPS:
                    print(
                        f"Contour at {center_2d} ROUGH_R={radius_px} CLASS. as {detected_color}. Mean HSV: H={hue_roi:.1f} S={saturation_roi:.1f} V={value_roi:.1f}")

            data['type'] = detected_color if ENABLE_POST_FILTER_COLOR_CLASSIFICATION else "generic_shape"
            classified_tomatoes_data.append(data)

    detected_tomatoes_final = []
    if classified_tomatoes_data:
        if ONLY_DETECT_LARGEST_CONTOUR:
            classified_tomatoes_data.sort(key=lambda x: x['area'], reverse=True)
            if classified_tomatoes_data: detected_tomatoes_final.append(classified_tomatoes_data[0])
        else:
            detected_tomatoes_final = classified_tomatoes_data

        for data in detected_tomatoes_final:
            # Sử dụng tâm và bán kính từ minEnclosingCircle đã lưu
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

            cv2.circle(original_img_display, center_2d, radius_px, draw_color, 2)  # Vẽ vòng tròn ngoại tiếp
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
            for i, item in enumerate(detected_tomatoes_final):
                if item['center_2d'] == center_2d and item['radius_px'] == radius_px:
                    detected_tomatoes_final[i].update(tomato_info)
                    break

            info_text = f"{detected_color.capitalize()} D:{estimated_distance_m:.2f}m R:{radius_px}px"
            cv2.putText(original_img_display, info_text,
                        (center_2d[0] - radius_px, center_2d[1] - radius_px - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 2)

    if SHOW_PROCESSING_STEPS:
        print("\n--- Kết thúc xử lý ---")
        cv2.imshow("7. Final Detections (Contour Features)", original_img_display)
        if cv2.waitKey(0) & 0xFF == ord('q'): pass
        cv2.destroyAllWindows()

    return detected_tomatoes_final, original_img_display


def process_video(video_source=0, use_uart=True):
    """Xử lý video từ camera hoặc file video
    Args:
        video_source: 0 cho webcam, hoặc đường dẫn file video
        use_uart: True nếu muốn gửi dữ liệu qua UART
    """
    ser = init_uart() if use_uart else None
    
    try:
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            logging.error(f"Không thể mở video source: {video_source}")
            return
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        logging.info(f"Video source: {video_source}")
        logging.info(f"Frame size: {frame_width}x{frame_height}, FPS: {fps}")
    except Exception as e:
        logging.error(f"Lỗi khi khởi tạo video capture: {e}")
        return

    frame_count = 0
    start_time = time.time()
    logging.info("Bắt đầu xử lý video. Nhấn 'q' để thoát.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Không đọc được frame")
                break

            frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            try:
                results, result_image = detect_tomatoes_contour_features(frame)
                
                if result_image is not None:
                    fps_text = f"FPS: {frame_count/elapsed_time:.1f}"
                    cv2.putText(result_image, fps_text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("Tomato Detection", result_image)
                
                if results and use_uart:
                    for tomato in results:
                        color_code = 1 if tomato['type'] == "red" else 0
                        center_x, center_y = tomato['center_2d']
                        radius = tomato['radius_px']
                        
                        logging.info(f"Phát hiện cà chua - Màu: {'Chín' if color_code == 1 else 'Xanh'}, "
                                   f"Tâm: ({center_x}, {center_y}), Bán kính: {radius}px")
                        
                        send_tomato_data(ser, color_code, center_x, center_y, radius)
                
            except Exception as e:
                logging.error(f"Lỗi khi xử lý frame {frame_count}: {e}")
                continue
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Người dùng yêu cầu thoát")
                break
            
    except KeyboardInterrupt:
        logging.info("Chương trình bị ngắt bởi người dùng")
    except Exception as e:
        logging.error(f"Lỗi không mong muốn: {e}")
    finally:
        logging.info(f"Kết thúc xử lý video. Đã xử lý {frame_count} frames trong {elapsed_time:.1f} giây")
        cap.release()
        cv2.destroyAllWindows()
        if ser and ser.is_open:
            ser.close()
            logging.info("Đã đóng kết nối UART")

if __name__ == "__main__":
    try:
        video_source = 0
        logging.info("Khởi động chương trình phát hiện cà chua")
        process_video(video_source, use_uart=True)
    except Exception as e:
        logging.error(f"Lỗi chương trình: {e}")
    finally:
        logging.info("Kết thúc chương trình")