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

# --- Cấu hình (CẬP NHẬT DỰA TRÊN LOG) ---
LOWER_RED_HSV1 = np.array((0, 120, 70))
UPPER_RED_HSV1 = np.array((8, 255, 255))  # Giảm H_MAX của đỏ để tránh chồng lấn
LOWER_RED_HSV2 = np.array((165, 120, 70))
UPPER_RED_HSV2 = np.array((180, 255, 255))

# --- NGƯỠNG CHO MÀU MỤC TIÊU (VÀNG) - CẬP NHẬT DỰA TRÊN LOG: Mean H:31.8 S:114.5 V:235.3 ---
LOWER_TARGET_COLOR_HSV = np.array((20, 70, 180))  # H_MIN có thể từ 20-25 nếu nó luôn là vàng
UPPER_TARGET_COLOR_HSV = np.array((36, 200, 255))  # H_MAX > 31.8, S_MAX có thể điều chỉnh, V_MAX=255

MEDIAN_BLUR_KERNEL_SIZE = 5
OPENING_KERNEL_SIZE = 5
OPENING_ITERATIONS = 1
DILATE_ITERATIONS_AFTER_OPENING = 5

MIN_CONTOUR_AREA = 1000
MIN_SOLIDITY = 0.80
MIN_ASPECT_RATIO = 0.65
MAX_ASPECT_RATIO = 1.35
MIN_EXTENT = 0.5
MIN_CIRCULARITY_RELAXED = 0.80  # Tăng vì log Circ:0.86

ENABLE_POST_FILTER_COLOR_CLASSIFICATION = True
# --- Ngưỡng phân loại màu ---
CLASSIFY_RED_MIN_HUE1 = 0
CLASSIFY_RED_MAX_HUE1 = 8  # Phù hợp với UPPER_RED_HSV1
CLASSIFY_RED_MIN_HUE2 = 165
CLASSIFY_RED_MAX_HUE2 = 180
CLASSIFY_RED_MIN_SATURATION = 100
CLASSIFY_RED_MIN_VALUE = 60

# --- NGƯỠNG PHÂN LOẠI CHO MÀU MỤC TIÊU (VÀNG) - CẬP NHẬT DỰA TRÊN LOG ---
CLASSIFY_TARGET_MIN_HUE = 20  # Dựa trên H=31.8, có thể bắt đầu từ khoảng này
CLASSIFY_TARGET_MAX_HUE = 38  # Mở rộng hơn một chút so với UPPER_TARGET_COLOR_HSV
CLASSIFY_TARGET_MIN_SATURATION = 80  # Mean S là 114.5, nên 80 là hợp lý
CLASSIFY_TARGET_MIN_VALUE = 200  # Mean V là 235.3, nên 200 là hợp lý

DOMINANT_PIXEL_RATIO_THRESHOLD = 0.35
SIGNIFICANT_PIXEL_RATIO_THRESHOLD = 0.20

RADIUS_AT_KNOWN_DISTANCE_PX = 26
KNOWN_DISTANCE_M = 0.4
RADIUS_TO_METERS_CALIB = RADIUS_AT_KNOWN_DISTANCE_PX * KNOWN_DISTANCE_M
OBJECT_REAL_WIDTH_M = 0.05
PIX_TO_METERS_CALIB = OBJECT_REAL_WIDTH_M / (RADIUS_AT_KNOWN_DISTANCE_PX * 2)
ASSUMED_Z_HEIGHT_M = 0.35

SHOW_PROCESSING_STEPS = True
SHOW_FINAL_RESULT_IMAGE = True
ONLY_DETECT_LARGEST_CONTOUR = True  # Nên giữ True nếu chỉ có 1 đối tượng chính


def get_dominant_color_in_roi(hsv_image_roi, original_mask_roi):
    # Tạo mask cho màu đỏ trong ROI
    temp_mask_red1 = cv2.inRange(hsv_image_roi,
                                 np.array([CLASSIFY_RED_MIN_HUE1, CLASSIFY_RED_MIN_SATURATION, CLASSIFY_RED_MIN_VALUE]),
                                 np.array([CLASSIFY_RED_MAX_HUE1, 255, 255]))
    temp_mask_red2 = cv2.inRange(hsv_image_roi,
                                 np.array([CLASSIFY_RED_MIN_HUE2, CLASSIFY_RED_MIN_SATURATION, CLASSIFY_RED_MIN_VALUE]),
                                 np.array([CLASSIFY_RED_MAX_HUE2, 255, 255]))
    mask_red_pixels_in_roi = cv2.bitwise_or(temp_mask_red1, temp_mask_red2)
    mask_red_pixels_in_roi = cv2.bitwise_and(mask_red_pixels_in_roi, mask_red_pixels_in_roi, mask=original_mask_roi)

    # Tạo mask cho MÀU MỤC TIÊU (VÀNG) trong ROI
    temp_mask_target_color = cv2.inRange(hsv_image_roi,
                                         np.array([CLASSIFY_TARGET_MIN_HUE, CLASSIFY_TARGET_MIN_SATURATION,
                                                   CLASSIFY_TARGET_MIN_VALUE]),
                                         np.array([CLASSIFY_TARGET_MAX_HUE, 255, 255]))
    mask_target_color_pixels_in_roi = cv2.bitwise_and(temp_mask_target_color, temp_mask_target_color,
                                                      mask=original_mask_roi)

    red_pixel_count = cv2.countNonZero(mask_red_pixels_in_roi)
    target_color_pixel_count = cv2.countNonZero(mask_target_color_pixels_in_roi)
    total_pixels_in_contour = cv2.countNonZero(original_mask_roi)

    if total_pixels_in_contour == 0:
        return "unknown", 0, 0, 0

    mean_hsv_roi_val = cv2.mean(hsv_image_roi, mask=original_mask_roi)
    hue_roi, saturation_roi, value_roi = mean_hsv_roi_val[0], mean_hsv_roi_val[1], mean_hsv_roi_val[2]

    is_red_by_mean = ((CLASSIFY_RED_MIN_HUE1 <= hue_roi <= CLASSIFY_RED_MAX_HUE1 or
                       CLASSIFY_RED_MIN_HUE2 <= hue_roi <= CLASSIFY_RED_MAX_HUE2) and
                      saturation_roi >= CLASSIFY_RED_MIN_SATURATION and
                      value_roi >= CLASSIFY_RED_MIN_VALUE)

    is_target_color_by_mean = (CLASSIFY_TARGET_MIN_HUE <= hue_roi <= CLASSIFY_TARGET_MAX_HUE and
                               saturation_roi >= CLASSIFY_TARGET_MIN_SATURATION and
                               value_roi >= CLASSIFY_TARGET_MIN_VALUE)

    red_ratio = red_pixel_count / total_pixels_in_contour if total_pixels_in_contour > 0 else 0
    target_color_ratio = target_color_pixel_count / total_pixels_in_contour if total_pixels_in_contour > 0 else 0

    # --- Logic phân loại màu (đỏ vs target_color) ---
    if is_red_by_mean and red_ratio >= DOMINANT_PIXEL_RATIO_THRESHOLD:
        if is_target_color_by_mean and target_color_ratio >= SIGNIFICANT_PIXEL_RATIO_THRESHOLD:
            if red_ratio > target_color_ratio * 1.5:
                return "red", hue_roi, saturation_roi, value_roi
        else:
            return "red", hue_roi, saturation_roi, value_roi

    if is_target_color_by_mean and target_color_ratio >= DOMINANT_PIXEL_RATIO_THRESHOLD:
        if is_red_by_mean and red_ratio >= SIGNIFICANT_PIXEL_RATIO_THRESHOLD:
            if target_color_ratio > red_ratio * 1.5:
                return "target_color", hue_roi, saturation_roi, value_roi
        else:  # Nếu chỉ có target color khớp mạnh
            return "target_color", hue_roi, saturation_roi, value_roi

    if is_red_by_mean and not is_target_color_by_mean and red_ratio >= SIGNIFICANT_PIXEL_RATIO_THRESHOLD:
        return "red", hue_roi, saturation_roi, value_roi

    if is_target_color_by_mean and not is_red_by_mean and target_color_ratio >= SIGNIFICANT_PIXEL_RATIO_THRESHOLD:
        return "target_color", hue_roi, saturation_roi, value_roi

    if is_red_by_mean and is_target_color_by_mean:  # Cả hai đều khớp màu trung bình
        if red_ratio > target_color_ratio + 0.1:
            return "red", hue_roi, saturation_roi, value_roi
        if target_color_ratio > red_ratio + 0.1:
            return "target_color", hue_roi, saturation_roi, value_roi

    # In thông tin debug chi tiết hơn khi sắp trả về "unknown"
    if SHOW_PROCESSING_STEPS:
        print(
            f"    ROI Color Analysis (before returning 'unknown' or if conditions not met strongly):")
        print(f"      Mean HSV: H={hue_roi:.1f} S={saturation_roi:.1f} V={value_roi:.1f}")
        print(f"      is_red_by_mean: {is_red_by_mean}, red_ratio: {red_ratio:.2f}")
        print(f"      is_target_color_by_mean: {is_target_color_by_mean}, target_color_ratio: {target_color_ratio:.2f}")
        print(
            f"      Thresh Target: H({CLASSIFY_TARGET_MIN_HUE}-{CLASSIFY_TARGET_MAX_HUE}), S(>{CLASSIFY_TARGET_MIN_SATURATION}), V(>{CLASSIFY_TARGET_MIN_VALUE})")
        print(
            f"      Thresh Red1: H({CLASSIFY_RED_MIN_HUE1}-{CLASSIFY_RED_MAX_HUE1}), S(>{CLASSIFY_RED_MIN_SATURATION}), V(>{CLASSIFY_RED_MIN_VALUE})")

    return "unknown", hue_roi, saturation_roi, value_roi


def detect_objects_contour_features(image_input):
    """Xử lý ảnh hoặc frame video để phát hiện cà chua
    Args:
        image_input: Có thể là đường dẫn file ảnh (str) hoặc frame video (numpy array)
    """
    # Xử lý input là đường dẫn file ảnh
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            logging.error(f"Loi doc anh: '{image_input}'")
            return None, None
    # Xử lý input là frame video
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    else:
        logging.error("Dinh dang input khong hop le")
        return None, None

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
    mask_red = cv2.bitwise_or(mask_r1, mask_r2)

    mask_target_color = cv2.inRange(hsv_image, LOWER_TARGET_COLOR_HSV, UPPER_TARGET_COLOR_HSV)

    # QUAN TRỌNG: Nếu bạn CHỈ muốn phát hiện màu vàng (target_color) cho ảnh này, hãy dùng:
    combined_mask = mask_target_color
    # Nếu bạn muốn phát hiện CẢ màu đỏ VÀ màu vàng, dùng:
    # combined_mask = cv2.bitwise_or(mask_red, mask_target_color)

    if SHOW_PROCESSING_STEPS:
        cv2.imshow("Debug Target Color Mask", mask_target_color)
        if 'mask_red' in locals() and combined_mask is not mask_target_color: cv2.imshow("Debug Red Mask", mask_red)
        cv2.imshow("3. Combined Raw Mask (Current)", combined_mask)

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
                cv2.drawContours(feature_debug_img, [cnt], -1, (0, 255, 255), 2)

            (x_mc, y_mc), radius_mc = cv2.minEnclosingCircle(cnt)
            center_2d_mc = (int(x_mc), int(y_mc))
            radius_px_mc = int(radius_mc)
            detected_color = "unknown"

            if ENABLE_POST_FILTER_COLOR_CLASSIFICATION:
                contour_roi_mask = np.zeros(dilated_mask.shape, dtype="uint8")
                cv2.drawContours(contour_roi_mask, [cnt], -1, 255, -1)
                hsv_image_roi_pixels = cv2.bitwise_and(hsv_image, hsv_image, mask=contour_roi_mask)

                detected_color, hue_roi, saturation_roi, value_roi = get_dominant_color_in_roi(hsv_image_roi_pixels,
                                                                                               contour_roi_mask)

                if detected_color == "unknown":
                    if SHOW_PROCESSING_STEPS: print(
                        f"  Contour {i} at {center_2d_mc} ROUGH_R={radius_px_mc} FAILED COLOR CLASSIFICATION (Final decision was 'unknown').")
                    cv2.drawContours(feature_debug_img, [cnt], -1, (0, 0, 0), 2)
                    continue
                elif SHOW_PROCESSING_STEPS:
                    print(
                        f"  Contour {i} at {center_2d_mc} ROUGH_R={radius_px_mc} CLASSIFIED as {detected_color} (Mean H:{hue_roi:.1f} S:{saturation_roi:.1f} V:{value_roi:.1f}).")

            valid_contours_data.append({
                'contour': cnt, 'area': area,
                'center_2d': center_2d_mc,
                'radius_px': radius_px_mc,
                'type': detected_color if ENABLE_POST_FILTER_COLOR_CLASSIFICATION else "generic_shape",
                'bbox': (x_b, y_b, w_b, h_b)
            })
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
                draw_color = (0, 255, 255)  # Vàng BGR để vẽ cho màu vàng
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

            object_info = {
                'center_2d': center_2d, 'radius_px': radius_px, 'type': detected_color,
                'estimated_distance_m': round(estimated_distance_m, 3),
                'estimated_offset_y_m': round(estimated_offset_y_m, 3),
                'estimated_pos_z_m': estimated_pos_z_m
            }
            for i_final, item_final in enumerate(detected_objects_final):
                if item_final['center_2d'] == center_2d and item_final['radius_px'] == radius_px:
                    detected_objects_final[i_final].update(object_info)
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
        if key == ord('q'):
            cv2.destroyAllWindows()
    elif SHOW_FINAL_RESULT_IMAGE and original_img_display is not None:
        cv2.imshow("Final Result", original_img_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return detected_objects_final, original_img_display


def process_video(video_source=0, use_uart=True, target_fps=30):
    """Xử lý video từ camera hoặc file video
    Args:
        video_source: 0 cho webcam, hoặc đường dẫn file video
        use_uart: True nếu muốn gửi dữ liệu qua UART
        target_fps: FPS mục tiêu để xử lý video
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
    last_frame_time = start_time
    frame_interval = 1.0 / target_fps
    logging.info(f"Bắt đầu xử lý video với target FPS: {target_fps}. Nhấn 'q' để thoát.")
    
    try:
        while True:
            current_time = time.time()
            elapsed_since_last_frame = current_time - last_frame_time
            
            # Kiểm soát FPS
            if elapsed_since_last_frame < frame_interval:
                time.sleep(frame_interval - elapsed_since_last_frame)
                continue
                
            ret, frame = cap.read()
            if not ret:
                logging.warning("Không đọc được frame")
                break

            frame_count += 1
            last_frame_time = current_time
            total_elapsed_time = current_time - start_time
            
            try:
                results, result_image = detect_objects_contour_features(frame)
                
                if result_image is not None:
                    # Hiển thị thông tin FPS và số frame đã xử lý
                    current_fps = frame_count / total_elapsed_time
                    fps_text = f"FPS: {current_fps:.1f}"
                    frame_text = f"Frame: {frame_count}"
                    cv2.putText(result_image, fps_text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(result_image, frame_text, (10, 70),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Hiển thị thời gian chạy
                    runtime_text = f"Runtime: {total_elapsed_time:.1f}s"
                    cv2.putText(result_image, runtime_text, (10, 110),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    cv2.imshow("Tomato Detection", result_image)
                
                if results and use_uart:
                    for obj in results:
                        color_code = 1 if obj['type'] == "red" else 0
                        center_x, center_y = obj['center_2d']
                        radius = obj['radius_px']
                        
                        # Log chi tiết hơn về đối tượng được phát hiện
                        logging.info(
                            f"Frame {frame_count} - Phát hiện cà chua:\n"
                            f"  - Màu: {'Chín' if color_code == 1 else 'Xanh'}\n"
                            f"  - Vị trí: ({center_x}, {center_y})\n"
                            f"  - Bán kính: {radius}px\n"
                            f"  - Khoảng cách: {obj['estimated_distance_m']:.2f}m\n"
                            f"  - Offset Y: {obj['estimated_offset_y_m']:.2f}m"
                        )
                        
                        send_tomato_data(ser, color_code, center_x, center_y, radius)
                
            except Exception as e:
                logging.error(f"Lỗi khi xử lý frame {frame_count}: {e}")
                continue
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logging.info("Người dùng yêu cầu thoát")
                break
            elif key == ord('s'):  # Thêm chức năng lưu ảnh
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"detection_{timestamp}.jpg"
                cv2.imwrite(filename, result_image)
                logging.info(f"Đã lưu ảnh: {filename}")
            
    except KeyboardInterrupt:
        logging.info("Chương trình bị ngắt bởi người dùng")
    except Exception as e:
        logging.error(f"Lỗi không mong muốn: {e}")
    finally:
        logging.info(f"Kết thúc xử lý video. Đã xử lý {frame_count} frames trong {total_elapsed_time:.1f} giây")
        logging.info(f"FPS trung bình: {frame_count/total_elapsed_time:.1f}")
        cap.release()
        cv2.destroyAllWindows()
        if ser and ser.is_open:
            ser.close()
            logging.info("Đã đóng kết nối UART")

if __name__ == "__main__":
    try:
        video_source = 0  # 0 cho webcam, hoặc đường dẫn file video
        logging.info("Khởi động chương trình phát hiện cà chua")
        process_video(video_source, use_uart=True, target_fps=30)  # Có thể điều chỉnh target_fps
    except Exception as e:
        logging.error(f"Lỗi chương trình: {e}")
    finally:
        logging.info("Kết thúc chương trình")