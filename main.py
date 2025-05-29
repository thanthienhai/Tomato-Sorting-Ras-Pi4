import cv2
import numpy as np
import os
import glob


def preprocess_image(frame):
    """
    Tiền xử lý ảnh để tăng chất lượng nhận diện.
    - Cân bằng sáng sử dụng CLAHE.
    - Làm mịn ảnh để giảm nhiễu, giúp các bước nhận diện contours sau này tốt hơn.
    """
    # 1. Cân bằng sáng sử dụng CLAHE (Contrast Limited Adaptive Histogram Equalization)
    frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(frame_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    frame_lab = cv2.merge((l, a, b))
    frame = cv2.cvtColor(frame_lab, cv2.COLOR_LAB2BGR)

    # 2. Làm mịn ảnh
    frame = cv2.medianBlur(frame, 5)
    return frame


def detect_tomatoes(frame):
    """
    Nhận diện cà chua dựa trên màu sắc (của điểm trung tâm), hình dạng và diện tích.
    Trả về danh sách các detection (x, y, w, h, nhãn, diện tích).
    """
    frame_processed = preprocess_image(frame)
    hsv = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2HSV)

    # Khoảng màu cho cà chua chín (đỏ) - Đã điều chỉnh
    lower_red1 = np.array([0, 70, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 70])
    upper_red2 = np.array([180, 255, 255])

    # Khoảng màu cho cà chua xanh - Đã điều chỉnh
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])

    # Tạo mask cho từng khoảng màu (dùng cho tìm contour)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask_red1 + mask_red2

    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)

    detections = []

    # --- Xử lý cà chua chín (đỏ) ---
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_red:
        area = cv2.contourArea(contour)
        min_area = 150
        max_area = frame.shape[0] * frame.shape[1] / 4

        if min_area < area < max_area:
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h

                if circularity > 0.35 and 0.6 < aspect_ratio < 1.6:
                    center_x = x + w // 2
                    center_y = y + h // 2

                    if 0 <= center_y < hsv.shape[0] and 0 <= center_x < hsv.shape[1]:
                        center_pixel_hsv = hsv[center_y, center_x]

                        is_red_center1 = np.all((center_pixel_hsv >= lower_red1) & (center_pixel_hsv <= upper_red1))
                        is_red_center2 = np.all((center_pixel_hsv >= lower_red2) & (center_pixel_hsv <= upper_red2))

                        if is_red_center1 or is_red_center2:
                            detections.append((x, y, w, h, "Chín", area))

    # --- Xử lý cà chua xanh ---
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_green:
        area = cv2.contourArea(contour)
        min_area = 150
        max_area = frame.shape[0] * frame.shape[1] / 4

        if min_area < area < max_area:
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h

                if circularity > 0.35 and 0.6 < aspect_ratio < 1.6:
                    center_x = x + w // 2
                    center_y = y + h // 2

                    if 0 <= center_y < hsv.shape[0] and 0 <= center_x < hsv.shape[1]:
                        center_pixel_hsv = hsv[center_y, center_x]

                        is_green_center = np.all((center_pixel_hsv >= lower_green) & (center_pixel_hsv <= upper_green))

                        if is_green_center:
                            detections.append((x, y, w, h, "Xanh", area))

    detections.sort(key=lambda x: x[5], reverse=True)
    return detections


def draw_detections(frame, detections):
    """
    Xác định trạng thái tổng thể của ảnh (Chín/Xanh/Không có cà chua)
    và vẽ duy nhất nhãn đó ở giữa màn hình.
    """
    result_frame = frame.copy()

    chin_count = sum(1 for det in detections if det[4] == "Chín")
    xanh_count = sum(1 for det in detections if det[4] == "Xanh")

    # Xác định nhãn tổng thể và màu sắc
    overall_label = ""
    overall_color = (255, 255, 255)  # Mặc định trắng
    text_border_color = (0, 0, 0)  # Mặc định đen

    if chin_count > 0:
        overall_label = "CHIN"
        overall_color = (0, 0, 255)  # Đỏ
    elif xanh_count > 0:
        overall_label = "XANH"
        overall_color = (0, 255, 0)  # Xanh lá
    else:
        overall_label = "KHONG PHAT HIEN CA CHUA"
        # Giữ màu mặc định trắng cho chữ, và đen cho viền

    # Thiết lập font và kích thước cho nhãn tổng thể (làm lớn)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5  # Có thể tăng lên 2.0 hoặc 2.5 tùy kích thước ảnh
    thickness = 3  # Độ dày của chữ

    # Tính toán kích thước của text để căn giữa
    (text_width, text_height), baseline = cv2.getTextSize(overall_label, font, font_scale, thickness)

    # Lấy kích thước ảnh
    img_height, img_width = result_frame.shape[:2]

    # Tính toán vị trí để căn giữa text
    text_x = (img_width - text_width) // 2
    text_y = (img_height + text_height) // 2

    # Vẽ chữ lên ảnh với viền đen để dễ đọc
    # Vẽ viền trước
    cv2.putText(result_frame, overall_label, (text_x, text_y),
                font, font_scale, text_border_color, thickness + 2)  # Viền dày hơn
    # Vẽ chữ chính
    cv2.putText(result_frame, overall_label, (text_x, text_y),
                font, font_scale, overall_color, thickness)

    return result_frame


def process_images_auto_slideshow(folder_path="images_tomato", delay_seconds=1):
    """
    Xử lý tất cả ảnh từ thư mục và hiển thị tự động dưới dạng slideshow.
    """
    if not os.path.exists(folder_path):
        print(f"Thư mục '{folder_path}' không tồn tại!")
        print("Hãy tạo thư mục đó và đặt các file ảnh vào.")
        return

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []

    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, extension)))
        image_files.extend(glob.glob(os.path.join(folder_path, extension.upper())))

    if not image_files:
        print(f"Không tìm thấy file ảnh nào trong thư mục '{folder_path}'!")
        print("Các định dạng được hỗ trợ: JPG, JPEG, PNG, BMP, TIFF")
        return

    image_files.sort()

    print(f"Tìm thấy {len(image_files)} file ảnh trong thư mục '{folder_path}'")
    print(f"Chạy slideshow tự động - mỗi ảnh hiển thị {delay_seconds} giây")
    print("Nhấn ESC để thoát bất kỳ lúc nào")
    print("=" * 60)

    window_name = "Nhan dien ca chua - Auto Slideshow"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    for current_index, image_path in enumerate(image_files):
        filename = os.path.basename(image_path)

        print(f"[{current_index + 1}/{len(image_files)}] Đang xử lý: {filename}")

        frame = cv2.imread(image_path)
        if frame is None:
            print(f"  -> Lỗi: Không thể đọc file {filename}")
            continue

        height, width = frame.shape[:2]
        max_display_width = 1000
        max_display_height = 700

        if width > max_display_width or height > max_display_height:
            scale = min(max_display_width / width, max_display_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))

        detections = detect_tomatoes(frame)
        result_frame = draw_detections(frame, detections)  # Hàm này giờ sẽ vẽ nhãn tổng thể

        # Có thể tùy chọn bỏ thông tin tiến độ/file dưới đây nếu muốn màn hình hoàn toàn "sạch"
        progress_text = f"{current_index + 1}/{len(image_files)} - {filename}"
        cv2.putText(result_frame, progress_text, (20, result_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(result_frame, progress_text, (20, result_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        chin_count = sum(1 for det in detections if det[4] == "Chín")
        xanh_count = sum(1 for det in detections if det[4] == "Xanh")

        # In thông tin tổng thể ra console
        overall_status_console = ""
        if chin_count > 0:
            overall_status_console = "CHIN"
        elif xanh_count > 0:
            overall_status_console = "XANH"
        else:
            overall_status_console = "KHONG PHAT HIEN CA CHUA"

        print(f"  -> Trang thai tong the: {overall_status_console} (Chin: {chin_count}, Xanh: {xanh_count})")

        cv2.imshow(window_name, result_frame)

        key = cv2.waitKey(int(delay_seconds * 1000)) & 0xFF
        if key == 27:
            print("\n*** Người dùng nhấn ESC - Dừng slideshow ***")
            break

    cv2.destroyAllWindows()
    print("=" * 60)
    print("Đã hoàn thành slideshow!")


def process_images_manual(folder_path="images_tomato"):
    """
    Xử lý tất cả ảnh từ thư mục và hiển thị với điều khiển thủ công.
    """
    if not os.path.exists(folder_path):
        print(f"Thư mục '{folder_path}' không tồn tại!")
        print("Hãy tạo thư mục đó và đặt các file ảnh vào.")
        return

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []

    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, extension)))
        image_files.extend(glob.glob(os.path.join(folder_path, extension.upper())))

    if not image_files:
        print(f"Không tìm thấy file ảnh nào trong thư mục '{folder_path}'!")
        print("Các định dạng được hỗ trợ: JPG, JPEG, PNG, BMP, TIFF")
        return

    image_files.sort()

    print(f"Tìm thấy {len(image_files)} file ảnh trong thư mục '{folder_path}'")
    print("Hướng dẫn:")
    print("- Nhấn SPACE hoặc ENTER để chuyển ảnh tiếp theo")
    print("- Nhấn B để quay lại ảnh trước")
    print("- Nhấn S để lưu ảnh hiện tại")
    print("- Nhấn ESC hoặc Q để thoát")
    print("-" * 50)

    current_index = 0
    window_name = "Nhan dien ca chua - Manual Control"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while current_index < len(image_files):
        image_path = image_files[current_index]
        filename = os.path.basename(image_path)

        print(f"\nĐang xử lý: {filename} ({current_index + 1}/{len(image_files)})")

        frame = cv2.imread(image_path)
        if frame is None:
            print(f"  -> Lỗi: Không thể đọc file: {filename}")
            current_index += 1
            continue

        height, width = frame.shape[:2]
        max_display_width = 1200
        max_display_height = 800

        if width > max_display_width or height > max_display_height:
            scale = min(max_display_width / width, max_display_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))

        detections = detect_tomatoes(frame)
        result_frame = draw_detections(frame, detections)  # Hàm này giờ sẽ vẽ nhãn tổng thể

        # Có thể tùy chọn bỏ thông tin tiến độ/file dưới đây nếu muốn màn hình hoàn toàn "sạch"
        progress_text = f"{current_index + 1}/{len(image_files)} - {filename}"
        cv2.putText(result_frame, progress_text, (20, result_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(result_frame, progress_text, (20, result_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        chin_count = sum(1 for det in detections if det[4] == "Chín")
        xanh_count = sum(1 for det in detections if det[4] == "Xanh")

        # In thông tin tổng thể ra console
        overall_status_console = ""
        if chin_count > 0:
            overall_status_console = "CHIN"
        elif xanh_count > 0:
            overall_status_console = "XANH"
        else:
            overall_status_console = "KHONG PHAT HIEN CA CHUA"

        print(f"  -> Trang thai tong the: {overall_status_console} (Chin: {chin_count}, Xanh: {xanh_count})")

        cv2.imshow(window_name, result_frame)

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == 27 or key == ord('q'):
                cv2.destroyAllWindows()
                print("\nĐã thoát chương trình.")
                return

            elif key == 32 or key == 13:
                break

            elif key == ord('s'):
                save_folder = "results"
                os.makedirs(save_folder, exist_ok=True)
                save_path = os.path.join(save_folder, f"result_{filename}")
                cv2.imwrite(save_path, result_frame)
                print(f"  -> Đã lưu ảnh kết quả: {save_path}")

            elif key == ord('b'):
                if current_index > 0:
                    current_index -= 2
                    break
                else:
                    print("  -> Đây là ảnh đầu tiên, không thể quay lại.")

            else:
                print("  -> Phím không hợp lệ. Sử dụng SPACE/ENTER (tiếp theo), B (quay lại), S (lưu), Q/ESC (thoát).")

        current_index += 1

    cv2.destroyAllWindows()
    print("\nĐã xử lý hết tất cả ảnh trong thư mục!")


def main():
    """Hàm main với lựa chọn chế độ hoạt động."""
    print("=== CHƯƠNG TRÌNH NHẬN DIỆN CÀ CHUA ===")
    print("Chọn chế độ hoạt động:")
    print("1. Chế độ tự động (slideshow) - mỗi ảnh hiển thị 1 giây")
    print("2. Chế độ thủ công - nhấn phím để điều khiển chuyển ảnh, lưu ảnh")
    print("3. Chế độ tự động với thời gian tùy chỉnh")

    image_folder = "images_tomato"

    while True:
        try:
            choice = input("\nNhập lựa chọn (1/2/3): ").strip()

            if choice == '1':
                print(f"\n🚀 Bắt đầu slideshow tự động trong thư mục '{image_folder}' (1 giây/ảnh)...")
                process_images_auto_slideshow(image_folder, 1)
                break

            elif choice == '2':
                print(f"\n🎮 Chế độ thủ công trong thư mục '{image_folder}' - điều khiển bằng phím...")
                process_images_manual(image_folder)
                break

            elif choice == '3':
                delay_input = input("Nhập thời gian hiển thị mỗi ảnh (giây, mặc định 1.0): ").strip()
                try:
                    delay = float(delay_input) if delay_input else 1.0
                    if delay < 0.1:
                        delay = 0.1
                    print(f"\n⏱️ Bắt đầu slideshow với {delay} giây/ảnh trong thư mục '{image_folder}'...")
                    process_images_auto_slideshow(image_folder, delay)
                    break
                except ValueError:
                    print("Thời gian không hợp lệ, sử dụng mặc định 1 giây.")
                    process_images_auto_slideshow(image_folder, 1)
                    break

            else:
                print("Lựa chọn không hợp lệ! Vui lòng nhập 1, 2, hoặc 3.")

        except KeyboardInterrupt:
            print("\n\nĐã thoát chương trình.")
            break


if __name__ == "__main__":
    main()