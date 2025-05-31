import cv2
import numpy as np


def filter_circle_color(image_path, lower_color, upper_color):
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Không thể đọc ảnh từ đường dẫn cung cấp")

    # Chuyển sang không gian màu HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Tạo mask cho dải màu
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Áp dụng mask lên ảnh gốc
    color_filtered = cv2.bitwise_and(image, image, mask=mask)

    # Chuyển sang ảnh xám để phát hiện hình tròn
    gray = cv2.cvtColor(color_filtered, cv2.COLOR_BGR2GRAY)

    # Làm mờ để giảm nhiễu
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Phát hiện hình tròn bằng Hough Circle Transform
    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=100
    )

    # Tạo ảnh kết quả
    result = image.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Vẽ hình tròn
            center = (i[0], i[1])
            radius = i[2]
            cv2.circle(result, center, radius, (0, 255, 0), 2)

    return result, color_filtered


# Ví dụ sử dụng
if __name__ == "__main__":
    # Đường dẫn đến ảnh
    image_path = "img.png"

    # Định nghĩa dải màu cần lọc (ví dụ: màu đỏ trong không gian HSV)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])

    # Gọi hàm xử lý
    result, color_filtered = filter_circle_color(image_path, lower_red, upper_red)

    # Hiển thị kết quả
    cv2.imshow("Original with Circles", result)
    cv2.imshow("Color Filtered", color_filtered)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Lưu ảnh kết quả
    cv2.imwrite("result_circles.jpg", result)
    cv2.imwrite("color_filtered.jpg", color_filtered)