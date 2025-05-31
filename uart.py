import serial
import time

# Cấu hình cổng UART
port = '/dev/ttyUSB0'  # Thay đổi thành cổng UART của bạn, ví dụ: '/dev/ttyUSB0' trên Linux
baudrate = 115200  # Tốc độ baud, thay đổi nếu cần
timeout = 1  # Thời gian chờ (giây)

# Khởi tạo kết nối UART
try:
    ser = serial.Serial(port, baudrate, timeout=timeout)
    print(f"Đã kết nối với {port} tại {baudrate} baud")
except serial.SerialException as e:
    print(f"Lỗi kết nối: {e}")
    exit()

# Chuỗi cần gửi
message = "Hello, UART!\n"

# Gửi chuỗi qua UART
try:
    # Chuyển chuỗi thành bytes (encode sang UTF-8)
    ser.write(message.encode('utf-8'))
    print(f"Đã gửi: {message.strip()}")
except Exception as e:
    print(f"Lỗi khi gửi dữ liệu: {e}")

# Đóng cổng UART
ser.close()
print("Đã đóng cổng UART")