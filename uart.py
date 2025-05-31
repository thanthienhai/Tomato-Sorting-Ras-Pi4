import serial
import time

# Cấu hình cổng serial (COMx đối với Windows hoặc /dev/ttyUSBx đối với Linux)
serial_port = '/dev/ttyUSB0'  # Ví dụ trên Linux, có thể thay đổi tùy theo hệ thống
baud_rate = 115200  # Tốc độ truyền
timeout = 1  # Thời gian chờ để nhận dữ liệu (giây)

# Mở cổng serial
ser = serial.Serial(serial_port, baudrate=baud_rate, timeout=timeout)

# Kiểm tra xem cổng có mở thành công không
if ser.is_open:
    print(f"Đã kết nối với cổng: {serial_port}")
else:
    print(f"Lỗi khi kết nối với cổng {serial_port}")
    exit()

# Gửi dữ liệu qua UART
ser.write(b'Hello UART!\n')  # Gửi một chuỗi dữ liệu

# Đợi một chút trước khi đọc dữ liệu phản hồi
time.sleep(1)

ser.close()
