import os

num_runs = 30  # Số lần muốn train lại
for i in range(num_runs):
    print(f"\n🚀 Run {i+1}/{num_runs}")
    os.system("python deep_walmart.py")  # Nếu file bạn tên khác thì sửa lại
