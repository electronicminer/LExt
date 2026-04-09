import os
import shutil
import subprocess

root_dir = r".\csr_1_LDC93S6A\csr_1"
target_dir = r".\wsj0"

ffmpeg_path = r"D:\ffmpeg\bin\ffmpeg.exe"

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

print("开始合并并解码 15 张 CD 的音频...")

count = 0
for i in range(1, 16):
    cd_folder = os.path.join(root_dir, f"11-{i}.1", "wsj0")
    if not os.path.exists(cd_folder):
        continue
        
    print(f"正在处理 第 {i} 张 CD: {cd_folder} ...")
    
    for root, dirs, files in os.walk(cd_folder):
        for file in files:
            if file.endswith(".wv1"):
                # 源文件路径
                wv1_path = os.path.join(root, file)
                
                # 计算目标文件夹路径并创建
                rel_path = os.path.relpath(root, cd_folder)
                target_sub_dir = os.path.join(target_dir, rel_path)
                os.makedirs(target_sub_dir, exist_ok=True)
                
                # 目标 wav 文件路径
                wav_path = os.path.join(target_sub_dir, file.replace(".wv1", ".wav"))
                
                # 调用 FFmpeg 进行解码转换
                # -y: 覆盖同名文件, -loglevel error: 保持终端干净, -f wav: 强制输出标准 wav
                cmd = [ffmpeg_path, '-y', '-loglevel', 'error', '-i', wv1_path, '-f', 'wav', wav_path]
                subprocess.run(cmd)
                
                count += 1

print(f"共成功解码了 {count} 个 .wav ！")