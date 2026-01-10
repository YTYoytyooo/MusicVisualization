import os

# 支持的音频格式
exts = ('.flac', '.wav', '.mp3')

files = []

# 遍历当前目录
for f in os.listdir('.'):
    if f.lower().endswith(exts):
        files.append(f)


# 写入为一行： "a.mp3" "b.flac"
with open('list.txt', 'w', encoding='utf-8') as out:
    out.write(' '.join(f'"{f}"' for f in files))

print(f"已写入 {len(files)} 个文件到 list.txt")
