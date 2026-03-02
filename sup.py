import os
import re

def delete_specific_images():
    # 目标目录
    ai_dir = "/root/autodl-tmp/data2/train/ai"
    # 匹配格式：数字_adm_数字.PNG（支持任意位数的数字）
    # pattern = re.compile(r'^GLIDE_\d+_\d+_\d+_\d+_glide_\d+\.png$', re.IGNORECASE)
    pattern = re.compile(r'^\d+_adm_\d+\.PNG$', re.IGNORECASE)
    # pattern = re.compile(r'^\d+_biggan_\d+\.PNG$', re.IGNORECASE)
    # 要删除的文件数量
    delete_count = 1000

    # 1. 遍历目录，筛选符合条件的文件
    target_files = []
    for filename in os.listdir(ai_dir):
        # 检查文件名是否匹配格式，且是文件（不是目录）
        if pattern.match(filename) and os.path.isfile(os.path.join(ai_dir, filename)):
            target_files.append(os.path.join(ai_dir, filename))

    # 2. 校验文件数量
    if len(target_files) == 0:
        print("未找到符合 '数字_adm_数字.PNG' 格式的文件")
        return
    if len(target_files) < delete_count:
        print(f"符合条件的文件只有 {len(target_files)} 张，不足1000张，将删除所有符合条件的文件")
        delete_count = len(target_files)
    print(f"符合条件的文件有 {len(target_files)} 张")
    # 执行删除前增加确认提示，防止误操作
    confirm = input(f"即将删除 /root/autodl-tmp/data2/train/ai 目录下符合 '数字_adm_数字.PNG' 格式的{delete_count}张图片，是否确认？(y/n)：")
    if confirm.lower() == 'y':
        # # 3. 执行删除操作
        deleted_files = []
        error_files = []
        for i in range(delete_count):
            file_path = target_files[i]
            try:
                os.remove(file_path)
                deleted_files.append(file_path)
                # 每删除100个文件打印一次进度
                if (i + 1) % 100 == 0:
                    print(f"已删除 {i + 1} 个文件")
            except Exception as e:
                error_files.append((file_path, str(e)))

        # 4. 打印删除结果
        print(f"\n删除完成！")
        print(f"成功删除文件数量：{len(deleted_files)}")
        if error_files:
            print(f"删除失败文件数量：{len(error_files)}")
            print("失败文件列表：")
            for file, err in error_files:
                print(f"  - {file}: {err}")
    else:
        print("操作已取消")


if __name__ == "__main__":
    delete_specific_images()

