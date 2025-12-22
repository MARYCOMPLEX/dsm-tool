import os

def modify_rpb_file(input_path, output_path, x):
    # 读取原始 .RPB 文件内容
    with open(input_path, 'r') as file:
        lines = file.readlines()

    # 创建一个新的内容列表
    new_lines = []

    # 遍历每一行并进行修改
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith("lineScale"):
            key, value = stripped_line.split('=')
            new_value = float(value.strip(';')) * x
            new_lines.append(f"{key.strip()} = {new_value:.4f};\n")
        elif stripped_line.startswith("sampScale"):
            key, value = stripped_line.split('=')
            new_value = float(value.strip(';')) * x
            new_lines.append(f"{key.strip()} = {new_value:.4f};\n")
        elif stripped_line.startswith("lineOffset"):
            key, value = stripped_line.split('=')
            original_value = float(value.strip(';'))
            new_value = original_value * x + (x - 1) / 2
            new_lines.append(f"{key.strip()} = {new_value:.4f};\n")
        elif stripped_line.startswith("sampOffset"):
            key, value = stripped_line.split('=')
            original_value = float(value.strip(';'))
            new_value = original_value * x + (x - 1) / 2
            new_lines.append(f"{key.strip()} = {new_value:.4f};\n")
        else:
            new_lines.append(line)

    # 调试打印以检查结果


    # 将修改后的内容写回到目标 .RPB 文件
    with open(output_path, 'w') as file:
        file.writelines(new_lines)

# 设置参数
#left_input = 'E:/stereosr/geoeye/2/3D/3733LR.tif'
#output = '3733SR.tif'
#x = 4  # 替换为需要的倍数

# 生成输入 .RPB 文件路径
#input_rpb_file = os.path.splitext(left_input)[0] + '.RPB'

# 生成输出 .RPB 文件路径
#output_rpb_file = os.path.splitext(output)[0] + '.RPB'

# 修改 .RPB 文件并保存到输出路径
#modify_rpb_file(input_rpb_file, output_rpb_file, x)