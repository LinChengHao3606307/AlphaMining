import pandas as pd
import glob
import os



def get_stock_data(stock_idx_range: tuple[int,int],
                   source_dir_path: str = r"alpha_mining/data") -> list[pd.DataFrame]:
    all_stocks_df = []

    # 获取指定目录下的所有 Excel 文件
    excel_files = glob.glob(os.path.join(source_dir_path, "*.xlsx"))

    for file in excel_files:
        # 提取文件名中的索引
        file_name = os.path.basename(file)
        stock_idx = int(file_name.split('_')[1].split('.')[0])

        # 检查索引是否在指定范围内
        if stock_idx_range[0] <= stock_idx < stock_idx_range[1]:
            # 读取 Excel 文件并追加到列表中
            df = pd.read_excel(file)
            all_stocks_df.append(df)

    return all_stocks_df





if __name__ == "__main__":
    from main.train import get_sep_idx
    stock_idx_range = (0, 10)
    source_dir_path = "TODO"
    df = get_stock_data(stock_idx_range, source_dir_path)
    print()
    print(get_sep_idx(df))
    print()
    print(df)
