import pandas as pd

dta_file = "./CGED-Q 1850-1864.dta"
df = pd.read_stata(dta_file)

col_name_map = {
    "阳历年份": "year",
    "季节号": "season",
    "序号": "xuhao",
    "插补号": "interpol",
    "版本年号": "banben_nianhao",
    "版本年代": "banben_niandai",
    "版本季节": "banben_jijie",
    "地区": "diqu",
    "机构一": "jigou_1",
    "机构二": "jigou_2",
    "官职一": "guanzhi_1",
    "姓": "xing",
    "名": "ming",
    "字号": "zihao",
    "籍贯省": "ren_sheng",
    "籍贯县": "ren_xian",
    "出身一": "chushen_1",
}

df = df[col_name_map.keys()].rename(columns=col_name_map)

print(df.head())

df["year"] = df["year"].astype(int) + (df["season"].astype(int) - 1) / 4
df["assigned_edition"] = df["banben_nianhao"] + df["banben_niandai"].astype(str) + df["banben_jijie"].astype(str)

df.to_csv("CGED-Q 1850-1864.csv", index=False)