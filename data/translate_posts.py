import pandas as pd
from googletrans import Translator
import time

translator = Translator()

def safe_translate(text, src='ko', dest='en'):
    if not isinstance(text, str) or not text.strip():
        return ""
    try:
        return translator.translate(text, src=src, dest=dest).text
    except Exception as e:
        print(f"Exception: {e}")
        return ""

# 메뉴 출력 및 입력
print("==== 번역할 게시판 선택 ====")
print("1. missyusa")
print("2. gototheusa")
choice = input("번호를 입력하세요 (1 또는 2): ").strip()

if choice == "1":
    input_file = "data/mu_posts.csv"
    output_file = "data/mu_posts_translated.csv"
    encoding = "euc-kr"
elif choice == "2":
    input_file = "data/gu_posts.csv"
    output_file = "data/gu_posts_translated.csv"
    encoding = "utf-8"
else:
    print("잘못된 입력입니다.")
    exit(1)

df = pd.read_csv(input_file, encoding=encoding)

# 진행상황 표시하며 번역
eng_titles = []
eng_contents = []
total = len(df)
for i, row in df.iterrows():
    print(f"{i+1}/{total} 번역중...")  # 진행상황 출력
    eng_titles.append(safe_translate(row["title"]))
    eng_contents.append(safe_translate(row["content"]))

# 번역 결과 저장할 컬럼 추가
df["Eng_title"] = eng_titles
df["Eng_Contents"] = eng_contents

# 결과 저장
df.to_csv(output_file, index=False, encoding="cp949")
print(f"{output_file} 파일을 확인하세요.") 