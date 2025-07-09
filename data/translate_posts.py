import pandas as pd
from googletrans import Translator
import time
from transformers import pipeline

translator = Translator()

# 모델명: xlm-roberta-large-xnli (한국어 zero-shot 분류 지원)
classifier = pipeline(
    "zero-shot-classification",
    model="joeddav/xlm-roberta-large-xnli",
    device=0  # GPU 사용시, 없으면 -1
)

text = "이게 스캠인지 궁금합니다. 조언 부탁드려요."
candidate_labels = [
    "사기 여부가 불확실하여 묻는 질문",
    "사기임을 확신한 후 대응법을 묻는 질문",
    "직접 피해 경험 후 경고 목적",
    "영상/뉴스 등 외부 자료 공유 경고",
    "판단이 불가능한 경우 (정보 부족, 맥락 모호 등)"
]

result = classifier(text, candidate_labels, multi_label=False)
print(result)

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

# 결과 저장 (utf-8-sig 인코딩 사용 - 한국어/Excel 호환성)
df.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"{output_file} 파일을 확인하세요.") 