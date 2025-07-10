import pandas as pd
import requests
import json
import time
from typing import Dict, List, Optional
import re
import yaml
import os
import sys
from transformers.pipelines import pipeline

class TextClassifier:
    def __init__(self, api_token: Optional[str] = None, config: Optional[dict] = None):
        """
        텍스트 분류기 초기화
        api_token: Hugging Face API 토큰 (선택사항)
        config: 설정 딕셔너리 (선택사항)
        """
        # self.api_token, self.api_url, self.headers 등 API 관련 코드 제거
        # config에서 categories/patterns 불러오는 부분은 유지
        self.local_classifier = pipeline(
            "zero-shot-classification",
            model="joeddav/xlm-roberta-large-xnli",
            device=0  # GPU 사용시 0, CPU만 있으면 -1
        )
        
        # confidence threshold 설정 (키워드 매칭률이 이 값보다 낮으면 Transformer 사용)
        self.confidence_threshold = 0.3  # 기본값
        if config and 'text_classification' in config:
            tc = config['text_classification']
            if 'confidence_threshold' in tc:
                self.confidence_threshold = tc['confidence_threshold']

        # config에서 분류 설정 읽기
        self.type_categories = {}
        self.type_patterns = {}
        self.topic_categories = {}
        self.topic_patterns = {}
        self.method_categories = {}
        self.method_patterns = {}
        if config and 'text_classification' in config:
            tc = config['text_classification']
            if 'type' in tc:
                self.type_categories = tc['type']['categories']
                self.type_patterns = tc['type']['patterns']
            if 'scam_topic' in tc:
                self.topic_categories = tc['scam_topic']['categories']
                self.topic_patterns = tc['scam_topic']['patterns']
            if 'scam_method' in tc:
                self.method_categories = tc['scam_method']['categories']
                self.method_patterns = tc['scam_method']['patterns']
        else:
            # 기본값 설정 - YAML에서 읽어오므로 빈 딕셔너리로 설정
            self.type_categories = {}
            self.type_patterns = {}
            self.topic_categories = {}
            self.topic_patterns = {}
            self.method_categories = {}
            self.method_patterns = {}

    def classify_with_keywords(self, text: str, patterns_dict: dict, default: str = "other", multi: bool = False) -> tuple:
        """
        키워드가 한번이라도 포함되면 해당 카테고리로 분류 (multi=True면 모든 매칭 카테고리 +로 연결)
        Returns: (classification, 매칭여부, 매칭된 키워드)
        """
        if not text or pd.isna(text):
            return default, 0, ""
        text = str(text).lower()
        matched = []
        matched_keywords = []
        for category, patterns in patterns_dict.items():
            for pattern in patterns:
                if pattern.lower() in text:
                    matched.append(category)
                    matched_keywords.append(pattern)
                    if not multi:
                        return category, 1, pattern  # 최초 매칭 카테고리, 키워드 반환
        if multi and matched:
            return "+".join(sorted(set(matched))), 1, "+".join(sorted(set(matched_keywords)))
        return default, 0, ""

    def classify_type(self, text: str) -> tuple:
        """
        타입 분류 (confidence 포함)
        """
        return self.classify_with_keywords(text, self.type_patterns, default="unclear")

    def classify_topic(self, text: str) -> tuple:
        """
        사기 주제 분류 (confidence 포함)
        """
        return self.classify_with_keywords(text, self.topic_patterns, default="other")

    def classify_method(self, text: str) -> tuple:
        """
        사기 방법 분류 (confidence 포함, 복수 선택 가능)
        """
        return self.classify_with_keywords(text, self.method_patterns, default="", multi=True)

    def classify_texts(self, text: str, use_api: bool = False) -> dict:
        # 1단계: 키워드 기반 분류 시도
        type_result, type_matched, type_kw = self.classify_type(text)
        topic_result, topic_matched, topic_kw = self.classify_topic(text)
        method_result, method_matched, method_kw = self.classify_method(text)
        # 2단계: 하나라도 매칭 실패하면 해당 matched_*_keyword만 'API'로 기록, 값은 fallback 결과 사용
        api_result = None
        if (type_matched == 0 or topic_matched == 0 or method_matched == 0) and use_api:
            print(f"일부 카테고리 키워드 매칭 실패. 부족한 부분은 Transformer 분류기를 사용합니다.")
            api_result = self.classify_with_api(text)
        return {
            "type": api_result["type"] if type_matched == 0 and api_result else type_result,
            "scam_topic": api_result["scam_topic"] if topic_matched == 0 and api_result else topic_result,
            "scam_method": api_result["scam_method"] if method_matched == 0 and api_result else method_result,
            "matched_type_keyword": "API" if type_matched == 0 else type_kw,
            "matched_topic_keyword": "API" if topic_matched == 0 else topic_kw,
            "matched_method_keyword": "API" if method_matched == 0 else method_kw
        }

    def classify_with_api(self, text: str) -> Optional[dict]:
        """
        로컬 zero-shot classification을 사용한 텍스트 분류
        """
        if not text or pd.isna(text):
            return None

        type_label_map = {v: k for k, v in self.type_categories.items()}
        topic_label_map = {v: k for k, v in self.topic_categories.items()}
        method_label_map = {v: k for k, v in self.method_categories.items()}
        candidate_labels = list(self.type_categories.values()) + list(self.topic_categories.values()) + list(self.method_categories.values())

        try:
            result = self.local_classifier(text, candidate_labels, multi_label=False)
            labels = result["labels"] if isinstance(result, dict) else []
            scores = result["scores"] if isinstance(result, dict) else []
            type_label = next((type_label_map[lbl] for lbl in labels if lbl in type_label_map), "unclear")
            topic_label = next((topic_label_map[lbl] for lbl in labels if lbl in topic_label_map), "other")
            method_label = next((method_label_map[lbl] for lbl in labels if lbl in method_label_map), "")
            return {
                "type": type_label,
                "scam_topic": topic_label,
                "scam_method": method_label
            }
        except Exception as e:
            print(f"로컬 모델 분류 오류: {e}")
            return {
                "type": "API_TIMEOUT",
                "scam_topic": "API_TIMEOUT",
                "scam_method": "API_TIMEOUT"
            }

def process_csv_file(input_file: str, output_file: str, api_token: Optional[str] = None, use_api: bool = False, config: Optional[dict] = None):
    """
    CSV 파일을 읽어서 분류 결과를 추가하여 저장
    """
    # 여러 인코딩을 시도하여 CSV 파일 읽기
    encodings = ['euc-kr', 'utf-8-sig', 'utf-8', 'cp949', 'latin1']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(input_file, encoding=encoding)
            print(f"파일을 {encoding} 인코딩으로 성공적으로 읽었습니다.")
            break
        except UnicodeDecodeError:
            print(f"{encoding} 인코딩으로 읽기 실패, 다음 인코딩 시도...")
            continue
        except Exception as e:
            print(f"{encoding} 인코딩으로 읽기 중 오류: {e}")
            continue
    
    if df is None:
        raise Exception("모든 인코딩으로 파일 읽기 실패")
    
    print(f"총 {len(df)}개의 행을 처리합니다...")
    
    # 기존 분류된 파일이 있는지 확인
    existing_df = None
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file, encoding='utf-8-sig')
            print(f"기존 분류 파일을 찾았습니다. {len(existing_df)}개의 행이 있습니다.")
        except Exception as e:
            print(f"기존 파일 읽기 실패: {e}")
    
    # 분류기 초기화
    classifier = TextClassifier(api_token, config)
    
    # 분류 결과를 저장할 리스트
    classifications_type = []
    classifications_topic = []
    classifications_method = []
    matched_type_keywords = []
    matched_topic_keywords = []
    matched_method_keywords = []
    
    output_exists = os.path.exists(output_file)
    for idx, (_, row) in enumerate(df.iterrows()):
        # 기존 파일이 있고, 현재 행이 이미 분류되어 있는지 확인
        if existing_df is not None:
            existing_row = existing_df[existing_df['id'] == row['id']]
            if not existing_row.empty and 'type' in existing_row.columns and 'scam_topic' in existing_row.columns and 'scam_method' in existing_row.columns:
                # 기존 분류 결과 사용
                classifications_type.append(existing_row.iloc[0]['type'])
                classifications_topic.append(existing_row.iloc[0]['scam_topic'])
                classifications_method.append(existing_row.iloc[0]['scam_method'])
                matched_type_keywords.append(existing_row.iloc[0].get('matched_type_keyword', ''))
                matched_topic_keywords.append(existing_row.iloc[0].get('matched_topic_keyword', ''))
                matched_method_keywords.append(existing_row.iloc[0].get('matched_method_keyword', ''))
                continue
        
        # 제목과 내용을 결합하여 분류
        title = str(row.get('title', ''))
        content = str(row.get('content', ''))
        eng_title = str(row.get('Eng_title', ''))
        eng_content = str(row.get('Eng_Contents', ''))
        
        # 한국어 텍스트 우선, 없으면 영어 텍스트 사용
        text_to_classify = ""
        if title and title != 'nan':
            text_to_classify += title + " "
        if content and content != 'nan':
            text_to_classify += content + " "
        
        # 한국어 텍스트가 없으면 영어 텍스트 사용
        if not text_to_classify.strip():
            if eng_title and eng_title != 'nan':
                text_to_classify += eng_title + " "
            if eng_content and eng_content != 'nan':
                text_to_classify += eng_content + " "
        
        # 분류 수행
        result = classifier.classify_texts(text_to_classify.strip(), use_api)
        classifications_type.append(result['type'])
        classifications_topic.append(result['scam_topic'])
        classifications_method.append(result['scam_method'])
        matched_type_keywords.append(result.get('matched_type_keyword', ''))
        matched_topic_keywords.append(result.get('matched_topic_keyword', ''))
        matched_method_keywords.append(result.get('matched_method_keyword', ''))
        
        # 진행상황 출력
        if (idx + 1) % 10 == 0:
            print(f"진행률: {idx + 1}/{len(df)} ({((idx + 1)/len(df)*100):.1f}%)")
        
        # API 사용 시 요청 간격 조절
        if use_api:
            time.sleep(0.5)  # 0.5초 대기
        
        result_row = dict(row)  # 원본 row의 모든 컬럼 포함
        result_row.update({
            "type": result['type'],
            "scam_topic": result['scam_topic'],
            "scam_method": result['scam_method'],
            "matched_type_keyword": result.get('matched_type_keyword', ''),
            "matched_topic_keyword": result.get('matched_topic_keyword', ''),
            "matched_method_keyword": result.get('matched_method_keyword', '')
        })
        result_df = pd.DataFrame([result_row])
        result_df.to_csv(
            output_file,
            mode='a',
            header=not output_exists and idx == 0,  # 첫 행만 헤더
            index=False,
            encoding='utf-8-sig'
        )
        output_exists = True
    
    # 분류 결과를 데이터프레임에 추가
    df['type'] = classifications_type
    df['scam_topic'] = classifications_topic
    df['scam_method'] = classifications_method
    df['matched_type_keyword'] = matched_type_keywords
    df['matched_topic_keyword'] = matched_topic_keywords
    df['matched_method_keyword'] = matched_method_keywords
    
    # 결과 저장 (utf-8-sig 인코딩 사용)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    # 분류 결과 통계 출력
    print("\n=== 분류 결과 통계 ===")
    type_counts = df['type'].value_counts()
    for category, count in type_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{category}: {count}개 ({percentage:.1f}%)")
    
    print(f"\n분류 완료! 결과가 {output_file}에 저장되었습니다.")

def main():
    """
    메인 실행 함수
    """
    # 현재 스크립트의 디렉토리를 기준으로 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 커맨드라인 인자 처리
    args = sys.argv[1:]
    def resolve_path(arg, default_name):
        if os.path.isabs(arg):
            return arg
        # data/로 시작하는 경우 워크스페이스 루트 기준으로 처리
        if arg.startswith('data' + os.sep) or arg.startswith('data/'):
            return os.path.join(project_root, arg)
        # ./data/로 시작하는 경우도 처리
        if arg.startswith('.' + os.sep + 'data' + os.sep) or arg.startswith('./data/'):
            return os.path.join(project_root, arg[2:])  # ./ 제거
        # 그 외의 경우 스크립트 디렉토리 기준으로 처리
        return os.path.join(script_dir, arg)
    if len(args) >= 1:
        input_file = resolve_path(args[0], "gu_posts_translated.csv")
    else:
        input_file = os.path.join(script_dir, "gu_posts_translated.csv")
    if len(args) >= 2:
        output_file = resolve_path(args[1], "gu_posts_classified.csv")
    else:
        # 입력 파일명에서 _translated를 _classified로 변경
        input_basename = os.path.basename(input_file)
        if "_translated.csv" in input_basename:
            output_basename = input_basename.replace("_translated.csv", "_classified.csv")
        else:
            output_basename = input_basename.replace(".csv", "_classified.csv")
        output_file = os.path.join(os.path.dirname(input_file), output_basename)
    config_file = os.path.join(project_root, "config.yaml")
    config_secret_file = os.path.join(project_root, "config_secret.yaml")
    
    # config.yaml에서 기본 설정 읽기
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"config.yaml 읽기 실패: {e}")
        config = {}
    
    # config_secret.yaml에서 민감한 정보 읽기
    try:
        with open(config_secret_file, 'r', encoding='utf-8') as f:
            config_secret = yaml.safe_load(f)
        # 민감한 정보를 기본 설정에 병합
        if config_secret:
            config.update(config_secret)
    except Exception as e:
        print(f"config_secret.yaml 읽기 실패: {e}")
    
    api_token = config.get('huggingface_api_key')
    
    # API 사용 여부 (토큰이 없으면 False로 설정)
    use_api = False if not api_token else True
    
    print("=== 텍스트 분류 시작 ===")
    print(f"입력 파일: {input_file}")
    print(f"출력 파일: {output_file}")
    print(f"API 사용: {use_api}")
    
    if use_api:
        print("Hugging Face API를 사용하여 분류합니다...")
    else:
        print("키워드 기반 분류를 사용합니다...")
    
    # CSV 파일 처리
    process_csv_file(input_file, output_file, api_token, use_api, config)

if __name__ == "__main__":
    main() 