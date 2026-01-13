"""Query Preprocessor - Enhanced entity mapping"""
from sqlalchemy import inspect, text
from langchain_community.utilities.sql_database import SQLDatabase
from config.settings import DATABASE_CONFIG
import re

class QueryPreprocessor:
    """질문 전처리 및 엔티티 매핑 - 강화 버전"""
    
    def __init__(self):
        self.entity_cache = {}
        self.db_connections = {}
    
    def build_entity_cache(self, project_name, db_uri):
        """DB에서 엔티티 캐시 구축 - 다중 매핑"""
        try:
            db = SQLDatabase.from_uri(db_uri, sample_rows_in_table_info=0)
            
            # fury_projects 테이블에서 프로젝트 정보
            query = "SELECT projectId, projectName, teamId FROM fury_projects"
            result = db.run(query)
            
            project_map = {}
            
            if result and result != "[]":
                # 결과 파싱 (3개 컬럼)
                matches = re.findall(r"\('([^']*)',\s*'([^']*)',\s*'([^']*)'\)", result)
                
                for project_id, project_name, team_id in matches:
                    project_info = {
                        'projectId': project_id,
                        'projectName': project_name,
                        'teamId': team_id
                    }
                    
                    # 1. projectId로 검색 (정확)
                    if project_id:
                        project_map[project_id.lower()] = project_info
                    
                    # 2. projectName으로 검색 (전체)
                    if project_name:
                        project_map[project_name.lower()] = project_info
                        
                        # 3. projectName 일부 단어로도 검색
                        # "🐾 Paw Spa X KNightFury Pre-Registration Event 🐾"
                        # → "paw spa", "paw", "spa" 등으로도 찾을 수 있게
                        words = self._extract_keywords(project_name)
                        for word in words:
                            if len(word) >= 3:  # 3글자 이상만
                                key = word.lower()
                                # 중복 방지: 이미 있으면 리스트로 변환
                                if key in project_map:
                                    if not isinstance(project_map[key], list):
                                        project_map[key] = [project_map[key]]
                                    project_map[key].append(project_info)
                                else:
                                    project_map[key] = project_info
                    
                    # 4. teamId로 검색
                    if team_id:
                        key = team_id.lower()
                        if key in project_map:
                            if not isinstance(project_map[key], list):
                                project_map[key] = [project_map[key]]
                            project_map[key].append(project_info)
                        else:
                            project_map[key] = project_info
            
            self.entity_cache[project_name] = {
                'projects': project_map
            }
            
            unique_projects = len([p for p in project_map.values() if isinstance(p, dict)])
            print(f"      ✅ {unique_projects}개 프로젝트 매핑 완료!")
            
        except Exception as e:
            print(f"      ⚠️  엔티티 캐시 구축 실패: {e}")
            import traceback
            traceback.print_exc()
            self.entity_cache[project_name] = {'projects': {}}
    
    def _extract_keywords(self, text):
        """텍스트에서 의미있는 키워드 추출"""
        # 특수문자 제거
        cleaned = re.sub(r'[^\w\s]', ' ', text)
        
        # 단어 분리
        words = cleaned.split()
        
        # 불용어 제거 (너무 일반적인 단어)
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were',
            'x', 'event', 'campaign', 'quest', 'mission',
            '프로젝트', '이벤트', '캠페인', '퀘스트', '미션'
        }
        
        keywords = []
        for word in words:
            word_lower = word.lower()
            if word_lower not in stopwords and len(word) >= 3:
                keywords.append(word)
        
        # 2단어 조합도 추가
        for i in range(len(keywords)-1):
            bigram = f"{keywords[i]} {keywords[i+1]}"
            keywords.append(bigram)
        
        return keywords
    
    def preprocess(self, project_name, question):
        """질문 전처리 및 엔티티 해석 - 강화 버전
        
        Returns:
            dict: {
                'original_question': 원본 질문,
                'processed_question': 처리된 질문,
                'entities': 발견된 엔티티,
                'hints': SQL 생성 힌트,
                'ambiguous': 애매한 경우 True
            }
        """
        if project_name not in self.entity_cache:
            return {
                'original_question': question,
                'processed_question': question,
                'entities': {},
                'hints': [],
                'ambiguous': False
            }
        
        question_lower = question.lower()
        entities = {}
        hints = []
        ambiguous = False
        
        cache = self.entity_cache[project_name]
        
        # 가능한 모든 매칭 찾기
        matches = []
        
        for key, project_info in cache['projects'].items():
            if key in question_lower:
                # 리스트인 경우 (여러 프로젝트 매칭)
                if isinstance(project_info, list):
                    matches.extend(project_info)
                else:
                    matches.append(project_info)
        
        # 중복 제거 (projectId 기준)
        unique_matches = {}
        for match in matches:
            pid = match['projectId']
            if pid not in unique_matches:
                unique_matches[pid] = match
        
        matches = list(unique_matches.values())
        
        # 매칭 결과 처리
        if len(matches) == 1:
            # 정확히 1개 매칭
            entities['project'] = matches[0]
            hints.append(f"Use projectId = '{matches[0]['projectId']}'")
            
        elif len(matches) > 1:
            # 여러 개 매칭 (애매함)
            ambiguous = True
            entities['project_candidates'] = matches
            
            # 가장 정확한 매칭 선택 (projectId가 질문에 정확히 있으면)
            for match in matches:
                if match['projectId'].lower() in question_lower:
                    entities['project'] = match
                    hints.append(f"Use projectId = '{match['projectId']}'")
                    ambiguous = False
                    break
            
            # 그래도 애매하면 첫 번째 선택
            if ambiguous and matches:
                entities['project'] = matches[0]
                hints.append(f"Use projectId = '{matches[0]['projectId']}' (assumed)")
                hints.append(f"Note: Multiple projects matched: {[m['projectId'] for m in matches]}")
        
        return {
            'original_question': question,
            'processed_question': question,
            'entities': entities,
            'hints': hints,
            'ambiguous': ambiguous
        }
