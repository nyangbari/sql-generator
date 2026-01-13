"""Query Preprocessor - Complete entity mapping"""
from sqlalchemy import inspect, text
from langchain_community.utilities.sql_database import SQLDatabase
from config.settings import DATABASE_CONFIG
import re

class QueryPreprocessor:
    """질문 전처리 및 엔티티 매핑"""
    
    def __init__(self):
        self.entity_cache = {}
        self.db_connections = {}
    
    def build_entity_cache(self, project_name, db_uri):
        """DB에서 엔티티 캐시 구축 - 모든 필드 지원"""
        try:
            db = SQLDatabase.from_uri(db_uri, sample_rows_in_table_info=0)
            
            # fury_projects 테이블에서 모든 관련 정보
            query = "SELECT projectId, projectName, teamId, displayTeamName FROM fury_projects"
            result = db.run(query)
            
            project_map = {}
            
            if result and result != "[]":
                # 결과 파싱 (4개 컬럼)
                matches = re.findall(r"\('([^']*)',\s*'([^']*)',\s*'([^']*)',\s*'([^']*)'\)", result)
                
                for project_id, project_name, team_id, display_team_name in matches:
                    project_info = {
                        'projectId': project_id,
                        'projectName': project_name,
                        'teamId': team_id,
                        'displayTeamName': display_team_name
                    }
                    
                    # 모든 필드로 검색 가능하도록
                    self._add_mapping(project_map, project_id, project_info)
                    self._add_mapping(project_map, project_name, project_info)
                    self._add_mapping(project_map, team_id, project_info)
                    self._add_mapping(project_map, display_team_name, project_info)
                    
                    # 키워드 추출 (projectName, displayTeamName)
                    for text in [project_name, display_team_name]:
                        keywords = self._extract_keywords(text)
                        for keyword in keywords:
                            if len(keyword) >= 3:
                                self._add_mapping(project_map, keyword, project_info)
            
            self.entity_cache[project_name] = {
                'projects': project_map
            }
            
            unique_projects = len(set(
                p['projectId'] for p in project_map.values() 
                if isinstance(p, dict)
            ))
            print(f"      ✅ {unique_projects}개 프로젝트 매핑 완료!")
            
        except Exception as e:
            print(f"      ⚠️  엔티티 캐시 구축 실패: {e}")
            import traceback
            traceback.print_exc()
            self.entity_cache[project_name] = {'projects': {}}
    
    def _add_mapping(self, project_map, key, project_info):
        """매핑 추가 (중복 처리)"""
        if not key:
            return
        
        key_lower = key.lower().strip()
        if not key_lower:
            return
        
        if key_lower in project_map:
            # 이미 있으면 리스트로 변환
            if not isinstance(project_map[key_lower], list):
                project_map[key_lower] = [project_map[key_lower]]
            # 중복 방지 (projectId 기준)
            if project_info['projectId'] not in [p['projectId'] for p in project_map[key_lower]]:
                project_map[key_lower].append(project_info)
        else:
            project_map[key_lower] = project_info
    
    def _extract_keywords(self, text):
        """텍스트에서 의미있는 키워드 추출"""
        # 특수문자 제거
        cleaned = re.sub(r'[^\w\s가-힣]', ' ', text)
        
        # 단어 분리
        words = cleaned.split()
        
        # 불용어 제거
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were',
            'x', 'event', 'campaign', 'quest', 'mission', 'pre', 'launch',
            '프로젝트', '이벤트', '캠페인', '퀘스트', '미션'
        }
        
        keywords = []
        for word in words:
            word_lower = word.lower()
            if word_lower not in stopwords and len(word) >= 2:
                keywords.append(word)
        
        # 2단어 조합
        for i in range(len(keywords)-1):
            bigram = f"{keywords[i]} {keywords[i+1]}"
            keywords.append(bigram)
        
        return keywords
    
    def preprocess(self, project_name, question):
        """질문 전처리 및 엔티티 해석
        
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
        matched_keys = []
        
        for key, project_info in cache['projects'].items():
            if key in question_lower:
                if isinstance(project_info, list):
                    matches.extend(project_info)
                    matched_keys.extend([key] * len(project_info))
                else:
                    matches.append(project_info)
                    matched_keys.append(key)
        
        # 중복 제거 (projectId 기준)
        unique_matches = {}
        for i, match in enumerate(matches):
            pid = match['projectId']
            if pid not in unique_matches:
                unique_matches[pid] = (match, matched_keys[i])
        
        matches = [m[0] for m in unique_matches.values()]
        matched_keys = [m[1] for m in unique_matches.values()]
        
        # 매칭 결과 처리
        if len(matches) == 1:
            # 정확히 1개 매칭
            entities['project'] = matches[0]
            hints.append(f"Use projectId = '{matches[0]['projectId']}'")
            entities['matched_by'] = matched_keys[0]
            
        elif len(matches) > 1:
            # 여러 개 매칭
            ambiguous = True
            entities['project_candidates'] = matches
            
            # 가장 정확한 매칭 우선순위:
            # 1. projectId 정확 매칭
            # 2. displayTeamName 정확 매칭
            # 3. teamId 정확 매칭
            # 4. 첫 번째
            
            for match in matches:
                if match['projectId'].lower() in question_lower:
                    entities['project'] = match
                    hints.append(f"Use projectId = '{match['projectId']}'")
                    entities['matched_by'] = 'projectId (exact)'
                    ambiguous = False
                    break
            
            if ambiguous:
                for match in matches:
                    if match['displayTeamName'] and match['displayTeamName'].lower() in question_lower:
                        entities['project'] = match
                        hints.append(f"Use projectId = '{match['projectId']}'")
                        entities['matched_by'] = 'displayTeamName'
                        ambiguous = False
                        break
            
            if ambiguous:
                for match in matches:
                    if match['teamId'] and match['teamId'].lower() in question_lower:
                        entities['project'] = match
                        hints.append(f"Use projectId = '{match['projectId']}'")
                        entities['matched_by'] = 'teamId'
                        ambiguous = False
                        break
            
            # 그래도 애매하면 첫 번째
            if ambiguous and matches:
                entities['project'] = matches[0]
                hints.append(f"Use projectId = '{matches[0]['projectId']}' (assumed)")
                hints.append(f"Note: Multiple projects matched: {[m['projectId'] for m in matches]}")
                entities['matched_by'] = 'ambiguous'
        
        return {
            'original_question': question,
            'processed_question': question,
            'entities': entities,
            'hints': hints,
            'ambiguous': ambiguous
        }
