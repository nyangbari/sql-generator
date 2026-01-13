"""Query Preprocessor - Project-specific handling"""
from sqlalchemy import inspect
from langchain_community.utilities.sql_database import SQLDatabase
from config.settings import DATABASE_CONFIG
import re

class QueryPreprocessor:
    """질문 전처리 및 엔티티 매핑"""
    
    def __init__(self):
        self.entity_cache = {}
    
    def build_entity_cache(self, project_name, db_uri):
        """DB에서 엔티티 캐시 구축 - 프로젝트별 처리"""
        try:
            db = SQLDatabase.from_uri(db_uri, sample_rows_in_table_info=0)
            inspector = inspect(db._engine)
            
            all_tables = inspector.get_table_names()
            
            # knightfury 전용 처리
            if project_name.lower() == 'knightfury' and 'fury_projects' in all_tables:
                self._build_knightfury_cache(project_name, db)
            else:
                # 다른 프로젝트는 엔티티 캐시 스킵
                print(f"      ℹ️  엔티티 매핑 스킵 (프로젝트 구조 다름)")
                self.entity_cache[project_name] = {'projects': {}}
            
        except Exception as e:
            print(f"      ⚠️  엔티티 캐시 구축 실패: {e}")
            self.entity_cache[project_name] = {'projects': {}}
    
    def _build_knightfury_cache(self, project_name, db):
        """KnightFury 전용 엔티티 캐시"""
        try:
            query = "SELECT projectId, projectName, teamId, displayTeamName FROM fury_projects"
            result = db.run(query)
            
            project_map = {}
            
            if result and result != "[]":
                matches = re.findall(r"\('([^']*)',\s*'([^']*)',\s*'([^']*)',\s*'([^']*)'\)", result)
                
                for project_id, project_name_val, team_id, display_team_name in matches:
                    project_info = {
                        'projectId': project_id,
                        'projectName': project_name_val,
                        'teamId': team_id,
                        'displayTeamName': display_team_name
                    }
                    
                    self._add_mapping(project_map, project_id, project_info)
                    self._add_mapping(project_map, project_name_val, project_info)
                    self._add_mapping(project_map, team_id, project_info)
                    self._add_mapping(project_map, display_team_name, project_info)
                    
                    for text in [project_name_val, display_team_name]:
                        keywords = self._extract_keywords(text)
                        for keyword in keywords:
                            if len(keyword) >= 3:
                                self._add_mapping(project_map, keyword, project_info)
            
            self.entity_cache[project_name] = {'projects': project_map}
            
            unique_projects = len(set(
                p['projectId'] for p in project_map.values() 
                if isinstance(p, dict)
            ))
            
            print(f"      ✅ {unique_projects}개 프로젝트 매핑 완료!")
            
        except Exception as e:
            print(f"      ⚠️  KnightFury 캐시 구축 실패: {e}")
            self.entity_cache[project_name] = {'projects': {}}
    
    def _add_mapping(self, project_map, key, project_info):
        """매핑 추가"""
        if not key:
            return
        
        key_lower = key.lower().strip()
        if not key_lower:
            return
        
        if key_lower in project_map:
            if not isinstance(project_map[key_lower], list):
                project_map[key_lower] = [project_map[key_lower]]
            if project_info['projectId'] not in [p['projectId'] for p in project_map[key_lower]]:
                project_map[key_lower].append(project_info)
        else:
            project_map[key_lower] = project_info
    
    def _extract_keywords(self, text):
        """키워드 추출"""
        cleaned = re.sub(r'[^\w\s가-힣]', ' ', text)
        words = cleaned.split()
        
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
        
        for i in range(len(keywords)-1):
            bigram = f"{keywords[i]} {keywords[i+1]}"
            keywords.append(bigram)
        
        return keywords
    
    def preprocess(self, project_name, question):
        """질문 전처리"""
        if project_name not in self.entity_cache:
            return {
                'original_question': question,
                'processed_question': question,
                'entities': {},
                'hints': [],
                'ambiguous': False
            }
        
        # 엔티티 캐시가 비어있으면 스킵
        cache = self.entity_cache[project_name]
        if not cache['projects']:
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
        
        # 매칭 찾기
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
        
        # 중복 제거
        unique_matches = {}
        for i, match in enumerate(matches):
            pid = match['projectId']
            if pid not in unique_matches:
                unique_matches[pid] = (match, matched_keys[i])
        
        matches = [m[0] for m in unique_matches.values()]
        matched_keys = [m[1] for m in unique_matches.values()]
        
        # 결과 처리
        if len(matches) == 1:
            entities['project'] = matches[0]
            hints.append(f"Use projectId = '{matches[0]['projectId']}'")
            entities['matched_by'] = matched_keys[0]
            
        elif len(matches) > 1:
            ambiguous = True
            entities['project_candidates'] = matches
            
            # 우선순위 매칭
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
            
            if ambiguous and matches:
                entities['project'] = matches[0]
                hints.append(f"Use projectId = '{matches[0]['projectId']}' (assumed)")
                entities['matched_by'] = 'ambiguous'
        
        return {
            'original_question': question,
            'processed_question': question,
            'entities': entities,
            'hints': hints,
            'ambiguous': ambiguous
        }
