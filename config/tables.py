"""Table descriptions and metadata"""

# 테이블별 상세 설명
TABLE_DESCRIPTIONS = {
    'fury_action_configs': {
        'description': '''
            KnightFury dashboard basic actions/quests/missions.
            Platform-level missions like connect Telegram, Discord, Twitter.
            System missions, not project-specific quests.
            플랫폼 기본 미션. 텔레그램 연결, 디스코드 연결 등.
        ''',
        'keywords': ['action', 'dashboard', 'platform mission', 'connect', '기본 미션', '플랫폼'],
        'use_cases': [
            'What dashboard actions are available?',
            'Show platform missions',
            '대시보드 미션이 뭐야?',
            '플랫폼 기본 미션이 뭐야?'
        ]
    },
    
    'fury_mission_configs': {
        'description': '''
            Mission type definitions and templates.
            Defines KINDS of missions/quests: quiz, visit, NFT mint, swap.
            Mission categories and types.
            미션 타입 정의. 퀴즈, 방문, NFT 민팅 등의 종류.
        ''',
        'keywords': ['mission type', 'mission category', 'kind of mission', '미션 종류', '미션 타입'],
        'use_cases': [
            'What types of missions exist?',
            'Show mission categories',
            '미션 종류가 뭐가 있어?'
        ]
    },
    
    'fury_projects': {
        'description': '''
            Main projects table. Primary project information.
            All projects registered with KnightFury platform.
            Companies wanting users to complete missions/quests.
            프로젝트 메인 테이블. 전체 프로젝트 정보.
        ''',
        'keywords': ['project', 'all projects', 'total projects', '프로젝트', '전체 프로젝트'],
        'use_cases': [
            'How many projects?',
            'List all projects',
            '프로젝트가 몇 개야?'
        ]
    },
    
    'fury_airdrop_projects': {
        'description': '''
            Airdrop-specific subset of projects.
            Only projects with airdrop plans.
            Subset of fury_projects table for airdrops only.
            에어드롭 하는 프로젝트만 (fury_projects의 부분집합).
        ''',
        'keywords': ['airdrop', 'airdrop project', '에어드롭', '에어드롭 프로젝트'],
        'use_cases': [
            'How many airdrop projects?',
            'List airdrop projects',
            '에어드롭 프로젝트 몇 개?'
        ]
    },
    
    'fury_project_missions': {
        'description': '''
            Quests that projects registered.
            Project-specific missions/quests.
            Which missions each project has.
            프로젝트가 등록한 퀘스트. 프로젝트별 미션 목록.
        ''',
        'keywords': ['project quest', 'project mission', 'registered mission', '프로젝트 퀘스트', '어떤 미션'],
        'use_cases': [
            'What quests do projects have?',
            'Show project missions',
            '프로젝트가 등록한 퀘스트 뭐야?',
            '2pic 프로젝트는 어떤 미션을 해?',
        ]
    },
    
    'fury_user_project_missions': {
        'description': '''
            User mission completion tracking.
            Which users completed which missions/quests.
            User progress on quests/missions.
            유저의 미션 완료 기록. 유저 진행 상황.
        ''',
        'keywords': ['user progress', 'completed mission', 'user mission', '유저 진행', '완료 미션'],
        'use_cases': [
            'Which users completed missions?',
            'Show user progress',
            '유저가 완료한 미션 뭐야?'
        ]
    },
    
    'fury_users': {
        'description': '''
            User accounts and profiles.
            User information: wallet address, username, social connections.
            유저 계정 정보. 지갑 주소, 사용자명, 소셜 연결.
        ''',
        'keywords': ['user', 'account', 'profile', '사용자', '유저', '계정'],
        'use_cases': [
            'How many users?',
            'List all users',
            '사용자가 몇 명이야?'
        ]
    },
}

# 테이블 우선순위 (질문 패턴별)
TABLE_PRIORITY = {
    'project_count': ['fury_projects', 'fury_airdrop_projects'],
    'airdrop_count': ['fury_airdrop_projects', 'fury_projects'],
    'user_count': ['fury_users', 'fury_user_project_missions'],
    'mission_types': ['fury_mission_configs', 'fury_action_configs'],
    'platform_missions': ['fury_action_configs', 'fury_mission_configs'],
    'project_quests': ['fury_project_missions', 'fury_mission_configs'],
    'project_missions': ['fury_project_missions', 'fury_mission_configs'],
}

# 기본 테이블 (fallback)
DEFAULT_TABLE = 'fury_users'
