"""Table descriptions and metadata - Enhanced Version"""

# 테이블별 상세 설명 (강화 버전)
TABLE_DESCRIPTIONS = {
    'fury_action_configs': {
        'description': '''
            KnightFury platform dashboard basic actions and system missions.
            Platform-level missions that ALL users can do: connect Telegram, Discord, Twitter, visit website.
            These are NOT project-specific quests.
            플랫폼 공통 기본 미션. 모든 유저가 할 수 있는 시스템 미션.
            텔레그램 연결, 디스코드 연결, 트위터 팔로우 등.
        ''',
        'keywords': [
            'action', 'dashboard', 'platform mission', 'system mission',
            'connect', 'follow', 'visit', 'basic mission',
            '액션', '대시보드', '플랫폼 미션', '시스템 미션', '기본 미션',
            'telegram', 'discord', 'twitter', '텔레그램', '디스코드', '트위터'
        ],
        'use_cases': [
            'What dashboard actions are available?',
            'Show platform missions',
            'List system missions',
            '플랫폼 기본 미션이 뭐야?',
            '대시보드 액션 보여줘'
        ],
        'columns_description': {
            'actionId': 'Unique action identifier',
            'actionType': 'Type of action (telegram, discord, twitter, etc)',
            'desc': 'Action description'
        }
    },
    
    'fury_mission_configs': {
        'description': '''
            Mission TYPE definitions and templates.
            Defines what KINDS/CATEGORIES/TYPES of missions exist in the system.
            Examples: quiz mission, visit mission, NFT mint mission, swap mission, vote mission.
            This is NOT about specific projects or specific mission instances.
            미션 타입/종류 정의. 어떤 카테고리의 미션이 존재하는지.
            예시: 퀴즈 미션, 방문 미션, NFT 민팅 미션, 스왑 미션.
            특정 프로젝트나 구체적 미션이 아님.
        ''',
        'keywords': [
            'mission type', 'mission category', 'mission template',
            'kind of mission', 'type of mission', 'mission definition',
            '미션 종류', '미션 타입', '미션 카테고리', '미션 템플릿',
            'quiz', 'visit', 'mint', 'swap', 'vote',
            '퀴즈', '방문', '민팅', '스왑', '투표'
        ],
        'use_cases': [
            'What types of missions exist?',
            'Show mission categories',
            'List mission templates',
            '미션 종류가 뭐가 있어?',
            '어떤 타입의 미션이 있어?'
        ],
        'columns_description': {
            'missionType': 'Type identifier (quiz, visit, mint, etc)',
            'missionCategory1': 'Primary category',
            'desc': 'Mission type description'
        }
    },
    
    'fury_projects': {
        'description': '''
            Main projects table. PRIMARY source for ALL project information.
            All projects registered with KnightFury platform.
            Companies/teams that want users to complete missions and quests.
            Use this table to COUNT projects, LIST projects, or get project details.
            프로젝트 메인 테이블. 모든 프로젝트 정보의 기본 소스.
            나이트퓨리에 등록된 전체 프로젝트.
            프로젝트 개수 세기, 프로젝트 목록, 프로젝트 정보에 사용.
        ''',
        'keywords': [
            'project', 'all projects', 'total projects', 'how many projects',
            'list projects', 'project info', 'project details',
            '프로젝트', '전체 프로젝트', '프로젝트 개수', '프로젝트 목록',
            'company', 'team', '회사', '팀'
        ],
        'use_cases': [
            'How many projects?',
            'List all projects',
            'Show project details',
            'Count total projects',
            '프로젝트가 몇 개야?',
            '프로젝트 목록 보여줘'
        ],
        'columns_description': {
            'projectId': 'Unique project identifier',
            'projectName': 'Project name',
            'teamId': 'Team identifier',
            'showFront': 'Display on frontend (1=yes, 0=no)'
        }
    },
    
    'fury_airdrop_projects': {
        'description': '''
            Airdrop-specific SUBSET of projects.
            ONLY projects with airdrop plans.
            This is a FILTERED subset of fury_projects, not all projects.
            Use ONLY when specifically asking about AIRDROP projects.
            에어드롭 프로젝트만. fury_projects의 부분집합.
            에어드롭을 하는 프로젝트만 포함. 전체 프로젝트 아님.
            "에어드롭" 명시적으로 물어볼 때만 사용.
        ''',
        'keywords': [
            'airdrop', 'airdrop project', 'token airdrop',
            '에어드롭', '에어드롭 프로젝트', '토큰 에어드롭'
        ],
        'use_cases': [
            'How many airdrop projects?',
            'List airdrop projects',
            'Show projects with airdrops',
            '에어드롭 프로젝트 몇 개?',
            '에어드롭 하는 프로젝트'
        ],
        'columns_description': {
            'projectId': 'Project identifier (links to fury_projects)',
            'chainId': 'Blockchain identifier',
            'desc': 'Airdrop description'
        }
    },
    
    'fury_project_missions': {
        'description': '''
            SPECIFIC missions/quests that EACH PROJECT has registered.
            Links projects to their missions. Project-mission relationships.
            Use this to find: "What missions does PROJECT X have?"
            Each row = one mission for one project.
            각 프로젝트가 등록한 구체적 미션/퀘스트.
            프로젝트-미션 관계. "X 프로젝트는 어떤 미션이 있어?" 에 사용.
        ''',
        'keywords': [
            'project mission', 'project quest', 'missions for project',
            'what missions does', 'which missions', 'project has missions',
            '프로젝트 미션', '프로젝트 퀘스트', '프로젝트가 등록한',
            '어떤 미션', '무슨 미션', '미션 목록',
            'specific mission', 'actual mission', '구체적 미션', '실제 미션'
        ],
        'use_cases': [
            'What missions does project X have?',
            'Show missions for 2pic project',
            'List quests in project Y',
            '2pic 프로젝트는 어떤 미션을 해?',
            'X 프로젝트 미션 목록',
            'What kind of missions does 2pic have?',  # ← 추가!
        ],
        'columns_description': {
            'projectId': 'Project identifier (e.g., "2pic")',
            'missionCategory1': 'Mission category',
            'missionSeq': 'Mission sequence number',
            'point': 'Points awarded',
            'desc': 'Mission description'
        }
    },
    
    'fury_user_project_missions': {
        'description': '''
            User progress and completion tracking on missions.
            Which USERS completed which PROJECT MISSIONS.
            User-mission completion history and status.
            유저의 미션 완료 기록. 어떤 유저가 어떤 미션을 완료했는지.
            유저 진행 상황 추적.
        ''',
        'keywords': [
            'user progress', 'user completion', 'completed mission',
            'user mission', 'user quest', 'user finished',
            '유저 진행', '유저 완료', '완료한 미션', '유저가 한 미션',
            'who completed', 'which users', '누가 완료',
        ],
        'use_cases': [
            'Which users completed missions?',
            'Show user progress',
            'Who finished mission X?',
            '누가 이 미션을 완료했어?',
            '유저가 완료한 미션은?'
        ],
        'columns_description': {
            'address': 'User wallet address',
            'projectId': 'Project identifier',
            'missionSeq': 'Mission sequence',
            'status': 'Completion status'
        }
    },
    
    'fury_users': {
        'description': '''
            User accounts, profiles, and wallet information.
            User personal information: wallet address, username, social connections.
            유저 계정 정보. 지갑 주소, 사용자명, 소셜 연결.
        ''',
        'keywords': [
            'user', 'account', 'profile', 'wallet',
            '사용자', '유저', '계정', '지갑',
            'username', 'address', '주소',
        ],
        'use_cases': [
            'How many users?',
            'List all users',
            'Show user profiles',
            '사용자가 몇 명이야?',
            '유저 목록'
        ],
        'columns_description': {
            'address': 'Wallet address (primary key)',
            'username': 'Username',
            'isAdmin': 'Admin status'
        }
    },
}

# 테이블 우선순위 (질문 패턴별) - 강화 버전
TABLE_PRIORITY = {
    # 프로젝트 개수/목록 (일반)
    'project_count': ['fury_projects', 'fury_airdrop_projects'],
    
    # 에어드롭 프로젝트 (명시적)
    'airdrop_count': ['fury_airdrop_projects', 'fury_projects'],
    
    # 유저 개수/목록
    'user_count': ['fury_users', 'fury_user_project_missions'],
    
    # 미션 타입/종류 (카테고리)
    'mission_types': ['fury_mission_configs', 'fury_action_configs'],
    
    # 플랫폼 기본 미션
    'platform_missions': ['fury_action_configs', 'fury_mission_configs'],
    
    # 프로젝트의 구체적 미션 (가장 중요!)
    'project_missions': ['fury_project_missions', 'fury_mission_configs', 'fury_projects'],
    
    # 프로젝트 퀘스트 (동의어)
    'project_quests': ['fury_project_missions', 'fury_mission_configs'],
}

# 기본 테이블
DEFAULT_TABLE = 'fury_projects'
