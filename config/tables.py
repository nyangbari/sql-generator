"""Table descriptions and metadata - Enhanced"""

TABLE_DESCRIPTIONS = {
    'fury_action_configs': {
        'description': '''
            Platform dashboard basic actions (Telegram, Discord, Twitter connections).
            System-wide missions available to all users.
        ''',
        'keywords': ['action', 'dashboard', 'platform', 'telegram', 'discord', 'twitter'],
        'use_cases': ['What dashboard actions?', 'Show platform missions']
    },
    
    'fury_mission_configs': {
        'description': '''
            Mission TYPE definitions (quiz, visit, mint, swap, vote).
            Defines categories of missions, not specific mission instances.
        ''',
        'keywords': ['mission type', 'mission category', 'kind', 'type'],
        'use_cases': ['What types of missions?', 'Mission categories']
    },
    
    'fury_projects': {
        'description': '''
            Main projects table. ALL registered projects.
            Use this to count projects or get project information.
        ''',
        'keywords': ['project', 'all projects', 'how many projects'],
        'use_cases': ['How many projects?', 'List projects']
    },
    
    'fury_airdrop_projects': {
        'description': '''
            Airdrop-specific subset. ONLY projects with airdrops.
            Use only when specifically asking about airdrops.
        ''',
        'keywords': ['airdrop'],
        'use_cases': ['Airdrop projects', 'How many airdrops?']
    },
    
    'fury_project_missions': {
        'description': '''
            ACTUAL missions for each project. Each row = one mission.
            To count missions for a project, use: COUNT(*) WHERE projectId = 'X'
            The missionSeq column is the mission identifier within a project.
            The groupId and weekId are for grouping/organization only.
        ''',
        'keywords': ['project mission', 'quest', 'what missions', 'how many missions', '퀘스트', '미션'],
        'use_cases': [
            'What missions does X have?',
            'How many quests in project Y?',
            'SuperWalk 퀘스트?',
            'List missions for 2pic'
        ],
        'columns_description': {
            'projectId': 'Project identifier (use for WHERE clause)',
            'missionSeq': 'Mission sequence number (unique per project)',
            'groupId': 'Grouping identifier (NOT for counting missions!)',
            'weekId': 'Week identifier (NOT for counting missions!)'
        }
    },
    
    'fury_project_weeks': {
        'description': '''
            Campaign/week schedule for projects. Contains start and end dates.
            Use this table to find when a project's campaign starts or ends.
            Each project can have multiple weeks/campaigns.
        ''',
        'keywords': ['campaign', 'week', 'schedule', 'when', 'start', 'end', '언제', '시작', '종료'],
        'use_cases': [
            'When does X campaign end?',
            'When did Y start?',
            'Show me campaign schedule'
        ],
        'columns_description': {
            'projectId': 'Project identifier',
            'weekId': 'Week/campaign identifier',
            'weekName': 'Campaign name',
            'startDate': 'Campaign start date',
            'endDate': 'Campaign end date (use this for "when does it end")'
        }
    },
    
    'fury_user_project_missions': {
        'description': '''
            User completion tracking. Which users completed which missions.
        ''',
        'keywords': ['user completed', 'user progress', 'who finished'],
        'use_cases': ['Who completed mission X?', 'User progress']
    },
    
    'fury_users': {
        'description': '''
            User accounts and profiles.
        ''',
        'keywords': ['user', 'account', 'how many users'],
        'use_cases': ['How many users?', 'List users']
    },
}

TABLE_PRIORITY = {
    'project_count': ['fury_projects', 'fury_airdrop_projects'],
    'airdrop_count': ['fury_airdrop_projects', 'fury_projects'],
    'user_count': ['fury_users'],
    'mission_types': ['fury_mission_configs', 'fury_action_configs'],
    'platform_missions': ['fury_action_configs'],
    'project_missions': ['fury_project_missions', 'fury_mission_configs', 'fury_projects'],
    'project_quests': ['fury_project_missions', 'fury_mission_configs'],
    'campaign_dates': ['fury_project_weeks', 'fury_projects'],  # New!
}

DEFAULT_TABLE = 'fury_projects'
