# explore_data.py

import json
from collections import Counter, defaultdict

print("📊 Spider Dataset Analysis")
print("="*70)

# Load data
with open('../data/train.json', 'r') as f:
    train_data = json.load(f)

with open('../data/validation.json', 'r') as f:
    val_data = json.load(f)

print(f"\n📚 Total examples:")
print(f"- Train: {len(train_data)}")
print(f"- Validation: {len(val_data)}")

# Database statistics
print(f"\n🗄️ Database statistics:")
db_counts = Counter([item['db_id'] for item in train_data])
print(f"- Unique databases: {len(db_counts)}")
print(f"\nTop 5 databases:")
for db, count in db_counts.most_common(5):
    print(f"  {db}: {count} examples")

# SQL keyword analysis
print(f"\n🔍 SQL keyword distribution:")
keywords = defaultdict(int)
for item in train_data:
    query = item['query'].upper()
    if 'SELECT' in query:
        keywords['SELECT'] += 1
    if 'WHERE' in query:
        keywords['WHERE'] += 1
    if 'JOIN' in query:
        keywords['JOIN'] += 1
    if 'GROUP BY' in query:
        keywords['GROUP BY'] += 1
    if 'ORDER BY' in query:
        keywords['ORDER BY'] += 1
    if 'HAVING' in query:
        keywords['HAVING'] += 1

for keyword, count in sorted(keywords.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / len(train_data)) * 100
    print(f"  {keyword}: {count} ({percentage:.1f}%)")

# Query complexity
print(f"\n📏 Query complexity:")
simple = 0  # SELECT, WHERE only
medium = 0  # + JOIN or GROUP BY
complex_q = 0  # + subquery or multiple joins

for item in train_data:
    query = item['query'].upper()
    
    if 'SELECT' in query and 'FROM' in query:
        join_count = query.count('JOIN')
        has_subquery = '(' in query and 'SELECT' in query[query.index('('):]
        has_group = 'GROUP BY' in query
        
        if join_count >= 2 or has_subquery:
            complex_q += 1
        elif join_count == 1 or has_group:
            medium += 1
        else:
            simple += 1

print(f"  Simple (SELECT, WHERE): {simple} ({simple/len(train_data)*100:.1f}%)")
print(f"  Medium (+ JOIN/GROUP BY): {medium} ({medium/len(train_data)*100:.1f}%)")
print(f"  Complex (+ subquery/multi-join): {complex_q} ({complex_q/len(train_data)*100:.1f}%)")

# Question length
print(f"\n📝 Question length:")
lengths = [len(item['question'].split()) for item in train_data]
avg_length = sum(lengths) / len(lengths)
min_length = min(lengths)
max_length = max(lengths)

print(f"  Average: {avg_length:.1f} words")
print(f"  Min: {min_length} words")
print(f"  Max: {max_length} words")

# Sample by complexity
print(f"\n💡 Sample queries by complexity:")

print(f"\n[SIMPLE]")
for item in train_data:
    if 'JOIN' not in item['query'].upper() and 'GROUP BY' not in item['query'].upper():
        print(f"Q: {item['question']}")
        print(f"SQL: {item['query']}")
        break

print(f"\n[MEDIUM]")
for item in train_data:
    if 'JOIN' in item['query'].upper() or 'GROUP BY' in item['query'].upper():
        if item['query'].upper().count('JOIN') <= 1:
            print(f"Q: {item['question']}")
            print(f"SQL: {item['query']}")
            break

print(f"\n[COMPLEX]")
for item in train_data:
    if item['query'].upper().count('JOIN') >= 2 or '(' in item['query']:
        print(f"Q: {item['question']}")
        print(f"SQL: {item['query']}")
        break

print("\n" + "="*70)
print("✅ Analysis complete!")
