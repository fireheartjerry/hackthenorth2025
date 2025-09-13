from app import app

c = app.test_client()

# Basic mode fetch with sorting
r = c.get('/api/classified_mode?mode=balanced_growth&sort_by=account_name&sort_dir=asc')
print('resp1', r.status_code)
j = r.get_json()
print('count1', j.get('count'), 'len1', len(j.get('data') or []))

# Verify filtering by appetite status returns data
r2 = c.get('/api/classified_mode?mode=balanced_growth&status=IN')
print('resp2', r2.status_code)
j2 = r2.get_json()
print('count2', j2.get('count'), 'len2', len(j2.get('data') or []))

