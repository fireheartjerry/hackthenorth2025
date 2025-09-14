#!/usr/bin/env python3
"""
Test script to verify the new priority scoring system
"""
import requests
import json

def test_priority_accounts():
    """Test the priority accounts API endpoint"""
    try:
        url = "http://127.0.0.1:5050/api/priority-accounts"
        params = {
            'per_page': 10,
            'sort_by': 'priority_score',
            'sort_dir': 'desc'
        }
        
        print("🧪 Testing Priority Accounts API...")
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Response successful!")
            print(f"📊 Total accounts: {data['pagination']['total_count']}")
            print(f"📄 Current page: {data['pagination']['page']}")
            
            print("\n🎯 Top Priority Accounts:")
            print("=" * 80)
            
            for i, account in enumerate(data['data'][:5], 1):
                status_emoji = {'TARGET': '🎯', 'IN': '✅', 'OUT': '❌'}.get(account['appetite_status'], '❓')
                print(f"{i}. {account['account_name'][:30]:<30} | "
                      f"{status_emoji} {account['appetite_status']:<6} | "
                      f"Score: {account['priority_score']:.2f} | "
                      f"Premium: ${account['total_premium']:,.0f} | "
                      f"State: {account['primary_risk_state']}")
            
            # Check score distribution
            scores = [acc['priority_score'] for acc in data['data']]
            statuses = [acc['appetite_status'] for acc in data['data']]
            
            print(f"\n📈 Score Statistics:")
            print(f"   Min Score: {min(scores):.2f}")
            print(f"   Max Score: {max(scores):.2f}")
            print(f"   Avg Score: {sum(scores)/len(scores):.2f}")
            
            status_counts = {}
            for status in statuses:
                status_counts[status] = status_counts.get(status, 0) + 1
            
            print(f"\n📊 Status Distribution:")
            for status, count in sorted(status_counts.items()):
                percentage = (count / len(statuses)) * 100
                print(f"   {status}: {count} ({percentage:.1f}%)")
                
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Flask app. Make sure it's running on localhost:5050")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_classified_data():
    """Test the classified data to see overall scoring"""
    try:
        url = "http://127.0.0.1:5050/api/classified"
        params = {'sort_by': 'priority_score', 'sort_dir': 'desc'}
        
        print("\n🧪 Testing Classified Data API...")
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Response successful!")
            print(f"📊 Total submissions: {data['count']}")
            
            # Analyze score distribution
            scores = [sub.get('priority_score', 0) for sub in data['data']]
            statuses = [sub.get('appetite_status', 'UNKNOWN') for sub in data['data']]
            
            print(f"\n📈 Overall Score Statistics:")
            print(f"   Min Score: {min(scores):.2f}")
            print(f"   Max Score: {max(scores):.2f}")
            print(f"   Avg Score: {sum(scores)/len(scores):.2f}")
            print(f"   High Scores (>7): {sum(1 for s in scores if s > 7)}")
            print(f"   Medium Scores (4-7): {sum(1 for s in scores if 4 <= s <= 7)}")
            print(f"   Low Scores (<4): {sum(1 for s in scores if s < 4)}")
            
            status_counts = {}
            for status in statuses:
                status_counts[status] = status_counts.get(status, 0) + 1
            
            print(f"\n📊 Overall Status Distribution:")
            for status, count in sorted(status_counts.items()):
                percentage = (count / len(statuses)) * 100
                print(f"   {status}: {count} ({percentage:.1f}%)")
                
        else:
            print(f"❌ API Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 Testing New Priority Scoring System\n")
    test_priority_accounts()
    test_classified_data()
    print("\n✅ Testing completed!")