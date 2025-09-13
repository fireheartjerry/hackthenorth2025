import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()


AUTH_URL = os.environ.get(
    "FEDERATO_AUTH_URL", "https://product-federato.us.auth0.com/oauth/token"
)
AUDIENCE = os.environ.get("FEDERATO_AUDIENCE", "https://product.federato.ai/core-api")
CLIENT_ID = os.environ.get("FEDERATO_CLIENT_ID")
CLIENT_SECRET = os.environ.get("FEDERATO_CLIENT_SECRET")
POLICIES_URL_TYPO = os.environ.get(
    "FEDERATO_POLICIES_URL_TYPO",
    "https://product.federato.ai/integrations-api/handlers/all-pollicies?outputOnly=true",
)
POLICIES_URL = os.environ.get(
    "FEDERATO_POLICIES_URL",
    "https://product.federato.ai/integrations-api/handlers/all-policies?outputOnly=true",
)


def get_access_token():
    token = os.environ.get("FEDERATO_TOKEN")
    if token:
        return token
    if not CLIENT_ID or not CLIENT_SECRET:
        raise RuntimeError("Set FEDERATO_TOKEN or CLIENT_ID/SECRET in .env")
    r = requests.post(
        AUTH_URL,
        json={
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "audience": AUDIENCE,
            "grant_type": "client_credentials",
        },
        headers={"Content-Type": "application/json"},
    )
    r.raise_for_status()
    return r.json().get("access_token")


def fetch_policies(token: str):
    h = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    r = requests.post(POLICIES_URL_TYPO, headers=h, json={})
    if r.status_code in (404, 405):
        r = requests.post(POLICIES_URL, headers=h, json={})
    r.raise_for_status()
    return r.json()


def save_data():
    token = get_access_token()
    data = fetch_policies(token)
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("Saved data.json with", len(data.get("data", [])), "records (or wrapped output)")


if __name__ == "__main__":
    save_data()

