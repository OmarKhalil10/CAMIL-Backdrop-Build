"""
Send a request to flask application
"""
import requests

url = "https://camil-ltnijdawbq-oa.a.run.app"

# Method 1
resp = requests.get(f"{url}/", verify=False)
print(resp.content.decode())

# Method 2

r = requests.get("https://camil-ltnijdawbq-oa.a.run.app")
print(r)

# <Response [200]>
