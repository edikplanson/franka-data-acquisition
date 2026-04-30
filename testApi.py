import requests

API_KEY = "a2ec289b-1807-431d-aff3-92a0e62808e9"

url = "https://api.tisseo.fr/v2/stops_schedules.json"

params = {
    "key": API_KEY,
    "stopPointId": "SP_Jean_Jaures"
}

data = requests.get(url, params=params).json()

departures = data.get("departures", {}).get("departure", [])

print("🚇 Prochains passages vers Balma Gramont :\n")

for d in departures:
    print(departures)
    destination = d.get("destination", "")
    if "Balma" in destination:
        print(d.get("dateTime"), "→", destination)