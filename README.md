# 🏆 Pointless Squash Stats

> *Where squash isn't just a vegetable - and your data tells the real story!*

Eine umfassende Streamlit-Anwendung zur Verwaltung und Analyse von Squash-Spielergebnissen. Perfekt für Turniere, Ligen oder einfach nur um deine Freunde mit detaillierten Statistiken zu beeindrucken.

## 🎯 Was macht die App?

**Pointless Squash Stats** ist deine digitale Squash-Zentrale mit zwei Hauptbereichen:

### 📊 1. Pointless Racquet Records
- **Spielergebnisse erfassen**: Einfache Eingabe von Matches mit Spielern, Scores und Datum
- **Zwei Modi**: Season Mode (reguläre Spiele) und Tournament Mode (Turnier-spezifisch)
- **Spielerverwaltung**: Automatische Spielerlisten oder manuelle Eingabe
- **Admin-Funktionen**: Bearbeiten und Löschen von Matches (passwortgeschützt)
- **Live-Synchronisation**: Alle Daten werden in Google Sheets gespeichert

### 🔍 2. Pointless Overanalysis Oasis  
- **Detaillierte Statistiken**: Umfassende Analyse aller Spielergebnisse
- **Erweiterte Metriken**: Win-Rate, Durchschnittswerte, Trends
- **Visualisierungen**: Interaktive Charts und Grafiken
- **Spielervergleiche**: Head-to-Head Statistiken
- **Historische Daten**: Entwicklung über Zeit

## 🚀 Features

- ✅ **Einfache Bedienung**: Intuitive Streamlit-Oberfläche
- ✅ **Cloud-Speicher**: Google Sheets als Backend-Datenbank
- ✅ **Zwei Modi**: Season und Tournament getrennt verwaltet
- ✅ **Responsive Design**: Funktioniert auf Desktop und Mobile
- ✅ **Sichere Admin-Funktionen**: Passwortgeschützte Bearbeitung
- ✅ **Automatische Backups**: Google Sheets speichert alles automatisch
- ✅ **Keine Datenbank nötig**: Läuft komplett über Google Sheets

## 🛠️ Installation & Setup

### Voraussetzungen
- Python 3.7+
- Google Account (für Google Sheets)

### 1. Repository klonen
```bash
git clone <repository-url>
cd squorsh
```

### 2. Virtual Environment erstellen
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# oder
venv\Scripts\activate  # Windows
```

### 3. Dependencies installieren
```bash
pip install -r requirements.txt
```

### 4. Google Sheet erstellen
1. Gehe zu [sheets.google.com](https://sheets.google.com)
2. Erstelle eine neue Tabelle (z.B. "Squash Tournament Daten")
3. Kopiere die URL aus der Adressleiste
4. Klicke "Teilen" → "Für alle Nutzer mit dem Link freigeben"
5. Berechtigung auf "Bearbeiter" setzen

### 5. Streamlit Secrets konfigurieren
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Bearbeite `.streamlit/secrets.toml` und trage deine Daten ein:
```toml
[connections.gsheets]
spreadsheet = "https://docs.google.com/spreadsheets/d/DEINE_SPREADSHEET_ID/edit#gid=0"

admin_password = "dein-sicheres-passwort"
```

### 6. App starten
```bash
./run_app.sh
```

Die App läuft dann auf `http://localhost:8502`

## 📁 Projektstruktur

```
squorsh/
├── Welcome_to_Pointless_Squash_Stats.py  # Hauptseite
├── pages/
│   ├── 1_Pointless_Racquet_Records.py    # Match-Eingabe
│   └── 2_Pointless_Overanalysis_Oasis.py # Statistiken
├── .streamlit/
│   └── secrets.toml.example              # Konfiguration
├── requirements.txt                       # Dependencies
├── run_app.sh                            # Start-Script
└── README.md                             # Diese Datei
```

## 🎮 Verwendung

### Matches eingeben
1. Gehe zu "Pointless Racquet Records"
2. Wähle Season oder Tournament Mode
3. Gib Spieler, Scores und Datum ein
4. Klicke "Log Match Result"

### Statistiken anzeigen
1. Gehe zu "Pointless Overanalysis Oasis"
2. Wähle den gewünschten Modus
3. Erkunde die verschiedenen Statistiken und Visualisierungen

### Matches bearbeiten
1. Scrolle in "Pointless Racquet Records" nach unten
2. Wähle ein Match aus der Liste
3. Gib das Admin-Passwort ein
4. Bearbeite die Daten und speichere

## 🔧 Konfiguration

### Google Sheets Arbeitsblätter
Die App erstellt automatisch diese Arbeitsblätter:
- `match_results` - Saison-Spielergebnisse
- `match_results_tournament` - Turnier-Spielergebnisse  
- `player_names` - Saison-Spielernamen
- `player_names_tournament` - Turnier-Spielernamen

### Sicherheit
- `secrets.toml` ist in `.gitignore` und wird nicht committed
- Verwende ein starkes Admin-Passwort
- Google Sheet sollte nur mit vertrauenswürdigen Personen geteilt werden

## 🐛 Troubleshooting

| Problem | Lösung |
|---------|--------|
| "Worksheet not found" | App erstellt Arbeitsblätter automatisch beim ersten Start |
| "Permission denied" | Prüfe Google Sheet Freigabe-Einstellungen |
| "Connection failed" | Prüfe Spreadsheet-URL und Internetverbindung |
| "IndexError" | Tritt bei leeren Spielerlisten auf - wurde bereits behoben |

## 🤝 Contributing

1. Fork das Repository
2. Erstelle einen Feature Branch
3. Committe deine Änderungen
4. Push zum Branch
5. Erstelle einen Pull Request

## 📝 License

Dieses Projekt steht unter der MIT License.

## 🎾 Credits

Entwickelt für alle Squash-Enthusiasten, die ihre Spiele genauso ernst nehmen wie ihre Statistiken!

*"Where numbers narrate the nonsense and amusement is always ace."*
