# AROGYA Website — Setup Instructions

## Folder Structure
```
arogya_website/
├── app.py
├── requirements.txt
├── templates/
│   └── index.html
└── logs/              (created automatically)
```

## Setup Steps

### 1. Install Python (if not installed)
Download from python.org — version 3.9 or 3.10 recommended

### 2. Install dependencies
Open Command Prompt in the arogya_website folder:
```
pip install -r requirements.txt
```

### 3. Run the website
```
python app.py
```

### 4. Open in browser
```
http://localhost:5000
```

## Notes
- Models load automatically on startup (takes 30-60 seconds)
- All patient data saved to logs/patients.jsonl
- No internet needed after first run
- Works on any PC or laptop
