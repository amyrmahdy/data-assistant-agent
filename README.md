## KPI-Report Service — Quick README

A super-small FastAPI endpoint that converts KPI JSON ( + optional targets ) into a neat Markdown report via an autonomous LLM workflow.

---

### ① Install

```bash
pip install fastapi uvicorn autogen
```

*(Python 3.9+ required)*

---

### ② Run the API

```bash
uvicorn service:app --host 0.0.0.0 --port 8000
```

The service now listens on **http\://\<your-ip>:8000/report**

---

### ③ Call with cURL

```bash
curl -X POST http://localhost:8000/report \
     -H "Content-Type: application/json" \
     -d '{
           "data":[
             { "name":"impressions",
               "data":[
                 {"date":"2025-07-01","value":108200},
                 {"date":"2025-07-08","value":114500}
               ]},
             { "name":"conversions",
               "data":[
                 {"date":"2025-07-01","value":1040},
                 {"date":"2025-07-08","value":1105}
               ]}
           ],
           "targets":"impressions = 150000\nconversions = 1500"
         }'
```

Response:

```json
{
  "report": "## Executive Summary\n…\n\n## KPI Details\n…\n\n## Recommendations\n…\n\n**REPORT_DONE**"
}
```

---

### Request schema

| key       | type              | required | notes                                 |
| --------- | ----------------- | -------- | ------------------------------------- |
| `data`    | list of KPI blobs | yes      | each blob: `{ "name", "data":[{…}] }` |
| `targets` | plain-text string | no       | one “kpi = value” per line            |
