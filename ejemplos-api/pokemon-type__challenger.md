# pokemon-type @ challenger

- URI: `models:/pokemon-type@challenger`
- Versión: 1
- Endpoint: `POST http://127.0.0.1:5001/invocations`

## Esquema de entrada

```json
[
  {
    "type": "double",
    "name": "HP",
    "required": true
  },
  {
    "type": "double",
    "name": "Attack",
    "required": true
  },
  {
    "type": "double",
    "name": "Defense",
    "required": true
  },
  {
    "type": "double",
    "name": "Sp. Atk",
    "required": true
  },
  {
    "type": "double",
    "name": "Sp. Def",
    "required": true
  },
  {
    "type": "double",
    "name": "Speed",
    "required": true
  },
  {
    "type": "double",
    "name": "Generation",
    "required": true
  }
]
```

## Esquema de salida

```json
[
  {
    "type": "string",
    "required": true
  }
]
```

## Petición

```bash
curl http://127.0.0.1:5001/invocations \
  -H 'Content-Type: application/json' \
  -d '{"dataframe_split": {"columns": ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed", "Generation"], "data": [[0.5441593383904934, -0.4103211105494686, -0.09718827258877333, 1.0910994158848046, -0.23195776969853676, 1.2025799941540203, 0.38032817305836625]]}}'
```

```json
{
  "dataframe_split": {
    "columns": [
      "HP",
      "Attack",
      "Defense",
      "Sp. Atk",
      "Sp. Def",
      "Speed",
      "Generation"
    ],
    "data": [
      [
        0.5441593383904934,
        -0.4103211105494686,
        -0.09718827258877333,
        1.0910994158848046,
        -0.23195776969853676,
        1.2025799941540203,
        0.38032817305836625
      ]
    ]
  }
}
```

## Respuesta

```json
{
  "predictions": [
    "Electric"
  ]
}
```
