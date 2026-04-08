# Data Layout

Place your aerial dataset in the following structure:

```text
data/
|-- raw/
|   |-- images/
|   `-- annotations/
`-- processed/
    |-- train/
    |   |-- images/
    |   `-- masks/
    |-- val/
    |   |-- images/
    |   `-- masks/
    `-- test/
        |-- images/
        `-- masks/
```

Annotation JSON format:

```json
{
  "polygons": [
    {
      "label": "house",
      "points": [[10, 20], [90, 20], [90, 80], [10, 80]]
    }
  ]
}
```

To create a self-contained demo dataset instead of using real aerial images:

```powershell
python -m lab.prepare_dataset --generate-synthetic 120
```
