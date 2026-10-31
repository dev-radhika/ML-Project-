# Data Dictionary — poi_3_0

Each row represents a software class from the Apache POI 3.0 dataset version in the supplied PROMISE/BPD workbook.

| Column | Type | Role |
|---|---|---|
| ClassName | text | Class identifier; excluded from model input |
| CBO | numeric | Coupling Between Objects |
| DCC | numeric | Direct Class Coupling |
| ExportCoupling | numeric | Export/outgoing coupling metric |
| ImportCoupling | numeric | Import/incoming coupling metric |
| NOM | numeric | Number of Methods |
| WMC | numeric | Weighted Methods per Class |
| Defect | binary integer | Target: 0 or 1 |

Target labels represent historical dataset labels. They are not real-time bug findings.
