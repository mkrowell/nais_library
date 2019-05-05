# nais_library
library for working with NAIS data

# Usage
Create Postgres tables from publicly available data. The AIS data is from https://marinecadastre.gov/.
```
import nais_library as nais

db = nais.NAIS_Database(zone='10', year='2017', password='pswrd')

# Build TSS, Shoreline, NAIS Points and NAIS Tracks tables in the postgres database.
db.build_tables()
```
