# nais_library
library for working with NAIS data

# Usage
```
import nais_library as nais

db = nais.NAIS_Database(zone='10', year='2017', password='pswrd')

# Build points and tracks tables in the postgres database.
db.build_tables()
```
