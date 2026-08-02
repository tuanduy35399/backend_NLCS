import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from indexing.build_index import BuildIndex


print("Bat dau chay")
builder = BuildIndex()
builder.run()
