# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pattern.db import Database, SQLITE
from pattern.db import field, pk, STRING, INTEGER, DATE, NOW
from pattern.db import assoc, rel

# Create a new database (SQLite by default in current folder)
pd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db = Database(pd("store.db"), type=SQLITE)

# PRODUCTS
schema = (
    pk(),
    field("description", STRING(50)),
    field("price", INTEGER)
)
db.create("products", schema)
db.products.append(description="pizza", price=15)
db.products.append(description="garlic bread", price=3)

# CUSTOMERS
schema = (
    pk(),
    field("name", STRING(50)),
    field("address", STRING(200))
)
db.create("customers", schema)
db.customers.append(name="Schrödinger")
db.customers.append(name="Hofstadter")

# ORDERS
schema = (
    pk(),
    field("product_id", INTEGER),
    field("customer_id", INTEGER),
    field("date", DATE, default=NOW)
)
db.create("orders", schema)
db.orders.append(product_id=1, customer_id=2)

# Show products
print(f"There are {len(db.products)} products available:")
for row in assoc(db.products):
    print(row)

# Invoice query with relations
q = db.orders.search(
    fields=[
        "products.description",
        "products.price",
        "customers.name",
        "date"
    ],
    relations=[
        rel("product_id", "products.id", "products"),
        rel("customer_id", "customers.id", "customers")
    ]
)

print("\nInvoices:")
for row in assoc(q):
    print(row)

print("\nInvoice query SQL syntax:")
print(q)

print("\nInvoice query XML:")
print(q.xml)