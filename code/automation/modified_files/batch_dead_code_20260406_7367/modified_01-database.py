# -*- coding: utf-8 -*-

from pattern.db import Database, SQLITE
from pattern.db import field, pk, STRING, INTEGER, DATE, NOW
from pattern.db import assoc
from pattern.db import rel

# Create a new database.
db = Database("store.db", type=SQLITE)

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

# Show all the products in the database.
print("There are %s products available:" % len(db.products))
for row in assoc(db.products):
    print(row)

# Orders table with related data
q = db.orders.search(
    fields = (
       "products.description",
       "products.price",
       "customers.name",
       "date"
    ),
    relations = (
        rel("product_id", "products.id", "products"),
        rel("customer_id", "customers.id", "customers")
    ))

print("")
print("Invoices:")
for row in assoc(q):
    print(row)

print("")
print("Invoice query SQL syntax:")
print(q)

print("")
print("Invoice query XML:")
print(q.xml)