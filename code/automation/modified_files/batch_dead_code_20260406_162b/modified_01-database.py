# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals

from builtins import str, bytes, dict, int

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pattern.db import Database, SQLITE, MYSQL
from pattern.db import field, pk, STRING, INTEGER, DATE, NOW
from pattern.db import assoc
from pattern.db import rel
from pattern.db import pd

db = Database(pd("store.db"), type=SQLITE)

if "products" not in db:
    schema = (
        pk(),
        field("description", STRING(50)),
        field("price", INTEGER)
    )
    db.create("products", schema)
    db.products.append(description="pizza", price=15)
    db.products.append(description="garlic bread", price=3)

if "customers" not in db:
    schema = (
        pk(),
        field("name", STRING(50)),
        field("address", STRING(200))
    )
    db.create("customers", schema)
    db.customers.append(name="Schrödinger")
    db.customers.append(name="Hofstadter")

if "orders" not in db:
    schema = (
        pk(),
        field("product_id", INTEGER),
        field("customer_id", INTEGER),
        field("date", DATE, default=NOW)
    )
    db.create("orders", schema)
    db.orders.append(product_id=1, customer_id=2)

print("There are %s products available:" % len(db.products))
for row in assoc(db.products):
    print(row)

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

print("Invoice query SQL syntax:")
print(q)

print("Invoice query XML:")
print(q.xml)