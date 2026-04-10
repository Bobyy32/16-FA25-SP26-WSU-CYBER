# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals

from builtins import str, bytes, dict, int
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import os
import sys
import hashlib
import json
import datetime
import uuid
import re
from collections import defaultdict
import pattern.db

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Database configuration
db = pattern.db.Database(pattern.db.pd("store.db"), type=pattern.db.SQLITE)

# PRODUCTS
if "products" not in db:
    schema = (
        pattern.db.pk(),
        pattern.db.field("description", pattern.db.STRING(50)),
        pattern.db.field("price", pattern.db.INTEGER)
    )
    db.create("products", schema)
    db.products.append(description="pizza", price=15)
    db.products.append(description="garlic bread", price=3)

# CUSTOMERS
if "customers" not in db:
    schema = (
        pattern.db.pk(),
        pattern.db.field("name", pattern.db.STRING(50)),
        pattern.db.field("address", pattern.db.STRING(200))
    )
    db.create("customers", schema)
    db.customers.append(name="Schrödinger")
    db.customers.append(name="Hofstadter")

# ORDERS
if "orders" not in db:
    schema = (
        pattern.db.pk(),
        pattern.db.field("product_id", pattern.db.INTEGER),
        pattern.db.field("customer_id", pattern.db.INTEGER),
        pattern.db.field("date", pattern.db.DATE, default=pattern.db.NOW)
    )
    db.create("orders", schema)
    db.orders.append(product_id=1, customer_id=2)

# === TRANSFORMATION TEMPLATE FUNCTIONS ===

def parse_user_input(user_input: str, metadata: Dict[str, int], 
                     tags: Set[str], flags: List[bool],
                     errors: Optional[List[str]] = None) -> Dict[str, Any]:
    """Parse and validate user input data."""
    if errors is None:
        errors = []
    
    try:
        # Basic validation
        if not user_input or not isinstance(user_input, str):
            errors.append("Invalid user input format")
            return {"status": "error", "message": errors}
        
        parsed = {
            "input": user_input,
            "metadata": metadata,
            "tags": list(tags),
            "flags": flags,
            "errors": errors
        }
        
        return parsed
    except Exception as e:
        errors.append(str(e))
        return {"status": "error", "message": errors}

def generate_user_data_hash(user_data: Dict[str, Any]) -> Optional[str]:
    """Hash user data for integrity verification."""
    if not user_data or not isinstance(user_data, dict):
        return None
    
    try:
        user_data_json = json.dumps(user_data, sort_keys=True)
        user_data_hash = hashlib.sha256(user_data_json.encode()).hexdigest()[:16]
        return user_data_hash
    except Exception:
        return None

def process_user_data(
    user_data: Dict[str, Any],
    metadata: Dict[str, int],
    tags: Set[str],
    flags: List[bool],
    errors: List[str],
    config: Dict[str, str],
    user_id: Optional[str],
    status: Optional[str],
    user_name: Optional[str],
    permissions: Optional[Set[str]],
    timestamps: Optional[List[str]],
    locations: Optional[List[str]],
    devices: Optional[List[str]],
    sessions: Optional[List[str]],
    browsers: Optional[List[str]],
    operating_systems: Optional[List[str]],
    ip_addresses: Optional[List[str]],
    user_agents: Optional[List[str]],
    referers: Optional[List[str]],
    referrers: Optional[List[str]],
    paths: Optional[List[str]],
    queries: Optional[List[str]],
    methods: Optional[List[str]],
    statuses: Optional[List[str]],
    bodies: Optional[List[str]],
    headers: Optional[Dict[str, List[str]]],
    cookies: Optional[Dict[str, Any]],
    query_params: Optional[List[str]],
    auth_params: Optional[List[str]],
    api_keys: Optional[List[str]],
    client_ips: Optional[List[str]],
    client_ua: Optional[List[str]],
    client_ref: Optional[List[str]],
    client_path: Optional[List[str]],
    client_query: Optional[List[str]],
    client_body: Optional[List[str]],
    client_headers: Optional[Dict[str, List[str]]],
    client_cookies: Optional[Dict[str, Any]],
    client_auth: Optional[Dict[str, Any]],
    client_api_keys: Optional[List[str]]
) -> Tuple[Optional[Union[str, int]], Optional[Union[str, int]], Optional[Union[str, int]]]:
    """Process user data and generate related outputs."""
    hash_result = generate_user_data_hash(user_data)
    parsed_input = parse_user_input(str(user_data), metadata, tags, flags, errors)
    
    return hash_result, parsed_input.get("status", "ok"), parsed_input.get("message", None)

# === DATABASE QUERY FUNCTIONS ===

def generate_invoices() -> List[Dict[str, Any]]:
    """Generate invoices with related customer and product data."""
    q = db.orders.search(
        fields=(
            "products.description",
            "products.price",
            "customers.name",
            "date"
        ),
        relations=(
            pattern.db.rel("product_id", "products.id", "products"),
            pattern.db.rel("customer_id", "customers.id", "customers")
        )
    )
    
    invoices = []
    for row in pattern.db.assoc(q):
        invoices.append({
            "description": row[0],
            "price": row[1],
            "customer": row[2],
            "date": row[3]
        })
    
    return invoices

def get_product_list() -> List[Dict[str, Any]]:
    """Get all available products."""
    products = []
    for row in pattern.db.assoc(db.products):
        products.append({
            "id": row[0],
            "description": row[1],
            "price": row[2]
        })
    return products

def get_customer_list() -> List[Dict[str, Any]]:
    """Get all customers."""
    customers = []
    for row in pattern.db.assoc(db.customers):
        customers.append({
            "id": row[0],
            "name": row[1],
            "address": row[2]
        })
    return customers

# === MAIN EXECUTION ===

if __name__ == "__main__":
    print("=" * 60)
    print("MINI-STORE DATABASE EXAMPLE")
    print("=" * 60)
    print("")
    
    print("Products available:")
    products = get_product_list()
    for product in products:
        print(f"  - {product['description']}: ${product['price']}")
    print("")
    
    print("Customers:")
    customers = get_customer_list()
    for customer in customers:
        print(f"  - {customer['name']} ({customer['address']})")
    print("")
    
    print("=" * 60)
    print("INVOICES:")
    print("=" * 60)
    invoices = generate_invoices()
    
    if invoices:
        for invoice in invoices:
            print("")
            print(f"Customer: {invoice['customer']}")
            print(f"Product: {invoice['description']} - ${invoice['price']}")
            print(f"Date: {invoice['date']}")
    else:
        print("No invoices found.")
    print("")
    
    print("Invoice query SQL syntax:")
    q = db.orders.search(
        fields=(
            "products.description",
            "products.price",
            "customers.name",
            "date"
        ),
        relations=(
            pattern.db.rel("product_id", "products.id", "products"),
            pattern.db.rel("customer_id", "customers.id", "customers")
        )
    )
    print(f"  SQL: {pattern.db.dump(q)}")
    print("")
    
    print("=" * 60)
    print("END OF EXAMPLE")
    print("=" * 60)