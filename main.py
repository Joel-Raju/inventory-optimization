from flask import Flask, request, jsonify
from sqlalchemy import (
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from db import Customer, Order, Product

DATABASE_URL = "postgres://postgres:123456@127.0.0.1:5432/test"


engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

Base = declarative_base()

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/products", methods=["GET"])
def get_products():
    session = Session()
    products = session.query(Product).all()
    session.close()
    return jsonify(
        [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "unit_price": p.unit_price,
            }
            for p in products
        ]
    )


@app.route("/products/<int:product_id>", methods=["GET"])
def get_product(product_id):
    session = Session()
    product = session.query(Product).get(product_id)
    session.close()
    if product:
        return jsonify(
            {
                "id": product.id,
                "name": product.name,
                "description": product.description,
                "unit_price": product.unit_price,
            }
        )
    return jsonify({"message": "Product not found"}), 404


@app.route("/products", methods=["POST"])
def create_product():
    data = request.get_json()
    session = Session()
    new_product = Product(
        name=data["name"],
        description=data["description"],
        unit_price=data["unit_price"],
    )
    session.add(new_product)
    session.commit()
    session.close()
    return jsonify({"id": new_product.id, "message": "Product created"}), 201


@app.route("/customers", methods=["GET"])
def get_customers():
    session = Session()
    customers = session.query(Customer).all()
    session.close()
    return jsonify([{"id": c.id, "name": c.name, "email": c.email} for c in customers])


@app.route("/orders", methods=["GET"])
def get_orders():
    session = Session()
    orders = session.query(Order).all()
    session.close()
    return jsonify(
        [
            {"id": o.id, "customer_id": o.customer_id, "order_date": o.order_date}
            for o in orders
        ]
    )


if __name__ == "__main__":
    app.run(debug=True)
