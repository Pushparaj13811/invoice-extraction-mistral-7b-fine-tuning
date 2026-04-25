from pydantic import BaseModel


class LineItem(BaseModel):
    description: str
    quantity: float
    unit_price: float
    line_total: float


class Invoice(BaseModel):
    vendor_name: str
    invoice_number: str
    invoice_date: str
    due_date: str
    total_amount: float
    currency: str
    tax_amount: float | None = None
    discount: float | None = None
    billing_address: str | None = None
    payment_terms: str | None = None
    line_items: list[LineItem] = []
